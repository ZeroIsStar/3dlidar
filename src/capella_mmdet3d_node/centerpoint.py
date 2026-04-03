#!/usr/bin/env python3
import os

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'


import tempfile
import numpy as np
import math
import io

import time
import torch
import threading
import copy
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import rclpy.duration
from sensor_msgs.msg import PointCloud2, PointField
from ament_index_python.packages import get_package_share_directory
from mmdet3d.apis import inference_detector, init_model

# TF2 相关导入
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, Point, Twist
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA

# 用这个映射表把 msg.fields 里的 datatype 转换成 NumPy 能理解的 dtype
_PC_DTYPE_MAP = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2',
                 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}

# 将模型输出的原始类名映射到业务类名（car / truck）
CLASS_MAPPING = {
    'car':                  'car',
    'Car':                  'car',
    'truck':                'truck',
    'Truck':                'truck',
    # 'construction_vehicle': 'truck',
    # 'bus':                  'truck',
    # 'trailer':              'truck',
}

class VehicleTrack:
    """车辆跟踪数据结构"""
    
    def __init__(self, first_center, class_name, size, track_id):
        self.id = track_id          # 显式传入ID
        self.first_center = np.array(first_center)  # 基框中心
        self.last_center = np.array(first_center)   # 上一帧位置
        self.class_name = class_name
        self.size = size  # (l, w, h) 动态生成的基框尺寸
        self.missed_frames = 0
        self.is_matched = False
        # 静态车验证相关
        # is_moving: 当前帧是否被判定为动态（中心点出基框）
        # 注意：这只是瞬时状态，连续3帧 is_moving=True 才会真正转到 dynamic_tracks
        # 在转到 dynamic_tracks 之前，该目标仍保留在 static_tracks 中
        self.is_moving = False
        self.first_yaw = 0.0          # 首次检测时map系下的朝向角
        # 当前状态
        self.curr_center = np.array(first_center)   # 当前map系位置
        self.curr_yaw = 0.0                         # 当前map系朝向
        self.curr_size = size                       # 当前尺寸
        # 首次检测的原始尺寸
        self.first_size = None                      # 将在创建时设置
        # 穿透确认帧数计数器
        self.penetration_confirm_frames = 0
        # 运动怀疑帧数计数器
        self.moving_suspect_frames = 0

class Mmdet3dNode(Node):
    def __init__(self):
        super().__init__('mmdet3d_node')

        # ---- 参数声明 ----
        _pkg_dir = os.path.dirname(os.path.abspath(__file__))
        _demo_pth = os.path.join(_pkg_dir, 'configs', 'pth', 'centerpoint_voxel0075_nus.pth')
        
        # 优先使用包内bundled config，留空时才回退到从pth自动提取
        _bundled_cfg = os.path.join(
            _pkg_dir, 'configs', 'centerpoint',
            'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'
        )
        _default_cfg = _bundled_cfg if os.path.exists(_bundled_cfg) else ''
        self.declare_parameter('config_file',       _default_cfg)
        self.declare_parameter('checkpoint_file',   _demo_pth)
        self.declare_parameter('infer_device',      'cuda:0')

        self.declare_parameter('score_threshold',   0.4)  # 置信度
        self.declare_parameter('pointcloud_topic',  '/vanjee/lidar')
        self.declare_parameter('point_cloud_qos',   'best_effort')
        self.declare_parameter('target_frame',      'map')  # 决定了所有数据从哪里开始转                                                                                 
        # 各话题独立坐标系控制
        self.declare_parameter('vehicle_boxes_frame',        'map')  # /vehicle_boxes 话题坐标系
        self.declare_parameter('vehicle_raw_cloud_frame',    'map')  # /vehicle_raw_cloud 话题坐标系
        self.declare_parameter('vehicle_outlines',  'map')                # /vehicle_outlines 话题坐标系
        # ID匹配、运动判断参数
        self.declare_parameter('match_distance_threshold', 3.0) # 运动超过3m
        self.declare_parameter('max_missed_frames', 6)  # 多少帧看不到就删除id
        self.declare_parameter('process_every_n_frames', 1)  # 处理数据的帧数
        
        # 基框延展参数
        self.declare_parameter('base_box_expand_x', 2.0)  # X方向
        self.declare_parameter('base_box_expand_y', 4.0)  # Y方向
        
        # 静态车消失确认参数
        self.declare_parameter('lidar_max_range', 30.0)    # 激光雷达最大有效量程
        self.declare_parameter('max_static_tracks', 200)  # 静态车记忆上限
        self.declare_parameter('contour_slice_z', 1.0)    # 发布点云的切面高度
        self.declare_parameter('penetration_threshold', 0.5)  # 激光穿透比例阈值，超过此值认为车辆已消失
        
        # 抖动检测参数
        self.declare_parameter('jitter_linear_k', 3.0)    # 预期位移偏移多大
        self.declare_parameter('jitter_epsilon', 0.1)     # 最小位移容差(m)
        self.declare_parameter('jitter_stable_frames', 10)  # 连续稳定多少次才解除保护(20hz=0.5s)

        config_file     = self.get_parameter('config_file').value
        checkpoint_file = self.get_parameter('checkpoint_file').value
        device          = self.get_parameter('infer_device').value
        self.score_thr  = self.get_parameter('score_threshold').value
        self.pc_topic   = self.get_parameter('pointcloud_topic').value
        pc_qos_str      = self.get_parameter('point_cloud_qos').value
        self.target_frame = self.get_parameter('target_frame').value
        self.vehicle_boxes_frame = self.get_parameter('vehicle_boxes_frame').value
        self.vehicle_raw_cloud_frame = self.get_parameter('vehicle_raw_cloud_frame').value
        self.completed_cloud_map_frame = self.get_parameter('vehicle_outlines').value
        self.process_every_n_frames = self.get_parameter('process_every_n_frames').value
        
        self.match_threshold = self.get_parameter('match_distance_threshold').value
        self.max_missed_frames = self.get_parameter('max_missed_frames').value
        self.base_box_expand_x = self.get_parameter('base_box_expand_x').value
        self.base_box_expand_y = self.get_parameter('base_box_expand_y').value
        self.lidar_max_range = self.get_parameter('lidar_max_range').value
        self.max_static_tracks = self.get_parameter('max_static_tracks').value
        self.contour_slice_z = self.get_parameter('contour_slice_z').value
        self.penetration_threshold = self.get_parameter('penetration_threshold').value
        
        # 读取抖动检测参数
        self.jitter_linear_k = self.get_parameter('jitter_linear_k').value
        self.jitter_epsilon = self.get_parameter('jitter_epsilon').value
        self.jitter_stable_frames = self.get_parameter('jitter_stable_frames').value

        # ---- 加载模型 ----
        self._tmp_config_file = None
        if not config_file:
            ck = torch.load(checkpoint_file, map_location='cpu')
            config_str = ck.get('meta', {}).get('config', None)
            tmp = tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, prefix='mmdet3d_auto_cfg_')
            tmp.write(config_str)
            tmp.close()
            config_file = tmp.name
            self._tmp_config_file = tmp.name

        self.model = init_model(config_file, checkpoint_file, device=device)
        self.class_names: list[str] = list(self.model.dataset_meta.get('classes', []))
        self.get_logger().info(f'Model device: {next(self.model.parameters()).device}')


        # ---- Callback Groups ----
        self._infer_group = MutuallyExclusiveCallbackGroup()
        self._timer_group = MutuallyExclusiveCallbackGroup()
        self._jitter_group = MutuallyExclusiveCallbackGroup()  # 新增抖动检测回调组

        # ---- QoS: KEEP_LAST depth=1 + 根据参数设置 reliability ----
        if pc_qos_str.lower() == 'reliable':
            _reliability = ReliabilityPolicy.RELIABLE
        elif pc_qos_str.lower() == 'best_effort':
            _reliability = ReliabilityPolicy.BEST_EFFORT
        else:
            self.get_logger().warning(
                f"Invalid point_cloud_qos '{pc_qos_str}', fallback to BEST_EFFORT"
            )
            _reliability = ReliabilityPolicy.BEST_EFFORT
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=_reliability
        )

        # ---- 订阅原始点云 ----
        self.pc_sub = self.create_subscription(
            PointCloud2, self.pc_topic, self._pointcloud_callback, qos, callback_group=self._infer_group)
        
        # ---- 订阅cmd_vel用于抖动检测 ----
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_callback, 10, callback_group=self._jitter_group)
        
        # ---- 发布map系点云 ----
        self.completed_cloud_map_pub = self.create_publisher(
            PointCloud2, '/vehicle_outlines', 10)
        
        # ---- 发布车辆原始点云----
        self.vehicle_raw_cloud_pub = self.create_publisher(
            PointCloud2, '/vehicle_raw_cloud', 10)
        
        # ---- 发布先验长方体MarkerArray ----
        self.vehicle_boxes_pub = self.create_publisher(
            MarkerArray, '/vehicle_boxes', 10)

        # ---- TF2 初始化 ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # ---- 跟踪管理：动静分离 ----
        self.static_tracks = {}         # 只存静态车 {static_id: VehicleTrack}
        self.next_static_id = 0         # 静态车ID计数器
        
        self.dynamic_tracks = {}        # 动态车字典 {dyn_id: dict}，有跨帧ID但无保护，未匹配即删除
        self.next_dynamic_id = 0        # 动态车ID计数器
        
        # ---- 抖动检测相关状态变量 ----
        self.latest_cmd_linear = 0.0
        self.latest_cmd_angular = 0.0
        
        self.is_jittering = False
        self.jitter_frozen_static = None    # 抖动开始时的static_tracks快照
        self.jitter_stable_count = 0        # 连续稳定帧计数
        
        self.last_stable_translation = None   # np.ndarray shape(3,) 上一稳定帧平移
        self.last_stable_time = None          # rclpy Time
        
        self.jitter_lock = threading.Lock()
        
        self._last_lidar_frame_id = None   # 保存最后一次lidar的frame_id供抖动检测使用
        
        self._frame_counter = 0 
        self._last_marker_ids = set()
        # 缓存最新一帧的 MarkerArray，供定时器持续刷新用
        self._cached_marker_array = MarkerArray()
        self._cached_timestamp = None  # 缓存原始点云的时间戳
        self._cached_marker_lock = threading.Lock()
        # 日志限流：每3秒刷新一次
        self._last_timing_log_time = 0.0
        self._last_status_log_time = 0.0

        # ---- 可视化定时器：0.2s 刷一次，不依赖推理帧 ----
        self.viz_timer = self.create_timer(0.2, self._publish_cached_markers, callback_group=self._timer_group)
        
        # ---- 抖动检测定时器：10Hz ----
        self.jitter_timer = self.create_timer(0.1, self._tf_monitor_callback, callback_group=self._jitter_group)

        self.get_logger().info(
            f'VehicleDetectionNode ready | match_thr={self.match_threshold}m | '
            f'lidar_max_range={self.lidar_max_range}m | '
            f'jitter_k={self.jitter_linear_k}, epsilon={self.jitter_epsilon}m'
        )

    def _cmd_vel_callback(self, msg):
        """接收速度指令，用于抖动检测的预期位移计算"""
        self.latest_cmd_linear = msg.linear.x
        self.latest_cmd_angular = msg.angular.z
    
    def _tf_monitor_callback(self):
        """20Hz抖动检测核心函数，监测定位跳变并冻结/恢复静态车数据"""
        # 获取当前最新TF
        try:
            if self._last_lidar_frame_id is None:
                return
            
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self._last_lidar_frame_id,
                rclpy.time.Time()
            )
        except Exception as e:
            return
        
        # 提取平移向量
        t = transform.transform.translation
        current_translation = np.array([t.x, t.y, t.z])
        now = self.get_clock().now()
        
        # 初始化基准（第一次运行）
        if self.last_stable_translation is None:
            self.last_stable_translation = current_translation
            self.last_stable_time = now
            return
        
        # 计算实际位移和预期位移
        dt = (now - self.last_stable_time).nanoseconds / 1e9
        
        actual_displacement = np.linalg.norm(
            current_translation - self.last_stable_translation
        )
        
        expected_displacement = abs(self.latest_cmd_linear) * dt
        # 判断是否抖动  =  实际位移超过预期位移的k倍 + epsilon
        is_jitter = (
            actual_displacement >
            expected_displacement * self.jitter_linear_k + self.jitter_epsilon
        )
        
        # 状态机处理
        if not self.is_jittering:
            if is_jitter:
                # 进入抖动保护
                with self.jitter_lock:
                    self.is_jittering = True
                    self.jitter_frozen_static = copy.deepcopy(self.static_tracks)
                    self.dynamic_tracks.clear()    # 动态车全部清空
                self.jitter_stable_count = 0
                self.get_logger().warn(
                    f'[抖动检测] 定位跳变！'
                    f'实际位移={actual_displacement:.3f}m '
                    f'预期位移={expected_displacement:.3f}m '
                    f'冻结static_tracks，清空dynamic_tracks'
                )
                # 不更新last_stable_translation，保持抖动前的基准
            else:
                # 正常状态，更新稳定基准
                self.last_stable_translation = current_translation
                self.last_stable_time = now
        else:
            # 当前处于抖动保护期
            if not is_jitter:
                # TF趋于稳定，累计稳定帧数
                self.jitter_stable_count += 1
                
                if self.jitter_stable_count >= self.jitter_stable_frames:
                    # 解除保护，恢复冻结的static_tracks，重置时间基准
                    with self.jitter_lock:
                        self.is_jittering = False
                        self.static_tracks = self.jitter_frozen_static
                        self.jitter_frozen_static = None
                    self.last_stable_translation = current_translation
                    self.last_stable_time = now  # 重置时间基准，避免dt过大
                    self.get_logger().info(
                        f'[抖动检测] 定位恢复稳定，'
                        f'已恢复冻结的static_tracks'
                    )
            else:
                # 抖动仍在持续，重置稳定计数
                self.jitter_stable_count = 0
                # 不更新last_stable_translation，继续保持冻结

    # 读取点云
    def _read_pointcloud(self, msg: PointCloud2) -> np.ndarray | None:
        """
        将 PointCloud2 转换为 (N, 5) float32 数组 [x, y, z, intensity, ring].
        """
        n_pts = msg.width * msg.height
        if n_pts == 0:
            return None

        # 根据字段描述构建结构化 dtype，自动处理字段间的填充字节
        fields_sorted = sorted(msg.fields, key=lambda f: f.offset)
        dt_list = []
        pos = 0
        for f in fields_sorted:
            if f.offset > pos:
                dt_list.append(('_pad%d' % pos, 'u1', f.offset - pos))
            np_type = _PC_DTYPE_MAP.get(f.datatype, 'u1')
            dt_list.append((f.name, np_type))
            pos = f.offset + np.dtype(np_type).itemsize
        if msg.point_step > pos:
            dt_list.append(('_pad_end', 'u1', msg.point_step - pos))

        arr = np.frombuffer(msg.data, dtype=np.dtype(dt_list))

        field_names = {f.name for f in msg.fields}
        has_intensity = 'intensity' in field_names
        has_ring      = 'ring'      in field_names

        x = arr['x'].astype(np.float32)
        y = arr['y'].astype(np.float32)
        z = arr['z'].astype(np.float32)

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        n_valid = int(valid.sum())
        if n_valid == 0:
            return None

        cols = [x[valid], y[valid], z[valid]]
        cols.append(arr['intensity'].astype(np.float32)[valid] if has_intensity
                    else np.zeros(n_valid, dtype=np.float32))
        cols.append(arr['ring'].astype(np.float32)[valid] if has_ring
                    else np.zeros(n_valid, dtype=np.float32))

        return np.column_stack(cols)

    def _extract_transform_matrix(self, transform: TransformStamped) -> tuple:
        """从TransformStamped中提取旋转矩阵和平移向量"""
        t = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ])
        
        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w
        
        R = np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
            [    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx)],
            [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        
        return R, t


    def _id_tracks(self, detections, tracks_dict):
        """
        ID匹配逻辑：找同类别且距离最近的
        只对static_tracks做匹配
        """
        matches = []  # [(det_idx, track_id), ...]    # 成功匹配上的
         # 没匹配上的
        unmatched_dets = []    
        unmatched_tracks = set(tracks_dict.keys())
        
        if not tracks_dict:
            return [], list(range(len(detections))), set()
        
        det_used = set()   # 哪些 detection 已经匹配成功了
        matched_track_ids = set()  # 已匹配的track不再参与匹配
        
        for det_idx, det in enumerate(detections):
            det_center = np.array(det['center'])
            det_class = det['class_name']
            
            best_track_id = None
            min_dist = float('inf')
            
            # 找同类别且距离最近的未匹配track
            for track_id, track in tracks_dict.items():
                if track_id in matched_track_ids:
                    continue  # 该track已被其他检测占用
                if track.class_name != det_class:
                    continue
                
                dist = np.linalg.norm(det_center - track.last_center)
                if dist < self.match_threshold and dist < min_dist:
                    min_dist = dist
                    best_track_id = track_id
            
            if best_track_id is not None:
                matches.append((det_idx, best_track_id))
                det_used.add(det_idx)
                matched_track_ids.add(best_track_id)
                unmatched_tracks.discard(best_track_id)
        
        unmatched_dets = [i for i in range(len(detections)) if i not in det_used]
        return matches, unmatched_dets, unmatched_tracks

    def _match_dynamic_track(self, center, dynamic_tracks, det_class):
        """
        为动态车匹配现有的dynamic_tracks
        返回匹配到的dyn_id或None
        """
        center_np = np.array(center)
        best_id = None
        min_dist = float('inf')
        
        for dyn_id, dyn_info in dynamic_tracks.items():
            # 类别一致
            if dyn_info['label'] != det_class:
                continue
            dist = np.linalg.norm(center_np - np.array(dyn_info['center']))
            if dist < self.match_threshold and dist < min_dist:
                min_dist = dist
                best_id = dyn_id
        
        return best_id

    def _is_moving_check(self, current_center, first_center, prior_size, first_yaw):
        """
        判断当前检测框中心点是否在基框内部。
        如果中心点在基框XY投影范围内，认为是静态；否则认为是动态。
        
        Args:
            current_center: 当前检测框中心点 (x, y, z)
            first_center: 基框中心点 (x, y, z)
            prior_size: 基框尺寸 (l, w, h)
            first_yaw: 基框朝向角
        
        Returns:
            is_moving: bool, True表示中心点在基框外（动态）
            center_inside_base: bool, True表示中心点在基框内（静态）
        """
        # 计算当前中心点相对基框中心的位移
        dx = current_center[0] - first_center[0]
        dy = current_center[1] - first_center[1]
        
        # 用 first_yaw 把相对位移旋转到基框局部坐标系
        cos_b, sin_b = math.cos(-first_yaw), math.sin(-first_yaw)
        local_x = dx * cos_b - dy * sin_b
        local_y = dx * sin_b + dy * cos_b
        
        # 判断当前中心点是否在基框内部（只判断XY平面）
        bl, bw, bh = prior_size
        inside_x = abs(local_x) <= bl / 2
        inside_y = abs(local_y) <= bw / 2
        center_inside_base = inside_x and inside_y
        
        # 中心点在基框内 -> 静态；在基框外 -> 动态
        is_moving = not center_inside_base
        
        return is_moving, center_inside_base

    def _lidar_can_confirm_empty(self, pts_lidar: np.ndarray,
                                   region_center_lidar: np.ndarray,
                                   region_size: tuple,
                                   _precomputed=None) -> bool:
        """
        激光雷达能否确认目标区域是空的？通过判断该区域所在方向的激光是否大量穿透来确认。
        改进：过滤地面点，只看车体高度范围内的点。
        """
        d_target = float(np.linalg.norm(region_center_lidar))
        # 超出量程或目标过近，无法判断
        if d_target > self.lidar_max_range or d_target < 0.5:
            return False

        direction = region_center_lidar / d_target

        # 角度容差：目标区域最大水平半径对应的张角
        l, w, h = region_size
        half_span = max(l, w) / 2.0
        angle_tol = math.atan2(half_span, d_target)  # radians

        # 使用预计算的点云方向数据
        if _precomputed is not None:
            pts_valid, dists_valid, dirs = _precomputed
        else:
            dists = np.linalg.norm(pts_lidar, axis=1)
            valid = dists > 0.1
            pts_valid = pts_lidar[valid]
            dists_valid = dists[valid]
            if len(pts_valid) == 0:
                return False
            dirs = pts_valid / dists_valid[:, np.newaxis]

        if len(pts_valid) == 0:
            return False

        cos_angles = dirs @ direction
        cos_tol = math.cos(angle_tol)
        in_cone = cos_angles >= cos_tol

        if np.sum(in_cone) == 0:
            return False  # 该方向完全没有扫描数据，无法判断

        # 过滤地面点 — 只看车体高度范围内的点
        cone_pts = pts_valid[in_cone]
        cone_dists = dists_valid[in_cone]
        target_z = region_center_lidar[2]
        z_min = target_z - h / 2.0 + 0.3   # 车底 + 30cm余量（跳过地面）
        z_max = target_z + h / 2.0 + 0.3
        above_ground = (cone_pts[:, 2] > z_min) & (cone_pts[:, 2] < z_max)
        
        cone_dists_filtered = cone_dists[above_ground]
        
        # 锥体内几乎没有有效点 → 无法判断
        if len(cone_dists_filtered) == 0:
            return False
        
        # 区域内非地面点很少 → 确认为空
        d_near = d_target - half_span
        d_far = d_target + half_span
        n_in_region = np.sum(
            (cone_dists_filtered >= d_near) & (cone_dists_filtered <= d_far)
        )
        
        if n_in_region <= 3:
            return True  # 几乎没有有效点 = 空的
        
        n_through = np.sum(cone_dists_filtered > d_far)
        if n_through == 0:
            return False
        
        ratio = float(n_through) / (n_through + n_in_region)
        return ratio >= self.penetration_threshold
    
    def create_box_marker(self, center, size, color, marker_id, ns, frame_id, header_stamp, yaw=0.0):
        """创建2D footprint Marker"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = header_stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        half_yaw = yaw * 0.5
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = math.sin(half_yaw)
        marker.pose.orientation.w = math.cos(half_yaw)

        cx, cy, cz = center
        l, w, h = size
        marker.pose.position.x = cx
        marker.pose.position.y = cy
        marker.pose.position.z = cz
        marker.scale.x = l
        marker.scale.y = w
        marker.scale.z = 0.1  # 压扁为2D平面外形

        r, g, b, a = color
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a

        return marker

    def create_line_marker(self, start, end, color, marker_id, frame_id, header_stamp):
        """创建连线Marker"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = header_stamp
        marker.ns = "displacement_lines"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.pose.orientation.w = 1.0  
        
        r, g, b, a = color
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a
        
        p1 = Point()
        p1.x = start[0]
        p1.y = start[1]
        p1.z = start[2]
        marker.points.append(p1)
        
        p2 = Point()
        p2.x = end[0]
        p2.y = end[1]
        p2.z = end[2]
        marker.points.append(p2)
        
        return marker

    def create_center_point_marker(self, center, color, marker_id, ns, frame_id, header_stamp):
        """创建中心点 Sphere Marker"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = header_stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        cx, cy, _ = center
        marker.pose.position.x = cx
        marker.pose.position.y = cy
        marker.pose.position.z = self.contour_slice_z  # 固定高度，便于可视化
        marker.pose.orientation.w = 1.0
        
        # 中心点尺寸：一个清晰但不过分夸张的球体
        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.scale.z = 0.35
        
        r, g, b, a = color
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a
        
        return marker

    # 推理 + 过滤 + 跟踪 + 运动判断
    def _pointcloud_callback(self, msg: PointCloud2):
        # ---- 跳帧逻辑 ----
        self._frame_counter += 1
        if self._frame_counter % self.process_every_n_frames != 0:
            return
        
        # 保存lidar frame_id供抖动检测使用
        self._last_lidar_frame_id = msg.header.frame_id
        
        # 直接执行推理（在 _infer_group 中执行，不会阻塞 _timer_group）
        try:
            self._process_pointcloud(msg)
        except Exception as e:
            self.get_logger().error(f'Inference callback error: {e}')

    def _process_pointcloud(self, msg: PointCloud2):
        # ---- 抖动检查：如果处于抖动保护期，直接返回 ----
        with self.jitter_lock:
            if self.is_jittering:
                return
        
        # ---- 重置static_tracks的is_matched标志 ----
        for track in self.static_tracks.values():
            track.is_matched = False
        
        # ---- 重置dynamic_tracks的is_matched标志 ----
        for dyn in self.dynamic_tracks.values():
            dyn['is_matched'] = False
        
        _t0 = time.perf_counter()
        # ---- 获取TF变换 ----
        source_frame = msg.header.frame_id
        target_frame = self.target_frame
        
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time.from_msg(msg.header.stamp),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )
        except Exception as e:
            self.get_logger().debug(f'TF lookup failed: {e}')
            return
        
        R, t = self._extract_transform_matrix(transform)
        _t1 = time.perf_counter()
        
        # ---- 读取点云 ----
        pts = self._read_pointcloud(msg)
        if pts is None or len(pts) < 10:
            return

        # 记录原始点云数量
        n_pts_raw = len(pts)

        _t2 = time.perf_counter()

        # 距离过滤：只保留 lidar_max_range 范围内的点
        pts_xyz = pts[:, :3]
        dists = np.linalg.norm(pts_xyz, axis=1)
        mask = dists <= self.lidar_max_range
        pts = pts[mask]

        # 记录过滤后的点云数量
        n_pts_filtered = len(pts)

        # 如果过滤后点太少，直接返回
        if len(pts) < 10:
            return

        # ---- 推理 ----
        try:
            # result, _ = inference_detector(self.model, pts)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    result, _ = inference_detector(self.model, pts)            
                        
            
            
        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')
            torch.cuda.empty_cache()
            return
        _t3 = time.perf_counter()

        bboxes = result.pred_instances_3d.bboxes_3d
        scores = result.pred_instances_3d.scores_3d
        labels = result.pred_instances_3d.labels_3d

        # ---- 置信度过滤 ----
        keep = (scores > self.score_thr).nonzero(as_tuple=False).squeeze(1)
        if keep.numel() > 0:
            bboxes = bboxes[keep]
            scores = scores[keep]
            labels = labels[keep]
        else:
            labels = labels[:0]  # 空张量，跳过检测列表构建，但继续执行 track 更新

        # ---- 准备当前帧检测列表 ----
        current_detections = []

        if keep.numel() > 0:
            for i in range(len(labels)):
                label = int(labels[i].item())
                if label >= len(self.class_names):
                    continue
                class_name = self.class_names[label]
                if class_name not in CLASS_MAPPING:
                    continue
                class_name = CLASS_MAPPING[class_name]  # 统一为 'car' 或 'truck'
                
                # 获取lidar检测框信息
                bbox = bboxes[i]
                center = bbox.center
                if center.dim() == 2:
                    center = center[0]  # 如果是 (1, 3) 就取第一行
                cx_local = float(center[0])
                cy_local = float(center[1])
                cz_local = float(center[2])
                # 提取bbox偏航角   lidar系
                try:
                    yaw_local = float(bboxes[i].tensor[0, 6])
                except Exception:
                    yaw_local = 0.0
                
                # 转换到map系
                center_map = self.transform_point((cx_local, cy_local, cz_local), R, t)
                
                # 获取模型输出的尺寸
                try:
                    model_l = float(bboxes[i].tensor[0, 3])
                    model_w = float(bboxes[i].tensor[0, 4])
                    model_h = float(bboxes[i].tensor[0, 5])
                except Exception as e:
                    self.get_logger().warning(
                        f'Failed to extract bbox size for {class_name} at '
                        f'({cx_local:.1f}, {cy_local:.1f}): {e}. Skipping this detection.'
                    )
                    continue  # 跳过这个检测
                
                current_detections.append({
                    'center': center_map,
                    'class_name': class_name,
                    'model_bbox_size': (model_l, model_w, model_h),  # 模型真实尺寸
                    'local_center': (cx_local, cy_local, cz_local),
                    'yaw_local': yaw_local,
                    'bbox': bbox
                })
        
        _t4 = time.perf_counter()
        # ---- ID匹配：只对static_tracks做匹配 ----
        matches, unmatched_dets, unmatched_static_ids = self._id_tracks(
            current_detections, self.static_tracks
        )
        
        # ---- 提前定义删除列表（延迟删除，避免边遍历边删） ----
        tracks_to_delete = []
        static_ids_to_remove = []  # static转dynamic或激光穿透删除的ID列表
        
        # ---- 标记哪些dynamic被本帧匹配到了 ----
        matched_dynamic_ids = set()
        
        # ---- 处理匹配到的tracks ----
        for det_idx, track_id in matches:
            det = current_detections[det_idx]
            if track_id not in self.static_tracks:
                continue
            track = self.static_tracks[track_id]
            
            # 只算一次 yaw_map，后面复用
            yaw_map = self._transform_yaw_to_map(det.get('yaw_local', 0.0), R)
            det['yaw_map'] = yaw_map  # 缓存给 Marker 用
            
            # 更新跟踪点，用于下一帧匹配
            track.last_center = np.array(det['center'])
            track.missed_frames = 0
            track.is_matched = True
            track.penetration_confirm_frames = 0  # 匹配到即重置穿透计数器
            
            # 更新当前状态（用于生成点云，和框保持一致）
            track.curr_center = np.array(det['center'])
            track.curr_size = det['model_bbox_size']
            track.curr_yaw = yaw_map
            
            # 运动判断：中心点是否在基框内
            is_moving, center_inside_base = self._is_moving_check(
                det['center'],
                track.first_center,
                track.size,              # 基框尺寸（动态生成）
                track.first_yaw          # 基框yaw
            )
            
            # 静→动转换逻辑（连续3帧确认）
            if is_moving:
                track.moving_suspect_frames += 1
                if track.moving_suspect_frames >= 3:
                    if not track.is_moving:
                        self.get_logger().info(
                            f'Track {track_id} ({track.class_name}) confirmed MOVING '
                            f'after {track.moving_suspect_frames} consecutive frames. '
                            f'Converting to dynamic.'
                        )
                    track.is_moving = True
                    
                    # 标记从static_tracks删除
                    static_ids_to_remove.append(track_id)
                    
                    # 尝试匹配现有的dynamic_tracks
                    matched_dyn_id = self._match_dynamic_track(det['center'], self.dynamic_tracks, det['class_name'])
                    
                    if matched_dyn_id is not None:
                        # 更新现有的dynamic track
                        dyn_info = self.dynamic_tracks[matched_dyn_id]
                        dyn_info['center'] = det['center']
                        dyn_info['size'] = det['model_bbox_size']
                        dyn_info['yaw'] = yaw_map
                        dyn_info['label'] = det['class_name']
                        dyn_info['local_center'] = det['local_center']
                        dyn_info['local_yaw'] = det['yaw_local']
                        # 重新提取点云
                        # TODO: 优化建议 - 不在 dynamic_tracks 中长期存储 bbox_points，
                        # 改为在发布时按需计算，减少内存占用和重复计算
                        cx_l, cy_l, cz_l = det['local_center']
                        l, w, h = det['model_bbox_size']
                        y_l = det['yaw_local']
                        dyn_info['bbox_points'] = self._extract_bbox_points(
                            pts, cx_l, cy_l, cz_l, l, w, h, y_l
                        )
                        dyn_info['is_matched'] = True
                        matched_dynamic_ids.add(matched_dyn_id)
                    else:
                        # 创建新的dynamic track
                        new_dyn_id = self.next_dynamic_id % 10000
                        self.next_dynamic_id += 1
                        cx_l, cy_l, cz_l = det['local_center']
                        l, w, h = det['model_bbox_size']
                        y_l = det['yaw_local']
                        self.dynamic_tracks[new_dyn_id] = {
                            'center': det['center'],
                            'size': det['model_bbox_size'],
                            'yaw': yaw_map,
                            'label': det['class_name'],
                            'local_center': det['local_center'],
                            'local_yaw': det['yaw_local'],
                            # 不在 dynamic_tracks 中长期存储 bbox_points
                            'bbox_points': self._extract_bbox_points(
                                pts, cx_l, cy_l, cz_l, l, w, h, y_l
                            ),
                            'is_matched': True
                        }
                        matched_dynamic_ids.add(new_dyn_id)
                    
                    continue  # 跳过后续static处理逻辑
                
            else:
                # 非运动状态：中心点在基框内
                track.moving_suspect_frames = 0
            
            det['track_id'] = track_id
            det['is_moving'] = is_moving
            det['center_inside_base'] = center_inside_base
        
        # ---- 处理未匹配检测：先尝试匹配dynamic，再创建新static ----
        remaining_unmatched_dets = []
        for det_idx in unmatched_dets:
            det = current_detections[det_idx]
            
            # 尝试匹配现有的dynamic_tracks
            matched_dyn_id = self._match_dynamic_track(det['center'], self.dynamic_tracks, det['class_name'])
            
            if matched_dyn_id is not None:
                # 匹配到现有的dynamic，更新它
                yaw_map = self._transform_yaw_to_map(det.get('yaw_local', 0.0), R)
                det['yaw_map'] = yaw_map
                
                dyn_info = self.dynamic_tracks[matched_dyn_id]
                dyn_info['center'] = det['center']
                dyn_info['size'] = det['model_bbox_size']
                dyn_info['yaw'] = yaw_map
                dyn_info['label'] = det['class_name']
                dyn_info['local_center'] = det['local_center']
                dyn_info['local_yaw'] = det['yaw_local']
                # TODO: 优化建议 - 不在 dynamic_tracks 中长期存储 bbox_points
                cx_l, cy_l, cz_l = det['local_center']
                l, w, h = det['model_bbox_size']
                y_l = det['yaw_local']
                dyn_info['bbox_points'] = self._extract_bbox_points(
                    pts, cx_l, cy_l, cz_l, l, w, h, y_l
                )
                dyn_info['is_matched'] = True
                matched_dynamic_ids.add(matched_dyn_id)
                
                # 标记这个检测为动态车，用于后续可视化
                det['is_dynamic_matched'] = True
                det['dyn_id'] = matched_dyn_id
            else:
                # 未匹配到dynamic，作为新的static候选
                remaining_unmatched_dets.append(det_idx)
        
        # ---- 为真正的未匹配检测创建新static track ----
        for det_idx in remaining_unmatched_dets:
            det = current_detections[det_idx]
            
            # 只算一次 yaw_map，后面复用
            yaw_map = self._transform_yaw_to_map(det.get('yaw_local', 0.0), R)
            det['yaw_map'] = yaw_map  # 缓存给 Marker 用
            
            # 计算基框尺寸：第一帧检测box + 延展参数
            model_l, model_w, model_h = det['model_bbox_size']
            base_l = model_l + self.base_box_expand_x
            base_w = model_w + self.base_box_expand_y
            base_h = model_h  # 高度不延展
            
            # 分配新的static_id
            new_id = self.next_static_id
            self.next_static_id += 1
            
            new_track = VehicleTrack(
                first_center=det['center'],
                class_name=det['class_name'],
                size=(base_l, base_w, base_h),
                track_id=new_id
            )
            new_track.is_matched = True
            new_track.is_moving = False  # 首次出现默认静止
            new_track.first_yaw = yaw_map
            # 初始化当前状态
            new_track.curr_center = np.array(det['center'])
            new_track.curr_size = det['model_bbox_size']
            new_track.curr_yaw = yaw_map
            # 保存首次检测的原始尺寸
            new_track.first_size = det['model_bbox_size']
            
            self.static_tracks[new_id] = new_track
            self.get_logger().info(
                f'New static track {new_id} ({det["class_name"]}) created at '
                f'({det["center"][0]:.1f}, {det["center"][1]:.1f}). '
                f'Base box: ({base_l:.2f}x{base_w:.2f}x{base_h:.2f}m)'
            )
            
            det['track_id'] = new_id
            det['is_moving'] = False
            det['center_inside_base'] = True
        
        _t5 = time.perf_counter()
        # ---- 预计算激光点云方向向量----
        _pts_xyz3 = pts[:, :3]
        _dists_pre = np.linalg.norm(_pts_xyz3, axis=1)
        _valid_pre = _dists_pre > 0.1
        _pts_v_pre = _pts_xyz3[_valid_pre]
        _dists_v_pre = _dists_pre[_valid_pre]
        if len(_pts_v_pre) > 0:
            _dirs_pre = _pts_v_pre / _dists_v_pre[:, np.newaxis]
        else:
            _dirs_pre = np.empty((0, 3), dtype=np.float32)
        _precomp = (_pts_v_pre, _dists_v_pre, _dirs_pre)

        # ---- 更新未匹配static tracks（激光穿透判断） ----
        for track_id in unmatched_static_ids:
            # 跳过已在匹配阶段标记删除的track
            if track_id in tracks_to_delete:
                continue
                
            track = self.static_tracks[track_id]
            track.is_matched = False
            
            # 静态车：用激光穿透判断是否真的消失
            # 将 track.first_center 逆变换回 lidar 系
            center_lidar = R.T @ (track.first_center - t)
            actual_size = track.first_size if track.first_size is not None else track.size
            
            # 穿透确认需要连续 2 帧
            lidar_confirms_empty = self._lidar_can_confirm_empty(pts[:, :3], center_lidar, actual_size, _precomp)

            if lidar_confirms_empty:
                track.penetration_confirm_frames += 1
                if track.penetration_confirm_frames >= 2:
                    tracks_to_delete.append(track_id)
                    self.get_logger().info(
                        f'Static track {track_id} confirmed gone by lidar penetration '
                        f'({track.penetration_confirm_frames} consecutive frames), deleting'
                    )
            else:
                track.penetration_confirm_frames = 0  # 一帧没穿透就重置
        
        _t6 = time.perf_counter()
        # ---- 执行static tracks删除 ----
        for tid in tracks_to_delete:
            if tid in self.static_tracks:
                del self.static_tracks[tid]
        for tid in static_ids_to_remove:
            if tid in self.static_tracks:
                del self.static_tracks[tid]

        # 删除所有未被本帧匹配到的dynamic tracks
        all_dynamic_ids = set(self.dynamic_tracks.keys())
        unmatched_dynamic_ids = all_dynamic_ids - matched_dynamic_ids
        for dyn_id in unmatched_dynamic_ids:
            del self.dynamic_tracks[dyn_id]
        
        if unmatched_dynamic_ids:
            self.get_logger().debug(
                f'Removed {len(unmatched_dynamic_ids)} unmatched dynamic track(s): {unmatched_dynamic_ids}'
            )
        # 超过最大数量的
        static_ids = [tid for tid, tk in self.static_tracks.items() if not tk.is_moving]
        if len(static_ids) > self.max_static_tracks:
            # 按距离当前 lidar 位置从远到近排序，优先淘汰最远的
            static_ids_sorted = sorted(
                static_ids,
                key=lambda tid: np.linalg.norm(self.static_tracks[tid].first_center - t),
                reverse=True
            )
            overflow = len(static_ids) - self.max_static_tracks
            for tid in static_ids_sorted[:overflow]:
                self.get_logger().info(
                    f'Static track evict (farthest): evicting track {tid} '
                    f'(dist={np.linalg.norm(self.static_tracks[tid].first_center - t):.1f}m)'
                )
                del self.static_tracks[tid]
            self.get_logger().warn(
                f'Static track overflow: evicted {overflow} farthest track(s), '
                f'limit={self.max_static_tracks}'
            )

        # ---- 可视化和发布 ----
        marker_array = MarkerArray()
        marker_id = 0
        vehicle_raw_points_local = []  # 收集被检测为车辆的原始激光点（lidar系）
        
        # 判断是否需要打印状态日志（每3秒一次）
        _should_log_status = (time.time() - self._last_status_log_time >= 3.0)
        if _should_log_status:
            self._last_status_log_time = time.time()
        
        # 1. 处理匹配到的静态车（current_detections中包含track_id的）
        for det in current_detections:
            if 'track_id' not in det or 'is_dynamic_matched' in det:
                continue
            
            # 跳过已经不在static_tracks中的（可能已被删除）
            track_id = det['track_id']
            if track_id not in self.static_tracks:
                continue
                
            is_moving = det['is_moving']
            center_inside_base = det['center_inside_base']
            center_map = det['center']
            class_name = det['class_name']
            model_l, model_w, model_h = det['model_bbox_size']
            
            track = self.static_tracks[track_id]
            base_center = track.first_center
            base_l, base_w, base_h = track.size
            
            # 提取bbox内点云
            cx_local, cy_local, cz_local = det['local_center']
            yaw_local = det.get('yaw_local', 0.0)
            bbox_points_local = self._extract_bbox_points(
                pts, cx_local, cy_local, cz_local, model_l, model_w, model_h, yaw_local
            )
            
            if len(bbox_points_local) > 0:
                vehicle_raw_points_local.append(bbox_points_local)
            
            # 发Marker：基框（绿色半透明）
            base_marker = self.create_box_marker(
                center=base_center,
                size=(base_l, base_w, base_h),
                color=(0.0, 1.0, 0.0, 0.3),
                marker_id=track_id,  # 使用static_id保持稳定
                ns="base_boxes",
                frame_id=self.vehicle_boxes_frame,
                header_stamp=msg.header.stamp,
                yaw=track.first_yaw
            )
            marker_array.markers.append(base_marker)
            
            # 当前框：模型真实尺寸，红色(动)或蓝色(静)
            if is_moving:
                current_color = (1.0, 0.0, 0.0, 1.0)
            else:
                current_color = (0.0, 0.0, 1.0, 1.0)
            
            current_marker = self.create_box_marker(
                center=center_map,
                size=(model_l, model_w, model_h),
                color=current_color,
                marker_id=track_id + 10000,  # 偏移避免与基框ID冲突
                ns="current_boxes",
                frame_id=self.vehicle_boxes_frame,
                header_stamp=msg.header.stamp,
                yaw=det['yaw_map']
            )
            marker_array.markers.append(current_marker)
            
            # 位移连线
            if is_moving:
                line_marker = self.create_line_marker(
                    start=base_center,
                    end=center_map,
                    color=(1.0, 1.0, 0.0, 1.0),
                    marker_id=track_id + 20000,
                    frame_id=self.vehicle_boxes_frame,
                    header_stamp=msg.header.stamp
                )
                marker_array.markers.append(line_marker)
            
            # 中心点 marker：根据状态细分 namespace
            # - static: 中心点在基框内 (is_moving=False)
            # - suspect: 中心点出基框但未确认 (is_moving=True, 仍在 static_tracks)
            if not is_moving:
                center_point_color = (0.0, 0.0, 1.0, 1.0)  # 蓝色
                center_point_ns = "center_points_static"
            else:
                center_point_color = (1.0, 0.0, 0.0, 1.0)  # 红色
                center_point_ns = "center_points_suspect"
            
            center_marker = self.create_center_point_marker(
                center=center_map,
                color=center_point_color,
                marker_id=track_id + 30000,  # 偏移避免与现有 marker ID 冲突
                ns=center_point_ns,
                frame_id=self.vehicle_boxes_frame,
                header_stamp=msg.header.stamp
            )
            marker_array.markers.append(center_marker)
            
            status_str = "MOVING" if is_moving else "STATIC"
            if _should_log_status:
                self.get_logger().info(
                    f'[Static ID:{track_id}] {class_name} | '
                    f'Base:({base_center[0]:.1f},{base_center[1]:.1f}) | '
                    f'Curr:({center_map[0]:.1f},{center_map[1]:.1f}) | '
                    f'CenterInside:{center_inside_base} | {status_str}'
                )
        
        # 2. 处理未匹配但活着的静态车（记忆中的静态车）
        for track_id, track in self.static_tracks.items():
            if track.is_matched or track.is_moving:
                continue
            
            # 使用基框作为当前状态
            track.curr_center = track.first_center
            track.curr_yaw = track.first_yaw
            track.curr_size = track.first_size if track.first_size is not None else track.size
            l, w, h = track.size
            
            # 发Marker：绿色记忆中的静态车
            color = (0.0, 1.0, 0.0, 0.3)
            ns = "memory_static"
            
            marker = self.create_box_marker(
                center=track.first_center,
                size=(l, w, h),
                color=color,
                marker_id=track_id,
                ns=ns,
                frame_id=self.vehicle_boxes_frame,
                header_stamp=msg.header.stamp,
                yaw=track.first_yaw
            )
            marker_array.markers.append(marker)
        
        # 3. 处理动态车（dynamic_tracks）
        for dyn_id, dyn_info in self.dynamic_tracks.items():
            center = dyn_info['center']
            size = dyn_info['size']
            yaw = dyn_info['yaw']
            label = dyn_info['label']
            bbox_points = dyn_info['bbox_points']
            
            # 收集动态车的点云
            if len(bbox_points) > 0:
                vehicle_raw_points_local.append(bbox_points)
            
            # 动态车Marker：红色，使用dyn_id作为marker_id（保持稳定但每帧重新分配）
            # 使用独立ns避免与static冲突
            dyn_marker = self.create_box_marker(
                center=center,
                size=size,
                color=(1.0, 0.0, 0.0, 1.0),  # 红色
                marker_id=dyn_id,  # 使用dyn_id，在清理后会复用ID空间
                ns="dynamic_boxes",
                frame_id=self.vehicle_boxes_frame,
                header_stamp=msg.header.stamp,
                yaw=yaw
            )
            marker_array.markers.append(dyn_marker)
            
            # 动态车中心点 marker（红色）
            dyn_center_marker = self.create_center_point_marker(
                center=center,
                color=(1.0, 0.0, 0.0, 1.0),  # 红色
                marker_id=dyn_id + 40000,  # 偏移避免与现有 marker ID 冲突
                ns="center_points_dynamic",
                frame_id=self.vehicle_boxes_frame,
                header_stamp=msg.header.stamp
            )
            marker_array.markers.append(dyn_center_marker)
        
        # 日志输出状态
        self.get_logger().debug(
            f'Frame status: {len(self.static_tracks)} static, {len(self.dynamic_tracks)} dynamic'
        )

        # ---- 从所有 static_tracks 生成点云（只发静态车） ----
        all_outline_points = []
        for track_id, track in self.static_tracks.items():
            outline_pts = self._generate_outline_from_track(track)
            if len(outline_pts) > 0:
                all_outline_points.append(outline_pts)
        
        if all_outline_points:
            combined_outlines = np.vstack(all_outline_points)
            map_header = Header()
            map_header.stamp = msg.header.stamp
            map_header.frame_id = self.completed_cloud_map_frame
            self._publish_completed_cloud_map(combined_outlines, map_header)
        
        # ---- 发布车辆原始点云（包含静态车和动态车） ----
        if vehicle_raw_points_local:
            combined_raw = np.vstack(vehicle_raw_points_local)
            combined_raw_map = self.transform_pointcloud(combined_raw, R, t)
            raw_header = Header()
            raw_header.stamp = msg.header.stamp
            raw_header.frame_id = self.vehicle_raw_cloud_frame
            self._publish_vehicle_raw_cloud(combined_raw_map, raw_header)

        # ---- 处理Marker删除逻辑 ----
        current_marker_ids = {(m.ns, m.id) for m in marker_array.markers}
        for ns, mid in self._last_marker_ids:
            if (ns, mid) not in current_marker_ids:
                del_marker = Marker()
                del_marker.header.frame_id = self.vehicle_boxes_frame
                del_marker.header.stamp = msg.header.stamp
                del_marker.ns = ns
                del_marker.id = mid
                del_marker.action = Marker.DELETE
                marker_array.markers.append(del_marker)
        self._last_marker_ids = current_marker_ids

        _t7 = time.perf_counter()
        # ---- 更新缓存，供定时器持续刷新 ----
        with self._cached_marker_lock:
            self._cached_marker_array = marker_array
            self._cached_timestamp = msg.header.stamp

        _now = time.time()
        if _now - self._last_timing_log_time >= 1.0:
            self._last_timing_log_time = _now
            self.get_logger().info(
                f'[TIMING] total={(_t7-_t0)*1e3:.1f}ms | '
                f'tf={(_t1-_t0)*1e3:.1f}ms | '
                f'read={(_t2-_t1)*1e3:.1f}ms | '
                f'infer={(_t3-_t2)*1e3:.1f}ms | '
                f'postproc={(_t4-_t3)*1e3:.1f}ms | '
                f'match={(_t5-_t4)*1e3:.1f}ms | '
                f'track={(_t6-_t5)*1e3:.1f}ms | '
                f'publish={(_t7-_t6)*1e3:.1f}ms | '
                f'pts={n_pts_raw}->{n_pts_filtered} | '
                f'static={len(self.static_tracks)} | dynamic={len(self.dynamic_tracks)}'
            )

    def _publish_cached_markers(self):
        """0.2s 定时器回调：重发最新一帧的 MarkerArray，保持 RViz2 持续显示"""
        with self._cached_marker_lock:
            if not self._cached_marker_array.markers or self._cached_timestamp is None:
                return
            ma = copy.deepcopy(self._cached_marker_array)
        # 锁外修改时间戳并发布，避免持锁发消息阻塞
        now_stamp = self.get_clock().now().to_msg()
        for marker in ma.markers:
            marker.header.stamp = now_stamp
        self.vehicle_boxes_pub.publish(ma)

    def _extract_bbox_points(self, pts: np.ndarray, cx: float, cy: float, cz: float,
                             l: float, w: float, h: float, yaw: float) -> np.ndarray:
        """提取旋转 bbox 内的激光点云
        
        优化：先 AABB 粗筛，再 OBB 精筛，减少全量点云的旋转计算开销。
        """
        margin = 0.1
        xyz = pts[:, :3]

        # === AABB 粗筛 ===
        # cos(-yaw) = cos(yaw), sin(-yaw) = -sin(yaw)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        half_l = l / 2.0
        half_w = w / 2.0
        half_z = h / 2.0

        # 计算旋转框的外接 AABB 半长
        abs_cos = abs(cos_yaw)
        abs_sin = abs(sin_yaw)
        half_x = abs_cos * half_l + abs_sin * half_w
        half_y = abs_sin * half_l + abs_cos * half_w

        # AABB 包围盒范围（含 margin）
        x_min, x_max = cx - half_x - margin, cx + half_x + margin
        y_min, y_max = cy - half_y - margin, cy + half_y + margin
        z_min, z_max = cz - half_z - margin, cz + half_z + margin

        # 快速粗筛候选点
        aabb_mask = (
            (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) &
            (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max) &
            (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
        )
        cand_xyz = xyz[aabb_mask]
        if cand_xyz.size == 0:
            return cand_xyz  # 空数组，shape (0, 3)

        shifted = cand_xyz - np.array([cx, cy, cz], dtype=np.float32)

        cos_r = np.cos(-yaw)
        sin_r = np.sin(-yaw)
        local_x = shifted[:, 0] * cos_r - shifted[:, 1] * sin_r
        local_y = shifted[:, 0] * sin_r + shifted[:, 1] * cos_r
        local_z = shifted[:, 2]

        obb_mask = (
            (np.abs(local_x) <= half_l + margin) &
            (np.abs(local_y) <= half_w + margin) &
            (np.abs(local_z) <= half_z + margin)
        )
        return cand_xyz[obb_mask]
    
    def _transform_yaw_to_map(self, yaw_local: float, R: np.ndarray) -> float:
        """将 lidar 系的 yaw 通过旋转矩阵正确变换到 map 系"""
        dir_local = np.array([
            math.cos(yaw_local),
            math.sin(yaw_local),
            0.0
        ])
        dir_map = R @ dir_local
        return math.atan2(dir_map[1], dir_map[0])

    def transform_point(self, point_local: tuple, R: np.ndarray, t: np.ndarray) -> tuple:
        """把单个点转换到map系"""
        p_local = np.array(point_local)
        p_map = R @ p_local + t
        return (float(p_map[0]), float(p_map[1]), float(p_map[2]))

    def transform_pointcloud(self, points_local: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """把点云批量转换到map系"""
        if len(points_local) == 0:
            return points_local
        return (R @ points_local.T).T + t
    
    def _generate_outline_from_track(self, track: VehicleTrack) -> np.ndarray:
        """直接从 track 数据生成填充点云，和框保持完全一致"""
        cx, cy = track.curr_center[0], track.curr_center[1]
        l, w, h = track.curr_size
        yaw = track.curr_yaw
        
        resolution = 0.5 # 填充间距  
        
        # 生成局部坐标系下的网格点
        x_vals = np.arange(-l/2, l/2, resolution)
        y_vals = np.arange(-w/2, w/2, resolution)
        xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
        
        # 局部坐标
        local_x = xx.ravel()
        local_y = yy.ravel()
        
        # 旋转到 map 系
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        map_x = cx + local_x * cos_y - local_y * sin_y
        map_y = cy + local_x * sin_y + local_y * cos_y
        map_z = np.full_like(map_x, self.contour_slice_z)
        
        return np.column_stack([map_x, map_y, map_z]).astype(np.float32)

    def _publish_vehicle_raw_cloud(self, points: np.ndarray, header):
        """发布车辆原始点云，颜色设为紫色"""
        if len(points) == 0:
            return
        
        # 这里用 999.0 作为紫色标记值
        intensity = np.full((len(points), 1), 999.0, dtype=np.float32)
        points_with_intensity = np.hstack([points[:, :3].astype(np.float32), intensity])
        
        msg = PointCloud2()
        msg.header = header
        msg.header.frame_id = self.vehicle_raw_cloud_frame
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True
        msg.data = points_with_intensity.tobytes()
        
        self.vehicle_raw_cloud_pub.publish(msg)

    def _publish_completed_cloud_map(self, points: np.ndarray, header):
        """发布补全后的点云"""
        if len(points) == 0:
            return

        intensity = np.full((len(points), 1), 2048.0, dtype=np.float32)
        points_with_intensity = np.hstack([points[:, :3].astype(np.float32), intensity])
            
        msg = PointCloud2()
        msg.header = header
        msg.header.frame_id = self.completed_cloud_map_frame
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True
        msg.data = points_with_intensity.tobytes()
        
        self.completed_cloud_map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Mmdet3dNode()
    # 线程数增加到3个：_infer_group, _timer_group, _jitter_group
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        # 先取出临时文件路径，再销毁节点
        tmp_file = node._tmp_config_file
        node.destroy_node()
        # 清理自动提取生成的临时 config 文件
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)
        rclpy.shutdown()


if __name__ == '__main__':
    main()