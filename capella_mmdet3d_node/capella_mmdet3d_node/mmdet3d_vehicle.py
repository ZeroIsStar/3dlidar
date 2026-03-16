#!/usr/bin/env python3
import os
import tempfile
import yaml
import numpy as np
import math

import torch
import threading
import queue as _queue
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from ament_index_python.packages import get_package_share_directory
from mmdet3d.apis import inference_detector, init_model

# TF2 相关导入
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA

# 只检测该权重支持的类别 
TARGET_CLASSES = {'Car', 'car', 'Truck', 'truck'}

# 将模型输出的原始类名映射到业务类名（car / truck）
CLASS_MAPPING = {
    'car':                  'car',
    'Car':                  'car',
    'truck':                'truck',
    'Truck':                'truck',
    'construction_vehicle': 'truck',
    'bus':                  'truck',
    'trailer':              'truck',
}


class VehicleTrack:
    """车辆跟踪数据结构"""
    _id_counter = 0
    
    def __init__(self, first_center, class_name, size):
        
        VehicleTrack._id_counter += 1
        self.id = VehicleTrack._id_counter
        self.first_center = np.array(first_center)  # 基框中心
        self.last_center = np.array(first_center)   # 上一帧位置
        self.class_name = class_name
        self.size = size  # (l, w, h) 来自YAML
        self.missed_frames = 0
        self.is_matched = False
        # 静态车验证相关
        self.is_moving = False        # 是不是动态的
        self.first_yaw = 0.0          # 首次检测时map系下的朝向角

class VehicleDetectionNode(Node):
    def __init__(self):
        super().__init__('vehicle_detection_node')

        # ---- 参数声明 ----
        _pkg_dir = os.path.dirname(os.path.abspath(__file__))
        _demo_pth = os.path.join(_pkg_dir, 'mmdet3d_demo.pth')

        self.vehicle_priors = self._load_vehicle_priors()
        
        # 优先使用包内bundled config，留空时才回退到从pth自动提取
        _bundled_cfg = os.path.join(
            _pkg_dir, 'configs', 'centerpoint',
            'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'
        )
        _default_cfg = _bundled_cfg if os.path.exists(_bundled_cfg) else ''
        self.declare_parameter('config_file',       _default_cfg)
        self.declare_parameter('checkpoint_file',   _demo_pth)
        self.declare_parameter('infer_device',      'cuda:0')
        self.declare_parameter('score_threshold',   0.5)
        self.declare_parameter('pointcloud_topic',  '/vanjee/lidar')
        self.declare_parameter('point_cloud_qos',   'best_effort')
        self.declare_parameter('target_frame',      'map')
        
        # ID匹配、运动判断参数
        self.declare_parameter('match_distance_threshold', 3.0) # 运动超过3m
        self.declare_parameter('max_missed_frames', 6)  # 多少帧看不到就删除id
        self.declare_parameter('ls_moving_param', 0.8) 
        
        # 静态车消失确认参数
        self.declare_parameter('lidar_max_range', 80.0)    # 激光雷达最大有效量程
        self.declare_parameter('max_static_tracks', 200)  # 静态车记忆上限

        config_file     = self.get_parameter('config_file').value
        checkpoint_file = self.get_parameter('checkpoint_file').value
        device          = self.get_parameter('infer_device').value
        self.score_thr  = self.get_parameter('score_threshold').value
        self.pc_topic   = self.get_parameter('pointcloud_topic').value
        pc_qos_str      = self.get_parameter('point_cloud_qos').value
        self.target_frame = self.get_parameter('target_frame').value
        
        self.match_threshold = self.get_parameter('match_distance_threshold').value
        self.max_missed_frames = self.get_parameter('max_missed_frames').value
        self.ls_moving_param = self.get_parameter('ls_moving_param').value
        self.lidar_max_range = self.get_parameter('lidar_max_range').value
        self.max_static_tracks = self.get_parameter('max_static_tracks').value

        # ---- 加载模型 ----
        self._tmp_config_file = None
        if not config_file:
            # self.get_logger().info(f'config_file 未指定, 尝试从 checkpoint 自动提取: {checkpoint_file}')

            ck = torch.load(checkpoint_file, map_location='cpu')
            config_str = ck.get('meta', {}).get('config', None)
            tmp = tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, prefix='mmdet3d_auto_cfg_')
            tmp.write(config_str)
            tmp.close()
            config_file = tmp.name
            self._tmp_config_file = tmp.name
            # self.get_logger().info(f'自动提取 config 成功, 临时路径: {config_file}')

        # self.get_logger().info(f'Loading model: {config_file}  |  {checkpoint_file}')
        self.model = init_model(config_file, checkpoint_file, device=device)
        self.class_names: list[str] = list(self.model.dataset_meta.get('classes', []))
        self.get_logger().info(f'Model device: {next(self.model.parameters()).device}')
        # self.get_logger().info(f'Model loaded. classes={self.class_names}')


        # ---- QoS ----
        qos = QoSProfile(depth=5)
        if pc_qos_str == 'best_effort':
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
        else:
            qos.reliability = ReliabilityPolicy.RELIABLE

        # ---- 订阅原始点云 ----
        self.pc_sub = self.create_subscription(
            PointCloud2, self.pc_topic, self._pointcloud_callback, qos)
        
        # ---- 发布map系点云 ----
        self.completed_cloud_map_pub = self.create_publisher(
            PointCloud2, '/completed_vehicle_cloud_map', 10)
        
        # ---- 发布先验长方体MarkerArray ----
        self.vehicle_boxes_pub = self.create_publisher(
            MarkerArray, '/vehicle_boxes', 10)

        # ---- TF2 初始化 ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # ---- 跟踪管理 ----
        self.tracks = {}  # {track_id: VehicleTrack}
        self._tracks_lock = threading.Lock()  # 保护tracks字典的并发访问
        self._frame_counter = 0
        self._process_every_n_frames = 10
        self._last_marker_ids = set()
        # 缓存最新一帧的 MarkerArray，供定时器持续刷新用
        self._cached_marker_array = MarkerArray()
        self._cached_marker_lock = threading.Lock()

        # 静止车辆轮廓固定，避免逐帧重新生成
        self._static_contour_cache = {}

        # ---- 推理工作线程： ROS executor ----
        self._work_queue = _queue.Queue(maxsize=1)
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        # ---- 可视化定时器：0.2s 刷一次，不依赖推理帧 ----
        self.viz_timer = self.create_timer(0.2, self._publish_cached_markers)

        self.get_logger().info(
            f'VehicleDetectionNode ready | match_thr={self.match_threshold}m | '
            f'lidar_max_range={self.lidar_max_range}m'
        )

    
    def _load_vehicle_priors(self):
        # 硬编码路径
        yaml_path = '/capella/lib/python3.10/site-packages/3dlidar/src/capella_mmdet3d_node/capella_mmdet3d_node/configs/vehicle_priors/vehicle_priors.yaml'
        
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f'vehicle_priors.yaml not found at: {yaml_path}')
        
        self.get_logger().info(f'Loading vehicle priors from: {yaml_path}')
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data['vehicle_priors']

    # 读取点云
    def _read_pointcloud(self, msg: PointCloud2) -> np.ndarray | None:
        """
        将 PointCloud2 转换为 (N, 5) float32 数组 [x, y, z, intensity, ring].
        """
        field_names_avail = [f.name for f in msg.fields]
        has_intensity = 'intensity' in field_names_avail
        has_ring      = 'ring'      in field_names_avail

        read_fields = ['x', 'y', 'z']
        if has_intensity:
            read_fields.append('intensity')
        if has_ring:
            read_fields.append('ring')

        struct_arr = np.asarray(
            list(pc2.read_points(msg, field_names=read_fields, skip_nans=True))
        )
        if len(struct_arr) == 0:
            return None
        
        if struct_arr.dtype.names:
            pts = np.column_stack(
                [struct_arr[name].astype(np.float32) for name in struct_arr.dtype.names]
            )
        else:
            pts = struct_arr.astype(np.float32)

        if not has_intensity:
            pts = np.hstack([pts, np.zeros((len(pts), 1), dtype=np.float32)])
        if not has_ring:
            pts = np.hstack([pts, np.zeros((len(pts), 1), dtype=np.float32)])

        return pts[:, :5]

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
        
        yaw_offset = math.atan2(2.0 * (qw * qz + qx * qy), 
                                1.0 - 2.0 * (qy * qy + qz * qz))
        
        return R, t, yaw_offset

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

    def _id_tracks(self, detections, tracks_dict):
        """
        ID匹配逻辑：找同类别且距离最近的
        """
        matches = []  # [(det_idx, track_id), ...]
        unmatched_dets = []
        unmatched_tracks = set(tracks_dict.keys())
        
        if not tracks_dict:
            return [], list(range(len(detections))), set()
        
        det_used = set()
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

    def _is_moving_check(self, current_center, current_size, first_center, prior_size):
        """
        通过iou判断是否是在运动
        """
        prior_min = np.array([
            first_center[0] - prior_size[0]/2,
            first_center[1] - prior_size[1]/2,
            first_center[2] - prior_size[2]/2
        ])
        prior_max = np.array([
            first_center[0] + prior_size[0]/2,
            first_center[1] + prior_size[1]/2,
            first_center[2] + prior_size[2]/2
        ])

        det_min = np.array([
            current_center[0] - current_size[0]/2,
            current_center[1] - current_size[1]/2,
            current_center[2] - current_size[2]/2
        ])
        det_max = np.array([
            current_center[0] + current_size[0]/2,
            current_center[1] + current_size[1]/2,
            current_center[2] + current_size[2]/2
        ])

        # 计算交集体积
        intersect_min = np.maximum(prior_min, det_min)
        intersect_max = np.minimum(prior_max, det_max)
        intersect_dims = np.maximum(0, intersect_max - intersect_min)
        intersect_vol = np.prod(intersect_dims)

        # 当前 bbox 体积
        det_vol = np.prod(det_max - det_min)

        # 包含度：bbox 在先验框内的比例
        containment = intersect_vol / det_vol if det_vol > 0 else 0.0

        is_moving = containment < self.ls_moving_param

        return is_moving, containment

    def _lidar_can_confirm_empty(self, pts_lidar: np.ndarray,
                                   region_center_lidar: np.ndarray,
                                   region_size: tuple,
                                   _precomputed=None) -> bool:
        """
        激光雷达能否确认目标区域是空的？通过判断该区域所在方向的激光是否大量穿透来确认。

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

        # 即激光穿过了目标区域所在的位置，说明那里是空的
        cone_dists = dists_valid[in_cone]
        penetrating = np.sum(cone_dists > (d_target - half_span))
        ratio = float(penetrating) / len(cone_dists)

        # 50% 以上的锥内激光穿透了目标区域 → 确认为空
        return ratio >= 0.5
    
    def create_box_marker(self, center, size, color, marker_id, ns, frame_id, header_stamp):
        """创建2D footprint Marker"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = header_stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

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

    def _worker_loop(self):
        while rclpy.ok():
            try:
                msg = self._work_queue.get(timeout=0.5)
            except _queue.Empty:
                continue
            try:
                self._process_pointcloud(msg)
            except Exception as e:
                self.get_logger().error(f'Worker error: {e}')

    # 推理 + 过滤 + 跟踪 + 运动判断
    def _pointcloud_callback(self, msg: PointCloud2):
        # ---- 跳帧逻辑 ----
        self._frame_counter += 1
        if self._frame_counter % self._process_every_n_frames != 0:
            return
        # 放入队列，工作线程异步处理；队列满时丢弃当前帧
        try:
            self._work_queue.put_nowait(msg)
        except _queue.Full:
            pass

    def _process_pointcloud(self, msg: PointCloud2):
        # ---- 获取TF变换 ----
        source_frame = msg.header.frame_id
        target_frame = self.target_frame
        
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().debug(f'TF lookup failed: {e}')
            return
        
        R, t, yaw_offset = self._extract_transform_matrix(transform)
        
        # ---- 读取点云 ----
        pts = self._read_pointcloud(msg)
        if pts is None or len(pts) < 10:
            return

        # ---- 推理 ----
        try:
            result, _ = inference_detector(self.model, pts)
        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')
            torch.cuda.empty_cache()
            return

        bboxes = result.pred_instances_3d.bboxes_3d
        scores = result.pred_instances_3d.scores_3d
        labels = result.pred_instances_3d.labels_3d

        # ---- 置信度过滤 ----
        keep = (scores > self.score_thr).nonzero(as_tuple=False).squeeze(1)
        if keep.numel() == 0:
            return
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # ---- 准备当前帧检测列表 ----
        current_detections = []
        
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
            
            # 获取尺寸
            prior_key = class_name.capitalize()
            if prior_key in self.vehicle_priors:
                prior = self.vehicle_priors[prior_key]
                l, w, h = prior['length'], prior['width'], prior['height']
            else:
                l, w, h = 4.5, 1.8, 1.6  # 默认值

            try:
                model_l = float(bboxes[i].tensor[0, 3])
                model_w = float(bboxes[i].tensor[0, 4])
                model_h = float(bboxes[i].tensor[0, 5])
            except Exception:
                model_l, model_w, model_h = l, w, h
            
            current_detections.append({
                'center': center_map,
                'class_name': class_name,
                'size': (l, w, h),              # YAML先验尺寸，用于运动判断基准框、记忆静态车
                'model_bbox_size': (model_l, model_w, model_h),  # 模型真实尺寸
                'local_center': (cx_local, cy_local, cz_local),
                'yaw_local': yaw_local,
                'bbox': bbox
            })
        
        # ---- ID匹配----
        matches, unmatched_dets, unmatched_tracks = self._id_tracks(
            current_detections, self.tracks
        )
        
        # ---- 更新匹配到的tracks ----
        for det_idx, track_id in matches:
            det = current_detections[det_idx]
            track = self.tracks[track_id]
            
            # 更新跟踪点，用于下一帧匹配
            track.last_center = np.array(det['center'])
            track.missed_frames = 0
            track.is_matched = True
            
            # 运动判断
            is_moving, iou = self._is_moving_check(
                det['center'],
                det['model_bbox_size'],  # 模型真实bbox尺寸
                track.first_center,
                track.size              # YAML 先验尺寸
            )
            
            # 更新track状态：只允许静→动，不允许动→静
            if is_moving:
                track.is_moving = True
            
            det['track_id'] = track_id
            det['is_moving'] = is_moving
            det['iou'] = iou
        
        # ---- 为未匹配检测创建新track ----
        for det_idx in unmatched_dets:
            det = current_detections[det_idx]
            
            new_track = VehicleTrack(
                first_center=det['center'],
                class_name=det['class_name'],
                size=det['size']
            )
            new_track.is_matched = True
            new_track.is_moving = False  # 首次出现默认静止
            new_track.first_yaw = det.get('yaw_local', 0.0) + yaw_offset  # 存map系朝向
            self.tracks[new_track.id] = new_track
            self.get_logger().info(
                f'New track {new_track.id} ({det["class_name"]}) created at '
                f'({det["center"][0]:.1f}, {det["center"][1]:.1f}). '
                f'No existing track within {self.match_threshold}m.'
            )
            
            det['track_id'] = new_track.id
            det['is_moving'] = False
            det['iou'] = 1.0
        
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

        # ---- 更新未匹配tracks----
        tracks_to_delete = []  # 待删除列表
        
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.is_matched = False
            
            if track.is_moving:
                # 动态车：正常累计丢失帧，超期删除
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    tracks_to_delete.append(track_id)
                    self.get_logger().info(
                        f'Dynamic track {track_id} ({track.class_name}) deleted: '
                        f'not seen for {track.missed_frames} frames '
                        f'(limit={self.max_missed_frames}). '
                        f'Last pos: ({track.last_center[0]:.1f}, {track.last_center[1]:.1f})'
                    )
            else:
                # 静态车：用激光穿透判断是否真的消失
                # 将 track.first_center  逆变换回 lidar 系
                center_lidar = R.T @ (track.first_center - t)
                if self._lidar_can_confirm_empty(pts[:, :3], center_lidar, track.size, _precomp):
                    tracks_to_delete.append(track_id)
                    self.get_logger().info(
                        f'Static track {track_id} confirmed gone by lidar penetration, deleting'
                    )
        
        # ---- 执行删除 ----
        for tid in tracks_to_delete:
            if tid in self.tracks:
                del self.tracks[tid]
            self._static_contour_cache.pop(tid, None)

        # ---- 静态车数量上限保护
        static_ids = [tid for tid, tk in self.tracks.items() if not tk.is_moving]
        if len(static_ids) > self.max_static_tracks:
            overflow = len(static_ids) - self.max_static_tracks
            for tid in static_ids[:overflow]:
                del self.tracks[tid]
                self._static_contour_cache.pop(tid, None)
            self.get_logger().warn(
                f'Static track FIFO: evicted {overflow} oldest track(s), '
                f'limit={self.max_static_tracks}'
            )

        # ---- 处理每辆车：匹配的车正常发，未匹配的静态车也发----
        marker_array = MarkerArray()
        marker_id = 0
        all_map_points = []  # 收集所有车辆的map系点云，最后合并发布
        
        #  先发匹配到的车
        for det in current_detections:
            if 'track_id' not in det:
                continue
                
            track_id = det['track_id']
            is_moving = det['is_moving']
            iou = det['iou']
            center_map = det['center']
            class_name = det['class_name']
            prior_l, prior_w, prior_h = det['size']           # YAML先验尺寸
            model_l, model_w, model_h = det['model_bbox_size']  # 模型真实尺寸
            
            track = self.tracks[track_id]
            base_center = track.first_center
            
            # 提取bbox内点云
            cx_local, cy_local, cz_local = det['local_center']
            bbox_points_local = self._extract_bbox_points(
                pts, cx_local, cy_local, cz_local, model_l, model_w, model_h, 0.0
            )
            
            # 使用模型输出的bbox朝向补全轮廓
            completed_contour_local = self._fill_completed_contour(
                partial_points=bbox_points_local,
                class_name=class_name,
                bbox_center=(cx_local, cy_local, cz_local),
                bbox_size=(model_l, model_w, model_h),
                yaw=det.get('yaw_local', 0.0)
            )
            
            # 收集map系点云
            if len(completed_contour_local) > 0:
                completed_contour_map = self.transform_pointcloud(
                    completed_contour_local, R, t
                )
                all_map_points.append(completed_contour_map)
            
            # 发Marker
            # 基框：YAML先验尺寸，绿色半透明
            base_marker = self.create_box_marker(
                center=base_center,
                size=(prior_l, prior_w, prior_h),
                color=(0.0, 1.0, 0.0, 0.3),
                marker_id=marker_id,
                ns="base_boxes",
                frame_id=self.target_frame,
                header_stamp=msg.header.stamp
            )
            marker_array.markers.append(base_marker)
            marker_id += 1
            
            # 当前框：模型真实尺寸，红色(动)或蓝色(静)
            if is_moving:
                current_color = (1.0, 0.0, 0.0, 1.0)
            else:
                current_color = (0.0, 0.0, 1.0, 1.0)
            
            current_marker = self.create_box_marker(
                center=center_map,
                size=(model_l, model_w, model_h),
                color=current_color,
                marker_id=marker_id,
                ns="current_boxes",
                frame_id=self.target_frame,
                header_stamp=msg.header.stamp
            )
            marker_array.markers.append(current_marker)
            marker_id += 1
            
            # 位移连线
            if is_moving:
                line_marker = self.create_line_marker(
                    start=base_center,
                    end=center_map,
                    color=(1.0, 1.0, 0.0, 1.0),
                    marker_id=marker_id,
                    frame_id=self.target_frame,
                    header_stamp=msg.header.stamp
                )
                marker_array.markers.append(line_marker)
                marker_id += 1
            
            status_str = "MOVING" if is_moving else "STATIC"
            self.get_logger().info(
                f'[ID:{track_id}] {class_name} | '
                f'Base:({base_center[0]:.1f},{base_center[1]:.1f}) | '
                f'Curr:({center_map[0]:.1f},{center_map[1]:.1f}) | '
                f'Contain:{iou:.2f} | {status_str}'
            )
        
        #  再发没匹配但活着的静态车
        for track_id, track in self.tracks.items():
            if track.is_matched or track.is_moving:
                continue  # 匹配过的或动态的跳过
            
            l, w, h = track.size
            cx, cy, cz = track.first_center
            
            # 缓存静态车轮廓
            if track_id not in self._static_contour_cache:
                completed_contour = self._fill_completed_contour(
                    partial_points=np.array([]),
                    class_name=track.class_name,
                    bbox_center=(cx, cy, cz),
                    bbox_size=(l, w, h),
                    yaw=track.first_yaw
                )
                self._static_contour_cache[track_id] = completed_contour
            else:
                completed_contour = self._static_contour_cache[track_id]
            
            # 收集map系点云
            if len(completed_contour) > 0:
                all_map_points.append(completed_contour)
            
            # 发Marker：绿色   记忆中的静态车
            color = (0.0, 1.0, 0.0, 0.3)
            ns = "memory_static"
            
            marker = self.create_box_marker(
                center=track.first_center,
                size=(l, w, h),
                color=color,
                marker_id=marker_id,
                ns=ns,
                frame_id=self.target_frame,
                header_stamp=msg.header.stamp
            )
            marker_array.markers.append(marker)
            marker_id += 1

        # ---- 合并所有车辆点云，统一发布一次 ----
        if all_map_points:
            combined_map = np.vstack(all_map_points)
            map_header = Header()
            map_header.stamp = msg.header.stamp
            map_header.frame_id = self.target_frame
            self._publish_completed_cloud_map(combined_map, map_header)

        current_marker_ids = {(m.ns, m.id) for m in marker_array.markers}
        for ns, mid in self._last_marker_ids:
            if (ns, mid) not in current_marker_ids:
                del_marker = Marker()
                del_marker.header.frame_id = self.target_frame
                del_marker.header.stamp = msg.header.stamp
                del_marker.ns = ns
                del_marker.id = mid
                del_marker.action = Marker.DELETE
                marker_array.markers.append(del_marker)
        self._last_marker_ids = current_marker_ids

        # ---- 更新缓存，供定时器持续刷新 ----
        with self._cached_marker_lock:
            self._cached_marker_array = marker_array

    def _publish_cached_markers(self):
        """0.2s 定时器回调：重发最新一帧的 MarkerArray，保持 RViz2 持续显示"""
        with self._cached_marker_lock:
            ma = self._cached_marker_array
        if not ma.markers:
            return
        now = self.get_clock().now().to_msg()
        for m in ma.markers:
            m.header.stamp = now
        self.vehicle_boxes_pub.publish(ma)

    def _extract_bbox_points(self, pts: np.ndarray, cx: float, cy: float, cz: float,
                             l: float, w: float, h: float, yaw: float) -> np.ndarray:
        """提取 bbox 内的激光点云"""
        margin = 0.1
        mask = (
            (pts[:, 0] >= cx - l/2 - margin) & (pts[:, 0] <= cx + l/2 + margin) &
            (pts[:, 1] >= cy - w/2 - margin) & (pts[:, 1] <= cy + w/2 + margin) &
            (pts[:, 2] >= cz - h/2 - margin) & (pts[:, 2] <= cz + h/2 + margin)
        )
        return pts[mask, :3]
    
    def _fill_completed_contour(self, partial_points: np.ndarray, class_name: str,
                                 bbox_center: tuple, bbox_size: tuple, 
                                 yaw: float) -> np.ndarray:
        """补全车辆轮廓"""
        l, w, h = bbox_size
        cx, cy, cz = bbox_center
        resolution = 0.1  # 每10cm一个点

        x_vals = np.arange(-l/2, l/2, resolution)
        y_vals = np.arange(-w/2, w/2, resolution)
        z_vals = np.array([0.0])  # 单层2D切片
        
        # 生成所有网格点
        xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        local_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        
        # 根据yaw旋转这些点
        if abs(yaw) > 0.001:
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            rot = np.array([
                [cos_y, -sin_y, 0],
                [sin_y,  cos_y, 0],
                [0,      0,     1]
            ])
            local_points = (rot @ local_points.T).T
        
        # 平移到检测框中心位置
        filled_points = local_points + np.array([cx, cy, cz])

        # 如果 partial_points 有数据，合并它们；如果 empty，只返回填充点
        if len(partial_points) > 0:
            return np.vstack([filled_points, partial_points]).astype(np.float32)
        else:
            return filled_points.astype(np.float32)
    
    def _publish_completed_cloud_map(self, points: np.ndarray, header):
        """发布补全后的点云"""
        if len(points) == 0:
            return

        intensity = np.full((len(points), 1), 2048.0, dtype=np.float32)
        points_with_intensity = np.hstack([points[:, :3].astype(np.float32), intensity])
            
        msg = PointCloud2()
        msg.header = header
        msg.header.frame_id = self.target_frame
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
    node = VehicleDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        # 清理自动提取生成的临时 config 文件
        if node._tmp_config_file and os.path.exists(node._tmp_config_file):
            os.unlink(node._tmp_config_file)
        rclpy.shutdown()


if __name__ == '__main__':
    main()