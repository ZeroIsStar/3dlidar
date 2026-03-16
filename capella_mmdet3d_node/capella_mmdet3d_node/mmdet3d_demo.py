#!/usr/bin/env python3
import os
import tempfile
import yaml
import numpy as np
import math

import torch
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs_py import point_cloud2 as pc2
from ament_index_python.packages import get_package_share_directory
from mmdet3d.apis import inference_detector, init_model

# TF2 相关导入
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA

# 图像处理相关
from cv_bridge import CvBridge
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import io

# 只检测该权重支持的类别 
TARGET_CLASSES = {'car', 'truck', 'construction_vehicle', 'bus', 'trailer'}


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
        self.is_verifying = False     # 否进入距离验证模式
        self.verify_count = 0         # 验证帧计数
        self.first_yaw = 0.0          # 首次检测时map系下的朝向角（rad）
        self.speed = 0.0               # 用于显示的速度（可选）


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
        self.declare_parameter('score_threshold',   0.3)
        self.declare_parameter('pointcloud_topic',  '/vanjee/lidar')
        self.declare_parameter('point_cloud_qos',   'best_effort')
        self.declare_parameter('target_frame',      'map')
        
        # ID匹配、运动判断参数
        self.declare_parameter('match_distance_threshold', 3.0)
        self.declare_parameter('max_missed_frames', 6)
        self.declare_parameter('ls_moving_param', 0.8)
        
        # 静态车查岗参数
        self.declare_parameter('verify_distance', 50.0)   # 多近开始查岗
        self.declare_parameter('verify_frames', 5)        # 连续几帧没看到算真没了
        self.declare_parameter('timer_period', 10.0)      # 多久看一次距离

        # 图像发布相关参数
        self.declare_parameter('publish_image', True)          # 是否发布BEV图像
        self.declare_parameter('image_topic', '/detection_bev')
        self.declare_parameter('image_interval', 0.5)          # 发布间隔（秒）
        self.declare_parameter('image_width', 800)
        self.declare_parameter('image_height', 600)
        self.declare_parameter('image_range', 50.0)            # 显示范围（米）

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
        self.verify_distance = self.get_parameter('verify_distance').value
        self.verify_frames = self.get_parameter('verify_frames').value

        # 图像参数
        self.publish_image = self.get_parameter('publish_image').value
        self.image_topic = self.get_parameter('image_topic').value
        self.image_interval = self.get_parameter('image_interval').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.image_range = self.get_parameter('image_range').value

        # ---- 加载模型 ----
        self._tmp_config_file = None
        if not config_file:
            self.get_logger().info(f'config_file 未指定, 尝试从 checkpoint 自动提取: {checkpoint_file}')

            ck = torch.load(checkpoint_file, map_location='cpu')
            config_str = ck.get('meta', {}).get('config', None)
            tmp = tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, prefix='mmdet3d_auto_cfg_')
            tmp.write(config_str)
            tmp.close()
            config_file = tmp.name
            self._tmp_config_file = tmp.name
            self.get_logger().info(f'自动提取 config 成功, 临时路径: {config_file}')

        self.get_logger().info(f'Loading model: {config_file}  |  {checkpoint_file}')
        self.model = init_model(config_file, checkpoint_file, device=device)
        self.class_names: list[str] = list(self.model.dataset_meta.get('classes', []))
        self.get_logger().info(f'Model loaded. classes={self.class_names}')


        # ---- QoS ----
        qos = QoSProfile(depth=5)
        if pc_qos_str == 'best_effort':
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
        else:
            qos.reliability = ReliabilityPolicy.RELIABLE

        # ---- 订阅原始点云 ----
        self.pc_sub = self.create_subscription(
            PointCloud2, self.pc_topic, self._pointcloud_callback, qos)
        
        # ---- 发布补全后的车辆轮廓点云 ----
        self.completed_cloud_pub = self.create_publisher(
            PointCloud2, '/completed_vehicle_cloud', 10)
        
        # ---- 发布map系点云 ----
        self.completed_cloud_map_pub = self.create_publisher(
            PointCloud2, '/completed_vehicle_cloud_map', 10)
        
        # ---- 发布先验长方体MarkerArray ----
        self.vehicle_boxes_pub = self.create_publisher(
            MarkerArray, '/vehicle_boxes', 10)

        # ---- 发布BEV图像 ----
        if self.publish_image:
            self.image_pub = self.create_publisher(Image, self.image_topic, 10)
            self.bridge = CvBridge()
            self.last_image_time = self.get_clock().now()

        # ---- TF2 初始化 ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # ---- 跟踪管理 ----
        self.tracks = {}  # {track_id: VehicleTrack}
        self._tracks_lock = threading.Lock()  # 保护tracks字典的并发访问
        self._frame_counter = 0
        self._process_every_n_frames = 5 

        # ---- 静态车验证定时器 ----
        timer_period = self.get_parameter('timer_period').value
        self.verify_timer = self.create_timer(timer_period, self._check_distance_callback)
        self.get_logger().info(f'Static vehicle verification timer started: {timer_period}s')

        self.get_logger().info(
            f'VehicleDetectionNode ready | match_thr={self.match_threshold}m | '
            f'verify_dist={self.verify_distance}m | publish_image={self.publish_image}'
        )

    
    def _load_vehicle_priors(self):
        """加载车辆先验尺寸，若找不到文件则使用默认值"""
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'vehicle_priors', 'vehicle_priors.yaml'),
            os.path.join(get_package_share_directory('capella_mmdet3d_node'), 'configs', 'vehicle_priors', 'vehicle_priors.yaml'),
        ]
        
        yaml_path = None
        for path in possible_paths:
            if os.path.exists(path):
                yaml_path = path
                break
        
        if yaml_path is None:
            self.get_logger().warning('vehicle_priors.yaml not found, using default dimensions')
            # 返回默认尺寸
            return {
                'Car': {'length': 4.5, 'width': 1.8, 'height': 1.6},
                'Truck': {'length': 8.0, 'width': 2.5, 'height': 3.0},
                'Bus': {'length': 12.0, 'width': 2.5, 'height': 3.5},
                'Construction_vehicle': {'length': 6.0, 'width': 2.4, 'height': 2.8},
                'Trailer': {'length': 15.0, 'width': 2.5, 'height': 4.0},
            }
        
        self.get_logger().info(f'Loading vehicle priors from: {yaml_path}')
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data['vehicle_priors']

    def _read_pointcloud(self, msg: PointCloud2) -> np.ndarray | None:
        """将 PointCloud2 转换为 (N, 5) float32 数组 [x, y, z, intensity, ring]."""
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
        """将lidar系下的单点转换到map系"""
        p_local = np.array(point_local)
        p_map = R @ p_local + t
        return (float(p_map[0]), float(p_map[1]), float(p_map[2]))

    def transform_pointcloud(self, points_local: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """将lidar系下的点云批量转换到map系"""
        if len(points_local) == 0:
            return points_local
        return (R @ points_local.T).T + t

    def get_current_vehicle_pose(self) -> np.ndarray | None:
        """获取本车当前在map坐标系下的位置"""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            t = transform.transform.translation
            return np.array([t.x, t.y, t.z])
        except Exception as e:
            self.get_logger().debug(f'Failed to get vehicle pose: {e}')
            return None

    def match_tracks(self, detections, tracks_dict):
        """
        ID匹配逻辑：找同类别且距离最近的，强制一对一
        """
        matches = []  # [(det_idx, track_id), ...]
        unmatched_dets = []
        unmatched_tracks = set(tracks_dict.keys())
        
        if not tracks_dict:
            return [], list(range(len(detections))), set()
        
        det_used = set()
        matched_track_ids = set()  # 强制一对一：已匹配的track不再参与匹配
        
        for det_idx, det in enumerate(detections):
            det_center = np.array(det['center'])
            det_class = det['class_name']
            
            best_track_id = None
            min_dist = float('inf')
            
            # 找同类别且距离最近的未匹配track（强制一对一）
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

    def check_movement(self, current_center, first_center, size):
        """
        运动判断逻辑 - 通过 IOU 判断是否移动
        """
        base_min = np.array([
            first_center[0] - size[0]/2,
            first_center[1] - size[1]/2,
            first_center[2] - size[2]/2
        ])
        base_max = np.array([
            first_center[0] + size[0]/2,
            first_center[1] + size[1]/2,
            first_center[2] + size[2]/2
        ])

        # 检测框 小框，当前位置
        det_min = np.array([
            current_center[0] - size[0]/2,
            current_center[1] - size[1]/2,
            current_center[2] - size[2]/2
        ])
        det_max = np.array([
            current_center[0] + size[0]/2,
            current_center[1] + size[1]/2,
            current_center[2] + size[2]/2
        ])

        # 计算交集体积
        intersect_min = np.maximum(base_min, det_min)
        intersect_max = np.minimum(base_max, det_max)
        intersect_dims = np.maximum(0, intersect_max - intersect_min)
        intersect_vol = np.prod(intersect_dims)

        # 计算两个框的体积
        base_vol = np.prod(base_max - base_min)
        det_vol = np.prod(det_max - det_min)

        # 计算 IOU
        union_vol = base_vol + det_vol - intersect_vol
        iou = intersect_vol / union_vol if union_vol > 0 else 0.0

        is_moving = iou < self.ls_moving_param

        return is_moving, iou

    def _is_area_scanned(self, pts: np.ndarray, center: np.ndarray, size: tuple) -> bool:
        """
        检查指定区域是否被当前激光点云覆盖
        """
        l, w, h = size
        margin = 0.2
        
        mask = (
            (pts[:, 0] >= center[0] - l/2 - margin) & 
            (pts[:, 0] <= center[0] + l/2 + margin) &
            (pts[:, 1] >= center[1] - w/2 - margin) & 
            (pts[:, 1] <= center[1] + w/2 + margin) &
            (pts[:, 2] >= center[2] - h/2 - margin) & 
            (pts[:, 2] <= center[2] + h/2 + margin)
        )
        
        return np.sum(mask) > 10  # 超过10个点认为扫到了

    def _check_distance_callback(self):
        """
        定时器回调：每10秒检查静态车距离，决定是否进入验证模式
        """
        my_pose = self.get_current_vehicle_pose()
        if my_pose is None:
            return
        
        with self._tracks_lock:
            tracks_snapshot = dict(self.tracks)
        for track_id, track in tracks_snapshot.items():
            if track.is_moving:
                continue  # 动态车不需要
            
            # 计算与记忆位置的距离
            dist = np.linalg.norm(my_pose[:2] - track.first_center[:2])
            
            if dist < self.verify_distance:
                if not track.is_verifying:
                    track.is_verifying = True
                    track.verify_count = 0
                    self.get_logger().info(
                        f'Static vehicle {track_id} entering verification mode '
                        f'(distance: {dist:.1f}m)'
                    )
            else:
                if track.is_verifying:
                    track.is_verifying = False
                    track.verify_count = 0
                    self.get_logger().debug(
                        f'Static vehicle {track_id} leaving verification mode '
                        f'(distance: {dist:.1f}m)'
                    )
    
    def create_box_marker(self, center, size, color, marker_id, ns, frame_id, header_stamp):
        """创建3D框Marker"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = header_stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        
        r, g, b, a = color
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a
        
        cx, cy, cz = center
        l, w, h = size
        hl, hw, hh = l/2.0, w/2.0, h/2.0
        
        # 8个顶点
        corners = []
        for dz in [-hh, hh]:
            for dy in [-hw, hw]:
                for dx in [-hl, hl]:
                    corners.append([cx+dx, cy+dy, cz+dz])
        corners = np.array(corners)
        
        # 12条边
        edges = [
            (0,1), (1,3), (3,2), (2,0),  # 底面
            (4,5), (5,7), (7,6), (6,4),  # 顶面
            (0,4), (1,5), (2,6), (3,7),  # 垂直边
        ]
        
        for start_idx, end_idx in edges:
            p1 = Point()
            p1.x = corners[start_idx][0]
            p1.y = corners[start_idx][1]
            p1.z = corners[start_idx][2]
            marker.points.append(p1)
            
            p2 = Point()
            p2.x = corners[end_idx][0]
            p2.y = corners[end_idx][1]
            p2.z = corners[end_idx][2]
            marker.points.append(p2)
        
        return marker

    def create_text_marker(self, text, position, marker_id, ns, frame_id, header_stamp, color=(1.0,1.0,1.0,1.0), scale=1.0):
        """创建文本Marker"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = header_stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.w = 1.0
        marker.scale.z = scale
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.text = text
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

    # 推理 + 过滤 + 跟踪 + 运动判断
    def _pointcloud_callback(self, msg: PointCloud2):
        # ---- 跳帧逻辑 ----
        self._frame_counter += 1
        if self._frame_counter % self._process_every_n_frames != 0:
            return
        
        # ---- 获取TF变换 ----
        source_frame = msg.header.frame_id
        target_frame = self.target_frame
        
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),  # 使用最新变换，避免时间戳超前问题
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warning(f'TF lookup failed: {e}')
            return
        
        R, t, yaw_offset = self._extract_transform_matrix(transform)
        
        # ---- 读取点云 ----
        pts = self._read_pointcloud(msg)
        if pts is None or len(pts) < 10:
            return

        # ---- 将点云xyz变换到map系（静态车验证时与first_center坐标系一致）----
        pts_map_xyz = self.transform_pointcloud(pts[:, :3], R, t)

        # ---- 推理 ----
        try:
            result, _ = inference_detector(self.model, pts)
        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')
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
            if class_name not in TARGET_CLASSES:
                continue
            
            # 获取lidar检测框信息
            bbox = bboxes[i]
            center = bbox.center
            if center.dim() == 2:
                center = center[0]  # 如果是 (1, 3) 就取第一行
            cx_local = float(center[0])
            cy_local = float(center[1])
            cz_local = float(center[2])
            # 提取bbox偏航角（lidar系）
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
            
            current_detections.append({
                'center': center_map,
                'class_name': class_name,
                'size': (l, w, h),
                'local_center': (cx_local, cy_local, cz_local),
                'yaw_local': yaw_local,
                'bbox': bbox
            })
        
        # ---- ID匹配----
        matches, unmatched_dets, unmatched_tracks = self.match_tracks(
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
            is_moving, iou = self.check_movement(
                det['center'], 
                track.first_center,
                det['size']
            )
            
            # 更新track状态：只允许静→动，不允许动→静（动态车不考虑重新变静）
            if is_moving:
                track.is_moving = True
            
            # 如果在验证期间匹配到了，取消验证
            if track.is_verifying:
                track.is_verifying = False
                track.verify_count = 0
                self.get_logger().info(f'Track {track_id} verified exist, cancel verification')
            
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
            
            det['track_id'] = new_track.id
            det['is_moving'] = False
            det['iou'] = 1.0
        
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
            else:
                # 静态车：不累计missed_frames
                if track.is_verifying:
                    # 检查激光是否扫到了这个位置
                    if self._is_area_scanned(pts_map_xyz, track.first_center, track.size):
                        track.verify_count += 1
                        # 连续5帧扫到了但都没匹配到，确认真没了
                        if track.verify_count >= self.verify_frames:
                            tracks_to_delete.append(track_id)
                            self.get_logger().info(
                                f'Static track {track_id} verified gone after '
                                f'{self.verify_frames} frames, deleting'
                            )
                    else:
                        # 激光没扫到，重置计数
                        track.verify_count = 0
        
        # ---- 执行删除 ----
        for tid in tracks_to_delete:
            if tid in self.tracks:
                del self.tracks[tid]
        
        # ---- 构建MarkerArray用于RViz可视化 ----
        marker_array = MarkerArray()
        marker_id = 0
        
        # 每帧先清除上一帧的所有Marker，避免残影在RViz中积累
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        #  先发匹配到的车
        for det in current_detections:
            if 'track_id' not in det:
                continue
                
            track_id = det['track_id']
            is_moving = det['is_moving']
            iou = det['iou']
            center_map = det['center']
            class_name = det['class_name']
            l, w, h = det['size']
            
            track = self.tracks[track_id]
            base_center = track.first_center
            
            # 提取bbox内点云
            cx_local, cy_local, cz_local = det['local_center']
            bbox_points_local = self._extract_bbox_points(
                pts, cx_local, cy_local, cz_local, l, w, h, 0.0
            )
            
            # 补全轮廓（使用模型输出的bbox朝向）
            completed_contour_local = self._fill_completed_contour(
                partial_points=bbox_points_local,
                class_name=class_name,
                bbox_center=(cx_local, cy_local, cz_local),
                bbox_size=(l, w, h),
                yaw=det.get('yaw_local', 0.0)
            )
            
            # 发布lidar系点云
            self._publish_completed_cloud(completed_contour_local, msg.header)
            
            # 发布map系点云
            if len(completed_contour_local) > 0:
                completed_contour_map = self.transform_pointcloud(
                    completed_contour_local, R, t
                )
                map_header = Header()
                map_header.stamp = msg.header.stamp
                map_header.frame_id = self.target_frame
                self._publish_completed_cloud_map(completed_contour_map, map_header)
            
            # ----- RViz 框 Marker -----
            # 基框：绿色半透明
            base_marker = self.create_box_marker(
                center=base_center,
                size=(l, w, h),
                color=(0.0, 1.0, 0.0, 0.3),
                marker_id=marker_id,
                ns="base_boxes",
                frame_id=self.target_frame,
                header_stamp=msg.header.stamp
            )
            marker_array.markers.append(base_marker)
            marker_id += 1
            
            # 当前框：红色(动)或蓝色(静)
            if is_moving:
                current_color = (1.0, 0.0, 0.0, 1.0)
            else:
                current_color = (0.0, 0.0, 1.0, 1.0)
            
            current_marker = self.create_box_marker(
                center=center_map,
                size=(l, w, h),
                color=current_color,
                marker_id=marker_id,
                ns="current_boxes",
                frame_id=self.target_frame,
                header_stamp=msg.header.stamp
            )
            marker_array.markers.append(current_marker)
            marker_id += 1
            
            # ----- 添加文本标签（类别和ID）-----
            text_pos = (center_map[0], center_map[1], center_map[2] + h/2 + 1.0)
            text = f"{class_name}\nID:{track_id}"
            text_marker = self.create_text_marker(
                text=text,
                position=text_pos,
                marker_id=marker_id,
                ns="class_labels",
                frame_id=self.target_frame,
                header_stamp=msg.header.stamp,
                color=(1.0,1.0,1.0,1.0),
                scale=1.0
            )
            marker_array.markers.append(text_marker)
            marker_id += 1
            
            # 位移连线（仅运动车辆）
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
                f'IoU:{iou:.2f} | {status_str}'
            )
        
        #  再发没匹配但活着的静态车
        for track_id, track in self.tracks.items():
            if track.is_matched or track.is_moving:
                continue  # 匹配过的或动态的跳过
            
            l, w, h = track.size
            cx, cy, cz = track.first_center
            
            # 生成实心点云（map系，使用记录时的朝向）
            completed_contour = self._fill_completed_contour(
                partial_points=np.array([]),
                class_name=track.class_name,
                bbox_center=(cx, cy, cz),
                bbox_size=(l, w, h),
                yaw=track.first_yaw
            )
            
            # 发布map系点云
            if len(completed_contour) > 0:
                map_header = Header()
                map_header.stamp = msg.header.stamp
                map_header.frame_id = self.target_frame
                self._publish_completed_cloud_map(completed_contour, map_header)
            
            # 发Marker：绿色或黄色  
            if track.is_verifying:
                color = (1.0, 1.0, 0.0, 0.8)  # 黄：验证中
                ns = "memory_verifying"
            else:
                color = (0.0, 1.0, 0.0, 0.3)  # 绿：纯记忆
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

            # 为记忆中的静态车也添加文本标签（可选）
            text_pos = (cx, cy, cz + h/2 + 1.0)
            text = f"{track.class_name}\nID:{track_id}\n(memory)"
            text_marker = self.create_text_marker(
                text=text,
                position=text_pos,
                marker_id=marker_id,
                ns="class_labels_memory",
                frame_id=self.target_frame,
                header_stamp=msg.header.stamp,
                color=(1.0,1.0,1.0,0.8),
                scale=0.8
            )
            marker_array.markers.append(text_marker)
            marker_id += 1
        
        # ---- 发布MarkerArray ----
        if len(marker_array.markers) > 0:
            self.vehicle_boxes_pub.publish(marker_array)

        # ---- 发布BEV图像（如果启用） ----
        if self.publish_image:
            self._publish_detection_image(pts, msg.header, R, t)

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
        z_vals = np.arange(-h/2, h/2, resolution)
        
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
    
    def _publish_completed_cloud(self, points: np.ndarray, header):
        """发布lidar系补全点云"""
        if len(points) == 0:
            return
            
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True
        msg.data = points.astype(np.float32).tobytes()
        
        self.completed_cloud_pub.publish(msg)
    
    def _publish_completed_cloud_map(self, points: np.ndarray, header):
        """发布补全后的点云"""
        if len(points) == 0:
            return
            
        msg = PointCloud2()
        msg.header = header
        msg.header.frame_id = self.target_frame
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True
        msg.data = points.astype(np.float32).tobytes()
        
        self.completed_cloud_map_pub.publish(msg)

    def _publish_detection_image(self, pts_lidar: np.ndarray, header, R, t):
        """生成并发布BEV图像，显示点云和检测框，并标注类别"""
        now = self.get_clock().now()
        if (now - self.last_image_time).nanoseconds * 1e-9 < self.image_interval:
            return

        try:
            # 将点云转换到地图系用于绘图
            pts_map = self.transform_pointcloud(pts_lidar[:, :3], R, t)
            # 保留强度信息（如果有）
            intensity = pts_lidar[:, 3] if pts_lidar.shape[1] >= 4 else np.zeros(len(pts_lidar))

            fig, ax = plt.subplots(1, 1, figsize=(self.image_width/100, self.image_height/100))
            ax.set_xlim([-self.image_range, self.image_range])
            ax.set_ylim([-self.image_range, self.image_range])

            # 绘制点云（降采样）
            if len(pts_map) > 0:
                step = max(1, len(pts_map) // 5000)
                pts_down = pts_map[::step]
                int_down = intensity[::step]
                mask = (np.abs(pts_down[:,0]) < self.image_range) & (np.abs(pts_down[:,1]) < self.image_range)
                pts_plot = pts_down[mask]
                int_plot = int_down[mask]
                if len(pts_plot) > 0:
                    if int_plot.max() > int_plot.min():
                        colors = (int_plot - int_plot.min()) / (int_plot.max() - int_plot.min() + 1e-6)
                    else:
                        colors = np.zeros_like(int_plot)
                    ax.scatter(pts_plot[:,0], pts_plot[:,1], s=0.5, c=colors, cmap='gray', alpha=0.5)

            # 绘制检测框 - 使用锁保护 tracks 的读取
            with self._tracks_lock:
                tracks_snapshot = dict(self.tracks)  # 避免迭代时修改

            for track_id, track in tracks_snapshot.items():
                # 获取最新位置
                if not track.is_matched and not track.is_moving:
                    # 未匹配的静态车使用 first_center
                    cx, cy = track.first_center[0], track.first_center[1]
                    l, w, _ = track.size
                    yaw = track.first_yaw
                else:
                    # 匹配上的车使用 last_center
                    cx, cy = track.last_center[0], track.last_center[1]
                    l, w, _ = track.size
                    yaw = 0.0  # 如果需要朝向，可从历史记录获取，这里简化

                if abs(cx) > self.image_range or abs(cy) > self.image_range:
                    continue

                # 计算矩形角点
                corners_local = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
                rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                corners = (rot @ corners_local.T).T + np.array([cx, cy])

                # 根据状态选择颜色
                if track.is_moving:
                    color = 'red'
                elif track.is_verifying:
                    color = 'yellow'
                else:
                    color = 'green'

                polygon = Polygon(corners, closed=True, linewidth=2, edgecolor=color,
                                  facecolor='none', alpha=0.8)
                ax.add_patch(polygon)

                # 标注类别、ID（可选速度）
                label = f"{track.class_name}\nID:{track_id}"
                ax.text(cx, cy + w/2 + 1, label, ha='center', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

            ax.set_aspect('equal')
            ax.set_title(f'车辆检测 - {len(self.tracks)}个目标')
            ax.grid(True, alpha=0.3)

            # 统计信息
            moving = sum(1 for t in self.tracks.values() if t.is_moving)
            static = len(self.tracks) - moving
            verifying = sum(1 for t in self.tracks.values() if t.is_verifying)
            ax.text(0.02, 0.98, f'移动:{moving} 静止:{static} 验证:{verifying}',
                    transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # 保存图像
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)

            img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, (self.image_width, self.image_height))
                img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
                img_msg.header = header
                self.image_pub.publish(img_msg)
                self.last_image_time = now
        except Exception as e:
            self.get_logger().error(f'图像发布失败: {e}')


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