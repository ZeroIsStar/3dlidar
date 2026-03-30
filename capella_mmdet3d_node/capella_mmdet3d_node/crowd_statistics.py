#!/usr/bin/env python3
import os
import math
import tempfile
import numpy as np
from threading import Lock

import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Float32
from geometry_msgs.msg import Point, Quaternion

from mmdet3d.apis import inference_detector, init_model
import tf_transformations

TARGET_CLASSES = {'pedestrian'}

# 点云字段映射
_PC_DTYPE_MAP = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2',
                 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}

# 点云发布字段
_PC_FIELDS = [
    pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
    pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
    pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
    pc2.PointField(name='intensity', offset=12, datatype=pc2.PointField.FLOAT32, count=1),
]

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

class PersonDetectionNode(Node):
    """行人检测节点 - 10Hz版本，简化参数"""

    def __init__(self):
        super().__init__('person_detection_node')

        # ---- 参数声明 ----
        self._declare_parameters()
        
        # ---- 加载模型 ----
        self._load_model()
        
        # ---- 内部状态 ----
        self._frame_counter = 0
        
        self.last_max_density = 0.0
        self.last_pressure = 0.0
        
        # 保存当前检测结果用于可视化
        self._current_detections = []
        self._last_marker_header = None
        self._detection_lock = Lock()
        
        # Marker管理：使用唯一ID确保发布数量与检测数量一致
        self._next_marker_id = 0
        self._active_markers = {}  # {detection_index: marker_id}

        # ---- QoS配置 ----
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT if self.pc_qos_str == 'best_effort' 
                      else ReliabilityPolicy.RELIABLE
        )
        
        pc_pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        # ---- 订阅 ----
        self.pc_sub = self.create_subscription(
            PointCloud2, self.pc_topic, self._pointcloud_callback, qos)
        
        # ---- 发布器 ----
        self.pc_pub = self.create_publisher(PointCloud2, '/filtered_pointcloud', pc_pub_qos)
        self.marker_pub = self.create_publisher(MarkerArray, '/detection_boxes', 10)
        self.density_pub = self.create_publisher(Float32, '/max_density', 10)
        self.pressure_pub = self.create_publisher(Float32, '/crowd_pressure', 10)

        # 定时发布Marker（10Hz）
        self.create_timer(0.1, self._publish_detection_markers)  # 10Hz
        
        self.get_logger().info(
            f'PersonDetectionNode ready | score_thr={self.score_thr} | '
            f'radius={self.detection_radius}m | device={self.device}'
        )
    
    def _declare_parameters(self):
        """声明所有参数"""
        _pkg_dir = os.path.dirname(os.path.abspath(__file__))
        _demo_pth = os.path.join(_pkg_dir, 'configs', 'pth', 'pointpillars_hv_fpn_nus.pth')
        
        _bundled_cfg = os.path.join(
            _pkg_dir, 'configs', 'pointpillars',
            'pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'
        )
        _default_cfg = _bundled_cfg if os.path.exists(_bundled_cfg) else ''
        
        self.declare_parameter('config_file', _default_cfg)
        self.declare_parameter('checkpoint_file', _demo_pth)
        self.declare_parameter('infer_device', 'cuda:0')
        self.declare_parameter('score_threshold', 0.5)
        self.declare_parameter('min_score_threshold', 0.3)
        self.declare_parameter('score_decay_factor', 0.9)
        self.declare_parameter('pointcloud_topic', '/vanjee/lidar')
        self.declare_parameter('point_cloud_qos', 'best_effort')
        self.declare_parameter('detection_radius', 30.0)
        self.declare_parameter('grid_size', 2.0)
        self.declare_parameter('process_every_n_frames', 2)
        self.declare_parameter('max_points_per_frame', 20000)
        self.declare_parameter('publish_downsampled_cloud', False)
        self.declare_parameter('pointcloud_downsample_ratio', 0.15)
        self.declare_parameter('use_gpu_for_points', True)
        
        # 获取参数值
        self.config_file = self.get_parameter('config_file').value
        self.checkpoint_file = self.get_parameter('checkpoint_file').value
        self.device = self.get_parameter('infer_device').value
        self.score_thr = self.get_parameter('score_threshold').value
        self.min_score_thr = self.get_parameter('min_score_threshold').value
        self.score_decay = self.get_parameter('score_decay_factor').value
        self.pc_topic = self.get_parameter('pointcloud_topic').value
        self.pc_qos_str = self.get_parameter('point_cloud_qos').value
        self.detection_radius = self.get_parameter('detection_radius').value
        self.grid_size = self.get_parameter('grid_size').value
        self.process_every_n_frames = self.get_parameter('process_every_n_frames').value
        self.max_points_per_frame = self.get_parameter('max_points_per_frame').value
        self.publish_downsampled = self.get_parameter('publish_downsampled_cloud').value
        self.cloud_downsample_ratio = self.get_parameter('pointcloud_downsample_ratio').value
        self.use_gpu_for_points = self.get_parameter('use_gpu_for_points').value
    
    def _load_model(self):
        """加载模型并设置推理模式"""
        self._tmp_config_file = None
        if not self.config_file:
            ck = torch.load(self.checkpoint_file, map_location='cpu')
            config_str = ck.get('meta', {}).get('config', None)
            if config_str:
                tmp = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.py', delete=False, prefix='mmdet3d_auto_cfg_')
                tmp.write(config_str)
                tmp.close()
                self.config_file = tmp.name
                self._tmp_config_file = tmp.name
        
        # 设置设备
        if self.use_gpu_for_points and torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
            
        self.model = init_model(self.config_file, self.checkpoint_file, device=self.device)
        
        # 设置为评估模式并禁用梯度
        self.model.eval()
        torch.set_grad_enabled(False)
        
        # GPU优化
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        
        self.class_names = list(self.model.dataset_meta.get('classes', []))
        self.get_logger().info(f'Model loaded on {self.device}')
    
    def _read_pointcloud_fast(self, msg: PointCloud2) -> np.ndarray | None:
        """快速读取点云并降采样"""
        n_pts = msg.width * msg.height
        if n_pts == 0:
            return None
        
        # 解析点云
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
        
        x = arr['x'].astype(np.float32, copy=False)
        y = arr['y'].astype(np.float32, copy=False)
        z = arr['z'].astype(np.float32, copy=False)
        
        # 快速过滤
        dist_sq = x*x + y*y
        radius_sq = (self.detection_radius + 5.0) ** 2
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (dist_sq <= radius_sq)
        
        n_valid = np.count_nonzero(valid)
        if n_valid < 10:
            return None
        
        # 降采样
        valid_indices = np.where(valid)[0]
        if n_valid > self.max_points_per_frame:
            step = n_valid // self.max_points_per_frame
            valid_indices = valid_indices[::step][:self.max_points_per_frame]
        
        # 构建结果数组
        result = np.empty((len(valid_indices), 5), dtype=np.float32)
        result[:, 0] = x[valid_indices]
        result[:, 1] = y[valid_indices]
        result[:, 2] = z[valid_indices]
        result[:, 3] = 0.0
        result[:, 4] = 0.0
        
        # 发布点云（可选）
        if self.publish_downsampled and self.pc_pub.get_subscription_count() > 0:
            self._publish_pointcloud(msg.header, result)
        
        return result
    
    def _publish_pointcloud(self, header, points: np.ndarray):
        """发布点云"""
        if self.cloud_downsample_ratio < 1.0 and len(points) > 1000:
            step = max(1, int(1.0 / self.cloud_downsample_ratio))
            points = points[::step]
        
        cloud_msg = pc2.create_cloud(header, _PC_FIELDS, points[:, :4])
        self.pc_pub.publish(cloud_msg)
    
    def _create_detection_range_marker(self, header):
        """创建检测范围Marker"""
        marker = Marker()
        marker.header = header
        marker.ns = 'detection_range'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        points = []
        n_segments = 64
        for i in range(n_segments + 1):
            angle = 2 * math.pi * i / n_segments
            x = self.detection_radius * math.cos(angle)
            y = self.detection_radius * math.sin(angle)
            p = Point()
            p.x, p.y, p.z = x, y, 0.0
            points.append(p)
        marker.points = points
        
        return marker
    
    def _create_detection_box_marker(self, header, marker_id: int, center_x: float, center_y: float, 
                                      center_z: float, dx: float, dy: float, dz: float, 
                                      yaw: float, score: float):
        """创建单个检测框Marker - 使用唯一ID"""
        marker = Marker()
        marker.header = header
        marker.ns = 'detection_boxes'
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # 设置位置（将底部对齐到地面）
        marker.pose.position.x = center_x
        marker.pose.position.y = center_y
        marker.pose.position.z = center_z + dz / 2
        
        # 设置方向
        quat = tf_transformations.quaternion_from_euler(0, 0, yaw)
        marker.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        
        # 设置尺寸
        marker.scale.x = dx
        marker.scale.y = dy
        marker.scale.z = dz
        
        # 颜色基于分数（高分为绿色，低分为红色）
        green_ratio = min(1.0, max(0.0, (score - 0.3) / 0.7))
        red_ratio = 1.0 - green_ratio
        
        marker.color.r = red_ratio
        marker.color.g = green_ratio
        marker.color.b = 0.0
        # 透明度基于分数
        marker.color.a = min(0.8, score + 0.2)
        marker.color.a = max(0.5, marker.color.a)
        
        return marker
    
    def _delete_marker(self, header, marker_id: int):
        """创建删除指定Marker的消息"""
        marker = Marker()
        marker.header = header
        marker.ns = 'detection_boxes'
        marker.id = marker_id
        marker.action = Marker.DELETE
        return marker
    
    def _create_information_marker(self, header):
        """创建信息文本Marker"""
        marker = Marker()
        marker.header = header
        marker.ns = 'info_text'
        marker.id = 999
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 3.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.text = f"Detection Radius: {self.detection_radius}m\n" \
                     f"Score Threshold: {self.score_thr:.2f}\n" \
                     f"People Count: {len(self._current_detections)}\n" \
                     f"Density: {self.last_max_density:.2f} p/m²"
        
        return marker
    
    def _pointcloud_callback(self, msg: PointCloud2):
        """点云回调 - 主要处理逻辑"""
        self._frame_counter += 1
        if self._frame_counter % self.process_every_n_frames != 0:
            return
        
        # 读取点云
        pts = self._read_pointcloud_fast(msg)
        if pts is None:
            return
        
        try:
            with torch.no_grad():
                result, _ = inference_detector(self.model, pts)
        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')
            return
        
        bboxes = result.pred_instances_3d.bboxes_3d
        scores = result.pred_instances_3d.scores_3d
        labels = result.pred_instances_3d.labels_3d
        
        if len(scores) == 0:
            # 清空当前检测
            with self._detection_lock:
                self._current_detections = []
                self._last_marker_header = msg.header
            self._publish_density_pressure(0.0, 0.0)
            return
        
        # 过滤检测结果
        cx_vals = np.array([float(bboxes.center[i][0]) for i in range(len(scores))])
        cy_vals = np.array([float(bboxes.center[i][1]) for i in range(len(scores))])
        distances = np.sqrt(cx_vals**2 + cy_vals**2)
        
        # 动态阈值
        effective_dist = np.minimum(distances, self.detection_radius)
        dynamic_thr = self.score_thr * (1.0 - self.score_decay * effective_dist / self.detection_radius)
        dynamic_thr = np.maximum(self.min_score_thr, dynamic_thr)
        
        scores_np = scores.cpu().numpy()
        keep_mask = scores_np > dynamic_thr
        keep_indices = np.where(keep_mask)[0]
        
        if len(keep_indices) == 0:
            with self._detection_lock:
                self._current_detections = []
                self._last_marker_header = msg.header
            self._publish_density_pressure(0.0, 0.0)
            return
        
        keep = torch.tensor(keep_indices, device=scores.device)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        # 提取行人检测
        detections = []
        labels_np = labels.cpu().numpy()
        for i in range(len(labels)):
            label = int(labels_np[i])
            if label >= len(self.class_names):
                continue
            class_name = self.class_names[label]
            if class_name not in TARGET_CLASSES:
                continue
            
            score = float(scores[i])
            if score < self.min_score_thr:
                continue
                
            detections.append({
                'score': score,
                'cx': float(bboxes.center[i][0]),
                'cy': float(bboxes.center[i][1]),
                'cz': float(bboxes.center[i][2]),
                'dx': float(bboxes.dims[i][0]),
                'dy': float(bboxes.dims[i][1]),
                'dz': float(bboxes.dims[i][2]),
                'yaw': float(bboxes.tensor[i][-1]),
            })
        
        # 保存检测结果
        with self._detection_lock:
            self._current_detections = detections
            self._last_marker_header = msg.header
        
        # 计算密度
        positions = [(d['cx'], d['cy']) for d in detections if math.hypot(d['cx'], d['cy']) <= self.detection_radius]
        
        if positions:
            grid_cells = {}
            for x, y in positions:
                grid_x = int(x / self.grid_size)
                grid_y = int(y / self.grid_size)
                key = (grid_x, grid_y)
                grid_cells[key] = grid_cells.get(key, 0) + 1
            
            cell_area = self.grid_size * self.grid_size
            max_count = max(grid_cells.values())
            max_density = max_count / cell_area
        else:
            max_density = 0.0
        
        self.last_max_density = max_density
        self.last_pressure = max_density
        
        # 发布数据
        self._publish_density_pressure(max_density, max_density)
        
        # 日志
        self.get_logger().info(
            f"Detections: {len(detections)} | Density: {max_density:.3f} | "
            f"Points: {len(pts)}",
            throttle_duration_sec=2.0
        )
        
        # 清理GPU缓存（每10帧清理一次）
        if 'cuda' in self.device and self._frame_counter % 20 == 0:
            torch.cuda.empty_cache()
    
    def _publish_detection_markers(self):
        """定时发布检测框Marker - 使用唯一ID管理，确保发布数量与检测数量一致"""
        if self._last_marker_header is None:
            return
        
        marker_array = MarkerArray()
        
        # 1. 添加检测范围
        range_marker = self._create_detection_range_marker(self._last_marker_header)
        marker_array.markers.append(range_marker)
        
        # 2. 管理检测框
        with self._detection_lock:
            current_count = len(self._current_detections)
            
            # 为新的检测生成唯一ID
            new_active_markers = {}
            for idx, det in enumerate(self._current_detections):
                # 检查是否已有对应的Marker ID
                if idx in self._active_markers:
                    marker_id = self._active_markers[idx]
                else:
                    # 生成新的唯一ID
                    marker_id = self._next_marker_id
                    self._next_marker_id += 1
                
                # 创建检测框Marker
                box = self._create_detection_box_marker(
                    self._last_marker_header, marker_id,
                    det['cx'], det['cy'], det['cz'],
                    det['dx'], det['dy'], det['dz'],
                    det['yaw'], det['score']
                )
                marker_array.markers.append(box)
                new_active_markers[idx] = marker_id
            
            # 3. 删除不再存在的检测框
            for old_idx, old_id in self._active_markers.items():
                if old_idx not in new_active_markers:
                    delete_marker = self._delete_marker(self._last_marker_header, old_id)
                    marker_array.markers.append(delete_marker)
                    self.get_logger().debug(f'Deleted marker {old_id} (index {old_idx})')
            
            # 更新活动标记
            self._active_markers = new_active_markers
            
            # 验证数量一致性
            if len(marker_array.markers) > 0:
                # 统计当前发布的检测框数量（排除范围和信息marker）
                box_markers = [m for m in marker_array.markers 
                              if m.ns == 'detection_boxes' and m.action == Marker.ADD]
                published_count = len(box_markers)
                
                if published_count != current_count:
                    self.get_logger().warn(
                        f'Marker count mismatch! Published: {published_count}, '
                        f'Detections: {current_count}, Active markers: {len(self._active_markers)}'
                    )
        
        # 4. 添加信息文本
        info_marker = self._create_information_marker(self._last_marker_header)
        marker_array.markers.append(info_marker)
        
        # 发布
        self.marker_pub.publish(marker_array)
    
    def _publish_density_pressure(self, max_density: float, pressure: float):
        """发布密度和压力"""
        msg_density = Float32()
        msg_density.data = max_density
        self.density_pub.publish(msg_density)
        
        msg_pressure = Float32()
        msg_pressure.data = pressure
        self.pressure_pub.publish(msg_pressure)


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectionNode()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if node._tmp_config_file and os.path.exists(node._tmp_config_file):
            os.unlink(node._tmp_config_file)
        rclpy.shutdown()


if __name__ == '__main__':
    main()
