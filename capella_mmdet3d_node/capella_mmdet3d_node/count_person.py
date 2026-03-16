#!/usr/bin/env python3
import os
import math
import tempfile
import numpy as np
from collections import deque
from threading import Lock

import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Float32MultiArray, Float32, Int32MultiArray, String
from cv_bridge import CvBridge
import tf_transformations
import cv2
import matplotlib

matplotlib.use('Agg')  # 使用无GUI后端
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from mmdet3d.apis import inference_detector, init_model

TARGET_CLASSES = {'ppedestrian'}


class PersonTrack:
    """行人跟踪器"""
    def __init__(self, track_id: int, cx: float, cy: float, cz: float, ts: float):
        self.track_id = track_id
        # 历史位置: deque of (timestamp, cx, cy, cz)
        self.history: deque = deque(maxlen=20)
        self.history.append((ts, cx, cy, cz))
        self.last_seen: float = ts
        self.is_moving: bool = False
        self.speed: float = 0.0
        # 用于 Mode-1 激光点重合率
        self.last_lidar_set: set | None = None
        # 来自检测结果的附加信息
        self.class_id: str = ''
        self.score: float = 0.0
        self.bbox: tuple = (0.0, 0.0, 0.0)  # (dx, dy, dz)
        self.yaw: float = 0.0
        # 速度历史，用于平滑显示
        self.speed_history: deque = deque(maxlen=5)

    def update(self, cx: float, cy: float, cz: float, ts: float):
        self.history.append((ts, cx, cy, cz))
        self.last_seen = ts

    def calc_speed_and_motion(self, threshold: float) -> tuple[float, bool]:
        """用历史队列最老和最新的位置估算平均速度, 返回 (speed_m_s, is_moving)."""
        if len(self.history) < 2:
            return 0.0, False
        t0, x0, y0, _ = self.history[0]
        t1, x1, y1, _ = self.history[-1]
        dt = t1 - t0
        if dt < 1e-6:
            return 0.0, False
        dist = math.hypot(x1 - x0, y1 - y0)
        speed = dist / dt
        self.speed = speed
        self.speed_history.append(speed)
        self.is_moving = speed > threshold
        return speed, self.is_moving


class PersonDetectionNode(Node):
    """
    订阅原始 PointCloud2 → mmdet3d 推理 → 过滤 Pedestrian → 追踪 → 运动判断 → 发布可视化数据
    """

    def __init__(self):
        super().__init__('person_detection_node')

        # ---- 参数声明 ----
        _pkg_dir = os.path.dirname(os.path.abspath(__file__))
        _demo_pth = os.path.join(_pkg_dir, 'mmdet3d_demo.pth')
        # 优先使用包内bundled config（新版mmdet3d格式），留空时才回退到从pth自动提取
        _bundled_cfg = os.path.join(
            _pkg_dir, 'configs', 'centerpoint',
            'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'
        )
        _default_cfg = _bundled_cfg if os.path.exists(_bundled_cfg) else ''
        self.declare_parameter('config_file', _default_cfg)  # mmdet3d 配置路径
        self.declare_parameter('checkpoint_file', _demo_pth)  # 权重路径
        self.declare_parameter('infer_device', 'cuda:0')
        self.declare_parameter('score_threshold', 0.5)
        self.declare_parameter('pointcloud_topic', '/vanjee/lidar')
        self.declare_parameter('point_cloud_qos', 'best_effort')
        self.declare_parameter('motion_threshold', 0.2)  # Mode-2 速度阈值 (m/s)
        self.declare_parameter('track_timeout', 3.0)  # 轨迹过期时间 (s)
        self.declare_parameter('assoc_max_dist', 2.0)  # 关联最大距离 (m)
        self.declare_parameter('person_motion_detection_mode', 1)  # 1=重合率  2=速度
        self.declare_parameter('lidar_overlap_resolution', 0.20)  # Mode-1 网格分辨率 (m)
        self.declare_parameter('max_lidar_overlap_ratio', 0.05)  # Mode-1 重合率阈值

        # 图像发布相关参数
        self.declare_parameter('publish_image', True)  # 是否发布图像
        self.declare_parameter('image_publish_interval', 1.0)  # 图像发布间隔（秒）
        self.declare_parameter('image_range', 50.0)  # 图像显示范围（米）
        self.declare_parameter('image_width', 800)  # 图像宽度
        self.declare_parameter('image_height', 600)  # 图像高度

        # rqt_plot相关参数
        self.declare_parameter('publish_plot_data', True)  # 是否发布plot数据
        self.declare_parameter('plot_publish_interval', 0.1)  # plot数据发布间隔（秒）
        self.declare_parameter('max_tracks_for_plot', 10)  # plot中显示的最大轨迹数

        config_file = self.get_parameter('config_file').value
        checkpoint_file = self.get_parameter('checkpoint_file').value
        device = self.get_parameter('infer_device').value
        self.score_thr = self.get_parameter('score_threshold').value
        self.pc_topic = self.get_parameter('pointcloud_topic').value
        pc_qos_str = self.get_parameter('point_cloud_qos').value
        self.motion_thr = self.get_parameter('motion_threshold').value
        self.track_to = self.get_parameter('track_timeout').value
        self.assoc_dist = self.get_parameter('assoc_max_dist').value
        self.motion_mode = self.get_parameter('person_motion_detection_mode').value
        self.lidar_res = self.get_parameter('lidar_overlap_resolution').value
        self.max_overlap = self.get_parameter('max_lidar_overlap_ratio').value

        # 图像相关参数
        self.publish_image = self.get_parameter('publish_image').value
        self.image_interval = self.get_parameter('image_publish_interval').value
        self.image_range = self.get_parameter('image_range').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value

        # rqt_plot相关参数
        self.publish_plot_data = self.get_parameter('publish_plot_data').value
        self.plot_interval = self.get_parameter('plot_publish_interval').value
        self.max_tracks_plot = self.get_parameter('max_tracks_for_plot').value

        # ---- 加载模型 ----
        # 若未指定 config_file, 则从 checkpoint 的 meta 字段自动提取并写入临时文件
        self._tmp_config_file = None
        if not config_file:
            self.get_logger().info(f'config_file 未指定, 尝试从 checkpoint 自动提取: {checkpoint_file}')
            try:
                ck = torch.load(checkpoint_file, map_location='cpu')
                config_str = ck.get('meta', {}).get('config', None)
                if not config_str:
                    raise RuntimeError('checkpoint 中没有内嵌 config 信息, 请手动指定 config_file 参数')
                tmp = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.py', delete=False, prefix='mmdet3d_auto_cfg_')
                tmp.write(config_str)
                tmp.close()
                config_file = tmp.name
                self._tmp_config_file = tmp.name
                self.get_logger().info(f'自动提取 config 成功, 临时路径: {config_file}')
            except Exception as e:
                self.get_logger().fatal(
                    f'自动提取 config 失败: {e}\n'
                    f'请手动传参: ros2 run your_package person_detection_node --ros-args '
                    f'-p config_file:=/path/to/config.py'
                )
                raise SystemExit(1)
        self.get_logger().info(f'Loading model: {config_file}  |  {checkpoint_file}')
        self.model = init_model(config_file, checkpoint_file, device=device)
        self.class_names: list[str] = list(self.model.dataset_meta.get('classes', []))
        self.get_logger().info(f'Model loaded. classes={self.class_names}')

        # ---- 内部状态 ----
        self._lock = Lock()
        self._tracks: dict[int, PersonTrack] = {}
        self._next_id = 0

        # 用于rqt_plot的发布器字典
        self._track_pubs: dict[int, dict] = {}  # 存储每个track的发布器

        # ---- QoS ----
        qos = QoSProfile(depth=5)
        if pc_qos_str == 'best_effort':
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
        else:
            qos.reliability = ReliabilityPolicy.RELIABLE

        # ---- 订阅原始点云 ----
        self.pc_sub = self.create_subscription(
            PointCloud2, self.pc_topic, self._pointcloud_callback, qos)

        # ---- RViz可视化发布 ----
        self.marker_pub = self.create_publisher(MarkerArray, '/person_markers', 10)

        # ---- 图像发布 ----
        if self.publish_image:
            self.image_pub = self.create_publisher(Image, '/person_detection_image', 10)
            self.bridge = CvBridge()
            self.last_image_time = self.get_clock().now()

        # ---- rqt_plot数据发布 ----
        if self.publish_plot_data:
            # 聚合数据发布器
            self.speed_array_pub = self.create_publisher(Float32MultiArray, '/person_speeds', 10)
            self.count_pub = self.create_publisher(Int32MultiArray, '/person_counts', 10)
            self.stats_pub = self.create_publisher(String, '/person_stats', 10)

            # 为每个可能的track创建独立话题（动态创建）
            self._create_track_publishers()

            # 定时发布plot数据
            self.create_timer(self.plot_interval, self._publish_plot_data)

        # ---- 定时清理过期轨迹 ----
        self.create_timer(1.0, self._cleanup_tracks)

        self.get_logger().info(
            f'PersonDetectionNode ready | topic={self.pc_topic} | '
            f'score_thr={self.score_thr} | motion_mode={self.motion_mode} | '
            f'publish_image={self.publish_image} | publish_plot_data={self.publish_plot_data}'
        )
    #....................
    # 读取点云
    # ....................
    def _read_pointcloud(self, msg: PointCloud2) -> np.ndarray | None:
        """
        将 PointCloud2 转换为 (N, 5) float32 数组 [x, y, z, intensity, ring].
        CenterPoint nuScenes 模型要求 5 维输入 (load_dim=5, use_dim=5).
        """
        field_names_avail = [f.name for f in msg.fields]
        has_intensity = 'intensity' in field_names_avail
        has_ring = 'ring' in field_names_avail

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

    # 推理 + 过滤 + 轨迹更新 + 发布
    def _pointcloud_callback(self, msg: PointCloud2):
        pts = self._read_pointcloud(msg)
        if pts is None or len(pts) < 10:
            return

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
            # 没有检测到目标时，仍然发布空数据
            if self.publish_plot_data:
                self._publish_empty_plot_data()
            return
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # ---- 提取检测目标 ----
        persons = []
        for i in range(len(labels)):
            label = int(labels[i].item())
            if label >= len(self.class_names):
                continue
            class_name = self.class_names[label]
            if class_name not in TARGET_CLASSES:
                continue
            cx = float(bboxes.center[i][0])
            cy = float(bboxes.center[i][1])
            cz = float(bboxes.center[i][2])
            dx = float(bboxes.dims[i][0])
            dy = float(bboxes.dims[i][1])
            dz = float(bboxes.dims[i][2])
            yaw = float(bboxes.tensor[i][-1])
            persons.append({
                'class_id': class_name, 'score': float(scores[i]),
                'cx': cx, 'cy': cy, 'cz': cz,
                'dx': dx, 'dy': dy, 'dz': dz,
                'yaw': yaw,
            })

        # ---- 轨迹关联 + 运动判断 ----
        now = self.get_clock().now().nanoseconds * 1e-9
        with self._lock:
            self._associate_and_update(persons, now, pts)

        # ---- RViz 发布 ----
        self._publish_markers(msg.header)

        # ---- 图像发布 ----
        if self.publish_image:
            self._publish_detection_image(pts, msg.header)

    # 贪心最近邻关联
    def _associate_and_update(self, detections: list, now: float, pts: np.ndarray):
        unmatched_idx = list(range(len(detections)))

        for track in self._tracks.values():
            if not unmatched_idx:
                break
            _, tx, ty, tz = track.history[-1]
            best_idx, best_d = None, self.assoc_dist

            for idx in unmatched_idx:
                d = detections[idx]
                dist = math.hypot(d['cx'] - tx, d['cy'] - ty)
                if dist < best_d:
                    best_d, best_idx = dist, idx

            if best_idx is not None:
                d = detections[best_idx]
                track.update(d['cx'], d['cy'], d['cz'], now)
                track.class_id = d['class_id']
                track.score = d['score']
                track.bbox = (d['dx'], d['dy'], d['dz'])
                track.yaw = d['yaw']
                unmatched_idx.remove(best_idx)
                self._update_motion(track, d, pts)

        # 未匹配的检测 → 新建轨迹
        for idx in unmatched_idx:
            d = detections[idx]
            tid = self._next_id;
            self._next_id += 1
            t = PersonTrack(tid, d['cx'], d['cy'], d['cz'], now)
            t.class_id = d['class_id']
            t.score = d['score']
            t.bbox = (d['dx'], d['dy'], d['dz'])
            t.yaw = d['yaw']
            self._tracks[tid] = t

            # 为新轨迹创建plot发布器
            if self.publish_plot_data:
                self._create_track_publishers_for_id(tid)

        # 实时日志
        for tid, track in self._tracks.items():
            if track.last_seen >= now - 1.0:
                state = "MOVING" if track.is_moving else "static"
                self.get_logger().info(
                    f"[id={tid}] {track.class_id}  score={track.score:.2f}  "
                    f"pos=({track.history[-1][1]:.2f}, {track.history[-1][2]:.2f})  "
                    f"speed={track.speed:.2f} m/s  [{state}]",
                    throttle_duration_sec=1.0
                )

    # 运动状态判断分发
    def _update_motion(self, track: PersonTrack, det: dict, pts: np.ndarray):
        if self.motion_mode == 1:
            self._motion_lidar_overlap(track, det, pts)
        else:
            track.calc_speed_and_motion(self.motion_thr)

    # Mode 1: 激光点网格重合率
    def _motion_lidar_overlap(self, track: PersonTrack, det: dict, pts: np.ndarray):
        cx, cy, cz = det['cx'], det['cy'], det['cz']
        dx, dy, dz = det['dx'], det['dy'], det['dz']
        margin = 0.1

        mask = (
                (pts[:, 0] >= cx - dx / 2 - margin) & (pts[:, 0] <= cx + dx / 2 + margin) &
                (pts[:, 1] >= cy - dy / 2 - margin) & (pts[:, 1] <= cy + dy / 2 + margin) &
                (pts[:, 2] >= cz - dz / 2 - margin) & (pts[:, 2] <= cz + dz / 2 + margin)
        )
        cur_pts = pts[mask]

        if cur_pts.shape[0] < 3:
            track.calc_speed_and_motion(self.motion_thr)
            return

        def _quantise(p: np.ndarray) -> set:
            cells = (p[:, :2] / self.lidar_res).astype(np.int32)
            return set(map(tuple, cells.tolist()))

        cur_set = _quantise(cur_pts)

        if track.last_lidar_set is None:
            track.last_lidar_set = cur_set
            track.is_moving = False
            return

        prev_set = track.last_lidar_set
        track.last_lidar_set = cur_set

        if not cur_set:
            track.is_moving = False
            return

        overlap_ratio = len(cur_set & prev_set) / len(cur_set)
        track.is_moving = overlap_ratio < self.max_overlap
        # 同时计算速度用于显示
        track.calc_speed_and_motion(self.motion_thr)

    # 发布 RViz MarkerArray
    def _publish_markers(self, header):
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        with self._lock:
            for tid, track in self._tracks.items():
                if not track.history:
                    continue
                _, cx, cy, cz = track.history[-1]
                dx, dy, dz = track.bbox
                q = tf_transformations.quaternion_from_euler(0.0, 0.0, track.yaw)

                # 包围盒
                box = Marker()
                box.header = header
                box.ns = 'person_bbox'
                box.id = tid
                box.type = Marker.CUBE
                box.action = Marker.ADD
                box.pose.position.x = cx
                box.pose.position.y = cy
                box.pose.position.z = cz
                box.pose.orientation.x = q[0]
                box.pose.orientation.y = q[1]
                box.pose.orientation.z = q[2]
                box.pose.orientation.w = q[3]
                box.scale.x, box.scale.y, box.scale.z = max(dx, 0.1), max(dy, 0.1), max(dz, 0.1)
                if track.is_moving:
                    box.color.r, box.color.g, box.color.b, box.color.a = 0.2, 0.4, 1.0, 0.45
                else:
                    box.color.r, box.color.g, box.color.b, box.color.a = 0.2, 0.8, 0.8, 0.45
                ma.markers.append(box)

                # 文字标签
                lbl = Marker()
                lbl.header = header
                lbl.ns = 'person_label'
                lbl.id = tid + 100000
                lbl.type = Marker.TEXT_VIEW_FACING
                lbl.action = Marker.ADD
                lbl.pose.position.x = cx
                lbl.pose.position.y = cy
                lbl.pose.position.z = cz + dz / 2 + 0.4
                lbl.pose.orientation.w = 1.0
                lbl.scale.z = 0.5
                lbl.color.r = lbl.color.g = lbl.color.b = lbl.color.a = 1.0
                state = "MOVING" if track.is_moving else "static"
                lbl.text = (
                    f"ID:{tid} {track.speed:.2f}m/s\n"
                    f"[{state}]"
                )
                ma.markers.append(lbl)

        self.marker_pub.publish(ma)

    # 发布检测图像
    def _publish_detection_image(self, pts: np.ndarray, header):
        """生成并发布检测结果的BEV图像"""
        now = self.get_clock().now()

        if (now - self.last_image_time).nanoseconds * 1e-9 < self.image_interval:
            return

        try:
            # 设置matplotlib后端
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon
            import io

            # 创建图形 - 使用更简单的方法
            fig, ax = plt.subplots(1, 1, figsize=(self.image_width / 100, self.image_height / 100))

            # 设置坐标轴范围
            ax.set_xlim([-self.image_range, self.image_range])
            ax.set_ylim([-self.image_range, self.image_range])

            # 绘制点云（降采样以提高性能）
            if len(pts) > 0:
                step = max(1, len(pts) // 5000)  # 最多显示5000个点
                pts_downsampled = pts[::step]

                # 只显示范围内的点
                mask = (np.abs(pts_downsampled[:, 0]) < self.image_range) & \
                       (np.abs(pts_downsampled[:, 1]) < self.image_range)
                pts_filtered = pts_downsampled[mask]

                if len(pts_filtered) > 0:
                    # 用强度值着色
                    colors = pts_filtered[:, 3]
                    if colors.max() > colors.min():
                        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-6)
                    ax.scatter(pts_filtered[:, 0], pts_filtered[:, 1],
                               s=0.5, c=colors, cmap='gray', alpha=0.5)

            # 绘制检测框
            with self._lock:
                for tid, track in self._tracks.items():
                    if not track.history:
                        continue

                    _, cx, cy, cz = track.history[-1]
                    dx, dy, dz = track.bbox

                    # 只绘制范围内的目标
                    if abs(cx) > self.image_range or abs(cy) > self.image_range:
                        continue

                    # 计算矩形角点（考虑朝向）
                    half_w, half_l = dx / 2, dy / 2
                    corners_local = np.array([
                        [half_l, half_w],
                        [half_l, -half_w],
                        [-half_l, -half_w],
                        [-half_l, half_w]
                    ])

                    rot = np.array([
                        [np.cos(track.yaw), -np.sin(track.yaw)],
                        [np.sin(track.yaw), np.cos(track.yaw)]
                    ])

                    corners_global = corners_local @ rot.T + np.array([cx, cy])

                    # 根据运动状态选择颜色
                    color = 'red' if track.is_moving else 'green'

                    # 绘制矩形
                    polygon = Polygon(corners_global, closed=True,
                                      linewidth=2, edgecolor=color,
                                      facecolor='none', alpha=0.8)
                    ax.add_patch(polygon)

                    # 添加标签
                    ax.text(cx, cy + dy / 2 + 1,
                            f"ID:{tid}\n{track.speed:.1f}m/s",
                            ha='center', fontsize=8, color=color,
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor='white', alpha=0.7))

                    # 绘制中心点
                    ax.plot(cx, cy, 'o', color=color, markersize=4)

            # 设置图形属性
            ax.set_aspect('equal')
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.set_title(f'实时目标检测 - {len(self._tracks)}个目标', fontsize=12)
            ax.grid(True, alpha=0.3)

            # 添加统计信息
            moving_count = sum(1 for t in self._tracks.values() if t.is_moving)
            static_count = len(self._tracks) - moving_count
            info_text = f"移动: {moving_count} | 静止: {static_count} | 总目标: {len(self._tracks)}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

            # 添加时间戳
            time_str = f"时间戳: {header.stamp.sec}.{header.stamp.nanosec:09d}"
            ax.text(0.98, 0.02, time_str, transform=ax.transAxes,
                    fontsize=8, horizontalalignment='right',
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.6))

            # 方法1: 使用缓冲区保存图像（推荐）
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
            buf.seek(0)

            # 将PNG转换为numpy数组
            img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # 关闭图形和缓冲区
            plt.close(fig)
            buf.close()

            if img is not None:
                # 调整图像大小到指定尺寸
                img = cv2.resize(img, (self.image_width, self.image_height))

                # 转换为ROS图像消息并发布
                img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                img_msg.header = header
                self.image_pub.publish(img_msg)

                self.last_image_time = now
                self.get_logger().debug('检测图像已发布')
            else:
                self.get_logger().error('图像解码失败')

        except Exception as e:
            self.get_logger().error(f'生成或发布图像失败: {e}')
            import traceback
            traceback.print_exc()

    # 创建所有track的发布器
    def _create_track_publishers(self):
        """预先创建一些track发布器（动态创建也可以）"""
        for i in range(self.max_tracks_plot):
            self._track_pubs[i] = {
                'speed': self.create_publisher(Float32, f'/track_{i}/speed', 10),
                'state': self.create_publisher(Float32, f'/track_{i}/state', 10),
                'x': self.create_publisher(Float32, f'/track_{i}/x', 10),
                'y': self.create_publisher(Float32, f'/track_{i}/y', 10)
            }

    def _create_track_publishers_for_id(self, tid):
        """为特定ID创建发布器"""
        if tid not in self._track_pubs and tid < self.max_tracks_plot:
            self._track_pubs[tid] = {
                'speed': self.create_publisher(Float32, f'/track_{tid}/speed', 10),
                'state': self.create_publisher(Float32, f'/track_{tid}/state', 10),
                'x': self.create_publisher(Float32, f'/track_{tid}/x', 10),
                'y': self.create_publisher(Float32, f'/track_{tid}/y', 10)
            }

    # 发布plot数据
    def _publish_plot_data(self):
        """发布用于rqt_plot的数据"""
        if not self.publish_plot_data:
            return

        with self._lock:
            if not self._tracks:
                self._publish_empty_plot_data()
                return

            now = self.get_clock().now()

            # 1. 发布速度数组
            speed_msg = Float32MultiArray()
            speed_data = []
            track_ids = []

            for tid, track in list(self._tracks.items())[:self.max_tracks_plot]:
                if track.history:
                    speed_data.append(track.speed)
                    track_ids.append(tid)

                    # 为每个track发布独立话题
                    if tid in self._track_pubs:
                        # 速度
                        speed_single = Float32()
                        speed_single.data = track.speed
                        self._track_pubs[tid]['speed'].publish(speed_single)

                        # 运动状态
                        state_single = Float32()
                        state_single.data = 1.0 if track.is_moving else 0.0
                        self._track_pubs[tid]['state'].publish(state_single)

                        # 位置
                        if track.history:
                            _, x, y, _ = track.history[-1]
                            x_msg = Float32();
                            x_msg.data = x
                            y_msg = Float32();
                            y_msg.data = y
                            self._track_pubs[tid]['x'].publish(x_msg)
                            self._track_pubs[tid]['y'].publish(y_msg)

            speed_msg.data = speed_data
            self.speed_array_pub.publish(speed_msg)

            # 2. 发布计数数据
            count_msg = Int32MultiArray()
            moving_count = sum(1 for t in self._tracks.values() if t.is_moving)
            static_count = len(self._tracks) - moving_count
            count_msg.data = [len(self._tracks), moving_count, static_count]
            self.count_pub.publish(count_msg)

            # 3. 发布统计字符串（用于显示）
            stats_msg = String()
            stats_msg.data = f"总目标:{len(self._tracks)} 移动:{moving_count} 静止:{static_count}"
            self.stats_pub.publish(stats_msg)

    def _publish_empty_plot_data(self):
        """发布空数据"""
        speed_msg = Float32MultiArray()
        speed_msg.data = []
        self.speed_array_pub.publish(speed_msg)

        count_msg = Int32MultiArray()
        count_msg.data = [0, 0, 0]
        self.count_pub.publish(count_msg)

        stats_msg = String()
        stats_msg.data = "无目标"
        self.stats_pub.publish(stats_msg)

    # 定时清理过期轨迹
    def _cleanup_tracks(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        with self._lock:
            stale = [tid for tid, t in self._tracks.items()
                     if now - t.last_seen > self.track_to]
            for tid in stale:
                del self._tracks[tid]
                # 可以选择是否清理发布器（保留以便重用）
            if stale:
                self.get_logger().debug(f"清理过期轨迹: {stale}")


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if node._tmp_config_file and os.path.exists(node._tmp_config_file):
            os.unlink(node._tmp_config_file)
        rclpy.shutdown()


if __name__ == '__main__':
    main()