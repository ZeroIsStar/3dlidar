#!/usr/bin/env python3
"""
MMDet3D ROS2 测试节点 - 用于单独测试模型推理性能和日志帧率
基于 centerpoint.py 简化而来，只保留核心推理功能
"""

import os
import sys
import time
import threading
from collections import deque

# 设置线程数限制
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

import tempfile
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import PointCloud2, PointField
from mmdet3d.apis import inference_detector, init_model

# 用这个映射表把 msg.fields 里的 datatype 转换成 NumPy 能理解的 dtype
_PC_DTYPE_MAP = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2',
                 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


class SimpleMmdet3dNode(Node):
    """简化版 MMDet3D 测试节点 - 仅用于测试推理性能"""
    
    def __init__(self):
        super().__init__('simple_mmdet3d_node')

        # ---- 参数声明 ----
        _pkg_dir = os.path.dirname(os.path.abspath(__file__))
        _demo_pth = os.path.join(_pkg_dir, 'configs', 'pth', 'centerpoint_0075voxel_second_secfpn.pth')
        
        # 优先使用包内bundled config
        _bundled_cfg = os.path.join(
            _pkg_dir, 'configs', 'centerpoint',
            'centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
        )
        _default_cfg = _bundled_cfg if os.path.exists(_bundled_cfg) else ''
        
        self.declare_parameter('config_file', _default_cfg)
        self.declare_parameter('checkpoint_file', _demo_pth)
        self.declare_parameter('infer_device', 'cuda:0')
        self.declare_parameter('score_threshold', 0.4)
        self.declare_parameter('pointcloud_topic', '/vanjee/lidar')
        self.declare_parameter('point_cloud_qos', 'best_effort')
        self.declare_parameter('max_range', 80.0)
        self.declare_parameter('log_interval_sec', 3.0)  # 日志输出间隔
        self.declare_parameter('process_every_n_frames', 1)  # 处理数据的帧数

        config_file = self.get_parameter('config_file').value
        checkpoint_file = self.get_parameter('checkpoint_file').value
        device = self.get_parameter('infer_device').value
        self.score_thr = self.get_parameter('score_threshold').value
        self.pc_topic = self.get_parameter('pointcloud_topic').value
        pc_qos_str = self.get_parameter('point_cloud_qos').value
        self.max_range = self.get_parameter('max_range').value
        self.log_interval_sec = self.get_parameter('log_interval_sec').value
        self.process_every_n_frames = self.get_parameter('process_every_n_frames').value

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

        self.get_logger().info(f'Loading model from {checkpoint_file}...')
        t0 = time.perf_counter()
        self.model = init_model(config_file, checkpoint_file, device=device)
        t1 = time.perf_counter()
        self.get_logger().info(f'Model loaded in {(t1-t0)*1000:.2f} ms')
        self.get_logger().info(f'Model device: {next(self.model.parameters()).device}')
        
        self.class_names = list(self.model.dataset_meta.get('classes', []))
        self.get_logger().info(f'Class names: {self.class_names}')

        # ---- Callback Groups ----
        self._infer_group = MutuallyExclusiveCallbackGroup()

        # ---- QoS ----
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
            PointCloud2, self.pc_topic, self._pointcloud_callback, qos, 
            callback_group=self._infer_group)
        
        self.get_logger().info(f'Subscribed to {self.pc_topic}')

        # ---- 性能统计 ----
        self._frame_counter = 0
        self._process_counter = 0
        self._inference_times = deque(maxlen=100)  # 保存最近100次推理时间
        self._last_log_time = time.time()
        self._lock = threading.Lock()

        self.get_logger().info(
            f'SimpleMmdet3dNode ready | device={device} | '
            f'score_thr={self.score_thr} | max_range={self.max_range}m'
        )

    def _read_pointcloud(self, msg: PointCloud2) -> np.ndarray | None:
        """将 PointCloud2 转换为 (N, 5) float32 数组 [x, y, z, intensity, ring]."""
        n_pts = msg.width * msg.height
        if n_pts == 0:
            return None

        # 根据字段描述构建结构化 dtype
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
        has_ring = 'ring' in field_names

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

    def _apply_range_filter(self, pts: np.ndarray) -> np.ndarray:
        """应用距离过滤"""
        if self.max_range <= 0:
            return pts
        distances = np.linalg.norm(pts[:, :3], axis=1)
        mask = distances <= self.max_range
        return pts[mask]

    def _pointcloud_callback(self, msg: PointCloud2):
        """点云回调函数"""
        self._frame_counter += 1
        
        # 跳帧逻辑
        if self._frame_counter % self.process_every_n_frames != 0:
            return

        loop_start = time.perf_counter()
        
        try:
            # ---- 读取点云 ----
            t0 = time.perf_counter()
            pts = self._read_pointcloud(msg)
            t1 = time.perf_counter()
            read_time = (t1 - t0) * 1000
            
            if pts is None or len(pts) < 10:
                self.get_logger().warning(f'Empty or invalid point cloud: {pts.shape if pts is not None else None}')
                return
            
            n_original = len(pts)
            
            # ---- 距离过滤 ----
            pts = self._apply_range_filter(pts)
            n_filtered = len(pts)
            
            # ---- 推理 ----
            t0 = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            result, _ = inference_detector(self.model, pts)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            infer_time = (t1 - t0) * 1000  # ms
            
            # ---- 解析结果 ----
            bboxes = result.pred_instances_3d.bboxes_3d
            scores = result.pred_instances_3d.scores_3d
            labels = result.pred_instances_3d.labels_3d
            
            # 置信度过滤
            keep = (scores > self.score_thr).nonzero(as_tuple=False).squeeze(1)
            n_detections = keep.numel()
            
            # ---- 更新统计 ----
            with self._lock:
                self._process_counter += 1
                self._inference_times.append(infer_time)
            
            # ---- 日志输出 ----
            loop_end = time.perf_counter()
            loop_time = (loop_end - loop_start) * 1000
            
            # 定期输出统计信息
            current_time = time.time()
            if current_time - self._last_log_time >= self.log_interval_sec:
                self._print_stats()
                self._last_log_time = current_time
            else:
                # 单次处理日志
                self.get_logger().info(
                    f'Frame {self._frame_counter}: '
                    f'points={n_original}->{n_filtered}, '
                    f'detections={n_detections}, '
                    f'infer={infer_time:.2f}ms, '
                    f'total={loop_time:.2f}ms'
                )

        except Exception as e:
            self.get_logger().error(f'Inference error: {e}')
            import traceback
            self.get_logger().debug(traceback.format_exc())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _print_stats(self):
        """打印性能统计"""
        with self._lock:
            if not self._inference_times:
                return
            
            times = list(self._inference_times)
            n_frames = len(times)
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            p50 = np.percentile(times, 50)
            p90 = np.percentile(times, 90)
            p99 = np.percentile(times, 99)
            
            # 计算 FPS
            if avg_time > 0:
                fps = 1000.0 / avg_time
            else:
                fps = 0
            
            self.get_logger().info(
                f'\n=== Performance Stats (last {n_frames} frames) ===\n'
                f'  Frames processed: {self._process_counter}/{self._frame_counter}\n'
                f'  Inference time: avg={avg_time:.2f}ms, min={min_time:.2f}ms, max={max_time:.2f}ms\n'
                f'  Percentiles: p50={p50:.2f}ms, p90={p90:.2f}ms, p99={p99:.2f}ms\n'
                f'  Estimated FPS: {fps:.1f}\n'
                f'========================================'
            )

    def destroy_node(self):
        """节点销毁时清理"""
        # 打印最终统计
        self._print_stats()
        
        # 清理临时文件
        if self._tmp_config_file and os.path.exists(self._tmp_config_file):
            try:
                os.unlink(self._tmp_config_file)
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    """主函数入口"""
    rclpy.init(args=args)
    
    try:
        node = SimpleMmdet3dNode()
    except Exception as e:
        print(f'Failed to create node: {e}')
        import traceback
        traceback.print_exc()
        rclpy.shutdown()
        return 1
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
