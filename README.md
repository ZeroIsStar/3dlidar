# 3D 行人检测节点技术文档

## 1. 概述

该 ROS2 节点基于 **MMDetection3D** 框架和 **PointPillars** 模型，对机器人周围点云数据进行实时 3D 行人检测。节点支持 GPU 加速、动态阈值过滤、检测范围可视化，并输出人群密度与压力指标，适用于移动机器人环境感知与人群分析。

**主要特性：**
- 使用 PointPillars 模型（nuScenes 数据集预训练）检测 3D 行人边界框
- 动态置信度阈值：近处行人阈值低，远处阈值高，提高远距离召回率
- 点云降采样与半径过滤，降低计算负载
- 发布检测框可视化（MarkerArray）和统计指标（密度、压力）
- 可选发布降采样后的点云话题

## 2. 核心逻辑

### 2.1 整体流程
点云消息 → 读取并过滤（半径范围）→ 降采样 →
MMDetection3D 推理 → 获取 3D 边界框 → 动态阈值过滤 →
计算网格密度 → 更新可视化 Marker → 发布结果

### 2.2 关键模块详解

#### 2.2.1 模型加载
- 支持从配置文件和检查点加载 PointPillars 模型。
- 如果配置文件路径为空，则从检查点文件（`ckpt`）的 `meta` 字段中提取配置字符串，并保存为临时 `.py` 文件。
- 模型运行在指定设备（`cuda:0` 或 `cpu`），推理时关闭梯度计算（`torch.set_grad_enabled(False)`）。

#### 2.2.2 点云预处理
- **快速读取**：通过 `numpy.frombuffer` 解析 `PointCloud2` 消息，提取 `x, y, z` 坐标。
- **半径过滤**：保留距离原点 ≤ `detection_radius + 5.0` 的点，减少无关区域。
- **降采样**：若点数超过 `max_points_per_frame`，均匀步长采样至限制数量。
- **可选发布**：若 `publish_downsampled_cloud = true`，将降采样后的点云以 `PointCloud2` 形式发布到 `/filtered_pointcloud` 话题。

#### 2.2.3 推理与检测
- 调用 `inference_detector(model, points)` 得到 `Det3DDataSample`。
- 提取边界框中心坐标 `(cx, cy, cz)`、尺寸 `(dx, dy, dz)`、偏航角 `yaw` 以及置信度 `score` 和类别标签。

#### 2.2.4 动态阈值过滤
- 根据检测框到机器人的距离 `dist` 动态调整阈值：dynamic_thr = score_thr * (1 - score_decay * dist / detection_radius)
dynamic_thr = max(min_score_thr, dynamic_thr)

- 仅保留 `score > dynamic_thr` 且类别为 `pedestrian` 的检测结果。
- 作用：近处行人更易被检测，降低漏检；远处行人要求更高置信度，减少误检。

#### 2.2.5 密度与压力计算
- **网格密度**：
- 将检测框中心投影到 `x-y` 平面。
- 划分边长为 `grid_size`（米）的网格，统计每个网格内人数。
- 最大网格密度 = 最大人数 / 网格面积（人/平方米）。
- **人群压力**：当前实现直接使用 `max_density` 作为压力值发布（可扩展为更复杂模型）。

#### 2.2.6 可视化管理
- **Marker 唯一 ID 管理**：
- 使用 `_active_markers` 字典存储检测框索引与 Marker ID 的映射。
- 新增检测框分配递增 ID，移除的检测框发送 `DELETE` 消息。
- 确保发布框数量与检测数量严格一致，避免 RViz 中遗留旧框。
- **定时发布**：10Hz 定时器调用 `_publish_detection_markers()`，将当前检测结果渲染为 3D 立方体（颜色随置信度变化）。
- **辅助 Marker**：检测范围圆形（LINE_STRIP）和信息文本（TEXT_VIEW_FACING）。

## 3. 参数说明

所有参数可通过 ROS2 参数服务器配置（launch 文件或命令行）。

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `config_file` | string | 自动生成或空 | MMDetection3D 配置文件路径 |
| `checkpoint_file` | string | `configs/pth/pointpillars_hv_fpn_nus.pth` | 模型权重文件路径 |
| `infer_device` | string | `cuda:0` | 推理设备（`cuda:0` 或 `cpu`） |
| `score_threshold` | double | 0.5 | 基础置信度阈值 |
| `min_score_threshold` | double | 0.3 | 最小阈值（动态调整下限） |
| `score_decay_factor` | double | 0.9 | 阈值衰减系数（距离相关） |
| `pointcloud_topic` | string | `/vanjee/lidar` | 输入点云话题 |
| `point_cloud_qos` | string | `best_effort` | 点云订阅 QoS（`best_effort` 或 `reliable`） |
| `detection_radius` | double | 30.0 | 检测范围半径（米） |
| `grid_size` | double | 2.0 | 密度计算网格边长（米） |
| `process_every_n_frames` | int | 2 | 每 N 帧处理一次（跳帧） |
| `max_points_per_frame` | int | 20000 | 每帧最大处理点数 |
| `publish_downsampled_cloud` | bool | False | 是否发布降采样后的点云 |
| `pointcloud_downsample_ratio` | double | 0.15 | 发布点云时的降采样比例 |
| `use_gpu_for_points` | bool | True | 是否强制使用 GPU（若可用） |

## 4. 输出话题

| 话题名 | 消息类型 | 说明 |
|--------|----------|------|
| `/filtered_pointcloud` | `PointCloud2` | 降采样后的点云（仅当 `publish_downsampled_cloud=true` 时发布） |
| `/detection_boxes` | `MarkerArray` | 3D 检测框可视化（立方体 + 范围圆 + 信息文本） |
| `/max_density` | `Float32` | 最大网格密度（人/平方米） |
| `/crowd_pressure` | `Float32` | 人群压力值（当前等于最大密度） |

## 5. 依赖与环境要求

### 5.1 软件依赖
- ROS2 (Humble 或更高)
- Python 3.8+
- PyTorch 1.10+（支持 CUDA 可选）
- MMDetection3D >= 1.0.0
- 其他 Python 库：`numpy`, `sensor_msgs_py`, `tf2_ros`, `torch`

### 5.2 硬件要求
- **GPU**：推荐 NVIDIA GPU（至少 4GB 显存），CPU 模式推理速度较慢
- **内存**：≥ 8GB

### 5.3 模型文件
需要准备 PointPillars 模型配置和检查点文件。默认路径：
- 配置：`configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py`
- 权重：`configs/pth/pointpillars_hv_fpn_nus.pth`

## 6. 注意事项

1. **性能优化**：
 - 点云处理线程和推理在同一回调中进行，建议设置 `process_every_n_frames=2` 降低计算频率。
 - 通过 `max_points_per_frame` 限制点数量，避免 GPU 内存溢出。
 - GPU 内存每 20 帧清理一次（`torch.cuda.empty_cache()`）。

2. **动态阈值效果**：
 - 在 `detection_radius` 边界处，阈值会衰减到 `min_score_threshold`，确保远处行人仍有机会被检测。
 - 若远距离误检多，可提高 `score_threshold` 或降低 `score_decay_factor`。

3. **坐标系约定**：
 - 点云坐标系为机器人坐标系：x 前，y 左，z 上。
 - 检测框的偏航角 `yaw` 遵循右手定则（绕 Z 轴旋转）。

4. **可视化**：
 - 检测框颜色随置信度从红（低）到绿（高）渐变，透明度也随置信度变化。
 - 检测范围圆环半径 = `detection_radius`。

5. **TF 支持**：
 - 代码中初始化了 `tf2_ros.Buffer` 和 `TransformListener`，但未实际使用坐标变换。若点云不在机器人坐标系，可扩展实现转换逻辑。

6. **临时配置文件**：
 - 当未提供 `config_file` 时，节点会从检查点中提取配置并创建临时 `.py` 文件，退出时自动删除。

## 7. 启动示例

```bash
# 设置参数并运行
ros2 run capella_mmdet3d_node crowd_statistics \
--ros-args -p checkpoint_file:=/path/to/model.pth \
           -p config_file:=/path/to/config.py \
           -p detection_radius:=25.0 \
           -p publish_downsampled_cloud:=true
```
