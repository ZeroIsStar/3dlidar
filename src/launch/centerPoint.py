from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # 模型配置
        DeclareLaunchArgument('config_file', default_value='',
                              description='模型配置文件路径，留空使用默认配置'),
        DeclareLaunchArgument('checkpoint_file', default_value='',
                              description='权重文件路径，留空使用默认权重'),
        DeclareLaunchArgument('infer_device', default_value='cuda:0',
                              description='推理设备: cuda:0 或 cpu'),
        DeclareLaunchArgument('score_threshold', default_value='0.4',
                              description='检测置信度阈值'),

        # 输入输出话题
        DeclareLaunchArgument('pointcloud_topic', default_value='/vanjee/lidar',
                              description='输入点云话题'),
        DeclareLaunchArgument('point_cloud_qos', default_value='best_effort',
                              description='点云QoS: best_effort 或 reliable'),

        # 坐标系设置
        DeclareLaunchArgument('target_frame', default_value='map',
                              description='目标坐标系（TF变换目标）'),
        DeclareLaunchArgument('vehicle_boxes_frame', default_value='map',
                              description='vehicle_boxes 话题坐标系'),
        DeclareLaunchArgument('vehicle_raw_cloud_frame', default_value='map',
                              description='vehicle_raw_cloud 话题坐标系'),
        DeclareLaunchArgument('vehicle_outlines', default_value='map',
                              description='vehicle_outlines 话题坐标系'),

        # 跟踪参数
        DeclareLaunchArgument('match_distance_threshold', default_value='3.0',
                              description='ID匹配距离阈值(米)'),
        DeclareLaunchArgument('max_missed_frames', default_value='6',
                              description='最大丢失帧数，超过则删除跟踪'),
        DeclareLaunchArgument('ls_moving_param', default_value='0.3',
                              description='运动判断重合度阈值'),
        DeclareLaunchArgument('process_every_n_frames', default_value='10',
                              description='每N帧处理一次'),

        # 基框参数
        DeclareLaunchArgument('base_box_expand_x', default_value='2.0',
                              description='基框X方向延展(米)'),
        DeclareLaunchArgument('base_box_expand_y', default_value='5.0',
                              description='基框Y方向延展(米)'),

        # 静态车检测参数
        DeclareLaunchArgument('lidar_max_range', default_value='80.0',
                              description='激光雷达最大有效量程(米)'),
        DeclareLaunchArgument('max_static_tracks', default_value='200',
                              description='静态车记忆上限'),
        DeclareLaunchArgument('contour_slice_z', default_value='1.0',
                              description='发布点云的切面高度'),
        DeclareLaunchArgument('penetration_threshold', default_value='0.5',
                              description='激光穿透比例阈值'),

        Node(
            package='capella_mmdet3d_node',
            executable='centerPoint',
            name='centerpoint_detection_node',
            output='screen',
            parameters=[{
                'config_file': LaunchConfiguration('config_file'),
                'checkpoint_file': LaunchConfiguration('checkpoint_file'),
                'infer_device': LaunchConfiguration('infer_device'),
                'score_threshold': LaunchConfiguration('score_threshold'),
                'pointcloud_topic': LaunchConfiguration('pointcloud_topic'),
                'point_cloud_qos': LaunchConfiguration('point_cloud_qos'),
                'target_frame': LaunchConfiguration('target_frame'),
                'vehicle_boxes_frame': LaunchConfiguration('vehicle_boxes_frame'),
                'vehicle_raw_cloud_frame': LaunchConfiguration('vehicle_raw_cloud_frame'),
                'vehicle_outlines': LaunchConfiguration('vehicle_outlines'),
                'match_distance_threshold': LaunchConfiguration('match_distance_threshold'),
                'max_missed_frames': LaunchConfiguration('max_missed_frames'),
                'ls_moving_param': LaunchConfiguration('ls_moving_param'),
                'process_every_n_frames': LaunchConfiguration('process_every_n_frames'),
                'base_box_expand_x': LaunchConfiguration('base_box_expand_x'),
                'base_box_expand_y': LaunchConfiguration('base_box_expand_y'),
                'lidar_max_range': LaunchConfiguration('lidar_max_range'),
                'max_static_tracks': LaunchConfiguration('max_static_tracks'),
                'contour_slice_z': LaunchConfiguration('contour_slice_z'),
                'penetration_threshold': LaunchConfiguration('penetration_threshold'),
            }],
        ),
    ])
