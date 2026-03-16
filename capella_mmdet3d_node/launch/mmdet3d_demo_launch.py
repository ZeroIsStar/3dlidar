from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

_CONFIG_FILE = ''

_CHECKPOINT_FILE = ''

_LIDAR_TOPIC = '/lidar_points'

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config_file',      default_value=_CONFIG_FILE),
        DeclareLaunchArgument('checkpoint_file',  default_value=_CHECKPOINT_FILE),
        DeclareLaunchArgument('infer_device',     default_value='cuda:0'),
        DeclareLaunchArgument('score_threshold',  default_value='0.5'),
        DeclareLaunchArgument('pointcloud_topic', default_value=_LIDAR_TOPIC),
        DeclareLaunchArgument('point_cloud_qos',  default_value='best_effort'),
        DeclareLaunchArgument('vehicle_motion_detection_mode', default_value='1'),
        DeclareLaunchArgument('motion_threshold',              default_value='0.3'),
        DeclareLaunchArgument('max_lidar_overlap_ratio',       default_value='0.05'),

        Node(
            package='capella_mmdet3d_node',
            executable='mmdet3d_demo',
            name='vehicle_detection_node',
            output='screen',
            parameters=[{
                'config_file':      LaunchConfiguration('config_file'),
                'checkpoint_file':  LaunchConfiguration('checkpoint_file'),
                'infer_device':     LaunchConfiguration('infer_device'),
                'score_threshold':  LaunchConfiguration('score_threshold'),
                'pointcloud_topic': LaunchConfiguration('pointcloud_topic'),
                'point_cloud_qos':  LaunchConfiguration('point_cloud_qos'),
                'vehicle_motion_detection_mode':
                    LaunchConfiguration('vehicle_motion_detection_mode'),
                'motion_threshold': LaunchConfiguration('motion_threshold'),
                'max_lidar_overlap_ratio':
                    LaunchConfiguration('max_lidar_overlap_ratio'),
            }],
        ),
    ])
