from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'capella_mmdet3d_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        'capella_mmdet3d_node': [
            '*.pth',
            'configs/centerpoint/*.py',
            'configs/pointpillars/*.py',
            'configs/_base_/*.py',
            'configs/_base_/datasets/*.py',
            'configs/_base_/models/*.py',
            'configs/_base_/schedules/*.py',
            'configs/pth/*.pth',
            'configs/vehicle_priors/*.yaml',
            'demo_data/cjh_model/*',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.py'))),
        (os.path.join('share', package_name, 'demo_data', 'cjh_model', 'my_bag'),
            glob('capella_mmdet3d_node/demo_data/cjh_model/my_bag/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='linux',
    maintainer_email='3516551392@qq.com',
    description='MMDet3D ROS2 Node for 3D Object Detection',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'centerpoint = capella_mmdet3d_node.centerpoint:main',
            'pointpillar = capella_mmdet3d_node.pointpillar:main',
            'centerpointdcn = capella_mmdet3d_node.centerpointdcn:main',
            'democenter = capella_mmdet3d_node.democenter:main',
            'crowd_statistics = capella_mmdet3d_node.crowd_statistics:main',
        ],
    },
)
