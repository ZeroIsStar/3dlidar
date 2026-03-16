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
            'configs/_base_/*.py',
            'configs/_base_/datasets/*.py',
            'configs/_base_/models/*.py',
            'configs/_base_/schedules/*.py',
            'configs/vehicle_priors/*.yaml',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
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
            'mmdet3d_demo = capella_mmdet3d_node.mmdet3d_demo:main',
            'bin_publisher = capella_mmdet3d_node.bin_publisher:main',
            'mmdet3d_vehicle = capella_mmdet3d_node.mmdet3d_vehicle:main',
            'count_person = capella_mmdet3d_node.count_person:main'
        ],
    },
)
