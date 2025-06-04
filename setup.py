from setuptools import setup

package_name = 'midas_depth_publisher'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name + '/launch', ['launch/midas_depth_launch.py']),
        ('share/' + package_name + '/models', ['model/model-f6b98070.onnx']),
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='MiDaS ROS 2 depth publisher',
    license='MIT',
    entry_points={
        'console_scripts': [
            'midas_depth_node = midas_depth_publisher.midas_depth_node:main',
        ],
    },
)
