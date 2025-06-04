#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge

import numpy as np
import cv2
import onnxruntime as ort


class MiDaSDepthNode(Node):
    def __init__(self):
        super().__init__('midas_depth_node')
        self.bridge = CvBridge()

        # 订阅图像和高度
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.height_sub = self.create_subscription(Float32, '/drone/height', self.height_callback, 10)

        # 发布绝对深度图
        self.depth_pub = self.create_publisher(Image, '/camera/depth_absolute', 10)

        # ONNX 模型加载
        model_path = '/home/comb/midas_ws/src/model/model-f6b98070.onnx'  # 替换为你的模型路径
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

        # 初始高度
        self.current_height = 1.6  # 默认高度，可被 /drone/height 替换
        self.get_logger().info('MiDaS Depth Node 初始化完成.')

    def height_callback(self, msg: Float32):
        self.current_height = msg.data

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, (256, 256))  # 根据模型要求修改大小
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        return image.astype(np.float32)

    def image_callback(self, msg: Image):
        try:
            # 图像解码
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            input_tensor = self.preprocess_image(cv_image)

            # MiDaS 推理
            output = self.session.run(None, {self.input_name: input_tensor})
            depth_relative = output[0][0]  # (H, W)

            # 尺度恢复
            mean_depth = np.mean(depth_relative)
            scale = self.current_height / mean_depth if mean_depth > 0 else 1.0
            depth_absolute = depth_relative * scale  # 单位为米

            # 归一化为 mono8 发布（也可以改为 32FC1）
            depth_norm = cv2.normalize(depth_absolute, None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = depth_norm.astype(np.uint8)

            # 发布
            depth_msg = self.bridge.cv2_to_imgmsg(depth_uint8, encoding='mono8')
            depth_msg.header = msg.header
            self.depth_pub.publish(depth_msg)

        except Exception as e:
            self.get_logger().error(f'处理图像时出错: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = MiDaSDepthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
