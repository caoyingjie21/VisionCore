#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VisionCore 坐标转换计算器
专门负责从相机坐标系向机器人坐标系的转换
"""

import numpy as np
import json
import os
import math
import logging
from typing import List, Tuple, Optional, Any, Union
from pathlib import Path
from .decorators import retry
import time


class CoordinateCalculator:
    """
    坐标转换计算器
    负责处理相机坐标系到机器人坐标系的转换
    """
    
    def __init__(self, matrix_file_path: Optional[str] = None):
        """
        初始化坐标转换计算器
        
        Args:
            matrix_file_path: 变换矩阵文件路径，如果为None则使用默认路径
        """
        self.logger = logging.getLogger(__name__)
        
        # 设置变换矩阵文件路径
        if matrix_file_path:
            self.matrix_file_path = Path(matrix_file_path)
        else:
            # 默认路径：Config/transformation_matrix.json
            config_dir = Path(__file__).parent.parent / "Config"
            self.matrix_file_path = config_dir / "transformation_matrix.json"
        
        # 坐标转换相关属性
        self.transformation_matrix: Optional[np.ndarray] = None
        self.matrix_metadata: dict = {}
        
        # 加载变换矩阵
        self.load_transformation_matrix()
    
    def load_transformation_matrix(self) -> bool:
        """
        加载4x4变换矩阵
        
        Returns:
            bool: 是否成功加载
        """
        if not self.matrix_file_path.exists():
            self.logger.warning(f"变换矩阵文件不存在: {self.matrix_file_path}")
            return False
        
        try:
            with open(self.matrix_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查数据格式
            if 'matrix' not in data:
                self.logger.error("变换矩阵文件格式错误：缺少matrix字段")
                return False
            
            matrix_data = data['matrix']
            
            # 验证矩阵尺寸
            if (not isinstance(matrix_data, list) or 
                len(matrix_data) != 4 or 
                not all(len(row) == 4 for row in matrix_data)):
                self.logger.error("变换矩阵格式错误：应为4x4矩阵")
                return False
            
            # 转换为numpy数组
            self.transformation_matrix = np.array(matrix_data, dtype=np.float64)
            
            # 保存元数据
            self.matrix_metadata = data.get('metadata', {})
            
            self.logger.info(f"变换矩阵加载成功: {self.matrix_file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载变换矩阵失败: {e}")
            return False

    def perform_2d_detection(self, frame, detector):
        """
        执行2D检测 - 只进行目标检测，返回所有检测框的坐标
        用于图像稳定性检测，不进行ROI过滤
        
        Args:
            frame: 图像帧数据
            detector: 检测器实例
        
        Returns:
            dict: 2D检测结果字典，包含所有检测框的坐标
        """
        try:
            # 记录检测流程的开始时间
            total_start_time = time.time()
            
            if frame is None:
                return None
            
            # 执行检测
            detect_start = time.time()
            results = detector.detect(frame)
            detect_time = (time.time() - detect_start) * 1000  # 转换为毫秒
            
            # 处理检测结果
            process_start_time = time.time()
            detection_count = 0
            detection_boxes = []  # 存储所有检测框的坐标
            
            if results:
                for result in results:
                    if not (hasattr(result, 'pt1x') and hasattr(result, 'pt1y')):
                        continue
                    
                    # 计算检测框中心点
                    center_x = (result.pt1x + result.pt2x + result.pt3x + result.pt4x) / 4
                    center_y = (result.pt1y + result.pt2y + result.pt3y + result.pt4y) / 4
                    
                    # 添加到检测框列表
                    detection_count += 1
                    detection_boxes.append({
                        'center': [center_x, center_y],
                        'corners': [
                            [result.pt1x, result.pt1y],
                            [result.pt2x, result.pt2y],
                            [result.pt3x, result.pt3y],
                            [result.pt4x, result.pt4y]
                        ],
                        'result': result
                    })
            
            process_time = (time.time() - process_start_time) * 1000  # 转换为毫秒
            total_time = (time.time() - total_start_time) * 1000
            
            return {
                'detection_count': detection_count,
                'detection_boxes': detection_boxes,
                'timing': {
                    'detect_time': detect_time,
                    'process_time': process_time,
                    'total_time': total_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"2D检测失败: {e}")
            return None

    def _calculate_angle_fast(self, result):
        """
        快速计算检测结果的角度
        
        Args:
            result: 检测结果对象
            
        Returns:
            float: 角度值（度）
        """
        try:
            if hasattr(result, 'angle') and result.angle is not None:
                # 如果检测结果中已经包含角度信息，直接使用
                angle_rad = result.angle
                # 转换为角度
                angle_deg = math.degrees(angle_rad)
                # 规范化到 [0, 180) 范围
                angle_deg = angle_deg % 180
                if angle_deg < 0:
                    angle_deg += 180
                return angle_deg
            else:
                # 从四个角点计算角度
                points = [
                    [result.pt1x, result.pt1y],
                    [result.pt2x, result.pt2y], 
                    [result.pt3x, result.pt3y],
                    [result.pt4x, result.pt4y]
                ]
                
                # 计算所有边的长度和角度
                edges = []
                for i in range(4):
                    j = (i + 1) % 4
                    dx = points[j][0] - points[i][0]
                    dy = points[j][1] - points[i][1]
                    length = math.sqrt(dx*dx + dy*dy)
                    angle = math.atan2(dy, dx)
                    edges.append((length, angle))
                
                # 选择最长的边作为主方向
                longest_edge = max(edges, key=lambda x: x[0])
                main_angle = longest_edge[1]
                
                # 转换为角度并规范化
                angle_deg = math.degrees(main_angle)
                angle_deg = angle_deg % 180
                if angle_deg < 0:
                    angle_deg += 180
                
                return angle_deg
                
        except Exception as e:
            self.logger.warning(f"计算角度失败: {e}")
            return 0.0

    def _calculate_3d_fast(self, x, y, depth, cx, cy, fx, fy, k1, k2, f2rc, m_c2w):
        """
        快速计算3D坐标
        
        Args:
            x, y: 图像坐标
            depth: 深度数据
            cx, cy: 相机内参
            fx, fy: 焦距
            k1, k2: 畸变参数
            f2rc: 焦距到像素比例
            m_c2w: 相机到世界坐标变换矩阵
            
        Returns:
            list: [x, y, z] 3D坐标
        """
        try:
            # 这里应该实现具体的3D坐标计算逻辑
            # 由于原代码中没有完整的实现，这里提供一个基础框架
            # 实际使用时需要根据具体的相机参数和深度数据格式进行调整
            
            # 示例实现（需要根据实际情况修改）
            if depth is None or len(depth) == 0:
                return None
            
            # 获取深度值（这里需要根据实际的深度数据格式进行调整）
            depth_value = depth[int(y) * int(cx) + int(x)] if isinstance(depth, (list, np.ndarray)) else depth
            
            if depth_value <= 0 or depth_value == float('inf'):
                return None
            
            # 计算3D坐标（简化版本，实际需要更复杂的计算）
            z = depth_value
            x_3d = (x - cx) * z / fx
            y_3d = (y - cy) * z / fy
            
            return [x_3d, y_3d, z]
            
        except Exception as e:
            self.logger.error(f"3D坐标计算失败: {e}")
            return None

    def _transform_point_fast(self, camera_point, transformation_matrix):
        """
        快速坐标变换
        
        Args:
            camera_point: 相机坐标系点 [x, y, z]
            transformation_matrix: 变换矩阵
            
        Returns:
            list: 机器人坐标系点 [x, y, z]
        """
        try:
            if camera_point is None or transformation_matrix is None:
                return None
            
            # 转换为齐次坐标
            point_homogeneous = np.array([camera_point[0], camera_point[1], camera_point[2], 1.0])
            
            # 应用变换矩阵
            transformed_point = transformation_matrix @ point_homogeneous
            
            # 返回前三个坐标
            return [transformed_point[0], transformed_point[1], transformed_point[2]]
            
        except Exception as e:
            self.logger.error(f"坐标变换失败: {e}")
            return None

    def get_transformation_matrix_status(self) -> dict:
        """
        获取变换矩阵状态信息
        
        Returns:
            dict: 包含加载状态和元数据的字典
        """
        return {
            'loaded': self.transformation_matrix is not None,
            'matrix_file': str(self.matrix_file_path),
            'matrix_exists': self.matrix_file_path.exists(),
            'metadata': self.matrix_metadata,
            'error': None if self.transformation_matrix is not None else "变换矩阵未加载"
        }


# 兼容性别名，保持与旧代码的兼容
Calculator = CoordinateCalculator
