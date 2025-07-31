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
    负责相机坐标系到机器人坐标系的转换
    
    职责：
    - 从深度数据计算3D相机坐标
    - 相机坐标到机器人坐标的转换
    - 角度转换
    - 变换矩阵的加载和使用
    """
    
    def __init__(self, matrix_file_path: Optional[str] = None):
        """
        初始化坐标转换计算器
        
        Args:
            matrix_file_path: 变换矩阵文件路径，如果为None则使用默认路径
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 设置变换矩阵文件路径
        if matrix_file_path:
            self.matrix_file_path = Path(matrix_file_path)
        else:
            # 默认路径：Config/transformation_matrix.json
            config_dir = Path(__file__).parent.parent / "Config"
            self.matrix_file_path = config_dir / "transformation_matrix.json"
        
        # 4x4齐次变换矩阵
        self.transformation_matrix: Optional[np.ndarray] = None
        
        # 变换矩阵元数据
        self.matrix_metadata: dict = {}
        
        # 加载变换矩阵
        self.load_transformation_matrix()
        
        self.logger.info(f"坐标转换计算器初始化完成，变换矩阵文件: {self.matrix_file_path}")
    
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
                self.logger.error("变换矩阵文件格式错误：缺少'matrix'字段")
                return False
            
            matrix_data = data['matrix']
            
            # 验证矩阵尺寸
            if (not isinstance(matrix_data, list) or 
                len(matrix_data) != 4 or 
                not all(len(row) == 4 for row in matrix_data)):
                self.logger.error("变换矩阵必须是4x4矩阵")
                return False
            
            # 加载矩阵
            self.transformation_matrix = np.array(matrix_data, dtype=np.float64)
            
            # 保存元数据
            self.matrix_metadata = {
                'calibration_points_count': data.get('calibration_points_count', 0),
                'calibration_rmse': data.get('calibration_rmse', 0.0),
                'transformation_type': data.get('transformation_type', 'unknown'),
                'matrix_size': data.get('matrix_size', '4x4')
            }
            
            self.logger.info(f"成功加载变换矩阵，标定点数: {self.matrix_metadata['calibration_points_count']}, "
                           f"RMSE: {self.matrix_metadata['calibration_rmse']:.3f}mm")
            return True
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"加载变换矩阵失败: {e}")
            return False
        except Exception as e:
            self.logger.error(f"加载变换矩阵时发生未知错误: {e}")
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
            
            if frame is None or detector is None:
                self.logger.error("无效的输入参数：frame或detector为None")
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
            
            # 构造2D检测结果
            detection_result = {
                'detection_count': detection_count,
                'detection_boxes': detection_boxes,  # 所有检测框的坐标信息
                'timing': {
                    'detect_time': detect_time,
                    'process_time': process_time,
                    'total_time': total_time
                }
            }
            
            self.logger.debug(f"2D检测完成: 检测到{detection_count}个目标, "
                            f"模型检测={detect_time:.1f}ms, 处理={process_time:.1f}ms, "
                            f"总耗时={total_time:.1f}ms")
            
            return detection_result
            
        except Exception as e:
            self.logger.error(f"2D检测时出错: {str(e)}")
            return None
    
    def process_detection_to_coordinates_fast(self, results, depth_data, camera_params):
        """
        高性能版本的坐标计算 - 目标10ms内完成
        
        Args:
            results: 检测结果列表（DetectBox对象列表）
            depth_data: 深度数据
            camera_params: 相机参数
            
        Returns:
            dict or None: 最优目标的坐标信息
        """
        if not results or not depth_data or not camera_params:
            return None
        
        # 预计算常用值，避免重复计算
        width = camera_params.width
        height = camera_params.height
        cx, cy = camera_params.cx, camera_params.cy
        fx, fy = camera_params.fx, camera_params.fy
        k1, k2 = camera_params.k1, camera_params.k2
        f2rc = camera_params.f2rc
        
        # 预计算世界坐标变换矩阵（如果需要）
        m_c2w = None
        if hasattr(camera_params, 'cam2worldMatrix') and len(camera_params.cam2worldMatrix) == 16:
            m_c2w = np.array(camera_params.cam2worldMatrix).reshape(4, 4)
        
        # ROI过滤（优化版）
        roi_enabled = hasattr(self, 'roi_enable_checkbox') and self.roi_enable_checkbox.isChecked()
        if roi_enabled:
            # 预计算ROI边界
            roi_x1, roi_y1, roi_x2, roi_y2 = self.calculate_roi_coordinates()
            filtered_results = []
            for result in results:
                # 快速中心点计算
                center_x = (result.pt1x + result.pt2x + result.pt3x + result.pt4x) * 0.25
                center_y = (result.pt1y + result.pt2y + result.pt3y + result.pt4y) * 0.25
                if roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2:
                    filtered_results.append(result)
            results = filtered_results
            
        if not results:
            return None
        
        best_target = None
        min_camera_z = float('inf')  # 改为追踪相机坐标系中的最小Z值
        
        # 批量处理所有检测结果
        for i, result in enumerate(results):
            # 快速边界框计算
            center_x = int((result.pt1x + result.pt2x + result.pt3x + result.pt4x) * 0.25)
            center_y = int((result.pt1y + result.pt2y + result.pt3y + result.pt4y) * 0.25)
            
            # 边界检查
            if not (0 <= center_x < width and 0 <= center_y < height):
                continue
                
            # 获取深度值
            depth_index = center_y * width + center_x
            if depth_index >= len(depth_data):
                continue
                
            depth = depth_data[depth_index]
            if depth <= 0:
                continue
            
            # 简化的角度计算（基于主要边向量）
            angle = self._calculate_angle_fast(result)
            
            # 计算3D坐标
            success, camera_3d = self._calculate_3d_fast(center_x, center_y, depth, 
                                                       cx, cy, fx, fy, k1, k2, f2rc, m_c2w)
            
            if success and camera_3d[2] < min_camera_z:  # 比较相机坐标系中的Z值
                # 坐标系转换
                robot_3d = None
                if self.transformation_matrix is not None:
                    robot_3d = self._transform_point_fast(camera_3d, self.transformation_matrix)
                
                if robot_3d is not None:
                    min_camera_z = camera_3d[2]  # 更新最小Z值
                    best_target = {
                        'target_id': i + 1,
                        'center': [center_x, center_y],
                        'camera_3d': camera_3d,
                        'robot_3d': robot_3d,
                        'original_depth': depth,
                        'angle': angle,
                        'original_result': result
                    }
        
        return best_target
    
    def _calculate_angle_fast(self, result):
        """快速角度计算 - 简化版本"""
        try:
            # 使用对角线向量计算主要方向
            dx1 = result.pt3x - result.pt1x  # 对角线1
            dy1 = result.pt3y - result.pt1y
            dx2 = result.pt4x - result.pt2x  # 对角线2  
            dy2 = result.pt4y - result.pt2y
            
            # 选择较长的对角线作为主方向
            len1_sq = dx1*dx1 + dy1*dy1
            len2_sq = dx2*dx2 + dy2*dy2
            
            if len1_sq > len2_sq:
                angle_rad = math.atan2(dy1, dx1)
            else:
                angle_rad = math.atan2(dy2, dx2)
            
            # 转换为角度并规范化到[0, 180)
            angle_deg = math.degrees(angle_rad) % 180
            return angle_deg
            
        except:
            return 0.0
    
    def _calculate_3d_fast(self, x, y, depth, cx, cy, fx, fy, k1, k2, f2rc, m_c2w):
        """快速3D坐标计算"""
        try:
            # 计算相机坐标系下的坐标
            xp = (cx - x) / fx
            yp = (cy - y) / fy
            
            # 径向畸变校正
            r2 = xp*xp + yp*yp
            k = 1 + k1*r2 + k2*r2*r2
            
            xd = xp * k
            yd = yp * k
            
            # 3D坐标计算
            s0_inv = 1.0 / math.sqrt(xd*xd + yd*yd + 1)
            x_cam = xd * depth * s0_inv
            y_cam = yd * depth * s0_inv
            z_cam = depth * s0_inv - f2rc
            
            # 世界坐标系转换（如果需要）
            if m_c2w is not None:
                x_world = m_c2w[0,3] + z_cam*m_c2w[0,2] + y_cam*m_c2w[0,1] + x_cam*m_c2w[0,0]
                y_world = m_c2w[1,3] + z_cam*m_c2w[1,2] + y_cam*m_c2w[1,1] + x_cam*m_c2w[1,0]
                z_world = m_c2w[2,3] + z_cam*m_c2w[2,2] + y_cam*m_c2w[2,1] + x_cam*m_c2w[2,0]
                return True, [x_world, y_world, z_world]
            else:
                return True, [x_cam, y_cam, z_cam]
                
        except:
            return False, [0, 0, 0]
    
    def _transform_point_fast(self, camera_point, transformation_matrix):
        """快速坐标变换"""
        try:
            # 直接矩阵乘法，避免numpy数组创建开销
            x, y, z = camera_point
            T = transformation_matrix
            
            # 齐次坐标变换
            x_robot = T[0,0]*x + T[0,1]*y + T[0,2]*z + T[0,3]
            y_robot = T[1,0]*x + T[1,1]*y + T[1,2]*z + T[1,3]
            z_robot = T[2,0]*x + T[2,1]*y + T[2,2]*z + T[2,3]
            w = T[3,0]*x + T[3,1]*y + T[3,2]*z + T[3,3]
            
            if w != 0:
                return [x_robot/w, y_robot/w, z_robot/w]
            else:
                return [x_robot, y_robot, z_robot]
                
        except:
            return None


# 兼容性别名，保持与旧代码的兼容
Calculator = CoordinateCalculator
