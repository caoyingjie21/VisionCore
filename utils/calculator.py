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
    
    def is_calibrated(self) -> bool:
        """
        检查是否已完成标定
        
        Returns:
            bool: 是否已加载有效的变换矩阵
        """
        return self.transformation_matrix is not None
    
    def get_calibration_info(self) -> dict:
        """
        获取标定信息
        
        Returns:
            dict: 标定信息字典
        """
        return {
            'calibrated': self.is_calibrated(),
            'matrix_file_path': str(self.matrix_file_path),
            'metadata': self.matrix_metadata.copy() if self.matrix_metadata else {}
        }
    
    def calculate_3d_coordinates_from_depth(
        self, 
        pixel_x: int, 
        pixel_y: int, 
        depth_data: List[float], 
        camera_params: Any
    ) -> Tuple[bool, Tuple[float, float, float]]:
        """
        从深度数据和像素坐标计算3D相机坐标
        
        Args:
            pixel_x: 像素x坐标
            pixel_y: 像素y坐标
            depth_data: 深度数据数组（按行优先顺序）
            camera_params: 相机参数对象，包含内参和畸变参数
            
        Returns:
            Tuple[bool, Tuple[float, float, float]]: (成功标志, (x_3d, y_3d, z_3d))
        """
        try:
            # 参数验证
            if not self._validate_depth_calculation_params(pixel_x, pixel_y, depth_data, camera_params):
                return False, (0.0, 0.0, 0.0)
            
            # 计算深度数据索引
            depth_index = pixel_y * camera_params.width + pixel_x
            if depth_index >= len(depth_data):
                self.logger.debug(f"深度数据索引{depth_index}超出范围{len(depth_data)}")
                return False, (0.0, 0.0, 0.0)
            
            # 获取深度值
            depth_value = depth_data[depth_index]
            if depth_value <= 0:
                self.logger.debug(f"像素({pixel_x}, {pixel_y})的深度值{depth_value}无效")
                return False, (0.0, 0.0, 0.0)
            
            # 计算相机坐标系下的3D坐标
            success, camera_3d = self._depth_to_camera_coordinates(
                pixel_x, pixel_y, depth_value, camera_params
            )
            
            if not success:
                return False, (0.0, 0.0, 0.0)
            
            # 如果有相机到世界坐标系的变换矩阵，进行转换
            if (hasattr(camera_params, 'cam2worldMatrix') and 
                hasattr(camera_params.cam2worldMatrix, '__len__') and 
                len(camera_params.cam2worldMatrix) == 16):
                
                world_3d = self._camera_to_world_coordinates(camera_3d, camera_params.cam2worldMatrix)
                return True, world_3d
            else:
                return True, camera_3d
                
        except Exception as e:
            self.logger.error(f"计算3D坐标时发生错误: {e}")
            return False, (0.0, 0.0, 0.0)
    
    def _validate_depth_calculation_params(
        self, 
        pixel_x: int, 
        pixel_y: int, 
        depth_data: List[float], 
        camera_params: Any
    ) -> bool:
        """验证深度计算参数"""
        # 检查深度数据
        if depth_data is None or not hasattr(depth_data, '__len__'):
            self.logger.warning("深度数据无效")
            return False
        
        # 检查相机参数
        required_attrs = ['width', 'height', 'cx', 'cy', 'fx', 'fy', 'k1', 'k2', 'f2rc']
        for attr in required_attrs:
            if not hasattr(camera_params, attr):
                self.logger.warning(f"相机参数缺少必要属性: {attr}")
                return False
        
        # 检查像素坐标范围
        if (pixel_x < 0 or pixel_x >= camera_params.width or 
            pixel_y < 0 or pixel_y >= camera_params.height):
            self.logger.debug(f"像素坐标({pixel_x}, {pixel_y})超出图像范围"
                            f"({camera_params.width}, {camera_params.height})")
            return False
        
        return True
    
    def _depth_to_camera_coordinates(
        self, 
        pixel_x: int, 
        pixel_y: int, 
        depth_value: float, 
        camera_params: Any
    ) -> Tuple[bool, Tuple[float, float, float]]:
        """将深度值转换为相机坐标系坐标"""
        try:
            # 像素坐标转换为归一化图像坐标
            xp = (camera_params.cx - pixel_x) / camera_params.fx
            yp = (camera_params.cy - pixel_y) / camera_params.fy
            
            # 径向畸变补偿
            r2 = xp * xp + yp * yp
            r4 = r2 * r2
            distortion_factor = 1 + camera_params.k1 * r2 + camera_params.k2 * r4
            
            xd = xp * distortion_factor
            yd = yp * distortion_factor
            
            # 计算相机坐标系下的3D坐标
            s0 = np.sqrt(xd * xd + yd * yd + 1)
            x_cam = xd * depth_value / s0
            y_cam = yd * depth_value / s0
            z_cam = depth_value / s0 - camera_params.f2rc
            
            return True, (x_cam, y_cam, z_cam)
            
        except Exception as e:
            self.logger.error(f"深度到相机坐标转换失败: {e}")
            return False, (0.0, 0.0, 0.0)
    
    def _camera_to_world_coordinates(
        self, 
        camera_3d: Tuple[float, float, float], 
        cam2world_matrix: List[float]
    ) -> Tuple[float, float, float]:
        """将相机坐标转换为世界坐标"""
        try:
            # 重塑为4x4矩阵
            m_c2w = np.array(cam2world_matrix).reshape(4, 4)
            
            x_cam, y_cam, z_cam = camera_3d
            
            # 矩阵变换
            x_world = (m_c2w[0, 3] + z_cam * m_c2w[0, 2] + 
                      y_cam * m_c2w[0, 1] + x_cam * m_c2w[0, 0])
            y_world = (m_c2w[1, 3] + z_cam * m_c2w[1, 2] + 
                      y_cam * m_c2w[1, 1] + x_cam * m_c2w[1, 0])
            z_world = (m_c2w[2, 3] + z_cam * m_c2w[2, 2] + 
                      y_cam * m_c2w[2, 1] + x_cam * m_c2w[2, 0])
            
            return (x_world, y_world, z_world)
            
        except Exception as e:
            self.logger.error(f"相机到世界坐标转换失败: {e}")
            return camera_3d
    
    def transform_point_to_robot_coordinates(
        self, 
        camera_point: Union[List[float], Tuple[float, float, float]]
    ) -> Optional[Tuple[float, float, float]]:
        """
        将相机坐标系中的点转换为机器人坐标系
        
        Args:
            camera_point: 相机坐标系中的点 [x, y, z] 或 (x, y, z)
            
        Returns:
            Optional[Tuple[float, float, float]]: 机器人坐标系中的点 (x, y, z)，失败时返回None
        """
        if not self.is_calibrated():
            self.logger.warning("坐标转换矩阵未加载，请先完成标定")
            return None
        
        try:
            # 转换为numpy数组
            camera_point = np.array(camera_point, dtype=np.float64)
            
            if camera_point.shape != (3,):
                self.logger.error(f"相机坐标点必须是3维向量，当前维度: {camera_point.shape}")
                return None
            
            # 转换为齐次坐标
            camera_homogeneous = np.array([camera_point[0], camera_point[1], camera_point[2], 1.0])
            
            # 应用变换矩阵
            robot_homogeneous = self.transformation_matrix @ camera_homogeneous
            
            # 转换回3D坐标（归一化齐次坐标）
            if abs(robot_homogeneous[3]) < 1e-10:
                self.logger.warning("变换后的齐次坐标w分量接近零，可能存在问题")
                robot_point = robot_homogeneous[:3]
            else:
                robot_point = robot_homogeneous[:3] / robot_homogeneous[3]
            
            return tuple(robot_point)
            
        except Exception as e:
            self.logger.error(f"坐标转换失败: {e}")
            return None
    
    def transform_angle_to_robot_coordinates(self, camera_angle_deg: float) -> Optional[float]:
        """
        将相机坐标系中的角度转换为机器人坐标系中的角度
        
        Args:
            camera_angle_deg: 相机坐标系中的角度（度，通常在0-180范围）
            
        Returns:
            Optional[float]: 机器人坐标系中的角度（度，0-360范围），失败时返回None
        """
        if not self.is_calibrated():
            self.logger.warning("坐标转换矩阵未加载，请先完成标定")
            return None
        
        try:
            # 将角度转换为弧度
            camera_angle_rad = math.radians(camera_angle_deg)
            
            # 在相机坐标系中创建两个点来表示方向向量
            center_point = np.array([0.0, 0.0, 0.0])
            direction_point = np.array([
                math.cos(camera_angle_rad),
                math.sin(camera_angle_rad),
                0.0
            ])
            
            # 转换两个点到机器人坐标系
            center_robot = self.transform_point_to_robot_coordinates(center_point)
            direction_robot = self.transform_point_to_robot_coordinates(direction_point)
            
            if center_robot is None or direction_robot is None:
                self.logger.error("角度转换过程中坐标转换失败")
                return None
            
            # 计算机器人坐标系中的方向向量（只考虑XY平面）
            direction_vector = np.array([
                direction_robot[0] - center_robot[0],
                direction_robot[1] - center_robot[1]
            ])
            
            # 计算机器人坐标系中的角度
            robot_angle_rad = math.atan2(direction_vector[1], direction_vector[0])
            
            # 转换为度数并规范化到 [0, 360) 范围
            robot_angle_deg = math.degrees(robot_angle_rad) % 360
            
            return robot_angle_deg
            
        except Exception as e:
            self.logger.error(f"角度转换失败: {e}")
            return None
    
    def calculate_angle_compensation(
        self, 
        camera_angle_deg: float, 
        target_angle_deg: float = 0.0
    ) -> Optional[float]:
        """
        计算角度补偿值
        
        Args:
            camera_angle_deg: 相机坐标系中的角度（度）
            target_angle_deg: 目标角度（度，默认为0）
            
        Returns:
            Optional[float]: 角度补偿值（度），失败时返回None
        """
        robot_angle = self.transform_angle_to_robot_coordinates(camera_angle_deg)
        
        if robot_angle is None:
            return None
        
        # 计算到目标角度的最短旋转路径
        angle_diff = (target_angle_deg - robot_angle) % 360
        
        # 选择最短路径（±180度范围内）
        if angle_diff > 180:
            angle_diff -= 360
        
        return angle_diff
    
    @retry(max_retries=3, delay=0.1, exceptions=(FileNotFoundError, json.JSONDecodeError))
    def reload_transformation_matrix(self) -> bool:
        """
        重新加载变换矩阵（支持热重载）
        
        Returns:
            bool: 是否成功重新加载
        """
        self.logger.info("重新加载变换矩阵...")
        old_matrix = self.transformation_matrix.copy() if self.transformation_matrix is not None else None
        
        success = self.load_transformation_matrix()
        
        if success:
            if old_matrix is not None and self.transformation_matrix is not None:
                # 检查矩阵是否发生变化
                if np.allclose(old_matrix, self.transformation_matrix, rtol=1e-10):
                    self.logger.debug("变换矩阵内容未变化")
                else:
                    self.logger.info("变换矩阵已更新")
            else:
                self.logger.info("变换矩阵重新加载成功")
        else:
            self.logger.warning("变换矩阵重新加载失败")
        
        return success


# 兼容性别名，保持与旧代码的兼容
Calculator = CoordinateCalculator
