#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VisionCore 坐标系标定器
专门负责相机坐标系到机器人坐标系的标定
"""

import numpy as np
import json
import os
import logging
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
from .decorators import retry


class CoordinateCalibrator:
    """
    坐标系标定器
    负责计算相机坐标系到机器人坐标系的变换矩阵
    
    职责：
    - 收集标定点数据
    - 计算4x4齐次变换矩阵
    - 验证标定精度
    - 保存和管理标定结果
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化坐标系标定器
        
        Args:
            output_dir: 标定结果输出目录，如果为None则使用默认路径
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # 默认输出到Config目录
            self.output_dir = Path(__file__).parent.parent / "Config"
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 标定数据
        self.camera_points: List[List[float]] = []
        self.robot_points: List[List[float]] = []
        
        # 计算结果
        self.transformation_matrix: Optional[np.ndarray] = None
        self.calibration_metadata: Dict[str, Any] = {}
        
        self.logger.info(f"坐标系标定器初始化完成，输出目录: {self.output_dir}")
    
    def add_calibration_point(
        self, 
        camera_point: List[float], 
        robot_point: List[float]
    ) -> bool:
        """
        添加标定点对
        
        Args:
            camera_point: 相机坐标系中的点 [x, y, z]
            robot_point: 对应的机器人坐标系中的点 [x, y, z]
            
        Returns:
            bool: 是否成功添加
        """
        try:
            # 验证输入数据
            if not self._validate_point(camera_point, "相机坐标点"):
                return False
            
            if not self._validate_point(robot_point, "机器人坐标点"):
                return False
            
            # 添加到标定数据
            self.camera_points.append(list(camera_point))
            self.robot_points.append(list(robot_point))
            
            self.logger.info(f"添加标定点对 #{len(self.camera_points)}: "
                           f"相机{camera_point} -> 机器人{robot_point}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"添加标定点失败: {e}")
            return False
    
    def _validate_point(self, point: List[float], point_type: str) -> bool:
        """验证点数据格式"""
        if not isinstance(point, (list, tuple, np.ndarray)):
            self.logger.error(f"{point_type}必须是列表、元组或numpy数组")
            return False
        
        if len(point) != 3:
            self.logger.error(f"{point_type}必须包含3个坐标值(x, y, z)")
            return False
        
        try:
            # 尝试转换为浮点数
            float_point = [float(x) for x in point]
            # 检查是否有无效值
            if any(not np.isfinite(x) for x in float_point):
                self.logger.error(f"{point_type}包含无效值(NaN或Inf): {point}")
                return False
        except (ValueError, TypeError):
            self.logger.error(f"{point_type}包含非数值数据: {point}")
            return False
        
        return True
    
    def add_calibration_points_batch(
        self, 
        camera_points: List[List[float]], 
        robot_points: List[List[float]]
    ) -> int:
        """
        批量添加标定点对
        
        Args:
            camera_points: 相机坐标系中的点列表 [[x, y, z], ...]
            robot_points: 对应的机器人坐标系中的点列表 [[x, y, z], ...]
            
        Returns:
            int: 成功添加的点对数量
        """
        if len(camera_points) != len(robot_points):
            self.logger.error("相机坐标点和机器人坐标点数量必须相同")
            return 0
        
        success_count = 0
        for camera_point, robot_point in zip(camera_points, robot_points):
            if self.add_calibration_point(camera_point, robot_point):
                success_count += 1
        
        self.logger.info(f"批量添加标定点完成: 成功{success_count}/{len(camera_points)}个")
        return success_count
    
    def clear_calibration_points(self):
        """清除所有标定点数据"""
        self.camera_points.clear()
        self.robot_points.clear()
        self.transformation_matrix = None
        self.calibration_metadata.clear()
        self.logger.info("已清除所有标定点数据")
    
    def get_calibration_points_count(self) -> int:
        """获取当前标定点数量"""
        return len(self.camera_points)
    
    def calculate_transformation_matrix(self) -> Dict[str, Any]:
        """
        计算4x4齐次变换矩阵
        
        Returns:
            Dict[str, Any]: 标定结果字典
            
        Raises:
            ValueError: 标定点不足或计算失败时抛出
        """
        # 检查标定点数量
        point_count = len(self.camera_points)
        if point_count < 4:
            raise ValueError(f"至少需要4个标定点，当前只有{point_count}个")
        
        self.logger.info(f"开始计算变换矩阵，使用{point_count}个标定点...")
        
        try:
            # 转换为numpy数组
            camera_points = np.array(self.camera_points, dtype=np.float64)
            robot_points = np.array(self.robot_points, dtype=np.float64)
            
            # 使用齐次坐标进行变换矩阵计算
            camera_homogeneous = np.hstack([camera_points, np.ones((point_count, 1))])
            robot_homogeneous = np.hstack([robot_points, np.ones((point_count, 1))])
            
            # 使用伪逆求解变换矩阵
            # 求解：T @ camera_homogeneous.T = robot_homogeneous.T
            camera_homogeneous_T = camera_homogeneous.T
            robot_homogeneous_T = robot_homogeneous.T
            
            camera_pinv = np.linalg.pinv(camera_homogeneous_T)
            self.transformation_matrix = robot_homogeneous_T @ camera_pinv
            
            # 确保变换矩阵的最后一行是 [0, 0, 0, 1]
            self.transformation_matrix[3, :] = [0, 0, 0, 1]
            
            self.logger.info("变换矩阵计算完成")
            
            # 验证变换质量
            validation_results = self._validate_transformation(camera_points, robot_points)
            
            # 保存元数据
            self.calibration_metadata = {
                'calibration_points_count': point_count,
                'calibration_rmse': validation_results['total_rmse'],
                'transformation_type': 'complete_3d',
                'matrix_size': '4x4',
                'calibration_datetime': datetime.now().isoformat(),
                'validation_results': validation_results
            }
            
            self.logger.info(f"标定完成！RMSE: {validation_results['total_rmse']:.3f}mm")
            
            return {
                'transformation_matrix': self.transformation_matrix,
                'metadata': self.calibration_metadata,
                'validation_results': validation_results
            }
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"变换矩阵计算失败: {str(e)}")
        except Exception as e:
            raise ValueError(f"标定过程中发生错误: {str(e)}")
    
    def _validate_transformation(
        self, 
        camera_points: np.ndarray, 
        robot_points: np.ndarray
    ) -> Dict[str, Any]:
        """
        验证变换矩阵的精度
        
        Args:
            camera_points: 相机坐标点数组
            robot_points: 机器人坐标点数组
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        if self.transformation_matrix is None:
            raise ValueError("变换矩阵未计算")
        
        # 使用变换矩阵转换相机坐标
        transformed_points = []
        for camera_point in camera_points:
            # 转换为齐次坐标
            camera_homogeneous = np.array([camera_point[0], camera_point[1], camera_point[2], 1.0])
            
            # 应用变换矩阵
            robot_homogeneous = self.transformation_matrix @ camera_homogeneous
            
            # 转换回3D坐标
            if abs(robot_homogeneous[3]) < 1e-10:
                robot_point = robot_homogeneous[:3]
            else:
                robot_point = robot_homogeneous[:3] / robot_homogeneous[3]
            
            transformed_points.append(robot_point)
        
        transformed_points = np.array(transformed_points)
        
        # 计算误差
        errors = transformed_points - robot_points
        
        # 各轴RMSE
        x_rmse = np.sqrt(np.mean(errors[:, 0]**2))
        y_rmse = np.sqrt(np.mean(errors[:, 1]**2))
        z_rmse = np.sqrt(np.mean(errors[:, 2]**2))
        
        # 总体RMSE（3D空间中的欧氏距离）
        point_errors = np.sqrt(np.sum(errors**2, axis=1))
        total_rmse = np.sqrt(np.mean(point_errors**2))
        
        # 统计指标
        max_error = np.max(point_errors)
        min_error = np.min(point_errors)
        mean_error = np.mean(point_errors)
        std_error = np.std(point_errors)
        
        return {
            'x_rmse': float(x_rmse),
            'y_rmse': float(y_rmse),
            'z_rmse': float(z_rmse),
            'total_rmse': float(total_rmse),
            'max_error': float(max_error),
            'min_error': float(min_error),
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'errors': errors.tolist(),
            'point_errors': point_errors.tolist()
        }
    
    def save_transformation_matrix(self, filename: str = "transformation_matrix.json") -> bool:
        """
        保存变换矩阵到文件
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 是否成功保存
        """
        if self.transformation_matrix is None:
            self.logger.error("没有可保存的变换矩阵，请先进行标定")
            return False
        
        try:
            output_path = self.output_dir / filename
            
            # 准备保存数据
            save_data = {
                'matrix': self.transformation_matrix.tolist(),
                **self.calibration_metadata
            }
            
            # 保存到文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"变换矩阵已保存到: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存变换矩阵失败: {e}")
            return False
    
    def save_calibration_data(self, filename: str = "calibration_data.json") -> bool:
        """
        保存标定数据（包括所有标定点）
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 是否成功保存
        """
        try:
            output_path = self.output_dir / filename
            
            # 准备保存数据
            save_data = {
                'camera_points': self.camera_points,
                'robot_points': self.robot_points,
                'calibration_datetime': datetime.now().isoformat(),
                'points_count': len(self.camera_points)
            }
            
            # 如果已经计算了变换矩阵，也保存
            if self.transformation_matrix is not None:
                save_data.update({
                    'transformation_matrix': self.transformation_matrix.tolist(),
                    'metadata': self.calibration_metadata
                })
            
            # 保存到文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"标定数据已保存到: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存标定数据失败: {e}")
            return False
    
    def load_calibration_data(self, filename: str = "calibration_data.json") -> bool:
        """
        从文件加载标定数据
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 是否成功加载
        """
        try:
            data_path = self.output_dir / filename
            
            if not data_path.exists():
                self.logger.warning(f"标定数据文件不存在: {data_path}")
                return False
            
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 清除现有数据
            self.clear_calibration_points()
            
            # 加载标定点
            camera_points = data.get('camera_points', [])
            robot_points = data.get('robot_points', [])
            
            if len(camera_points) != len(robot_points):
                self.logger.error("标定数据文件中相机坐标点和机器人坐标点数量不匹配")
                return False
            
            # 批量添加标定点
            success_count = self.add_calibration_points_batch(camera_points, robot_points)
            
            # 如果文件中包含变换矩阵，也加载
            if 'transformation_matrix' in data:
                self.transformation_matrix = np.array(data['transformation_matrix'])
                self.calibration_metadata = data.get('metadata', {})
            
            self.logger.info(f"标定数据加载完成: {success_count}个标定点")
            return True
            
        except Exception as e:
            self.logger.error(f"加载标定数据失败: {e}")
            return False
    
    def get_calibration_quality_assessment(self) -> Dict[str, str]:
        """
        获取标定质量评估
        
        Returns:
            Dict[str, str]: 质量评估结果
        """
        if not self.calibration_metadata:
            return {'quality': 'unknown', 'message': '未进行标定'}
        
        rmse = self.calibration_metadata.get('calibration_rmse', float('inf'))
        point_count = self.calibration_metadata.get('calibration_points_count', 0)
        
        # RMSE质量评估
        if rmse < 2.0:
            quality = 'excellent'
            quality_msg = f'优秀 (RMSE: {rmse:.3f}mm)'
        elif rmse < 5.0:
            quality = 'good'
            quality_msg = f'良好 (RMSE: {rmse:.3f}mm)'
        elif rmse < 10.0:
            quality = 'acceptable'
            quality_msg = f'可接受 (RMSE: {rmse:.3f}mm)'
        else:
            quality = 'poor'
            quality_msg = f'较差 (RMSE: {rmse:.3f}mm)'
        
        # 标定点数量评估
        if point_count < 4:
            point_msg = '标定点不足'
        elif point_count < 6:
            point_msg = '标定点数量基本够用'
        elif point_count < 10:
            point_msg = '标定点数量良好'
        else:
            point_msg = '标定点数量充足'
        
        # 综合建议
        if quality == 'poor' or point_count < 4:
            recommendation = '建议重新标定或增加标定点'
        elif quality in ['acceptable', 'good'] and point_count < 6:
            recommendation = '建议增加标定点以提高精度'
        else:
            recommendation = '标定质量满足使用要求'
        
        return {
            'quality': quality,
            'quality_message': quality_msg,
            'points_message': point_msg,
            'recommendation': recommendation,
            'rmse': rmse,
            'points_count': point_count
        }
    
    def print_calibration_report(self):
        """打印标定报告"""
        print("=" * 60)
        print("VisionCore 坐标系标定报告")
        print("=" * 60)
        
        point_count = len(self.camera_points)
        print(f"标定点数量: {point_count}")
        
        if point_count == 0:
            print("⚠️  没有标定点数据")
            print("=" * 60)
            return
        
        # 显示标定点
        print("\n📍 标定点列表:")
        for i, (cam_pt, rob_pt) in enumerate(zip(self.camera_points, self.robot_points), 1):
            print(f"  {i:2d}. 相机{cam_pt} -> 机器人{rob_pt}")
        
        # 如果已经计算了变换矩阵，显示结果
        if self.transformation_matrix is not None and self.calibration_metadata:
            print(f"\n🎯 标定结果:")
            
            quality_info = self.get_calibration_quality_assessment()
            
            print(f"  精度评估: {quality_info['quality_message']}")
            print(f"  点数评估: {quality_info['points_message']}")
            print(f"  建议: {quality_info['recommendation']}")
            
            # 详细误差信息
            validation = self.calibration_metadata.get('validation_results', {})
            if validation:
                print(f"\n📊 详细误差分析:")
                print(f"  X轴RMSE: {validation.get('x_rmse', 0):.3f}mm")
                print(f"  Y轴RMSE: {validation.get('y_rmse', 0):.3f}mm")
                print(f"  Z轴RMSE: {validation.get('z_rmse', 0):.3f}mm")
                print(f"  最大误差: {validation.get('max_error', 0):.3f}mm")
                print(f"  最小误差: {validation.get('min_error', 0):.3f}mm")
                print(f"  平均误差: {validation.get('mean_error', 0):.3f}mm")
                print(f"  误差标准差: {validation.get('std_error', 0):.3f}mm")
            
            # 变换矩阵
            print(f"\n🔢 4x4变换矩阵:")
            for i, row in enumerate(self.transformation_matrix):
                print(f"  [{row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f} {row[3]:10.6f}]")
        else:
            print("\n⚠️  尚未计算变换矩阵")
            if point_count >= 4:
                print("   可以调用 calculate_transformation_matrix() 进行计算")
            else:
                print(f"   需要至少4个标定点，当前只有{point_count}个")
        
        print("=" * 60)

    def calculate_3d_transformation_matrix(self, camera_points, robot_points):
        """
        计算完整的3D变换矩阵
        
        Args:
            camera_points: 相机坐标系中的点 [[x, y, z], ...]
            robot_points: 机器人坐标系中的点 [[x, y, z], ...]
            
        Returns:
            dict: 包含变换结果的字典
        """
        if len(camera_points) != len(robot_points):
            raise ValueError("相机坐标点和机器人坐标点数量必须相同")
            
        if len(camera_points) < 4:
            raise ValueError("至少需要4组对应点来计算3D变换矩阵")
            
        camera_points = np.array(camera_points, dtype=np.float64)
        robot_points = np.array(robot_points, dtype=np.float64)
        
        # 保存标定点数量
        self.calibration_points_count = len(camera_points)
        
        # 使用齐次坐标进行变换矩阵计算
        # 将3D点转换为齐次坐标（添加第4维为1）
        camera_homogeneous = np.hstack([camera_points, np.ones((len(camera_points), 1))])
        robot_homogeneous = np.hstack([robot_points, np.ones((len(robot_points), 1))])
        
        try:
            # 使用伪逆求解变换矩阵
            # 对于齐次变换矩阵 T，有：robot_homogeneous.T = T @ camera_homogeneous.T
            # 求解：T = robot_homogeneous.T @ pinv(camera_homogeneous.T)
            camera_homogeneous_T = camera_homogeneous.T
            robot_homogeneous_T = robot_homogeneous.T
            
            # 计算伪逆
            camera_pinv = np.linalg.pinv(camera_homogeneous_T)
            self.transformation_matrix = robot_homogeneous_T @ camera_pinv
            
            # 确保变换矩阵的最后一行是 [0, 0, 0, 1]
            self.transformation_matrix[3, :] = [0, 0, 0, 1]
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"变换矩阵计算失败: {str(e)}")
        
        # 验证变换质量
        validation_results = self._validate_transformation(camera_points, robot_points)
        self.calibration_rmse = validation_results['total_rmse']
        
        # 设置标定元数据
        self.calibration_metadata = {
            'calibration_points_count': self.calibration_points_count,
            'calibration_rmse': self.calibration_rmse,
            'transformation_type': 'complete_3d',
            'matrix_size': '4x4',
            'calibration_datetime': datetime.now().isoformat(),
            'validation_results': validation_results
        }
        
        # 保存变换矩阵
        self.save_transformation_matrix()
        
        return {
            'transformation_matrix': self.transformation_matrix,
            'validation_results': validation_results,
            'calibration_points_count': self.calibration_points_count,
            'calibration_rmse': self.calibration_rmse
        }


# 兼容性别名
Calibrator = CoordinateCalibrator 