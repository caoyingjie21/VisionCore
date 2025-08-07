#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
标定数据实体类
用于解析坐标系标定命令中的数据结构
"""

from dataclasses import dataclass
from typing import List, Optional
import json


@dataclass
class Point3D:
    """3D坐标点"""
    x: float
    y: float
    z: float
    
    def to_list(self) -> List[float]:
        """转换为列表格式"""
        return [self.x, self.y, self.z]
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Point3D':
        """从字典创建Point3D对象"""
        return cls(
            x=float(data['x']),
            y=float(data['y']),
            z=float(data['z'])
        )


@dataclass
class CalibrationPoint:
    """标定点数据"""
    index: int
    camera_point: Point3D
    robot_point: Point3D
    image_path: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CalibrationPoint':
        """从字典创建CalibrationPoint对象"""
        return cls(
            index=int(data['index']),
            camera_point=Point3D.from_dict(data['cameraPoint']),
            robot_point=Point3D.from_dict(data['robotPoint']),
            image_path=data.get('imagePath')
        )


@dataclass
class CalibrationData:
    """标定数据集合"""
    calibration_points: List[CalibrationPoint]
    
    @classmethod
    def from_json_string(cls, json_str: str) -> 'CalibrationData':
        """从JSON字符串创建CalibrationData对象"""
        try:
            data_list = json.loads(json_str)
            calibration_points = [
                CalibrationPoint.from_dict(point_data) 
                for point_data in data_list
            ]
            return cls(calibration_points=calibration_points)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"解析标定数据失败: {e}")
    
    def get_camera_points(self) -> List[List[float]]:
        """获取所有相机坐标点"""
        return [point.camera_point.to_list() for point in self.calibration_points]
    
    def get_robot_points(self) -> List[List[float]]:
        """获取所有机器人坐标点"""
        return [point.robot_point.to_list() for point in self.calibration_points]
    
    def get_points_count(self) -> int:
        """获取标定点数量"""
        return len(self.calibration_points)
    
    def validate(self) -> bool:
        """验证数据有效性"""
        if not self.calibration_points:
            return False
        
        # 检查是否有重复的索引
        indices = [point.index for point in self.calibration_points]
        if len(indices) != len(set(indices)):
            return False
        
        # 检查坐标是否有效（非空且为数值）
        for point in self.calibration_points:
            camera_coords = point.camera_point.to_list()
            robot_coords = point.robot_point.to_list()
            
            if not all(isinstance(coord, (int, float)) for coord in camera_coords + robot_coords):
                return False
        
        return True


@dataclass
class CoordinateCalibrationCommand:
    """坐标系标定命令"""
    command: str
    data: CalibrationData
    date_time: str
    
    @classmethod
    def from_mqtt_payload(cls, payload: dict) -> 'CoordinateCalibrationCommand':
        """从MQTT消息载荷创建命令对象"""
        try:
            command = payload.get('command', '')
            data_str = payload.get('Data', '')
            date_time = payload.get('dateTime', '')
            
            if command != 'coordinate_calibration':
                raise ValueError(f"不支持的命令类型: {command}")
            
            calibration_data = CalibrationData.from_json_string(data_str)
            
            return cls(
                command=command,
                data=calibration_data,
                date_time=date_time
            )
        except Exception as e:
            raise ValueError(f"解析坐标标定命令失败: {e}") 