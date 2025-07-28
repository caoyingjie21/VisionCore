#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VisionCore 坐标系统使用示例
演示 CoordinateCalculator 和 CoordinateCalibrator 的使用
"""

import numpy as np
import logging
from utils.calculator import CoordinateCalculator
from utils.calibrator import CoordinateCalibrator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s | %(name)s | %(message)s'
)

class MockCameraParams:
    """模拟的相机参数对象"""
    def __init__(self):
        self.width = 256
        self.height = 256
        self.cx = 128.0
        self.cy = 128.0
        self.fx = 200.0
        self.fy = 200.0
        self.k1 = 0.0
        self.k2 = 0.0
        self.f2rc = 0.0
        # 相机到世界坐标系的变换矩阵（单位矩阵表示相机坐标系即为世界坐标系）
        self.cam2worldMatrix = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ]

def demo_coordinate_calibration():
    """演示坐标系标定流程"""
    print("🎯 === 坐标系标定演示 ===\n")
    
    # 1. 创建标定器
    calibrator = CoordinateCalibrator()
    
    # 2. 添加标定点（模拟从不同位置采集的对应点）
    print("📍 添加标定点...")
    
    # 模拟标定点数据：相机坐标系 -> 机器人坐标系
    calibration_data = [
        ([100, 50, 600], [300, 400, 100]),     # 点1
        ([200, 100, 650], [350, 450, 100]),   # 点2
        ([150, 150, 700], [325, 500, 100]),   # 点3
        ([50, 200, 750], [275, 550, 100]),    # 点4
        ([250, 200, 800], [375, 550, 100]),   # 点5
        ([120, 80, 620], [320, 420, 100]),    # 点6
    ]
    
    # 逐个添加标定点
    for camera_point, robot_point in calibration_data:
        success = calibrator.add_calibration_point(camera_point, robot_point)
        if not success:
            print(f"❌ 添加标定点失败: {camera_point} -> {robot_point}")
    
    print(f"✅ 标定点添加完成，共{calibrator.get_calibration_points_count()}个点\n")
    
    # 3. 计算变换矩阵
    print("🔄 计算变换矩阵...")
    try:
        result = calibrator.calculate_transformation_matrix()
        print(f"✅ 标定成功！RMSE: {result['metadata']['calibration_rmse']:.3f}mm\n")
    except ValueError as e:
        print(f"❌ 标定失败: {e}")
        return None
    
    # 4. 显示标定报告
    calibrator.print_calibration_report()
    
    # 5. 保存标定结果
    print("\n💾 保存标定结果...")
    calibrator.save_transformation_matrix()
    calibrator.save_calibration_data()
    print("✅ 标定结果已保存\n")
    
    return calibrator

def demo_coordinate_transformation():
    """演示坐标转换使用"""
    print("🔄 === 坐标转换演示 ===\n")
    
    # 1. 创建坐标转换计算器
    calculator = CoordinateCalculator()
    
    # 2. 检查标定状态
    print("📊 检查标定状态...")
    calibration_info = calculator.get_calibration_info()
    print(f"标定状态: {'✅ 已标定' if calibration_info['calibrated'] else '❌ 未标定'}")
    if calibration_info['calibrated']:
        metadata = calibration_info['metadata']
        print(f"标定点数: {metadata.get('calibration_points_count', 0)}")
        print(f"标定精度: {metadata.get('calibration_rmse', 0):.3f}mm")
    print()
    
    if not calibration_info['calibrated']:
        print("⚠️ 未找到标定数据，请先运行标定流程")
        return
    
    # 3. 演示深度数据到3D坐标计算
    print("🔍 深度数据到3D坐标计算演示...")
    
    # 模拟深度数据 (256x256图像)
    depth_data = []
    for y in range(256):
        for x in range(256):
            # 模拟深度值：中心较近，边缘较远
            distance_from_center = np.sqrt((x-128)**2 + (y-128)**2)
            depth_value = 500 + distance_from_center * 2
            depth_data.append(depth_value)
    
    camera_params = MockCameraParams()
    
    # 测试几个像素点
    test_pixels = [(100, 100), (150, 120), (200, 80)]
    
    for pixel_x, pixel_y in test_pixels:
        success, camera_3d = calculator.calculate_3d_coordinates_from_depth(
            pixel_x, pixel_y, depth_data, camera_params
        )
        
        if success:
            print(f"像素({pixel_x}, {pixel_y}) -> 相机3D坐标: ({camera_3d[0]:.2f}, {camera_3d[1]:.2f}, {camera_3d[2]:.2f})")
        else:
            print(f"像素({pixel_x}, {pixel_y}) -> 计算失败")
    
    print()
    
    # 4. 演示坐标转换
    print("🎯 坐标转换演示...")
    
    test_camera_points = [
        [100, 50, 600],
        [200, 100, 650],
        [150, 150, 700]
    ]
    
    for camera_point in test_camera_points:
        robot_point = calculator.transform_point_to_robot_coordinates(camera_point)
        
        if robot_point:
            print(f"相机坐标{camera_point} -> 机器人坐标({robot_point[0]:.2f}, {robot_point[1]:.2f}, {robot_point[2]:.2f})")
        else:
            print(f"相机坐标{camera_point} -> 转换失败")
    
    print()
    
    # 5. 演示角度转换
    print("📐 角度转换演示...")
    
    test_angles = [0, 45, 90, 135, 180]
    
    for camera_angle in test_angles:
        robot_angle = calculator.transform_angle_to_robot_coordinates(camera_angle)
        compensation = calculator.calculate_angle_compensation(camera_angle, target_angle_deg=0)
        
        if robot_angle is not None and compensation is not None:
            print(f"相机角度{camera_angle:3d}° -> 机器人角度{robot_angle:6.2f}° (补偿: {compensation:+6.2f}°)")
        else:
            print(f"相机角度{camera_angle:3d}° -> 转换失败")

def demo_hot_reload():
    """演示热重载功能"""
    print("\n🔄 === 热重载演示 ===\n")
    
    calculator = CoordinateCalculator()
    
    print("🔄 测试变换矩阵热重载...")
    success = calculator.reload_transformation_matrix()
    print(f"重载结果: {'✅ 成功' if success else '❌ 失败'}")

def demo_calibration_quality_assessment():
    """演示标定质量评估"""
    print("\n📊 === 标定质量评估演示 ===\n")
    
    # 加载已有的标定数据
    calibrator = CoordinateCalibrator()
    
    # 尝试加载标定数据
    if calibrator.load_calibration_data():
        print("✅ 标定数据加载成功")
        
        # 获取质量评估
        quality_info = calibrator.get_calibration_quality_assessment()
        
        print(f"标定质量: {quality_info['quality_message']}")
        print(f"点数评估: {quality_info['points_message']}")
        print(f"建议: {quality_info['recommendation']}")
    else:
        print("❌ 无法加载标定数据")

def main():
    """主函数"""
    print("🚀 VisionCore 坐标系统演示\n")
    
    # 1. 演示标定流程
    calibrator = demo_coordinate_calibration()
    
    if calibrator:
        # 2. 演示坐标转换
        demo_coordinate_transformation()
        
        # 3. 演示热重载
        demo_hot_reload()
        
        # 4. 演示质量评估
        demo_calibration_quality_assessment()
    
    print("\n🎉 演示完成！")
    print("\n💡 使用说明:")
    print("   - CoordinateCalibrator: 负责坐标系标定")
    print("   - CoordinateCalculator: 负责坐标转换")
    print("   - 两个类职责分离，功能明确")

if __name__ == "__main__":
    main() 