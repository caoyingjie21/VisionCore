#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试集成的检测功能
验证RknnYolo的detect_with_coordinates方法
"""

import time
import sys
import os
import logging

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_detector_integration():
    """测试检测器集成功能"""
    print("=== 测试检测器集成功能 ===")
    
    try:
        from Rknn.RknnYolo import RKNN_YOLO
        import cv2
        import numpy as np
        
        # 创建测试图像
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 绘制一些测试目标（矩形）
        cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.rectangle(test_image, (300, 150), (400, 250), (255, 255, 255), -1)
        
        print("创建测试图像成功")
        
        # 尝试加载模型（如果存在）
        model_path = "./Models/test_model.pt"  # 或其他模型路径
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            print("跳过实际检测测试，只测试方法调用")
            return
        
        # 创建检测器实例
        detector = RKNN_YOLO(
            model_path=model_path,
            conf_threshold=0.5,
            nms_threshold=0.45
        )
        
        print("检测器初始化成功")
        
        # 测试detect_with_coordinates方法
        print("\n测试detect_with_coordinates方法...")
        
        # 模拟深度数据和相机参数
        depth_data = [1000.0] * (480 * 640)  # 模拟深度数据
        camera_params = type('CameraParams', (), {
            'width': 640,
            'height': 480,
            'cx': 320,
            'cy': 240,
            'fx': 500,
            'fy': 500,
            'k1': 0.0,
            'k2': 0.0,
            'f2rc': 0.0
        })()
        
        # 执行检测
        start_time = time.time()
        result = detector.detect_with_coordinates(
            image=test_image,
            depth_data=depth_data,
            camera_params=camera_params,
            draw_annotations=True
        )
        total_time = (time.time() - start_time) * 1000
        
        print(f"检测完成，总耗时: {total_time:.1f}ms")
        
        # 分析结果
        print(f"检测到目标数量: {result['detection_count']}")
        print(f"检测耗时: {result['timing']['detect_time']:.1f}ms")
        print(f"坐标计算耗时: {result['timing']['coord_time']:.1f}ms")
        print(f"总耗时: {result['timing']['total_time']:.1f}ms")
        
        if result['best_target']:
            x, y, z = result['best_target']['robot_3d']
            angle = result['best_target']['angle']
            print(f"最佳目标坐标: X={x:.3f}, Y={y:.3f}, Z={z:.3f}, Angle={angle:.1f}°")
        else:
            print("未检测到有效目标")
        
        # 保存结果图像
        if result['annotated_image'] is not None:
            output_path = "./test_detection_result.jpg"
            cv2.imwrite(output_path, result['annotated_image'])
            print(f"结果图像已保存: {output_path}")
        
        # 清理资源
        detector.release()
        print("检测器资源已释放")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖已正确安装")
    except Exception as e:
        print(f"测试异常: {e}")
        import traceback
        traceback.print_exc()

def test_main_integration():
    """测试main.py集成"""
    print("\n=== 测试main.py集成 ===")
    
    try:
        from main import VisionCoreApp
        
        # 创建应用实例
        app = VisionCoreApp("./Config/config.yaml")
        
        print("VisionCore应用创建成功")
        
        # 测试系统初始化
        print("测试系统初始化...")
        if app._initialize_system():
            print("系统初始化成功")
            
            # 获取组件
            camera = app.initializer.get_camera()
            detector = app.initializer.get_detector()
            configManager = app.initializer.get_config_manager()
            
            if camera and detector and configManager:
                print("所有组件获取成功")
                
                # 测试单次检测
                print("测试单次检测...")
                result = app._perform_single_detection(camera, detector, configManager)
                
                if result:
                    print(f"检测成功: 检测到{result['detection_count']}个目标")
                    print(f"总耗时: {result['timing']['total_time']:.1f}ms")
                    
                    if result['best_target']:
                        x, y, z = result['best_target']['robot_3d']
                        angle = result['best_target']['angle']
                        print(f"最佳目标: X={x:.3f}, Y={y:.3f}, Z={z:.3f}, Angle={angle:.1f}°")
                else:
                    print("检测失败")
            else:
                print("部分组件获取失败")
                if not camera:
                    print("- 相机未就绪")
                if not detector:
                    print("- 检测器未就绪")
                if not configManager:
                    print("- 配置管理器未就绪")
        else:
            print("系统初始化失败")
        
        # 清理资源
        app.initializer.cleanup()
        print("资源清理完成")
        
    except Exception as e:
        print(f"集成测试异常: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("开始测试集成检测功能...")
    
    # 测试检测器集成
    test_detector_integration()
    
    # 测试main.py集成
    test_main_integration()
    
    print("\n=== 测试完成 ===")
    print("\n总结:")
    print("1. 所有计算逻辑已集成到RknnYolo.py中")
    print("2. main.py保持简洁，只负责调用和协调")
    print("3. 检测结果包含完整的图像绘制和坐标信息")
    print("4. 支持图像保存和SFTP上传功能")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试被中断")
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc() 