#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VisionCore 主程序入口
支持自动重启和永久运行模式，适合在开发板上长期运行
"""

import sys
import time
import signal
from System.SystemInitializer import SystemInitializer


class VisionCoreApp:
    """VisionCore应用程序类，支持自动重启和监控"""
    
    def __init__(self, config_path: str = "./Config/config.yaml"):
        """
        初始化应用程序
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.initializer = None
        self.running = True
        self.restart_on_failure = True
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n接收到信号 {signum}，正在优雅关闭...")
        self.running = False
        self.restart_on_failure = False
    
    def start(self):
        """启动应用程序主循环"""
        print("启动 VisionCore 系统...")
        
        while self.running:
            try:
                # 初始化系统
                if not self._initialize_system():
                    if self.restart_on_failure:
                        print("系统初始化失败，5秒后重试...")
                        time.sleep(5)
                        continue
                    else:
                        break
                
                # 运行主循环
                self._run_main_loop()
                
            except Exception as e:
                if self.initializer and self.initializer.logger:
                    self.initializer.logger.error(f"系统运行异常: {e}", exc_info=True)
                else:
                    print(f"系统运行异常: {e}")
                
                if self.restart_on_failure and self.running:
                    if self.initializer and self.initializer.logger:
                        self.initializer.logger.warning("系统将在10秒后自动重启...")
                    else:
                        print("系统将在10秒后自动重启...")
                    time.sleep(10)
                else:
                    break
            finally:
                # 清理资源
                if self.initializer:
                    self.initializer.cleanup()
                    self.initializer = None
        
        print("VisionCore 系统已关闭")
    
    def _initialize_system(self) -> bool:
        """初始化系统组件"""
        try:
            # 创建系统初始化器
            self.initializer = SystemInitializer(self.config_path)
            
            # 初始化配置
            if not self.initializer.initialize_config():
                return False
            
            # 初始化所有组件
            if not self.initializer.initialize_all_components():
                return False
            
            return True
            
        except Exception as e:
            print(f"系统初始化异常: {e}")
            return False
    
    def _run_main_loop(self):
        """运行主程序循环"""
        logger = self.initializer.logger
        camera = self.initializer.get_resource('camera')
        mqtt_client = self.initializer.get_resource('mqtt_client')
        detector = self.initializer.get_resource('detector')
        
        logger.info("系统运行中，按Ctrl+C停止...")
        
        # 主程序循环
        loop_count = 0
        while self.running:
            try:
                # 这里可以添加具体的业务逻辑
                # 例如：
                # 1. 从相机获取图像
                # 2. 进行目标检测
                # 3. 通过MQTT发布结果
                # 4. 处理MQTT命令
                
                # 示例业务逻辑（每10秒执行一次）
                if loop_count % 10 == 0:
                    self._execute_business_logic(camera, detector, mqtt_client, logger)
                
                # 检查系统健康状态（每60秒）
                if loop_count % 60 == 0:
                    self._check_system_health(logger)
                
                time.sleep(1)
                loop_count += 1
                
                # 防止计数器溢出
                if loop_count > 86400:  # 24小时重置
                    loop_count = 0
                
            except KeyboardInterrupt:
                logger.info("接收到停止信号")
                self.running = False
                break
            except Exception as e:
                logger.error(f"主循环异常: {e}", exc_info=True)
                # 继续运行，不退出主循环
                time.sleep(1)
    
    def _execute_business_logic(self, camera, detector, mqtt_client, logger):
        """执行主要业务逻辑"""
        try:
            # 示例业务逻辑
            logger.debug("执行业务逻辑检查...")
            
            # 检查相机状态
            if camera and hasattr(camera, 'is_connected') and camera.is_connected:
                logger.debug("相机状态正常")
                # 这里可以添加图像采集逻辑
                # image = camera.get_fresh_frame()
                # if image:
                #     results = detector.detect(image)
                #     # 发布检测结果到MQTT
            
            # 检查MQTT状态
            if mqtt_client and hasattr(mqtt_client, 'is_connected') and mqtt_client.is_connected:
                logger.debug("MQTT连接状态正常")
                # 这里可以添加状态发布逻辑
                # mqtt_client.publish("sickvision/system/heartbeat", {"status": "running", "timestamp": time.time()})
            
        except Exception as e:
            logger.error(f"业务逻辑执行异常: {e}")
    
    def _check_system_health(self, logger):
        """检查系统健康状态"""
        try:
            if self.initializer.monitor:
                status = self.initializer.get_system_status()
                
                # 记录系统状态
                healthy_components = sum(1 for comp in status.get("components", {}).values() if comp.get("healthy", False))
                total_components = len(status.get("components", {}))
                
                logger.info(f"系统健康检查: {healthy_components}/{total_components} 组件正常")
                
                # 检查是否有组件不健康
                unhealthy = [name for name, comp in status.get("components", {}).items() if not comp.get("healthy", False)]
                if unhealthy:
                    logger.warning(f"不健康的组件: {', '.join(unhealthy)}")
                
        except Exception as e:
            logger.error(f"健康检查异常: {e}")


def main():
    """
    主函数 - 支持自动重启的版本
    """
    # 创建并启动应用程序
    app = VisionCoreApp("./Config/config.yaml")
    app.start()


def main_simple():
    """
    简化版主函数 - 单次运行版本
    """
    # 创建系统初始化器
    initializer = SystemInitializer("./Config/config.yaml")
    
    try:
        # 初始化配置
        if not initializer.initialize_config():
            sys.exit(1)
        
        # 初始化所有组件
        if not initializer.initialize_all_components():
            sys.exit(1)
        
        # 获取关键组件
        logger = initializer.logger
        camera = initializer.get_resource('camera')
        mqtt_client = initializer.get_resource('mqtt_client')
        detector = initializer.get_resource('detector')
        
        # 主程序运行逻辑
        logger.info("系统运行中，按Ctrl+C停止...")
        
        # 简单的运行循环
        while True:
            time.sleep(1)
            # TODO: 添加具体的业务逻辑
            
    except KeyboardInterrupt:
        if initializer.logger:
            initializer.logger.info("接收到停止信号，正在关闭系统...")
        else:
            print("接收到停止信号，正在关闭系统...")
    except Exception as e:
        if initializer.logger:
            initializer.logger.error(f"系统运行出错: {e}", exc_info=True)
        else:
            print(f"系统运行出错: {e}")
    finally:
        # 清理所有资源
        initializer.cleanup()


if __name__ == "__main__":
    # 使用自动重启的应用程序类（推荐用于生产环境）
    main()
    
    # 或者使用简单版本（用于开发和测试）
    # main_simple()
    