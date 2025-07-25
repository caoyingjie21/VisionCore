#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
系统组件初始化器模块
包含所有系统组件的初始化函数，支持自动重启和健康检查
"""

import logging
import sys
import os
import asyncio
import time
import threading
from typing import Tuple, Optional, Any, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta

from Managers.ConfigManager import ConfigManager
from Managers.LogManager import LogManager
from Mqtt.MqttClient import MqttClient


@dataclass
class ComponentStatus:
    """组件状态数据类"""
    name: str
    is_healthy: bool
    last_check: datetime
    error_count: int
    last_error: Optional[str] = None


class SystemMonitor:
    """系统监控器，负责健康检查和自动重启"""
    
    def __init__(self, logger: logging.Logger, check_interval: int = 30):
        """
        初始化系统监控器
        
        Args:
            logger: 日志记录器
            check_interval: 健康检查间隔（秒）
        """
        self.logger = logger
        self.check_interval = check_interval
        self.components: Dict[str, Any] = {}
        self.status: Dict[str, ComponentStatus] = {}
        self.monitoring = False
        self.monitor_thread = None
        self.restart_callbacks: Dict[str, callable] = {}
        
    def register_component(self, name: str, component: Any, restart_callback: callable = None):
        """
        注册需要监控的组件
        
        Args:
            name: 组件名称
            component: 组件实例
            restart_callback: 重启回调函数
        """
        self.components[name] = component
        self.status[name] = ComponentStatus(
            name=name,
            is_healthy=True,
            last_check=datetime.now(),
            error_count=0
        )
        if restart_callback:
            self.restart_callbacks[name] = restart_callback
        
        self.logger.info(f"已注册监控组件: {name}")
    
    def start_monitoring(self):
        """启动监控线程"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("系统监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                self._check_all_components()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
                time.sleep(5)  # 短暂等待后继续
    
    def _check_all_components(self):
        """检查所有组件健康状态"""
        for name, component in self.components.items():
            try:
                is_healthy = self._check_component_health(name, component)
                status = self.status[name]
                
                if is_healthy:
                    if not status.is_healthy:
                        self.logger.info(f"组件 {name} 恢复正常")
                    status.is_healthy = True
                    status.error_count = 0
                else:
                    status.is_healthy = False
                    status.error_count += 1
                    self.logger.warning(f"组件 {name} 健康检查失败，错误次数: {status.error_count}")
                    
                    # 连续3次失败则尝试重启
                    if status.error_count >= 3:
                        self._restart_component(name)
                
                status.last_check = datetime.now()
                
            except Exception as e:
                self.logger.error(f"检查组件 {name} 时出错: {e}")
    
    def _check_component_health(self, name: str, component: Any) -> bool:
        """检查单个组件健康状态"""
        try:
            if name == "camera":
                # 检查相机连接状态
                return getattr(component, 'is_connected', False)
            
            elif name == "mqtt_client":
                # 检查MQTT连接状态
                return getattr(component, 'is_connected', False)
            
            elif name == "detector":
                # 检查检测器状态（简单检查是否存在）
                return component is not None
            
            else:
                # 默认检查：组件存在且不为None
                return component is not None
                
        except Exception as e:
            self.logger.error(f"健康检查异常 {name}: {e}")
            return False
    
    def _restart_component(self, name: str):
        """重启指定组件"""
        try:
            self.logger.warning(f"正在重启组件: {name}")
            
            if name in self.restart_callbacks:
                # 使用自定义重启回调
                success = self.restart_callbacks[name]()
                if success:
                    self.logger.info(f"组件 {name} 重启成功")
                    self.status[name].error_count = 0
                else:
                    self.logger.error(f"组件 {name} 重启失败")
            else:
                self.logger.warning(f"组件 {name} 没有重启回调函数")
                
        except Exception as e:
            self.logger.error(f"重启组件 {name} 时出错: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态信息"""
        return {
            "monitoring": self.monitoring,
            "components": {
                name: {
                    "healthy": status.is_healthy,
                    "error_count": status.error_count,
                    "last_check": status.last_check.isoformat(),
                    "last_error": status.last_error
                }
                for name, status in self.status.items()
            },
            "timestamp": datetime.now().isoformat()
        }


def initialize_logging(config_mgr: ConfigManager) -> Tuple[LogManager, logging.Logger]:
    """
    初始化日志管理器
    
    Args:
        config_mgr (ConfigManager): 配置管理器实例
        
    Returns:
        tuple: (log_manager, logger) 日志管理器实例和主日志器
    """
    try:
        # 获取日志配置
        logging_config = config_mgr.get_config("logging")
        
        if logging_config and logging_config.get("enabled", True):
            # 从配置中获取日志参数
            log_level_str = logging_config.get("level", "INFO")
            log_level = getattr(logging, log_level_str.upper(), logging.INFO)
            
            file_config = logging_config.get("file", {})
            console_config = logging_config.get("console", {})
            
            log_dir = file_config.get("path", "logs")
            file_output = file_config.get("enabled", True)
            console_output = console_config.get("enabled", True)
            max_backup_count = file_config.get("backup_count", 30)
            
            # 初始化日志管理器
            log_manager = LogManager(
                log_dir=log_dir,
                app_name="VisionCore",
                level=log_level,
                console_output=console_output,
                file_output=file_output,
                max_backup_count=max_backup_count
            )
            
            # 获取主日志器
            logger = log_manager.get_logger("VisionCore")
            logger.info("=== VisionCore 系统启动 ===")
            logger.info(f"日志级别: {log_level_str}")
            logger.info(f"控制台输出: {console_output}")
            logger.info(f"文件输出: {file_output}")
            logger.info(f"日志目录: {log_dir}")
            
            return log_manager, logger
            
        else:
            # 如果日志配置被禁用，使用基本的控制台输出
            print("日志功能已禁用，使用基本控制台输出")
            log_manager = LogManager(
                log_dir="logs",
                app_name="VisionCore",
                level=logging.INFO,
                console_output=True,
                file_output=False
            )
            logger = log_manager.get_logger("VisionCore")
            return log_manager, logger
            
    except Exception as e:
        # 如果配置加载失败，使用默认配置
        print(f"日志配置加载失败，使用默认配置: {e}")
        log_manager = LogManager(
            log_dir="logs",
            app_name="VisionCore", 
            level=logging.INFO,
            console_output=True,
            file_output=True
        )
        logger = log_manager.get_logger("VisionCore")
        logger.warning("使用默认日志配置")
        return log_manager, logger


def initialize_mqtt_client(config_mgr: ConfigManager, logger: logging.Logger, max_retries: int = 5) -> Optional[MqttClient]:
    """
    初始化MQTT客户端（支持重试）
    
    Args:
        config_mgr (ConfigManager): 配置管理器实例
        logger: 日志器实例
        max_retries: 最大重试次数
        
    Returns:
        MqttClient: MQTT客户端实例或None
    """
    mqtt_config = config_mgr.get_config("mqtt")
    
    if not mqtt_config or not mqtt_config.get("enabled", False):
        logger.info("MQTT功能已禁用")
        return None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"正在初始化MQTT客户端... (尝试 {attempt + 1}/{max_retries})")
            mqtt_client = MqttClient(mqtt_config)
            logger.info("MQTT客户端初始化成功")
            return mqtt_client
            
        except Exception as e:
            logger.error(f"MQTT客户端初始化失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 递增等待时间
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                logger.error("MQTT客户端初始化最终失败")
    
    return None


def initialize_camera(config_mgr: ConfigManager, logger: logging.Logger, max_retries: int = 5) -> Optional[Any]:
    """
    初始化相机（支持重试）
    
    Args:
        config_mgr (ConfigManager): 配置管理器实例
        logger: 日志器实例
        max_retries: 最大重试次数
        
    Returns:
        QtVisionSick: 相机实例或None
    """
    camera_config = config_mgr.get_config("camera")
    
    if not camera_config or not camera_config.get("enabled", False):
        logger.info("相机功能已禁用")
        return None
    
    # 获取连接配置
    connection_config = camera_config.get("connection", {})
    ip_address = connection_config.get("ip", "192.168.1.100")
    port = connection_config.get("port", 2122)
    
    for attempt in range(max_retries):
        try:
            logger.info(f"正在初始化SICK相机: {ip_address}:{port} (尝试 {attempt + 1}/{max_retries})")
            
            # 初始化相机实例
            from SickVision.SickSDK import QtVisionSick
            camera = QtVisionSick(
                ipAddr=ip_address,
                port=port,
                protocol="Cola2"
            )
            
            # 尝试连接相机
            logger.info("正在连接相机...")
            success = camera.connect()
            
            if success:
                logger.info(f"相机连接成功: {ip_address}:{port}")
                return camera
            else:
                logger.error(f"相机连接失败 (尝试 {attempt + 1}/{max_retries})")
                
        except Exception as e:
            logger.error(f"相机初始化异常 (尝试 {attempt + 1}/{max_retries}): {e}")
        
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 3  # 递增等待时间
            logger.info(f"等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
    
    logger.error("相机初始化最终失败")
    return None


def initialize_rknn(config_mgr: ConfigManager, logger: logging.Logger, max_retries: int = 3) -> Optional[Any]:
    """
    初始化RKNN检测器（支持重试）
    
    Args:
        config_mgr (ConfigManager): 配置管理器实例
        logger: 日志器实例
        max_retries: 最大重试次数
        
    Returns:
        RKNN_YOLO: 检测器实例或None
    """
    model_config = config_mgr.get_config("model")
    
    if not model_config or not model_config.get("enabled", False):
        logger.info("模型检测功能已禁用")
        return None
    
    # 获取平台和模型路径
    paths_config = model_config.get("paths", {})
    inference_config = model_config.get("inference", {})
    
    # 根据平台选择模型路径
    import platform as plt
    if plt.machine().lower() in ['aarch64', 'arm64']:
        model_path = paths_config.get("linux_arm64")
    elif plt.system().lower().startswith("windows"):
        model_path = paths_config.get("windows")
    else:
        model_path = paths_config.get("linux_x64")
    
    if not model_path or not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"正在初始化RKNN检测器: {model_path} (尝试 {attempt + 1}/{max_retries})")
            
            # 初始化检测器
            from Rknn.RknnYolo import RKNN_YOLO
            detector = RKNN_YOLO(
                model_path=model_path,
                conf_threshold=inference_config.get("conf_threshold", 0.7),
                nms_threshold=inference_config.get("nms_threshold", 0.45)
            )
            
            logger.info("RKNN检测器初始化成功")
            return detector
            
        except Exception as e:
            logger.error(f"RKNN检测器初始化失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
    
    logger.error("RKNN检测器初始化最终失败")
    return None


def cleanup_resources(logger: logging.Logger, **resources):
    """
    清理系统资源
    
    Args:
        logger: 日志器实例
        **resources: 需要清理的资源，键为资源名称，值为资源实例
    """
    logger.info("正在清理系统资源...")
    
    # 清理相机
    camera = resources.get('camera')
    if camera:
        try:
            camera.disconnect()
            logger.info("相机已断开连接")
        except Exception as e:
            logger.error(f"相机断开连接失败: {e}")
    
    # 清理RKNN检测器
    detector = resources.get('detector')
    if detector:
        try:
            detector.release()
            logger.info("RKNN检测器已释放")
        except Exception as e:
            logger.error(f"RKNN检测器释放失败: {e}")
    
    # 清理MQTT客户端
    mqtt_client = resources.get('mqtt_client')
    if mqtt_client:
        try:
            asyncio.run(mqtt_client.disconnect())
            logger.info("MQTT客户端已断开")
        except Exception as e:
            logger.error(f"MQTT客户端断开失败: {e}")


class SystemInitializer:
    """系统初始化器类，提供更高级的初始化管理和自动重启功能"""
    
    def __init__(self, config_path: str = "./Config/config.yaml"):
        """
        初始化系统初始化器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config_mgr = None
        self.logger = None
        self.resources = {}
        self.monitor = None
        self.max_restart_attempts = 3
        self.restart_count = 0
    
    def initialize_config(self) -> bool:
        """初始化配置管理器"""
        try:
            print("正在加载配置文件...")
            self.config_mgr = ConfigManager(self.config_path)
            print("配置文件加载成功")
            return True
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            return False
    
    def initialize_all_components(self) -> bool:
        """初始化所有系统组件"""
        try:
            # 初始化日志
            log_manager, self.logger = initialize_logging(self.config_mgr)
            self.resources['log_manager'] = log_manager
            
            self.logger.info("开始初始化系统组件...")
            
            # 初始化MQTT客户端
            mqtt_client = initialize_mqtt_client(self.config_mgr, self.logger)
            self.resources['mqtt_client'] = mqtt_client
            
            # 初始化相机
            camera = initialize_camera(self.config_mgr, self.logger)
            self.resources['camera'] = camera
            
            # 初始化RKNN检测器
            detector = initialize_rknn(self.config_mgr, self.logger)
            self.resources['detector'] = detector
            
            # 设置系统监控
            self._setup_monitoring()
            
            self.logger.info("系统组件初始化完成")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"系统组件初始化失败: {e}", exc_info=True)
            else:
                print(f"系统组件初始化失败: {e}")
            return False
    
    def _setup_monitoring(self):
        """设置系统监控"""
        if not self.logger:
            return
            
        self.monitor = SystemMonitor(self.logger, check_interval=30)
        
        # 注册组件和重启回调
        if self.resources.get('camera'):
            self.monitor.register_component(
                'camera', 
                self.resources['camera'],
                lambda: self._restart_camera()
            )
        
        if self.resources.get('mqtt_client'):
            self.monitor.register_component(
                'mqtt_client',
                self.resources['mqtt_client'],
                lambda: self._restart_mqtt()
            )
        
        if self.resources.get('detector'):
            self.monitor.register_component('detector', self.resources['detector'])
        
        # 启动监控
        self.monitor.start_monitoring()
    
    def _restart_camera(self) -> bool:
        """重启相机"""
        try:
            self.logger.info("正在重启相机...")
            
            # 清理旧相机
            old_camera = self.resources.get('camera')
            if old_camera:
                try:
                    old_camera.disconnect()
                except:
                    pass
            
            # 重新初始化相机
            camera = initialize_camera(self.config_mgr, self.logger, max_retries=3)
            self.resources['camera'] = camera
            
            if camera:
                self.monitor.components['camera'] = camera
                self.logger.info("相机重启成功")
                return True
            else:
                self.logger.error("相机重启失败")
                return False
                
        except Exception as e:
            self.logger.error(f"相机重启异常: {e}")
            return False
    
    def _restart_mqtt(self) -> bool:
        """重启MQTT客户端"""
        try:
            self.logger.info("正在重启MQTT客户端...")
            
            # 清理旧MQTT客户端
            old_mqtt = self.resources.get('mqtt_client')
            if old_mqtt:
                try:
                    asyncio.run(old_mqtt.disconnect())
                except:
                    pass
            
            # 重新初始化MQTT客户端
            mqtt_client = initialize_mqtt_client(self.config_mgr, self.logger, max_retries=3)
            self.resources['mqtt_client'] = mqtt_client
            
            if mqtt_client:
                self.monitor.components['mqtt_client'] = mqtt_client
                self.logger.info("MQTT客户端重启成功")
                return True
            else:
                self.logger.error("MQTT客户端重启失败")
                return False
                
        except Exception as e:
            self.logger.error(f"MQTT客户端重启异常: {e}")
            return False
    
    def get_resource(self, name: str) -> Optional[Any]:
        """获取指定的资源"""
        return self.resources.get(name)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        if self.monitor:
            return self.monitor.get_system_status()
        return {"monitoring": False, "components": {}}
    
    def restart_system(self):
        """重启整个系统"""
        self.restart_count += 1
        if self.restart_count > self.max_restart_attempts:
            self.logger.error("系统重启次数过多，程序退出")
            sys.exit(1)
        
        self.logger.warning(f"正在重启系统 (第 {self.restart_count} 次)...")
        
        # 清理资源
        self.cleanup()
        
        # 等待一段时间
        time.sleep(5)
        
        # 重新初始化
        if self.initialize_config() and self.initialize_all_components():
            self.logger.info("系统重启成功")
            self.restart_count = 0  # 重置重启计数
        else:
            self.logger.error("系统重启失败")
            sys.exit(1)
    
    def cleanup(self):
        """清理所有资源"""
        # 停止监控
        if self.monitor:
            self.monitor.stop_monitoring()
        
        if self.logger:
            cleanup_resources(self.logger, **self.resources)
            self.logger.info("=== VisionCore 系统关闭 ===")
        else:
            print("系统资源清理完成") 