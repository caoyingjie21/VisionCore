#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
系统初始化器模块
提供完整的系统组件初始化和管理功能
"""

import logging
import sys
import os
import asyncio
import time
import platform as plt
from typing import Tuple, Optional, Any, Dict, Callable

from Managers.ConfigManager import ConfigManager
from Managers.LogManager import LogManager
from Mqtt.MqttClient import MqttClient
from TcpServer.TcpServer import TcpServer
from .SystemMonitor import SystemMonitor


class SystemInitializer:
    """系统初始化器类，提供完整的系统初始化管理和自动重启功能"""
    
    def __init__(self, config_path: str = "./Config/config.yaml"):
        """
        初始化系统初始化器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config_mgr: Optional[ConfigManager] = None
        self.logger: Optional[logging.Logger] = None
        self.resources: Dict[str, Any] = {}
        self.monitor: Optional[SystemMonitor] = None
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
    
    def initialize_logging(self) -> bool:
        """
        初始化日志管理器
        
        Returns:
            bool: 初始化是否成功
        """
        if not self.config_mgr:
            print("配置管理器未初始化")
            return False
            
        try:
            # 获取日志配置
            logging_config = self.config_mgr.get_config("logging")
            
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
                self.logger = log_manager.get_logger("VisionCore")
                self.logger.info("=== VisionCore 系统启动 ===")
                self.logger.info(f"日志级别: {log_level_str}")
                self.logger.info(f"控制台输出: {console_output}")
                self.logger.info(f"文件输出: {file_output}")
                self.logger.info(f"日志目录: {log_dir}")
                
                # 保存日志管理器到资源
                self.resources['log_manager'] = log_manager
                return True
                
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
                self.logger = log_manager.get_logger("VisionCore")
                self.resources['log_manager'] = log_manager
                return True
                
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
            self.logger = log_manager.get_logger("VisionCore")
            self.logger.warning("使用默认日志配置")
            self.resources['log_manager'] = log_manager
            return True
    
    def initialize_mqtt_client(self, max_retries: int = 5) -> bool:
        """
        初始化MQTT客户端并连接到broker（支持重试）
        
        Args:
            max_retries: 最大重试次数
            
        Returns:
            bool: 初始化是否成功
        """
        if not self.config_mgr or not self.logger:
            return False
            
        mqtt_config = self.config_mgr.get_config("mqtt")
        
        if not mqtt_config or not mqtt_config.get("enabled", False):
            self.logger.info("MQTT功能已禁用")
            return True  # 禁用时也认为是成功的
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"正在初始化MQTT客户端... (尝试 {attempt + 1}/{max_retries})")
                mqtt_client = MqttClient(mqtt_config)
                
                # 连接到MQTT broker
                self.logger.info("正在连接MQTT broker...")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    connected = loop.run_until_complete(mqtt_client.connect())
                    if connected:
                        self.logger.info("MQTT客户端连接成功")
                        self.resources['mqtt_client'] = mqtt_client
                        return True
                    else:
                        self.logger.error(f"MQTT客户端连接失败 (尝试 {attempt + 1}/{max_retries})")
                finally:
                    loop.close()
                
            except Exception as e:
                self.logger.error(f"MQTT客户端初始化失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 递增等待时间
                    self.logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    self.logger.error("MQTT客户端初始化最终失败")
        
        return False
    
    def initialize_camera(self, max_retries: int = 5) -> bool:
        """
        初始化相机（支持重试）
        
        Args:
            max_retries: 最大重试次数
            
        Returns:
            bool: 初始化是否成功
        """
        if not self.config_mgr or not self.logger:
            return False
            
        camera_config = self.config_mgr.get_config("camera")
        
        if not camera_config or not camera_config.get("enabled", False):
            self.logger.info("相机功能已禁用")
            return True  # 禁用时也认为是成功的
        
        # 获取连接配置
        connection_config = camera_config.get("connection", {})
        ip_address = connection_config.get("ip", "192.168.1.100")
        port = connection_config.get("port", 2122)
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"正在初始化SICK相机: {ip_address}:{port} (尝试 {attempt + 1}/{max_retries})")
                
                # 初始化相机实例
                from SickVision.SickSDK import QtVisionSick
                camera = QtVisionSick(
                    ipAddr=ip_address,
                    port=port,
                    protocol="Cola2"
                )
                
                # 尝试连接相机
                self.logger.info("正在连接相机...")
                success = camera.connect()
                
                if success:
                    self.logger.info(f"相机连接成功: {ip_address}:{port}")
                    self.resources['camera'] = camera
                    return True
                else:
                    self.logger.error(f"相机连接失败 (尝试 {attempt + 1}/{max_retries})")
                    
            except Exception as e:
                self.logger.error(f"相机初始化异常 (尝试 {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 3  # 递增等待时间
                self.logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        
        self.logger.error("相机初始化最终失败")
        return False
    
    def initialize_detector(self, max_retries: int = 3) -> bool:
        """
        初始化检测器模型（根据平台自动选择模型）
        
        Args:
            max_retries: 最大重试次数
            
        Returns:
            bool: 初始化是否成功
        """
        if not self.logger:
            return False
        
        # 根据平台自动选择模型文件
        model_path = self._get_model_path()
        
        if not model_path or not os.path.exists(model_path):
            self.logger.error(f"模型文件不存在: {model_path}")
            return False
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"正在初始化检测器: {model_path} (尝试 {attempt + 1}/{max_retries})")
                
                # 根据模型类型选择检测器
                detector = self._create_detector(model_path)
                
                if detector:
                    # 预热模型
                    self.logger.info("正在预热模型...")
                    if self._warmup_model(detector):
                        self.logger.info("检测器初始化和预热成功")
                        self.resources['detector'] = detector
                        return True
                    else:
                        self.logger.warning("模型预热失败，但检测器仍可使用")
                        self.resources['detector'] = detector
                        return True
                else:
                    self.logger.error(f"检测器创建失败 (尝试 {attempt + 1}/{max_retries})")
                
            except Exception as e:
                self.logger.error(f"检测器初始化失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    self.logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
        
        self.logger.error("检测器初始化最终失败")
        return False
    
    def initialize_tcp_server(self, max_retries: int = 3) -> bool:
        """
        初始化TCP服务器
        
        Args:
            max_retries: 最大重试次数
            
        Returns:
            bool: 初始化是否成功
        """
        if not self.config_mgr or not self.logger:
            return False
            
        tcp_config = self.config_mgr.get_config("tcp_server")
        
        if not tcp_config or not tcp_config.get("enabled", False):
            self.logger.info("TCP服务器功能已禁用")
            return True  # 禁用时也认为是成功的
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"正在初始化TCP服务器... (尝试 {attempt + 1}/{max_retries})")
                
                # 创建TCP服务器实例（不设置回调，留给main.py处理）
                tcp_server = TcpServer(tcp_config, self.logger)
                
                # 启动TCP服务器
                if tcp_server.start():
                    self.logger.info("TCP服务器启动成功")
                    self.resources['tcp_server'] = tcp_server
                    return True
                else:
                    self.logger.error(f"TCP服务器启动失败 (尝试 {attempt + 1}/{max_retries})")
                
            except Exception as e:
                self.logger.error(f"TCP服务器初始化失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 递增等待时间
                    self.logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    self.logger.error("TCP服务器初始化最终失败")
        
        return False
    
    def create_tcp_message_handler(self, catch_handler: Optional[Callable] = None):
        """
        创建TCP消息处理器函数，供main.py使用
        
        Args:
            catch_handler: 外部提供的catch指令处理函数
        
        Returns:
            Callable: 消息处理函数
        """
        def handle_tcp_message(client_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """
            处理TCP消息的统一入口
            
            Args:
                client_id: 客户端ID
                message: 接收到的消息
                
            Returns:
                Optional[Dict]: 响应消息，如果为None则不回复
            """
            try:
                message_type = message.get("type", "unknown")
                
                if message_type == "catch":
                    if catch_handler:
                        return catch_handler(client_id, message)
                    else:
                        return {
                            "type": "catch_response",
                            "success": False,
                            "message": "catch处理器未设置",
                            "timestamp": time.time()
                        }
                else:
                    self.logger.debug(f"未处理的消息类型: {message_type} from {client_id}")
                    return {
                        "type": "error",
                        "message": f"不支持的消息类型: {message_type}",
                        "timestamp": time.time()
                    }
                    
            except Exception as e:
                self.logger.error(f"处理TCP消息时出错: {e}")
                return {
                    "type": "error",
                    "message": f"消息处理失败: {str(e)}",
                    "timestamp": time.time()
                }
        
        return handle_tcp_message
    
    def _get_model_path(self) -> Optional[str]:
        """
        根据当前平台自动选择模型路径（通过扫描Models目录）
        
        Returns:
            模型文件路径或None
        """
        models_dir = "./Models"
        
        if not os.path.exists(models_dir):
            if self.logger:
                self.logger.error(f"Models目录不存在: {models_dir}")
            else:
                print(f"Models目录不存在: {models_dir}")
            return None
        
        # 扫描Models目录中的文件
        try:
            files = os.listdir(models_dir)
        except Exception as e:
            if self.logger:
                self.logger.error(f"无法读取Models目录: {e}")
            else:
                print(f"无法读取Models目录: {e}")
            return None
        
        # 根据平台选择模型文件
        if plt.machine().lower() in ['aarch64']:
            # ARM平台使用RKNN模型
            rknn_files = [f for f in files if f.endswith('.rknn')]
            if rknn_files:
                model_file = rknn_files[0]  # 使用第一个找到的RKNN文件
                if self.logger:
                    self.logger.info(f"检测到ARM平台，使用RKNN模型: {model_file}")
                else:
                    print(f"检测到ARM平台，使用RKNN模型: {model_file}")
            else:
                if self.logger:
                    self.logger.error("ARM平台但未找到RKNN模型文件")
                else:
                    print("ARM平台但未找到RKNN模型文件")
                return None
        elif plt.system().lower() in ["windows", 'arm64']:
            # Windows平台使用PyTorch模型
            pt_files = [f for f in files if f.endswith('.pt')]
            if pt_files:
                model_file = pt_files[0]  # 使用第一个找到的PT文件
                if self.logger:
                    self.logger.info(f"检测到Windows平台，使用PyTorch模型: {model_file}")
                else:
                    print(f"检测到Windows平台，使用PyTorch模型: {model_file}")
            else:
                if self.logger:
                    self.logger.error("Windows平台但未找到PyTorch模型文件")
                else:
                    print("Windows平台但未找到PyTorch模型文件")
                return None
        else:
            # 其他平台默认使用PyTorch模型
            pt_files = [f for f in files if f.endswith('.pt')]
            if pt_files:
                model_file = pt_files[0]  # 使用第一个找到的PT文件
                if self.logger:
                    self.logger.info(f"检测到其他平台，使用PyTorch模型: {model_file}")
                else:
                    print(f"检测到其他平台，使用PyTorch模型: {model_file}")
            else:
                if self.logger:
                    self.logger.error("未找到PyTorch模型文件")
                else:
                    print("未找到PyTorch模型文件")
                return None
        
        return os.path.join(models_dir, model_file)
    
    def _create_detector(self, model_path: str):
        """
        根据模型文件类型创建对应的检测器
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            检测器实例或None
        """
        try:
            if model_path.endswith('.rknn'):
                # 使用RKNN检测器
                from Rknn.RknnYolo import RKNN_YOLO
                detector = RKNN_YOLO(
                    model_path=model_path,
                    conf_threshold=0.7,
                    nms_threshold=0.45
                )
                self.logger.info("创建RKNN检测器成功")
                return detector
                
            elif model_path.endswith('.pt'):
                # 使用PyTorch检测器（如果有的话）
                # 这里可以根据实际情况导入对应的PyTorch检测器
                try:
                    from ultralytics import YOLO
                    detector = YOLO(model_path)
                    self.logger.info("创建PyTorch检测器成功")
                    return detector
                except ImportError:
                    self.logger.error("ultralytics库未安装，无法使用PyTorch模型")
                    return None
            else:
                self.logger.error(f"不支持的模型格式: {model_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"创建检测器失败: {e}")
            return None
    
    def _warmup_model(self, detector) -> bool:
        """
        预热模型（使用test.jpg测试图片）
        
        Args:
            detector: 检测器实例
            
        Returns:
            预热是否成功
        """
        try:
            # 查找测试图片
            test_image_path = "./Models/test.jpg"
            
            if not os.path.exists(test_image_path):
                if self.logger:
                    self.logger.warning(f"测试图片不存在: {test_image_path}，使用随机数据预热")
                else:
                    print(f"测试图片不存在: {test_image_path}，使用随机数据预热")
                
                # 回退到随机数据
                import numpy as np
                test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            else:
                # 加载测试图片
                try:
                    import cv2
                    test_input = cv2.imread(test_image_path)
                    if test_input is None:
                        raise Exception("cv2无法读取图片")
                    
                    if self.logger:
                        self.logger.info(f"使用测试图片进行预热: {test_image_path}")
                        self.logger.info(f"图片尺寸: {test_input.shape}")
                    else:
                        print(f"使用测试图片进行预热: {test_image_path}")
                        print(f"图片尺寸: {test_input.shape}")
                    
                except ImportError:
                    if self.logger:
                        self.logger.warning("OpenCV未安装，尝试使用PIL")
                    try:
                        from PIL import Image
                        import numpy as np
                        pil_image = Image.open(test_image_path)
                        test_input = np.array(pil_image)
                        
                        # 确保是RGB格式
                        if len(test_input.shape) == 3 and test_input.shape[2] == 3:
                            # PIL默认是RGB，转换为BGR（如果需要）
                            test_input = cv2.cvtColor(test_input, cv2.COLOR_RGB2BGR) if 'cv2' in locals() else test_input
                        
                        if self.logger:
                            self.logger.info(f"使用PIL加载测试图片: {test_image_path}")
                            self.logger.info(f"图片尺寸: {test_input.shape}")
                    except ImportError:
                        if self.logger:
                            self.logger.warning("PIL也未安装，使用随机数据预热")
                        # 回退到随机数据
                        import numpy as np
                        test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"读取测试图片失败: {e}，使用随机数据预热")
                    # 回退到随机数据
                    import numpy as np
                    test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            if self.logger:
                self.logger.info("开始模型预热...")
            else:
                print("开始模型预热...")
            
            # 进行一次预热推理
            try:
                if hasattr(detector, 'predict'):
                    # PyTorch YOLO模型
                    results = detector.predict(test_input, verbose=False)
                    detection_count = self._parse_yolo_results(results)
                    
                elif hasattr(detector, 'inference'):
                    # RKNN模型
                    results = detector.inference(test_input)
                    detection_count = self._parse_rknn_results(results)
                    
                else:
                    if self.logger:
                        self.logger.warning("检测器没有已知的推理方法")
                    else:
                        print("检测器没有已知的推理方法")
                    return False
                
                # 输出检测结果
                if self.logger:
                    self.logger.info(f"模型预热完成，检测到 {detection_count} 个目标")
                else:
                    print(f"模型预热完成，检测到 {detection_count} 个目标")
                
                return True
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"模型预热失败: {e}")
                else:
                    print(f"模型预热失败: {e}")
                return False
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"模型预热失败: {e}")
            else:
                print(f"模型预热失败: {e}")
            return False
    
    def _parse_yolo_results(self, results) -> int:
        """
        解析PyTorch YOLO模型的检测结果
        
        Args:
            results: YOLO模型的检测结果
            
        Returns:
            检测到的目标数量
        """
        try:
            if results is None:
                return 0
            
            # ultralytics YOLO结果通常是一个列表，每个元素是一个Result对象
            if isinstance(results, list):
                if len(results) > 0 and hasattr(results[0], 'obb'):
                    # ultralytics格式
                    results[0].save()
                    return results[0].obb.shape[0]
                else:
                    return len(results) if results else 0
            
            # 单个结果对象
            elif hasattr(results, 'obb'):
                results.save()
                return results.obb.shape[0] if results.obb is not None else 0
            elif hasattr(results, 'boxes'):
                return len(results.boxes) if results.boxes is not None else 0
            
            # 其他格式，尝试获取长度
            elif hasattr(results, '__len__'):
                return len(results)
            
            else:
                return 0 if results is None else 1
                
        except Exception as e:
            if self.logger:
                self.logger.debug(f"解析YOLO结果失败: {e}")
            return 0
    
    def _parse_rknn_results(self, results) -> int:
        """
        解析RKNN模型的检测结果
        
        Args:
            results: RKNN模型的检测结果
            
        Returns:
            检测到的目标数量
        """
        try:
            if results is None:
                return 0
            
            # RKNN结果通常是一个包含检测框的列表或数组
            if isinstance(results, (list, tuple)):
                return len(results)
            
            # 如果是numpy数组
            elif hasattr(results, 'shape'):
                # 通常RKNN返回的是(N, 6)格式，N是检测数量
                if len(results.shape) == 2:
                    return results.shape[0]
                elif len(results.shape) == 1:
                    return len(results)
                else:
                    return 0
            
            # 如果结果是字典格式
            elif isinstance(results, dict):
                if 'detections' in results:
                    return len(results['detections'])
                elif 'boxes' in results:
                    return len(results['boxes'])
                else:
                    return 0
            
            else:
                return 0 if results is None else 1
                
        except Exception as e:
            if self.logger:
                self.logger.debug(f"解析RKNN结果失败: {e}")
            return 0
    
    def initialize_all_components(self) -> bool:
        """初始化所有系统组件"""
        try:
            # 确保配置已初始化
            if not self.config_mgr and not self.initialize_config():
                return False
            
            # 初始化日志
            if not self.initialize_logging():
                return False
            
            self.logger.info("开始初始化系统组件...")
            
            # 初始化各个组件
            mqtt_success = self.initialize_mqtt_client()
            camera_success = self.initialize_camera()
            detector_success = self.initialize_detector()  # 根据平台自动选择模型
            tcp_server_success = self.initialize_tcp_server()
            
            # 设置系统监控
            self._setup_monitoring()
            
            if mqtt_success and camera_success and detector_success and tcp_server_success:
                self.logger.info("系统组件初始化完成")
                return True
            else:
                self.logger.warning("部分组件初始化失败，但系统将继续运行")
                return True  # 允许部分组件失败
                
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
        
        if self.resources.get('tcp_server'):
            self.monitor.register_component(
                'tcp_server',
                self.resources['tcp_server'],
                lambda: self._restart_tcp_server()
            )
        
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
            if self.initialize_camera(max_retries=3):
                if self.monitor and 'camera' in self.monitor.components:
                    self.monitor.components['camera'] = self.resources['camera']
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
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(old_mqtt.disconnect())
                    finally:
                        loop.close()
                except:
                    pass
            
            # 重新初始化并连接MQTT客户端
            if self.initialize_mqtt_client(max_retries=3):
                mqtt_client = self.resources.get('mqtt_client')
                if mqtt_client and mqtt_client.is_connected:
                    if self.monitor and 'mqtt_client' in self.monitor.components:
                        self.monitor.components['mqtt_client'] = mqtt_client
                    self.logger.info("MQTT客户端重启成功")
                    return True
            
            self.logger.error("MQTT客户端重启失败")
            return False
                
        except Exception as e:
            self.logger.error(f"MQTT客户端重启异常: {e}")
            return False
    
    def _restart_tcp_server(self) -> bool:
        """重启TCP服务器"""
        try:
            self.logger.info("正在重启TCP服务器...")
            
            # 清理旧TCP服务器
            old_tcp_server = self.resources.get('tcp_server')
            if old_tcp_server:
                try:
                    old_tcp_server.stop()
                except:
                    pass
            
            # 重新初始化TCP服务器
            if self.initialize_tcp_server(max_retries=3):
                if self.monitor and 'tcp_server' in self.monitor.components:
                    self.monitor.components['tcp_server'] = self.resources['tcp_server']
                self.logger.info("TCP服务器重启成功")
                return True
            else:
                self.logger.error("TCP服务器重启失败")
                return False
                
        except Exception as e:
            self.logger.error(f"TCP服务器重启异常: {e}")
            return False
    
    def get_resource(self, name: str) -> Optional[Any]:
        """获取指定的资源"""
        return self.resources.get(name)
    
    # === 类型化资源获取方法 - 提供更好的IDE智能提示 ===
    
    def get_mqtt_client(self) -> Optional['MqttClient']:
        """
        获取MQTT客户端
        
        Returns:
            MqttClient实例或None
        """
        return self.resources.get('mqtt_client')
    
    def get_tcp_server(self) -> Optional['TcpServer']:
        """
        获取TCP服务器
        
        Returns:
            TcpServer实例或None
        """
        return self.resources.get('tcp_server')
    
    def get_camera(self) -> Optional[Any]:
        """
        获取相机实例
        
        Returns:
            相机实例或None (类型取决于具体的相机实现)
        """
        return self.resources.get('camera')
    
    def get_detector(self) -> Optional[Any]:
        """
        获取检测器实例
        
        Returns:
            检测器实例或None (类型取决于具体的检测器实现)
        """
        return self.resources.get('detector')
    
    def get_config_manager(self) -> Optional['ConfigManager']:
        """
        获取配置管理器
        
        Returns:
            ConfigManager实例或None
        """
        return self.config_mgr
    
    def get_logger(self) -> Optional[logging.Logger]:
        """
        获取日志记录器
        
        Returns:
            Logger实例或None
        """
        return self.logger
    
    def get_monitor(self) -> Optional['SystemMonitor']:
        """
        获取系统监控器
        
        Returns:
            SystemMonitor实例或None
        """
        return self.monitor
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        if self.monitor:
            return self.monitor.get_system_status()
        return {"monitoring": False, "components": {}}
    
    def restart_system(self):
        """重启整个系统"""
        self.restart_count += 1
        if self.restart_count > self.max_restart_attempts:
            if self.logger:
                self.logger.error("系统重启次数过多，程序退出")
            else:
                print("系统重启次数过多，程序退出")
            sys.exit(1)
        
        if self.logger:
            self.logger.warning(f"正在重启系统 (第 {self.restart_count} 次)...")
        else:
            print(f"正在重启系统 (第 {self.restart_count} 次)...")
        
        # 清理资源
        self.cleanup()
        
        # 等待一段时间
        time.sleep(5)
        
        # 重新初始化
        if self.initialize_config() and self.initialize_all_components():
            if self.logger:
                self.logger.info("系统重启成功")
            self.restart_count = 0  # 重置重启计数
        else:
            if self.logger:
                self.logger.error("系统重启失败")
            else:
                print("系统重启失败")
            sys.exit(1)
    
    def cleanup(self):
        """清理所有资源"""
        # 停止监控
        if self.monitor:
            self.monitor.stop_monitoring()
        
        if self.logger:
            self.logger.info("正在清理系统资源...")
        
        # 清理TCP服务器
        tcp_server = self.resources.get('tcp_server')
        if tcp_server:
            try:
                tcp_server.stop()
                if self.logger:
                    self.logger.info("TCP服务器已停止")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"TCP服务器停止失败: {e}")
        
        # 清理相机
        camera = self.resources.get('camera')
        if camera:
            try:
                camera.disconnect()
                if self.logger:
                    self.logger.info("相机已断开连接")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"相机断开连接失败: {e}")
        
        # 清理检测器（根据类型选择正确的清理方法）
        detector = self.resources.get('detector')
        if detector:
            try:
                # 检查检测器类型并选择合适的清理方法
                detector_type = type(detector).__name__
                
                if hasattr(detector, 'release'):
                    # RKNN_YOLO类型，有release方法
                    detector.release()
                    if self.logger:
                        self.logger.info(f"{detector_type}检测器已释放")
                elif hasattr(detector, 'model') and hasattr(detector.model, 'cpu'):
                    # ultralytics YOLO类型，将模型移到CPU并清理
                    try:
                        detector.model.cpu()  # 将模型移到CPU
                        if hasattr(detector, 'predictor'):
                            detector.predictor = None  # 清理预测器
                        if self.logger:
                            self.logger.info(f"{detector_type}检测器已清理")
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"{detector_type}检测器清理部分失败: {e}")
                else:
                    # 其他类型的检测器，尝试通用清理
                    if self.logger:
                        self.logger.info(f"{detector_type}检测器已移除（无特定清理方法）")
                        
            except Exception as e:
                if self.logger:
                    detector_type = type(detector).__name__ if detector else "Unknown"
                    self.logger.error(f"{detector_type}检测器释放失败: {e}")
        
        # 清理MQTT客户端
        mqtt_client = self.resources.get('mqtt_client')
        if mqtt_client:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(mqtt_client.disconnect())
                    if self.logger:
                        self.logger.info("MQTT客户端已断开")
                finally:
                    loop.close()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"MQTT客户端断开失败: {e}")
        
        # 清理资源字典
        self.resources.clear()
        
        if self.logger:
            self.logger.info("=== VisionCore 系统关闭 ===")
        else:
            print("系统资源清理完成") 