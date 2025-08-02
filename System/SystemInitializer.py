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
from SickVision.SickSDK import QtVisionSick
from TcpServer.TcpServer import TcpServer
from ClassModel.MqttResponse import MQTTResponse
from SystemEnums.VisionCoreCommands import VisionCoreCommands, MessageType
from Rknn.RknnYolo import RKNN_YOLO
from SFTP.QtSFTP import QtSFTP
from .SystemMonitor import SystemMonitor
from utils.decorators import handle_keyboard_interrupt, interruptible_retry


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
        
        # 板端程序配置 - 默认值，稍后从配置文件读取
        self.board_mode = True  # 默认启用板端模式
        self.infinite_retry = True  # 默认启用无限重试
        self.retry_delay = 5  # 固定重试延迟（秒）
        self.monitoring_check_interval = 30  # 监控检查间隔
        self.monitoring_failure_threshold = 2  # 失败阈值
        
        # 重启相关配置
        self.max_restart_attempts = float('inf')  # 默认无限重启
        self.restart_count = 0
        
        # 存储MQTT消息回调函数，用于重启时恢复
        self._mqtt_message_callback = None
    
    def initialize_config(self) -> bool:
        """初始化配置管理器"""
        try:
            print("正在加载配置文件...")
            self.config_mgr = ConfigManager(self.config_path)
            print("配置文件加载成功")
            
            # 读取板端模式配置
            self._load_board_mode_config()
            
            return True
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            return False
    
    def _load_board_mode_config(self):
        """从配置文件加载板端模式配置"""
        if not self.config_mgr:
            return
            
        try:
            board_config = self.config_mgr.get_config("board_mode")
            if board_config:
                self.board_mode = board_config.get("enabled", True)
                self.infinite_retry = board_config.get("infinite_retry", True)
                self.retry_delay = board_config.get("retry_delay", 5)
                
                # 读取监控配置
                monitoring_config = board_config.get("monitoring", {})
                self.monitoring_check_interval = monitoring_config.get("check_interval", 30)
                self.monitoring_failure_threshold = monitoring_config.get("failure_threshold", 2)
            
            # 更新重启配置
            self.max_restart_attempts = float('inf') if (self.board_mode and self.infinite_retry) else 3
            
            print(f"板端模式配置: enabled={self.board_mode}, infinite_retry={self.infinite_retry}, retry_delay={self.retry_delay}s")
            
        except Exception as e:
            print(f"加载板端模式配置失败: {e}，使用默认配置")
    
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
    
    @interruptible_retry(max_retries=None, log_prefix="初始化MQTT客户端")
    def initialize_mqtt_client(self, max_retries: int = None) -> bool:
        """
        初始化MQTT客户端并连接到broker（支持无限重试）
        
        Args:
            max_retries: 最大重试次数，None表示使用系统配置
            
        Returns:
            bool: 初始化是否成功
        """
        if not self.config_mgr or not self.logger:
            return False
            
        mqtt_config = self.config_mgr.get_config("mqtt")
        
        if not mqtt_config or not mqtt_config.get("enabled", False):
            self.logger.info("MQTT功能已禁用")
            return True  # 禁用时也认为是成功的
        
        try:
            # 检查是否有旧的MQTT客户端需要清理
            old_mqtt = self.resources.get('mqtt_client')
            if old_mqtt:
                self.logger.info("发现旧的MQTT客户端，正在清理...")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(old_mqtt.force_reinit())
                finally:
                    loop.close()
                # 移除旧客户端
                self.resources.pop('mqtt_client', None)
            
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
                    # 连接失败，尝试强制重新初始化
                    self.logger.warning("MQTT连接失败，尝试强制重新初始化...")
                    loop.run_until_complete(mqtt_client.force_reinit())
                    connected = loop.run_until_complete(mqtt_client.connect())
                    if connected:
                        self.logger.info("MQTT客户端重新初始化后连接成功")
                        self.resources['mqtt_client'] = mqtt_client
                        return True
                    else:
                        return False
                    
            
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"MQTT客户端初始化异常: {e}")
            return False
        return False

    def initialize_camera(self, max_retries: int = None) -> bool:
        """
        初始化相机（不使用重试，失败时通过MQTT通知）
        
        Args:
            max_retries: 最大重试次数（忽略，相机不重试）
            
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
        
        # 获取单步模式配置
        mode_config = camera_config.get("mode", {})
        use_single_step = mode_config.get("useSingleStep", True)
        
        try:
            # 初始化相机实例
            from SickVision.SickSDK import QtVisionSick
            camera = QtVisionSick(
                ipAddr=ip_address,
                port=port,
                protocol="Cola2"
            )
            
            # 尝试连接相机（快速超时，避免阻塞）
            self.logger.info(f"正在连接相机: {ip_address}:{port}...")
            
            # 使用简单的daemon线程进行快速超时
            import threading
            import time
            
            result = {"success": False, "error": None}
            
            def connect_thread():
                try:
                    result["success"] = camera.connect(use_single_step=use_single_step)
                except Exception as e:
                    result["error"] = str(e)
            
            # 创建daemon线程，超时后自动终止
            thread = threading.Thread(target=connect_thread, daemon=True)
            start_time = time.time()
            thread.start()
            
            # 轮询检查，每0.1秒检查一次，最多3秒
            timeout_seconds = 3.0
            while thread.is_alive() and (time.time() - start_time) < timeout_seconds:
                time.sleep(0.1)
            
            if thread.is_alive():
                # 超时了
                self.logger.error(f"相机连接超时: {ip_address}:{port} (3秒)")
                success = False
            elif result["error"]:
                # 有异常
                self.logger.error(f"相机连接异常: {result['error']}")
                success = False
            else:
                # 正常完成
                success = result["success"]
            
            if success:
                self.logger.info(f"相机连接成功: {ip_address}:{port}")
                
                # 进行相机预热
                self.logger.info("正在预热相机（获取第一帧图像）...")
                warmup_success = self._warmup_camera(camera)
                
                if warmup_success:
                    self.logger.info("相机预热成功")
                    self.resources['camera'] = camera
                    return True
                else:
                    self.logger.warning("相机预热失败，但相机连接正常，将继续运行")
                    self.resources['camera'] = camera
                    return True  # 即使预热失败也继续运行
            else:
                error_msg = f"相机连接失败: {ip_address}:{port}"
                self.logger.error(error_msg)
                self._notify_component_failure("camera", error_msg)
                return False
        except Exception as e:
            error_msg = f"相机初始化异常: {str(e)}"
            self.logger.error(error_msg)
            self._notify_component_failure("camera", error_msg)
            return False

    def _warmup_camera(self, camera) -> bool:
        """
        预热相机 - 获取第一帧图像以减少后续检测的延迟
        
        Args:
            camera: 相机实例
            
        Returns:
            bool: 预热是否成功
        """
        try:
            # 设置较短的超时时间进行预热
            import threading
            import time
            
            warmup_result = {"success": False, "error": None, "timing": 0}
            
            def warmup_thread():
                try:
                    start_time = time.time()
                    
                    # 尝试获取一帧图像
                    success, depth_data, frame, camera_params = camera.get_frame()
                    
                    end_time = time.time()
                    warmup_result["timing"] = (end_time - start_time) * 1000  # 转为毫秒
                    
                    if success and frame is not None:
                        # 检查图像有效性
                        if hasattr(frame, 'shape') and len(frame.shape) >= 2:
                            height, width = frame.shape[:2]
                            if height > 0 and width > 0:
                                warmup_result["success"] = True
                                self.logger.info(f"相机预热成功: 获取到图像 {width}x{height}, 耗时: {warmup_result['timing']:.1f}ms")
                            else:
                                warmup_result["error"] = f"获取到无效图像尺寸: {frame.shape}"
                        else:
                            warmup_result["error"] = f"获取到的不是有效图像数据: {type(frame)}"
                    else:
                        warmup_result["error"] = "无法获取相机图像数据"
                        
                except Exception as e:
                    warmup_result["error"] = str(e)
            
            # 创建预热线程
            thread = threading.Thread(target=warmup_thread, daemon=True)
            start_time = time.time()
            thread.start()
            
            # 等待预热完成，最多5秒
            timeout_seconds = 5.0
            while thread.is_alive() and (time.time() - start_time) < timeout_seconds:
                time.sleep(0.1)
            
            if thread.is_alive():
                self.logger.warning("相机预热超时(5秒)，但将继续运行")
                return False
            elif warmup_result["error"]:
                self.logger.warning(f"相机预热失败: {warmup_result['error']}")
                return False
            else:
                return warmup_result["success"]
                
        except Exception as e:
            self.logger.warning(f"相机预热异常: {e}")
            return False
    
    def _wait_for_model_file(self, model_path: str) -> bool:
        """等待模型文件出现"""
        while not os.path.exists(model_path):
            self.logger.warning(f"等待模型文件: {model_path}, 30秒后再次检查...")
            # 使用可中断的睡眠，每秒检查一次中断
            for _ in range(30):
                time.sleep(1)
        self.logger.info(f"模型文件已找到: {model_path}")
        return True
    
    def initialize_detector(self, max_retries: int = None) -> bool:
        """
        初始化检测器模型（不使用重试，失败时通过MQTT通知）
        
        Args:
            max_retries: 最大重试次数（忽略，检测器不重试）
            
        Returns:
            bool: 初始化是否成功
        """
        if not self.logger:
            return False
        
        try:
            # 根据平台自动选择模型文件
            model_path = self._get_model_path()
            
            if not model_path or not os.path.exists(model_path):
                error_msg = f"模型文件不存在: {model_path}"
                self.logger.error(error_msg)
                self._notify_component_failure("detector", error_msg)
                return False
            
            # 检查平台与模型兼容性
            platform_compatible, platform_error = self._check_platform_model_compatibility(model_path)
            if not platform_compatible:
                self.logger.error(platform_error)
                self._notify_component_failure("detector", platform_error)
                return False
            
            # 使用快速超时避免长时间阻塞
            import concurrent.futures
            
            def create_and_warmup_detector():
                try:
                    # 根据模型类型选择检测器
                    detector = self._create_detector(model_path)
                    if not detector:
                        return {"result": None, "error": "检测器创建失败", "error_type": "creation_failed"}
                    
                    # 预热模型
                    self.logger.info("正在预热模型...")
                    warmup_success, warmup_error = self._warmup_model_with_error(detector)
                    if warmup_success:
                        return {"result": detector, "error": None, "error_type": None}
                    else:
                        # 预热失败，释放检测器资源
                        self.logger.error("模型预热失败，检测器初始化失败")
                        try:
                            detector.release()
                        except:
                            pass
                        return {"result": None, "error": f"模型预热失败: {warmup_error}", "error_type": "warmup_failed"}
                except Exception as e:
                    return {"result": None, "error": f"检测器创建异常: {str(e)}", "error_type": "creation_exception"}
            
            # 使用10秒超时（模型加载可能需要更长时间）
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(create_and_warmup_detector)
                try:
                    result_data = future.result(timeout=10.0)  # 10秒超时
                    detector = result_data["result"]
                    error_msg = result_data["error"]
                    error_type = result_data["error_type"]
                except concurrent.futures.TimeoutError:
                    detector = None
                    error_msg = "检测器初始化超时"
                    error_type = "timeout"
                except Exception as e:
                    detector = None
                    error_msg = f"检测器初始化执行异常: {e}"
                    error_type = "execution_exception"
            
            if detector:
                self.logger.info("检测器初始化成功")
                self.resources['detector'] = detector
                return True
            else:
                # 根据错误类型生成详细错误信息
                if error_type in ["timeout", "execution_exception"]:
                    final_error_msg = error_msg
                else:
                    final_error_msg = error_msg if error_msg else "检测器初始化失败"
                
                self.logger.error(final_error_msg)
                self._notify_component_failure("detector", final_error_msg)
                return False
        except Exception as e:
            error_msg = f"检测器初始化异常: {str(e)}"
            self.logger.error(error_msg)
            self._notify_component_failure("detector", error_msg)
            return False
    
    def initialize_sftp_client(self, max_retries: int = None) -> bool:
        """
        初始化SFTP客户端（使用QtSFTP类）
        
        Args:
            max_retries: 最大重试次数（忽略，SFTP客户端不重试）
            
        Returns:
            bool: 初始化是否成功
        """
        if not self.config_mgr or not self.logger:
            return False
            
        sftp_config = self.config_mgr.get_config("sftp")
        
        if not sftp_config or not sftp_config.get("enabled", False):
            self.logger.info("SFTP功能已禁用")
            return True  # 禁用时也认为是成功的
        
        try:
            # 导入QtSFTP类
            from SFTP.QtSFTP import QtSFTP
            
            # 设置错误回调函数
            def sftp_error_callback(component_name: str, error_message: str):
                self._notify_component_failure(component_name, error_message)
            
            # 创建QtSFTP实例
            sftp_client = QtSFTP(sftp_config, self.logger)
            sftp_client.set_error_callback(sftp_error_callback)
            
            # 连接到SFTP服务器
            self.logger.info(f"正在连接SFTP服务器: {sftp_config.get('host', 'localhost')}:{sftp_config.get('port', 22)}")
            
            if sftp_client.connect():
                self.logger.info("SFTP客户端连接成功")
                self.resources['sftp_client'] = sftp_client
                return True
            else:
                error_msg = "SFTP客户端连接失败"
                self.logger.error(error_msg)
                self._notify_component_failure("sftp", error_msg)
                return False
                    
        except Exception as e:
            error_msg = f"SFTP客户端初始化异常: {str(e)}"
            self.logger.error(error_msg)
            self._notify_component_failure("sftp", error_msg)
            return False
    
    def initialize_tcp_server(self, max_retries: int = None) -> bool:
        """
        初始化TCP服务器（不使用重试，失败时通过MQTT通知）
        
        Args:
            max_retries: 最大重试次数（忽略，TCP服务器不重试）
            
        Returns:
            bool: 初始化是否成功
        """
        if not self.config_mgr or not self.logger:
            return False
            
        tcp_config = self.config_mgr.get_config("detectionServer")
        
        if not tcp_config or not tcp_config.get("enabled", False):
            self.logger.info("TCP服务器功能已禁用")
            return True  # 禁用时也认为是成功的
        
        try:
            # 创建TCP服务器实例（不设置回调，留给main.py处理）
            tcp_server = TcpServer(tcp_config, self.logger)
            
            # 启动TCP服务器（快速超时，避免阻塞）
            self.logger.info("正在启动TCP服务器...")
            
            # 使用线程池执行器进行非阻塞启动，设置短超时
            import concurrent.futures
            
            def start_with_timeout():
                try:
                    return tcp_server.start()
                except Exception as e:
                    self.logger.debug(f"TCP服务器启动异常: {e}")
                    return False
            
            # 使用2秒超时
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(start_with_timeout)
                try:
                    success = future.result(timeout=2.0)  # 2秒超时
                except concurrent.futures.TimeoutError:
                    self.logger.error("TCP服务器启动超时")
                    success = False
                except Exception as e:
                    self.logger.error(f"TCP服务器启动执行异常: {e}")
                    success = False
            
            if success:
                self.logger.info("TCP服务器启动成功")
                self.resources['detectionServer'] = tcp_server
                return True
            else:
                error_msg = "TCP服务器启动失败"
                self.logger.error(error_msg)
                self._notify_component_failure("tcp_server", error_msg)
                return False
        except Exception as e:
            error_msg = f"TCP服务器初始化异常: {str(e)}"
            self.logger.error(error_msg)
            self._notify_component_failure("tcp_server", error_msg)
            return False
    
    def create_tcp_message_handler(self, catch_handler: Optional[Callable] = None):
        """
        创建TCP消息处理器函数，供main.py使用
        
        Args:
            catch_handler: 外部提供的catch指令处理函数
        
        Returns:
            Callable: 消息处理函数
        """
        def handle_tcp_message(client_id: str, message: str) -> Optional[str]:
            """
            处理TCP消息的统一入口
            
            Args:
                client_id: 客户端ID
                message: 接收到的消息字符串
                
            Returns:
                Optional[str]: 响应消息字符串，如果为None则不回复
            """
            try:
                # 检查是否是catch命令
                if message.lower().strip() == "catch":
                    if catch_handler:
                        return catch_handler(client_id, message)
                    else:
                        # 通过MQTT发送错误通知，但不向TCP客户端返回任何数据
                        self._notify_component_failure("tcp_handler", f"catch处理器未设置 (客户端: {client_id})")
                        return None  # 不返回任何数据
                else:
                    self.logger.debug(f"未处理的消息: {message} from {client_id}")
                    # 通过MQTT发送错误通知，但不向TCP客户端返回任何数据
                    self._notify_component_failure("tcp_handler", f"不支持的消息: {message} (客户端: {client_id})")
                    return None  # 不返回任何数据，因为只有catch指令才应该有响应
                    
            except Exception as e:
                self.logger.error(f"处理TCP消息时出错: {e}")
                # 通过MQTT发送错误通知，但不向TCP客户端返回任何数据
                self._notify_component_failure("tcp_handler", f"消息处理失败: {str(e)} (客户端: {client_id})")
                return None  # 不返回任何数据
        
        return handle_tcp_message
    
    def _check_platform_model_compatibility(self, model_path: str) -> Tuple[bool, str]:
        """
        检查平台与模型的兼容性
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            Tuple[bool, str]: (是否兼容, 错误信息)
        """
        try:
            model_name = os.path.basename(model_path)
            current_platform = plt.system().lower()
            current_machine = plt.machine().lower()
            
            is_rknn_model = model_name.endswith('.rknn')
            is_pytorch_model = model_name.endswith('.pt')
            
            # ARM平台兼容性检查
            if current_machine in ['aarch64']:
                if is_pytorch_model:
                    error_msg = f"平台不兼容: ARM平台({current_machine})不支持PyTorch模型({model_name})，请使用RKNN模型"
                    return False, error_msg
                elif is_rknn_model:
                    return True, ""
                else:
                    error_msg = f"平台不兼容: ARM平台({current_machine})需要RKNN模型，当前模型({model_name})格式不支持"
                    return False, error_msg
            
            # Windows/x86平台兼容性检查
            elif current_platform in ["windows"] or current_machine in ['x86_64', 'amd64']:
                if is_rknn_model:
                    error_msg = f"平台不兼容: Windows/x86平台不支持RKNN模型({model_name})，请使用PyTorch模型(.pt)"
                    return False, error_msg
                elif is_pytorch_model:
                    return True, ""
                else:
                    error_msg = f"平台不兼容: Windows/x86平台需要PyTorch模型(.pt)，当前模型({model_name})格式不支持"
                    return False, error_msg
            
            # 其他平台默认使用PyTorch
            else:
                if is_rknn_model:
                    error_msg = f"平台不兼容: {current_platform}平台不支持RKNN模型({model_name})，请使用PyTorch模型(.pt)"
                    return False, error_msg
                elif is_pytorch_model:
                    return True, ""
                else:
                    error_msg = f"平台不兼容: {current_platform}平台需要PyTorch模型(.pt)，当前模型({model_name})格式不支持"
                    return False, error_msg
                    
        except Exception as e:
            error_msg = f"平台兼容性检查异常: {str(e)}"
            return False, error_msg
    
    def _get_model_path(self) -> Optional[str]:
        """
        根据配置中的selectedModel选择模型路径，如果未配置则根据平台自动选择
        
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
        
        # 首先检查配置中是否指定了selectedModel
        if self.config_mgr:
            model_config = self.config_mgr.get_config("model")
            if model_config and "selectedModel" in model_config:
                selected_model = model_config["selectedModel"]
                if selected_model in files:
                    if self.logger:
                        self.logger.info(f"使用配置中指定的模型: {selected_model}")
                    else:
                        print(f"使用配置中指定的模型: {selected_model}")
                    return os.path.join(models_dir, selected_model)
                else:
                    if self.logger:
                        self.logger.warning(f"配置中指定的模型文件不存在: {selected_model}，回退到自动选择")
                    else:
                        print(f"配置中指定的模型文件不存在: {selected_model}，回退到自动选择")
        
        # 如果没有配置或配置的模型不存在，则根据平台自动选择
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
            
            # 使用RKNN检测器
            from Rknn.RknnYolo import RKNN_YOLO
            detector = RKNN_YOLO(
                model_path=model_path,
                conf_threshold=0.5,
                nms_threshold=0.45
            )
            
            # 检查变换矩阵加载状态并记录日志
            matrix_status = detector.get_transformation_matrix_status()
            if matrix_status['loaded']:
                metadata = matrix_status['metadata']
                self.logger.info(f"变换矩阵加载成功，标定点数: {metadata.get('calibration_points_count', 0)}, "
                               f"RMSE: {metadata.get('calibration_rmse', 0.0):.3f}mm")
            else:
                self.logger.warning(f"变换矩阵加载失败: {matrix_status.get('error', '未知错误')}")
            
            self.logger.info("创建RKNN检测器成功")
            return detector
                
        except Exception as e:
            self.logger.error(f"创建检测器失败: {e}")
            return None
    
    def _warmup_model_with_error(self, detector) -> Tuple[bool, Optional[str]]:
        """
        预热模型并返回详细错误信息（使用test.jpg测试图片）
        
        Args:
            detector: 检测器实例
            
        Returns:
            Tuple[bool, Optional[str]]: (预热是否成功, 错误信息)
        """
        try:
            # 查找测试图片
            test_image_path = "./Models/test.jpg"
            
            if not os.path.exists(test_image_path):
                return False, f"测试图片不存在: {test_image_path}"
            
            # 加载测试图片
            try:
                import cv2
                test_input = cv2.imread(test_image_path)
                if test_input is None:
                    return False, f"无法读取测试图片: {test_image_path}"
                
                if self.logger:
                    self.logger.info(f"使用测试图片进行预热: {test_image_path}")
                    self.logger.info(f"图片尺寸: {test_input.shape}")
                else:
                    print(f"使用测试图片进行预热: {test_image_path}")
                    print(f"图片尺寸: {test_input.shape}")
                
            except ImportError:
                return False, "OpenCV未安装，无法读取测试图片"
            except Exception as e:
                return False, f"读取测试图片失败: {str(e)}"
            
            if self.logger:
                self.logger.info("开始模型预热（多次推理以充分预热）...")
            else:
                print("开始模型预热（多次推理以充分预热）...")
            
            # 进行多次预热推理以确保模型完全预热
            warmup_rounds = 2  # 进行5次预热
            try:
                for i in range(warmup_rounds):
                    _ = detector.detect(test_input)
                    
                if self.logger:
                    self.logger.info(f"模型预热完成，共执行{warmup_rounds}次推理")
                else:
                    print(f"模型预热完成，共执行{warmup_rounds}次推理")
                
                return True, None
                    
            except Exception as e:
                error_detail = str(e)
                if self.logger:
                    self.logger.warning(f"模型预热失败: {error_detail}")
                else:
                    print(f"模型预热失败: {error_detail}")
                return False, error_detail
            
        except Exception as e:
            error_detail = str(e)
            if self.logger:
                self.logger.error(f"模型预热异常: {error_detail}")
            else:
                print(f"模型预热异常: {error_detail}")
            return False, error_detail

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
            sftp_success = self.initialize_sftp_client()
            
            # 设置系统监控
            self._setup_monitoring()
            
            if mqtt_success and camera_success and detector_success and tcp_server_success and sftp_success:
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
        
        # 传递板端模式参数
        self.monitor = SystemMonitor(self.logger, check_interval=self.monitoring_check_interval, 
                                   failure_threshold=self.monitoring_failure_threshold, board_mode=self.board_mode)
        
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
            self.monitor.register_component(
                'detector', 
                self.resources['detector'],
                lambda: self._restart_detector()
            )
        
        if self.resources.get('detectionServer'):
            self.monitor.register_component(
                'tcp_server',
                self.resources['detectionServer'],
                lambda: self._restart_tcp_server()
            )
        
        if self.resources.get('sftp_client'):
            self.monitor.register_component(
                'sftp_client',
                self.resources['sftp_client'],
                lambda: self._restart_sftp_client()
            )
        
        # 启动监控
        self.monitor.start_monitoring()
    
    def _restart_camera(self) -> bool:
        """重启相机（单次尝试，失败时通过MQTT通知）"""
        try:
            self.logger.info("正在重启相机...")
            
            # 清理旧相机
            old_camera = self.resources.get('camera')
            if old_camera:
                try:
                    old_camera.disconnect()
                except:
                    pass
            
            # 重新初始化相机（单次尝试）
            if self.initialize_camera():
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
        """重启MQTT客户端（板端模式支持无限重试）"""
        try:
            self.logger.info("正在重启MQTT客户端...")
            
            # 清理旧MQTT客户端
            old_mqtt = self.resources.get('mqtt_client')
            if old_mqtt:
                try:
                    self.logger.info("正在断开旧的MQTT连接...")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(old_mqtt.disconnect())
                        # 额外等待，确保完全断开
                        loop.run_until_complete(asyncio.sleep(1.0))
                    finally:
                        loop.close()
                except Exception as e:
                    self.logger.warning(f"断开旧MQTT连接时出现异常: {e}")
                
                # 从资源中移除
                self.resources.pop('mqtt_client', None)
            
            # 等待一段时间，确保网络资源释放（可中断）
            import time
            for _ in range(2):
                time.sleep(1)
            
            # 重新初始化并连接MQTT客户端（板端模式使用无限重试）
            max_retries = None if self.board_mode else 3
            if self.initialize_mqtt_client(max_retries=max_retries):
                mqtt_client = self.resources.get('mqtt_client')
                if mqtt_client and mqtt_client.is_connected:
                    # 重新设置应用程序级别的消息回调
                    if self._mqtt_message_callback:
                        mqtt_client.set_general_callback(self._mqtt_message_callback)
                        self.logger.info("MQTT消息回调已重新设置")
                    
                    if self.monitor and 'mqtt_client' in self.monitor.components:
                        self.monitor.components['mqtt_client'] = mqtt_client
                    self.logger.info("MQTT客户端重启成功")
                    return True
            
            self.logger.error("MQTT客户端重启失败")
            return False
                
        except Exception as e:
            self.logger.error(f"MQTT客户端重启异常: {e}")
            return False
    
    def _restart_detector(self) -> bool:
        """重启检测器（单次尝试，失败时通过MQTT通知）"""
        try:
            self.logger.info("正在重启检测器...")
            
            # 清理旧检测器
            old_detector = self.resources.get('detector')
            if old_detector:
                try:
                    # 检查检测器类型并选择合适的清理方法
                    detector_type = type(old_detector).__name__
                    
                    # RKNN_YOLO类型，有release方法
                    old_detector.release()
                    self.logger.info(f"{detector_type}检测器已释放")
                        
                except Exception as e:
                    self.logger.warning(f"清理旧检测器时出现异常: {e}")
                
                # 从资源中移除
                self.resources.pop('detector', None)
            
            # 重新初始化检测器（单次尝试）
            if self.initialize_detector():
                if self.monitor and 'detector' in self.monitor.components:
                    self.monitor.components['detector'] = self.resources['detector']
                self.logger.info("检测器重启成功")
                
                return True
            else:
                # initialize_detector失败时已经发送了MQTT通知，这里只记录日志
                self.logger.error("检测器重启失败")
                return False
                
        except Exception as e:
            error_msg = f"检测器重启异常: {str(e)}"
            self.logger.error(error_msg)
            self._notify_component_failure("detector", error_msg)
            return False
    
    def _restart_tcp_server(self) -> bool:
        """重启TCP服务器（单次尝试，失败时通过MQTT通知）"""
        try:
            self.logger.info("正在重启TCP服务器...")
            
            # 清理旧TCP服务器
            old_tcp_server = self.resources.get('detectionServer')
            if old_tcp_server:
                try:
                    old_tcp_server.stop()
                except:
                    pass
            
            # 重新初始化TCP服务器（单次尝试）
            if self.initialize_tcp_server():
                if self.monitor and 'tcp_server' in self.monitor.components:
                    self.monitor.components['tcp_server'] = self.resources['detectionServer']
                self.logger.info("TCP服务器重启成功")
                return True
            else:
                self.logger.error("TCP服务器重启失败")
                return False
                
        except Exception as e:
            self.logger.error(f"TCP服务器重启异常: {e}")
            return False
    
    def _restart_sftp_client(self) -> bool:
        """重启SFTP客户端（单次尝试，失败时通过MQTT通知）"""
        try:
            self.logger.info("正在重启SFTP客户端...")
            
            # 清理旧SFTP客户端
            old_sftp_client = self.resources.get('sftp_client')
            if old_sftp_client:
                try:
                    # 使用QtSFTP类的disconnect方法
                    old_sftp_client.disconnect()
                    self.logger.info("旧SFTP连接已关闭")
                except Exception as e:
                    self.logger.warning(f"关闭旧SFTP连接时出现异常: {e}")
                
                # 从资源中移除
                self.resources.pop('sftp_client', None)
            
            # 重新初始化SFTP客户端（单次尝试）
            if self.initialize_sftp_client():
                if self.monitor and 'sftp_client' in self.monitor.components:
                    self.monitor.components['sftp_client'] = self.resources['sftp_client']
                self.logger.info("SFTP客户端重启成功")
                return True
            else:
                self.logger.error("SFTP客户端重启失败")
                return False
                
        except Exception as e:
            self.logger.error(f"SFTP客户端重启异常: {e}")
            return False
    
    def upload_file_to_sftp(self, local_file_path: str, remote_filename: str = None) -> bool:
        """
        通过SFTP上传文件到服务器（使用QtSFTP类）
        
        Args:
            local_file_path: 本地文件路径
            remote_filename: 远程文件名（可选，默认使用本地文件名）
            
        Returns:
            bool: 上传是否成功
        """
        try:
            sftp_client = self.get_sftp_client()
            if not sftp_client or not sftp_client.connected:
                self.logger.error("SFTP客户端未连接或不可用")
                return False
            
            # 使用QtSFTP类的upload_file方法
            result = sftp_client.upload_file(local_file_path, remote_filename)
            
            if result["success"]:
                self.logger.info(result["message"])
                return True
            else:
                self.logger.error(result["message"])
                return False
                
        except Exception as e:
            self.logger.error(f"SFTP文件上传失败: {e}")
            return False
    
    def upload_image_with_timestamp(self, image_data, image_format: str = "jpg", prefix: str = "detection") -> bool:
        """
        上传带时间戳的图像到SFTP服务器（使用QtSFTP类）
        
        Args:
            image_data: 图像数据 (numpy array 或 PIL Image)
            image_format: 图像格式 (jpg, png)
            prefix: 文件名前缀
            
        Returns:
            bool: 上传是否成功
        """
        try:
            sftp_client = self.get_sftp_client()
            if not sftp_client or not sftp_client.connected:
                self.logger.error("SFTP客户端未连接或不可用")
                return False
            
            # 使用QtSFTP类的upload_image方法
            result = sftp_client.upload_image(image_data, image_format, prefix)
            
            if result["success"]:
                self.logger.info(result["message"])
                return True
            else:
                self.logger.error(result["message"])
                return False
                
        except Exception as e:
            self.logger.error(f"上传图像失败: {e}")
            return False

    def get_resource(self, name: str) -> Optional[Any]:
        """获取指定的资源"""
        return self.resources.get(name)
    
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
        return self.resources.get('detectionServer')
    
    def get_camera(self) -> Optional['QtVisionSick']:
        """
        获取相机实例
        
        Returns:
            相机实例或None (类型取决于具体的相机实现)
        """
        return self.resources.get('camera')
    
    def get_detector(self) -> Optional['RKNN_YOLO']:
        """
        获取检测器实例
        
        Returns:
            检测器实例或None (类型取决于具体的检测器实现)
        """
        return self.resources.get('detector')
    
    def get_sftp_client(self) -> Optional['QtSFTP']:
        """
        获取SFTP客户端
        
        Returns:
            QtSFTP实例或None
        """
        return self.resources.get('sftp_client')
    
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
    
    def _notify_component_failure(self, component_name: str, error_message: str):
        """
        通过MQTT通知客户端组件初始化失败
        
        Args:
            component_name: 组件名称
            error_message: 错误消息
        """
        try:
            mqtt_client = self.get_mqtt_client()
            if mqtt_client and mqtt_client.is_connected:
                # 创建MQTTResponse对象
                response = MQTTResponse(
                    command=VisionCoreCommands.ERROR_TIP.value,
                    component=component_name,
                    messageType=MessageType.ERROR,
                    message=f"组件 {component_name} 初始化失败: {error_message}",
                    data={
                        "component": component_name,
                        "error": error_message,
                        "timestamp": time.time()
                    }
                )
                
                # 直接使用mqtt_client发送响应
                success = mqtt_client.send_mqtt_response(response)
                if success:
                    if self.logger:
                        self.logger.info(f"已通过MQTT通知组件失败: {component_name}")
                else:
                    if self.logger:
                        self.logger.warning(f"MQTT通知发送失败: {component_name}")
            else:
                if self.logger:
                    self.logger.warning(f"MQTT未连接，无法通知组件失败: {component_name}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"发送组件失败通知时出错: {e}")
    
    def set_mqtt_message_callback(self, callback: Callable):
        """
        设置MQTT消息回调函数并存储以供重启时恢复
        
        Args:
            callback: MQTT消息回调函数
        """
        self._mqtt_message_callback = callback
        mqtt_client = self.get_mqtt_client()
        if mqtt_client:
            mqtt_client.set_general_callback(callback)
            if self.logger:
                self.logger.info("MQTT消息回调已设置")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        if self.monitor:
            return self.monitor.get_system_status()
        return {"monitoring": False, "components": {}}
    
    @handle_keyboard_interrupt(exit_on_interrupt=True, log_message="停止系统重启")
    def restart_system(self):
        """重启整个系统（板端模式支持无限重启）"""
        self.restart_count += 1
        
        if self.max_restart_attempts != float('inf') and self.restart_count > self.max_restart_attempts:
            if self.logger:
                self.logger.error("系统重启次数过多，程序退出")
            else:
                print("系统重启次数过多，程序退出")
            sys.exit(1)
        
        if self.board_mode and self.max_restart_attempts == float('inf'):
            if self.logger:
                self.logger.warning(f"正在重启系统 (第 {self.restart_count} 次) - 板端模式，无限重试...")
            else:
                print(f"正在重启系统 (第 {self.restart_count} 次) - 板端模式，无限重试...")
        else:
            if self.logger:
                self.logger.warning(f"正在重启系统 (第 {self.restart_count} 次)...")
            else:
                print(f"正在重启系统 (第 {self.restart_count} 次)...")
        
        # 清理资源
        self.cleanup()
        
        # 计算重启延迟
        if self.board_mode:
            # 板端模式：使用固定重启延迟（可中断）
            restart_delay = self.retry_delay
            if self.logger:
                self.logger.info(f"等待 {restart_delay} 秒后重启系统...")
            # 使用可中断的睡眠
            for _ in range(int(restart_delay)):
                time.sleep(1)
        else:
            # 原有逻辑（可中断）
            for _ in range(5):
                time.sleep(1)
        
        # 重新初始化
        if self.initialize_config() and self.initialize_all_components():
            if self.logger:
                self.logger.info("系统重启成功")
            if not self.board_mode:
                self.restart_count = 0  # 非板端模式重置重启计数
        else:
            if self.logger:
                self.logger.error("系统重启失败")
            else:
                print("系统重启失败")
            
            if not self.board_mode:
                sys.exit(1)
            else:
                # 板端模式继续重试
                if self.logger:
                    self.logger.warning("板端模式：系统重启失败，将继续尝试...")
                # 递归调用自己继续重启
                self.restart_system()
    
    def cleanup(self):
        """清理所有资源"""
        # 停止监控
        if self.monitor:
            self.monitor.stop_monitoring()
        
        if self.logger:
            self.logger.info("正在清理系统资源...")
        
        # 清理TCP服务器
        tcp_server = self.resources.get('detectionServer')
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
                
                # RKNN_YOLO类型，有release方法
                detector.release()
                if self.logger:
                    self.logger.info(f"{detector_type}检测器已释放")
                        
            except Exception as e:
                if self.logger:
                    detector_type = type(detector).__name__ if detector else "Unknown"
                    self.logger.error(f"{detector_type}检测器释放失败: {e}")
        
        # 清理SFTP客户端
        sftp_client = self.resources.get('sftp_client')
        if sftp_client:
            try:
                # 使用QtSFTP类的disconnect方法
                sftp_client.disconnect()
                if self.logger:
                    self.logger.info("SFTP客户端已关闭")
                        
            except Exception as e:
                if self.logger:
                    self.logger.error(f"SFTP客户端关闭失败: {e}")
        
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