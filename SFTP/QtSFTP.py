#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QtSFTP 模块
提供完整的 SFTP 连接和文件上传功能
"""

import os
import time
import threading
import tempfile
import logging
from typing import Optional, Dict, Any, Tuple, Union
from datetime import datetime
import numpy as np

try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class QtSFTP:
    """SFTP 客户端类，提供连接管理和文件上传功能"""
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        """
        初始化 SFTP 客户端
        
        Args:
            config: SFTP 配置字典，包含 host, port, username, password 等
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # SFTP 连接相关
        self.ssh_client: Optional[paramiko.SSHClient] = None
        self.sftp_client: Optional[paramiko.SFTPClient] = None
        
        # 连接状态
        self.connected = False
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 22)
        self.username = config.get("username", "anonymous")
        self.password = config.get("password", "")
        self.private_key_path = config.get("private_key_path", None)
        self.remote_path = config.get("remote_path", "/uploads")
        
        # 连接超时设置
        self.connection_timeout = config.get("connection_timeout", 15.0)
        self.ssh_timeout = config.get("ssh_timeout", 10)
        
        # 错误回调
        self.error_callback = None
    
    def set_error_callback(self, callback):
        """设置错误回调函数"""
        self.error_callback = callback
    
    def _log(self, level: str, message: str):
        """统一的日志记录"""
        if self.logger:
            if level == "debug":
                self.logger.debug(message)
            elif level == "info":
                self.logger.info(message)
            elif level == "warning":
                self.logger.warning(message)
            elif level == "error":
                self.logger.error(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def connect(self, timeout: Optional[float] = None) -> bool:
        """
        连接到 SFTP 服务器
        
        Args:
            timeout: 连接超时时间（秒），None 使用默认值
            
        Returns:
            bool: 连接是否成功
        """
        if not PARAMIKO_AVAILABLE:
            error_msg = "paramiko库未安装，请运行: pip install paramiko"
            self._log("error", error_msg)
            self._notify_error("sftp", error_msg)
            return False
        
        if self.connected:
            self._log("info", "SFTP 已连接，跳过重复连接")
            return True
        
        timeout = timeout or self.connection_timeout
        self._log("info", f"正在连接SFTP服务器: {self.username}@{self.host}:{self.port}")
        
        # 使用线程进行非阻塞连接
        result = {"success": False, "error": None}
        
        def connect_thread():
            try:
                # 创建 SSH 客户端
                ssh_client = paramiko.SSHClient()
                ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                # 连接参数
                connect_kwargs = {
                    "hostname": self.host,
                    "port": self.port,
                    "username": self.username,
                    "timeout": self.ssh_timeout
                }
                
                # 认证方式：优先使用私钥，其次密码
                if self.private_key_path and os.path.exists(self.private_key_path):
                    try:
                        private_key = paramiko.RSAKey.from_private_key_file(self.private_key_path)
                        connect_kwargs["pkey"] = private_key
                        self._log("info", f"使用私钥认证: {self.private_key_path}")
                    except Exception as key_error:
                        self._log("warning", f"私钥加载失败，回退到密码认证: {key_error}")
                        if self.password:
                            connect_kwargs["password"] = self.password
                elif self.password:
                    connect_kwargs["password"] = self.password
                    self._log("info", "使用密码认证")
                else:
                    raise Exception("未提供有效的认证信息（私钥或密码）")
                
                # 连接 SSH
                ssh_client.connect(**connect_kwargs)
                
                # 创建 SFTP 客户端
                sftp_client = ssh_client.open_sftp()
                
                # 测试远程路径是否存在，不存在则尝试创建
                try:
                    sftp_client.stat(self.remote_path)
                    self._log("info", f"远程路径已存在: {self.remote_path}")
                except FileNotFoundError:
                    try:
                        # 尝试创建目录（递归创建）
                        self._create_remote_directory(sftp_client, self.remote_path)
                        self._log("info", f"远程路径已创建: {self.remote_path}")
                    except Exception as mkdir_error:
                        self._log("warning", f"无法创建远程路径: {mkdir_error}")
                
                result["success"] = True
                result["ssh_client"] = ssh_client
                result["sftp_client"] = sftp_client
                
            except Exception as e:
                result["error"] = str(e)
        
        # 创建 daemon 线程，超时后自动终止
        thread = threading.Thread(target=connect_thread, daemon=True)
        start_time = time.time()
        thread.start()
        
        # 轮询检查，每 0.1 秒检查一次
        while thread.is_alive() and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if thread.is_alive():
            # 超时了
            error_msg = f"SFTP连接超时: {self.host}:{self.port} ({timeout}秒)"
            self._log("error", error_msg)
            self._notify_error("sftp", error_msg)
            return False
        elif result["error"]:
            # 有异常
            error_msg = f"SFTP连接失败: {result['error']}"
            self._log("error", error_msg)
            self._notify_error("sftp", error_msg)
            return False
        else:
            # 正常完成
            if result["success"]:
                self.ssh_client = result["ssh_client"]
                self.sftp_client = result["sftp_client"]
                self.connected = True
                self._log("info", f"SFTP连接成功: {self.username}@{self.host}:{self.port}")
                return True
            else:
                error_msg = "SFTP连接失败: 未知错误"
                self._log("error", error_msg)
                self._notify_error("sftp", error_msg)
                return False
    
    def _create_remote_directory(self, sftp_client, remote_path: str):
        """
        递归创建远程目录
        
        Args:
            sftp_client: SFTP 客户端实例
            remote_path: 远程路径
        """
        try:
            # 标准化路径（使用正斜杠）
            remote_path = remote_path.replace('\\', '/')
            
            # 分割路径
            path_parts = remote_path.strip('/').split('/')
            current_path = '/'
            
            for part in path_parts:
                if part:  # 跳过空部分
                    current_path = current_path.rstrip('/') + '/' + part
                    try:
                        sftp_client.stat(current_path)
                    except FileNotFoundError:
                        sftp_client.mkdir(current_path)
                        self._log("debug", f"创建远程目录: {current_path}")
                        
        except Exception as e:
            self._log("error", f"创建远程目录失败: {e}")
            raise
    
    def disconnect(self):
        """断开 SFTP 连接"""
        try:
            if self.sftp_client:
                self.sftp_client.close()
                self._log("info", "SFTP客户端已关闭")
            
            if self.ssh_client:
                self.ssh_client.close()
                self._log("info", "SSH客户端已关闭")
                
        except Exception as e:
            self._log("error", f"断开SFTP连接时出错: {e}")
        finally:
            self.ssh_client = None
            self.sftp_client = None
            self.connected = False
    
    def upload_file(self, local_file_path: str, remote_filename: str = None, 
                   remote_path: str = None, verify_upload: bool = True) -> Dict[str, Any]:
        """
        上传文件到 SFTP 服务器
        
        Args:
            local_file_path: 本地文件路径
            remote_filename: 远程文件名（可选，默认使用本地文件名）
            remote_path: 远程路径（可选，默认使用配置中的路径）
            verify_upload: 是否验证上传结果
            
        Returns:
            Dict: 上传结果，包含 success, message, filename, remote_path, file_size 等
        """
        result = {
            "success": False,
            "message": "",
            "filename": "",
            "remote_path": "",
            "file_size": 0,
            "timestamp": time.time()
        }
        
        try:
            # 检查连接状态
            if not self.connected or not self.sftp_client:
                error_msg = "SFTP客户端未连接或不可用"
                self._log("error", error_msg)
                result["message"] = error_msg
                return result
            
            # 检查本地文件是否存在
            if not os.path.exists(local_file_path):
                error_msg = f"本地文件不存在: {local_file_path}"
                self._log("error", error_msg)
                result["message"] = error_msg
                return result
            
            # 生成远程文件名
            if not remote_filename:
                remote_filename = os.path.basename(local_file_path)
            
            # 构建完整的远程路径
            target_remote_path = remote_path or self.remote_path
            remote_full_path = f"{target_remote_path.rstrip('/')}/{remote_filename}"
            
            # 标准化远程路径
            remote_full_path = remote_full_path.replace('\\', '/')
            
            # 获取文件大小
            file_size = os.path.getsize(local_file_path)
            
            self._log("info", f"正在上传文件: {local_file_path} -> {remote_full_path}")
            
            # 执行文件上传
            self.sftp_client.put(local_file_path, remote_full_path)
            
            # 验证上传是否成功
            if verify_upload:
                try:
                    remote_stat = self.sftp_client.stat(remote_full_path)
                    if remote_stat.st_size == file_size:
                        success_msg = f"文件上传成功: {remote_filename} ({file_size} bytes)"
                        self._log("info", success_msg)
                        result.update({
                            "success": True,
                            "message": success_msg,
                            "filename": remote_filename,
                            "remote_path": remote_full_path,
                            "file_size": file_size
                        })
                    else:
                        error_msg = f"文件上传验证失败: 大小不匹配 (本地: {file_size}, 远程: {remote_stat.st_size})"
                        self._log("error", error_msg)
                        result["message"] = error_msg
                        
                except Exception as verify_error:
                    success_msg = f"文件上传完成: {remote_filename} -> {remote_full_path} (无法验证文件大小: {verify_error})"
                    self._log("info", success_msg)
                    result.update({
                        "success": True,
                        "message": success_msg,
                        "filename": remote_filename,
                        "remote_path": remote_full_path,
                        "file_size": file_size
                    })
            else:
                # 不验证，直接认为成功
                success_msg = f"文件上传完成: {remote_filename} ({file_size} bytes)"
                self._log("info", success_msg)
                result.update({
                    "success": True,
                    "message": success_msg,
                    "filename": remote_filename,
                    "remote_path": remote_full_path,
                    "file_size": file_size
                })
                
        except Exception as e:
            error_msg = f"SFTP文件上传失败: {e}"
            self._log("error", error_msg)
            result["message"] = error_msg
        
        return result
    
    def upload_image(self, image_data: Union[np.ndarray, Image.Image], 
                    image_format: str = "jpg", prefix: str = "detection",
                    remote_path: str = None, verify_upload: bool = True) -> Dict[str, Any]:
        """
        上传图像数据到 SFTP 服务器
        
        Args:
            image_data: 图像数据 (numpy array 或 PIL Image)
            image_format: 图像格式 (jpg, png)
            prefix: 文件名前缀
            remote_path: 远程路径（可选）
            verify_upload: 是否验证上传结果
            
        Returns:
            Dict: 上传结果
        """
        result = {
            "success": False,
            "message": "",
            "filename": "",
            "remote_path": "",
            "file_size": 0,
            "timestamp": time.time()
        }
        
        try:
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 毫秒精度
            filename = f"{prefix}_{timestamp}.{image_format}"
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=f".{image_format}", delete=False) as temp_file:
                temp_path = temp_file.name
                
                try:
                    # 保存图像到临时文件
                    if isinstance(image_data, np.ndarray):
                        if CV2_AVAILABLE:
                            cv2.imwrite(temp_path, image_data)
                        elif PIL_AVAILABLE:
                            # 转换 numpy array 到 PIL Image
                            if len(image_data.shape) == 3:
                                pil_image = Image.fromarray(image_data)
                            else:
                                pil_image = Image.fromarray(image_data)
                            pil_image.save(temp_path)
                        else:
                            error_msg = "需要安装OpenCV或PIL库来处理图像"
                            self._log("error", error_msg)
                            result["message"] = error_msg
                            return result
                    elif PIL_AVAILABLE and isinstance(image_data, Image.Image):
                        image_data.save(temp_path)
                    else:
                        error_msg = "不支持的图像数据格式"
                        self._log("error", error_msg)
                        result["message"] = error_msg
                        return result
                    
                    # 上传文件
                    upload_result = self.upload_file(temp_path, filename, remote_path, verify_upload)
                    result.update(upload_result)
                    
                finally:
                    # 清理临时文件
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                
        except Exception as e:
            error_msg = f"上传图像失败: {e}"
            self._log("error", error_msg)
            result["message"] = error_msg
        
        return result
    
    def upload_to_root(self, local_file_path: str, remote_filename: str = None,
                      verify_upload: bool = True) -> Dict[str, Any]:
        """
        上传文件到 SFTP 服务器根路径
        
        Args:
            local_file_path: 本地文件路径
            remote_filename: 远程文件名（可选）
            verify_upload: 是否验证上传结果
            
        Returns:
            Dict: 上传结果
        """
        return self.upload_file(local_file_path, remote_filename, "/", verify_upload)
    
    def test_connection(self, test_file_path: str = "./test.png") -> Dict[str, Any]:
        """
        测试 SFTP 连接，上传测试文件
        
        Args:
            test_file_path: 测试文件路径
            
        Returns:
            Dict: 测试结果
        """
        result = {
            "success": False,
            "message": "",
            "filename": "",
            "remote_path": "",
            "file_size": 0,
            "timestamp": time.time()
        }
        
        try:
            # 检查测试文件是否存在
            if not os.path.exists(test_file_path):
                error_msg = f"测试文件不存在: {test_file_path}"
                self._log("error", error_msg)
                result["message"] = error_msg
                return result
            
            self._log("info", f"正在测试SFTP连接，上传文件: {test_file_path}")
            
            # 上传到根路径
            upload_result = self.upload_to_root(test_file_path, "test.png")
            result.update(upload_result)
            
            if result["success"]:
                result["message"] = f"SFTP连接测试成功: {result['filename']} ({result['file_size']} bytes) -> 根路径"
            
        except Exception as e:
            error_msg = f"SFTP连接测试失败: {e}"
            self._log("error", error_msg)
            result["message"] = error_msg
        
        return result
    
    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        return {
            "connected": self.connected,
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "remote_path": self.remote_path,
            "ssh_client": self.ssh_client is not None,
            "sftp_client": self.sftp_client is not None
        }
    
    def _notify_error(self, component_name: str, error_message: str):
        """通知错误（如果有回调函数）"""
        if self.error_callback:
            try:
                self.error_callback(component_name, error_message)
            except Exception as e:
                self._log("error", f"错误回调执行失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        if not self.connected:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()
    
    def __del__(self):
        """析构函数，确保连接被关闭"""
        try:
            self.disconnect()
        except:
          pass
        
        
