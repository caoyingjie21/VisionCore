#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TCP服务器模块
提供TCP通信服务，支持多客户端连接、消息处理和健康检查
"""

import socket
import threading
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ClientInfo:
    """客户端信息"""
    socket: socket.socket
    address: tuple
    connect_time: datetime
    last_heartbeat: datetime
    is_active: bool = True


class TcpServer:
    """TCP服务器类，支持多客户端连接和消息处理"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None, message_callback: Optional[Callable] = None):
        """
        初始化TCP服务器
        
        Args:
            config: TCP服务器配置
            logger: 日志记录器
            message_callback: 消息处理回调函数，签名为 callback(client_id, message) -> response_message
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.message_callback = message_callback
        self.disconnect_callback: Optional[Callable] = None  # 添加断开连接回调
        
        # 服务器配置
        self.host = config.get("host", "0.0.0.0")
        self.port = config.get("port", 8888)
        self.max_connections = config.get("max_connections", 10)
        self.buffer_size = config.get("buffer_size", 4096)
        self.heartbeat_interval = config.get("heartbeat_interval", 30)
        self.connection_timeout = config.get("connection_timeout", 300)
        
        # 服务器状态
        self.is_running = False
        self.server_socket: Optional[socket.socket] = None
        self.clients: Dict[str, ClientInfo] = {}
        
        # 线程控制
        self.accept_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.cleanup_thread: Optional[threading.Thread] = None
        self.thread_lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            "start_time": None,
            "total_connections": 0,
            "current_connections": 0,
            "messages_received": 0,
            "messages_sent": 0,
            "errors": 0
        }
    
    def set_message_callback(self, callback: Callable):
        """设置消息处理回调函数"""
        self.message_callback = callback
    
    def set_disconnect_callback(self, callback: Callable):
        """设置客户端断开连接回调函数"""
        self.disconnect_callback = callback
    
    def start(self) -> bool:
        """
        启动TCP服务器
        
        Returns:
            bool: 启动是否成功
        """
        try:
            if self.is_running:
                self.logger.warning("TCP服务器已在运行中")
                return True
            
            # 创建服务器套接字
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(self.max_connections)
            
            self.is_running = True
            self.stats["start_time"] = datetime.now()
            
            # 启动线程
            self.accept_thread = threading.Thread(target=self._accept_connections, daemon=True)
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
            self.cleanup_thread = threading.Thread(target=self._cleanup_monitor, daemon=True)
            
            self.accept_thread.start()
            self.heartbeat_thread.start()
            self.cleanup_thread.start()
            
            self.logger.info(f"TCP服务器启动成功: {self.host}:{self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"TCP服务器启动失败: {e}")
            self.stop()
            return False
    
    def stop(self):
        """停止TCP服务器"""
        try:
            self.logger.info("正在停止TCP服务器...")
            self.is_running = False
            
            # 关闭所有客户端连接
            with self.thread_lock:
                for client_id, client_info in list(self.clients.items()):
                    self._disconnect_client(client_id, "服务器关闭")
            
            # 关闭服务器套接字
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
            
            self.logger.info("TCP服务器已停止")
            
        except Exception as e:
            self.logger.error(f"停止TCP服务器时出错: {e}")
    
    def _accept_connections(self):
        """接受客户端连接的线程"""
        while self.is_running:
            try:
                if not self.server_socket:
                    break
                
                client_socket, client_address = self.server_socket.accept()
                
                # 检查连接数限制
                if len(self.clients) >= self.max_connections:
                    self.logger.warning(f"达到最大连接数限制，拒绝连接: {client_address}")
                    client_socket.close()
                    continue
                
                # 创建客户端信息
                client_id = f"{client_address[0]}:{client_address[1]}:{int(time.time())}"
                client_info = ClientInfo(
                    socket=client_socket,
                    address=client_address,
                    connect_time=datetime.now(),
                    last_heartbeat=datetime.now()
                )
                
                with self.thread_lock:
                    self.clients[client_id] = client_info
                    self.stats["total_connections"] += 1
                    self.stats["current_connections"] = len(self.clients)
                
                # 启动客户端处理线程
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_id,),
                    daemon=True
                )
                client_thread.start()
                
                self.logger.info(f"新客户端连接: {client_address} -> {client_id}")
                
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"接受连接时出错: {e}")
                    self.stats["errors"] += 1
    
    def _handle_client(self, client_id: str):
        """处理单个客户端的线程"""
        client_info = self.clients.get(client_id)
        if not client_info:
            return
        
        client_socket = client_info.socket
        
        try:
            # 处理客户端消息
            buffer = ""
            while self.is_running and client_info.is_active:
                try:
                    # 接收消息
                    data = client_socket.recv(self.buffer_size)
                    if not data:
                        break
                    
                    # 解码数据并添加到缓冲区
                    buffer += data.decode('utf-8')
                    
                    # 处理完整的消息（以\r\n或\n结尾）
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if line:
                            self.stats["messages_received"] += 1
                            
                            # 更新心跳时间
                            client_info.last_heartbeat = datetime.now()
                            
                            # 调用外部消息处理器
                            if self.message_callback:
                                try:
                                    response = self.message_callback(client_id, line)
                                    if response:
                                        self._send_message(client_socket, response)
                                    # 移除错误响应，因为TCP服务端只应该在catch指令成功时回传数据
                                except Exception as e:
                                    self.logger.error(f"消息回调处理失败: {e}")
                                    # 不发送错误响应，保持静默
                            else:
                                # 如果没有设置回调，不发送任何响应，保持静默
                                # TCP服务端只应该在收到catch指令时回传检测数据
                                self.logger.debug(f"收到消息但未设置处理器: {line} from {client_id}")
                
                except socket.timeout:
                    continue
                except socket.error:
                    break
                    
        except Exception as e:
            self.logger.error(f"处理客户端 {client_id} 时出错: {e}")
        finally:
            self._disconnect_client(client_id, "连接断开")
    
    def _send_message(self, client_socket: socket.socket, message: str) -> bool:
        """发送消息给客户端"""
        try:
            # 确保消息以\r\n结尾
            if not message.endswith('\r\n'):
                message += '\r\n'
            
            data = message.encode('utf-8')
            client_socket.sendall(data)
            self.stats["messages_sent"] += 1
            return True
        except Exception as e:
            self.logger.error(f"发送消息失败: {e}")
            self.stats["errors"] += 1
            return False
    
    def _disconnect_client(self, client_id: str, reason: str):
        """断开客户端连接"""
        with self.thread_lock:
            client_info = self.clients.pop(client_id, None)
            if client_info:
                try:
                    client_info.socket.close()
                except:
                    pass
                client_info.is_active = False
                self.stats["current_connections"] = len(self.clients)
                self.logger.info(f"客户端断开: {client_id} - {reason}")
                
                # 调用断开连接回调函数
                if self.disconnect_callback:
                    try:
                        self.disconnect_callback(client_id, reason)
                    except Exception as e:
                        self.logger.error(f"断开连接回调处理失败: {e}")
    
    def _heartbeat_monitor(self):
        """心跳监控线程"""
        while self.is_running:
            try:
                current_time = datetime.now()
                timeout_clients = []
                
                with self.thread_lock:
                    for client_id, client_info in self.clients.items():
                        if client_info.is_active:
                            time_diff = (current_time - client_info.last_heartbeat).total_seconds()
                            if time_diff > self.connection_timeout:
                                timeout_clients.append(client_id)
                
                # 断开超时客户端
                for client_id in timeout_clients:
                    self._disconnect_client(client_id, "心跳超时")
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"心跳监控出错: {e}")
    
    def _cleanup_monitor(self):
        """清理监控线程"""
        while self.is_running:
            try:
                # 清理不活跃的连接
                inactive_clients = []
                with self.thread_lock:
                    for client_id, client_info in self.clients.items():
                        if not client_info.is_active:
                            inactive_clients.append(client_id)
                
                for client_id in inactive_clients:
                    self._disconnect_client(client_id, "连接不活跃")
                
                time.sleep(60)  # 每分钟清理一次
                
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"清理监控出错: {e}")
    
    # 公共方法
    def broadcast_message(self, message: str, exclude_client: Optional[str] = None):
        """广播消息给所有客户端"""
        with self.thread_lock:
            for client_id, client_info in self.clients.items():
                if client_id != exclude_client and client_info.is_active:
                    self._send_message(client_info.socket, message)
    
    def send_to_client(self, client_id: str, message: str) -> bool:
        """发送消息给指定客户端"""
        client_info = self.clients.get(client_id)
        if client_info and client_info.is_active:
            return self._send_message(client_info.socket, message)
        return False
    
    def get_client_list(self) -> List[Dict[str, Any]]:
        """获取客户端列表"""
        with self.thread_lock:
            return [
                {
                    "client_id": client_id,
                    "address": f"{info.address[0]}:{info.address[1]}",
                    "connect_time": info.connect_time.isoformat(),
                    "last_heartbeat": info.last_heartbeat.isoformat(),
                    "is_active": info.is_active
                }
                for client_id, info in self.clients.items()
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务器统计信息"""
        stats = self.stats.copy()
        stats["current_connections"] = len(self.clients)
        if stats["start_time"]:
            # 确保start_time转换为字符串格式
            if isinstance(stats["start_time"], datetime):
                stats["start_time"] = stats["start_time"].isoformat()
            stats["uptime_seconds"] = (datetime.now() - self.stats["start_time"]).total_seconds()
        return stats
    
    @property
    def is_connected(self) -> bool:
        """检查服务器是否正在运行（用于健康检查）"""
        return self.is_running and self.server_socket is not None
    
    @property
    def healthy(self) -> bool:
        """检查服务器是否健康"""
        return self.is_connected
