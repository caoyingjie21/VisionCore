#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
系统监控器模块
负责监控系统组件健康状态、自动重启和状态报告
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from .ComponentStatus import ComponentStatus


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
        self.monitor_thread: Optional[threading.Thread] = None
        self.restart_callbacks: Dict[str, callable] = {}
        
    def register_component(self, name: str, component: Any, restart_callback: Optional[Callable] = None):
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
    
    def unregister_component(self, name: str):
        """
        取消注册组件
        
        Args:
            name: 组件名称
        """
        if name in self.components:
            del self.components[name]
        if name in self.status:
            del self.status[name]
        if name in self.restart_callbacks:
            del self.restart_callbacks[name]
        
        self.logger.info(f"已取消注册监控组件: {name}")
    
    def start_monitoring(self):
        """启动监控线程"""
        if self.monitoring:
            self.logger.warning("监控已在运行中")
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
                self.status[name].last_error = str(e)
    
    def _check_component_health(self, name: str, component: Any) -> bool:
        """
        检查单个组件健康状态
        
        Args:
            name: 组件名称
            component: 组件实例
            
        Returns:
            组件是否健康
        """
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
        """
        重启指定组件
        
        Args:
            name: 组件名称
        """
        try:
            self.logger.warning(f"正在重启组件: {name}")
            
            if name in self.restart_callbacks:
                # 使用自定义重启回调
                success = self.restart_callbacks[name]()
                if success:
                    self.logger.info(f"组件 {name} 重启成功")
                    self.status[name].error_count = 0
                    self.status[name].last_error = None
                else:
                    self.logger.error(f"组件 {name} 重启失败")
                    self.status[name].last_error = "重启失败"
            else:
                self.logger.warning(f"组件 {name} 没有重启回调函数")
                self.status[name].last_error = "无重启回调"
                
        except Exception as e:
            self.logger.error(f"重启组件 {name} 时出错: {e}")
            self.status[name].last_error = f"重启异常: {e}"
    
    def get_component_status(self, name: str) -> Optional[ComponentStatus]:
        """
        获取指定组件的状态
        
        Args:
            name: 组件名称
            
        Returns:
            组件状态或None
        """
        return self.status.get(name)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态信息
        
        Returns:
            系统状态字典
        """
        return {
            "monitoring": self.monitoring,
            "check_interval": self.check_interval,
            "total_components": len(self.components),
            "healthy_components": sum(1 for status in self.status.values() if status.is_healthy),
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
    
    def force_check_component(self, name: str) -> bool:
        """
        强制检查指定组件
        
        Args:
            name: 组件名称
            
        Returns:
            检查结果
        """
        if name not in self.components:
            self.logger.error(f"组件 {name} 未注册")
            return False
        
        try:
            component = self.components[name]
            is_healthy = self._check_component_health(name, component)
            
            status = self.status[name]
            status.is_healthy = is_healthy
            status.last_check = datetime.now()
            
            if not is_healthy:
                status.error_count += 1
                self.logger.warning(f"强制检查组件 {name} 失败")
            else:
                status.error_count = 0
                self.logger.info(f"强制检查组件 {name} 成功")
            
            return is_healthy
            
        except Exception as e:
            self.logger.error(f"强制检查组件 {name} 异常: {e}")
            self.status[name].last_error = str(e)
            return False
    
    def is_system_healthy(self) -> bool:
        """
        检查整个系统是否健康
        
        Returns:
            系统是否健康
        """
        if not self.status:
            return True  # 没有组件则认为健康
        
        return all(status.is_healthy for status in self.status.values())
    
    def get_unhealthy_components(self) -> list:
        """
        获取不健康的组件列表
        
        Returns:
            不健康组件名称列表
        """
        return [name for name, status in self.status.items() if not status.is_healthy] 