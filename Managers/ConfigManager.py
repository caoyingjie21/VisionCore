#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置管理器 - ConfigManager
负责配置文件的加载、验证、动态更新和保存
"""

import os
import yaml
import json
import shutil
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from copy import deepcopy
import logging


@dataclass
class ConfigChangeEvent:
    """配置变更事件"""
    config_section: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    source: str  # "mqtt", "api", "reload"


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


class ConfigLoadError(Exception):
    """配置加载错误"""
    pass


class ConfigManager:
    """简化的配置管理器"""
    
    def __init__(self, config_path: str = "Config/config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 主配置文件路径
        """
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent
        
        # 确保目录存在
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置数据
        self._config_data: Dict[str, Any] = {}
        self._config_lock = asyncio.Lock()
        
        # 变更监听
        self._change_callbacks: List[Callable[[ConfigChangeEvent], None]] = []
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置数据字典
            
        Raises:
            ConfigLoadError: 配置加载失败
        """
        try:
            if not self.config_path.exists():
                raise ConfigLoadError(f"配置文件不存在: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif self.config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ConfigLoadError(f"不支持的配置文件格式: {self.config_path.suffix}")
            
            # 更新配置数据
            self._config_data = config_data or {}
            
            self.logger.info(f"配置文件加载成功: {self.config_path}")
            return deepcopy(self._config_data)
            
        except Exception as e:
            error_msg = f"配置文件加载失败: {e}"
            self.logger.error(error_msg)
            raise ConfigLoadError(error_msg)
    
    async def reload_config(self) -> bool:
        """
        触发器：重新加载配置文件
        
        Returns:
            重载是否成功
        """
        async with self._config_lock:
            try:
                # 保存旧配置
                old_config = deepcopy(self._config_data)
                
                # 重新加载配置
                new_config = self.load_config()
                
                # 检测变更并通知
                await self._notify_config_changes(old_config, new_config, "reload")
                
                self.logger.info("配置重载成功")
                return True
                
            except Exception as e:
                self.logger.error(f"配置重载失败: {e}")
                return False
    
    async def update_config_section(
        self, 
        section: str, 
        new_value: Any, 
        source: str = "mqtt",
        persist: bool = True
    ) -> bool:
        """
        更新配置的特定段
        
        Args:
            section: 配置段名称 (支持点分割路径，如 "camera.connection.ip")
            new_value: 新值
            source: 更新来源
            persist: 是否持久化到文件
            
        Returns:
            更新是否成功
        """
        async with self._config_lock:
            try:
                # 保存旧值
                old_value = self.get_config(section)
                
                # 更新配置
                self._set_nested_value(self._config_data, section, new_value)
                
                # 持久化到文件
                if persist:
                    await self._save_config()
                
                # 通知变更
                event = ConfigChangeEvent(
                    config_section=section,
                    old_value=old_value,
                    new_value=new_value,
                    timestamp=datetime.now(),
                    source=source
                )
                await self._notify_change(event)
                
                self.logger.info(f"配置段 '{section}' 更新成功")
                return True
                
            except Exception as e:
                self.logger.error(f"配置段 '{section}' 更新失败: {e}")
                return False
    
    async def update_full_config(
        self, 
        new_config: Dict[str, Any], 
        source: str = "mqtt"
    ) -> bool:
        """
        更新整个配置
        
        Args:
            new_config: 新的配置数据
            source: 更新来源
            
        Returns:
            更新是否成功
        """
        async with self._config_lock:
            try:
                # 保存旧配置
                old_config = deepcopy(self._config_data)
                
                # 更新配置
                self._config_data = new_config
                
                # 持久化到文件
                await self._save_config()
                
                # 检测变更并通知
                await self._notify_config_changes(old_config, new_config, source)
                
                self.logger.info("完整配置更新成功")
                return True
                
            except Exception as e:
                self.logger.error(f"完整配置更新失败: {e}")
                # 恢复旧配置
                self._config_data = old_config
                return False
    
    def get_config(self, path: Optional[str] = None) -> Any:
        """
        获取配置值
        
        Args:
            path: 配置路径 (支持点分割，如 "camera.connection.ip")
                 为None时返回整个配置
                 
        Returns:
            配置值
        """
        if path is None:
            return deepcopy(self._config_data)
        
        return self._get_nested_value(self._config_data, path)
    
    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """获取嵌套字典的值"""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return deepcopy(current) if isinstance(current, (dict, list)) else current
    
    def _set_nested_value(self, data: Dict, path: str, value: Any):
        """设置嵌套字典的值"""
        keys = path.split('.')
        current = data
        
        # 导航到目标位置
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # 设置值
        current[keys[-1]] = value
    
    async def _save_config(self):
        """保存配置到文件"""
        try:
            # 创建临时文件
            temp_path = self.config_path.with_suffix('.tmp')
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self._config_data, f, default_flow_style=False, 
                             allow_unicode=True, indent=2, sort_keys=False)
                elif self.config_path.suffix.lower() == '.json':
                    json.dump(self._config_data, f, indent=2, ensure_ascii=False)
            
            # 原子性替换
            shutil.move(str(temp_path), str(self.config_path))
            
            self.logger.debug("配置已保存到文件")
            
        except Exception as e:
            self.logger.error(f"配置保存失败: {e}")
            # 清理临时文件
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def add_change_callback(self, callback: Callable[[ConfigChangeEvent], None]):
        """添加配置变更回调"""
        self._change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[ConfigChangeEvent], None]):
        """移除配置变更回调"""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
    
    async def _notify_change(self, event: ConfigChangeEvent):
        """通知配置变更"""
        for callback in self._change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error(f"配置变更回调执行失败: {e}")
    
    async def _notify_config_changes(self, old_config: Dict, new_config: Dict, source: str):
        """检测并通知配置变更"""
        changes = self._detect_changes(old_config, new_config)
        
        for change_path, (old_val, new_val) in changes.items():
            event = ConfigChangeEvent(
                config_section=change_path,
                old_value=old_val,
                new_value=new_val,
                timestamp=datetime.now(),
                source=source
            )
            await self._notify_change(event)
    
    def _detect_changes(self, old_config: Dict, new_config: Dict, prefix: str = "") -> Dict[str, tuple]:
        """检测配置变更"""
        changes = {}
        
        # 检查新增和修改
        for key, new_value in new_config.items():
            current_path = f"{prefix}.{key}" if prefix else key
            
            if key not in old_config:
                # 新增
                changes[current_path] = (None, new_value)
            elif old_config[key] != new_value:
                if isinstance(new_value, dict) and isinstance(old_config[key], dict):
                    # 递归检查嵌套字典
                    nested_changes = self._detect_changes(
                        old_config[key], new_value, current_path
                    )
                    changes.update(nested_changes)
                else:
                    # 修改
                    changes[current_path] = (old_config[key], new_value)
        
        # 检查删除
        for key, old_value in old_config.items():
            if key not in new_config:
                current_path = f"{prefix}.{key}" if prefix else key
                changes[current_path] = (old_value, None)
        
        return changes
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "config_file": str(self.config_path),
            "config_dir": str(self.config_dir),
            "last_loaded": datetime.now().isoformat(),
            "sections": list(self._config_data.keys()) if self._config_data else []
        }


# 单例模式的配置管理器
_config_manager_instance: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    获取配置管理器单例
    
    Args:
        config_path: 配置文件路径（仅在首次调用时有效）
        
    Returns:
        配置管理器实例
    """
    global _config_manager_instance
    
    if _config_manager_instance is None:
        if config_path is None:
            config_path = "Config/config.yaml"
        _config_manager_instance = ConfigManager(config_path)
    
    return _config_manager_instance


# 便捷函数
def get_config(path: Optional[str] = None) -> Any:
    """获取配置值的便捷函数"""
    return get_config_manager().get_config(path)


async def update_config(section: str, value: Any, source: str = "mqtt") -> bool:
    """更新配置的便捷函数"""
    return await get_config_manager().update_config_section(section, value, source)


if __name__ == "__main__":
    # 测试代码
    import asyncio
    
    async def test_config_manager():
        # 初始化配置管理器
        config_mgr = ConfigManager("../Config/config.yaml")
        
        # 测试获取配置
        print("系统配置:", config_mgr.get_config("system"))
        print("相机IP:", config_mgr.get_config("camera.connection.ip"))
        
        # 测试更新配置
        await config_mgr.update_config_section("camera.connection.ip", "192.168.1.101", "mqtt")
        print("更新后的相机IP:", config_mgr.get_config("camera.connection.ip"))
        
        # 测试触发器重载
        await config_mgr.reload_config()
        
        # 获取系统信息
        print("系统信息:", config_mgr.get_system_info())
    
    # 运行测试
    asyncio.run(test_config_manager())
