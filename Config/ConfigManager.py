#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置管理器 - ConfigManager
负责配置文件的加载、验证、热重载和管理
"""

import os
import yaml
import json
import shutil
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from copy import deepcopy

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("Warning: watchdog not available, file watching disabled")

import logging


@dataclass
class ConfigChangeEvent:
    """配置变更事件"""
    config_section: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    source: str  # "file", "mqtt", "api"


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


class ConfigLoadError(Exception):
    """配置加载错误"""
    pass


class ConfigFileWatcher(FileSystemEventHandler):
    """配置文件监控器"""
    
    def __init__(self, config_manager, debounce_interval: float = 1.0):
        self.config_manager = config_manager
        self.debounce_interval = debounce_interval
        self.pending_changes = {}
        self.timer_lock = threading.Lock()
        
    def on_modified(self, event):
        """文件修改事件处理"""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # 检查是否是配置文件
        if file_path.suffix in ['.yaml', '.yml', '.json']:
            with self.timer_lock:
                # 取消之前的定时器
                if file_path in self.pending_changes:
                    self.pending_changes[file_path].cancel()
                
                # 设置新的定时器（防抖）
                timer = threading.Timer(
                    self.debounce_interval,
                    self._handle_file_change,
                    args=[file_path]
                )
                self.pending_changes[file_path] = timer
                timer.start()
    
    def _handle_file_change(self, file_path: Path):
        """处理文件变更"""
        try:
            # 从待处理列表中移除
            with self.timer_lock:
                if file_path in self.pending_changes:
                    del self.pending_changes[file_path]
            
            # 通知配置管理器文件已变更
            asyncio.create_task(
                self.config_manager._handle_file_change(str(file_path))
            )
            
        except Exception as e:
            self.config_manager.logger.error(f"处理文件变更失败: {e}")


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "VisionCore/Config/config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 主配置文件路径
        """
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent
        self.backup_dir = self.config_dir / "backups"
        
        # 确保目录存在
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置数据
        self._config_data: Dict[str, Any] = {}
        self._config_schemas: Dict[str, Dict] = {}
        self._config_lock = asyncio.Lock()
        
        # 变更监听
        self._change_callbacks: List[Callable[[ConfigChangeEvent], None]] = []
        
        # 文件监控
        self._file_observer: Optional[Observer] = None
        self._file_watcher: Optional[ConfigFileWatcher] = None
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化配置模式
        self._init_config_schemas()
        
        # 加载配置
        self.load_config()
        
        # 启动文件监控
        if WATCHDOG_AVAILABLE:
            self._start_file_watching()
    
    def _init_config_schemas(self):
        """初始化配置验证模式"""
        # 这里可以定义各个配置段的验证规则
        # 为了简化，这里使用基本的类型检查
        self._config_schemas = {
            "system": {
                "required": ["name", "version"],
                "properties": {
                    "name": {"type": "string"},
                    "version": {"type": "string"},
                    "debug": {"type": "boolean"},
                    "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                    "max_workers": {"type": "integer", "minimum": 1}
                }
            },
            "camera": {
                "required": ["enabled", "type"],
                "properties": {
                    "enabled": {"type": "boolean"},
                    "type": {"type": "string"},
                    "connection": {
                        "type": "object",
                        "required": ["ip", "port"],
                        "properties": {
                            "ip": {"type": "string"},
                            "port": {"type": "integer"},
                            "timeout": {"type": "integer"}
                        }
                    }
                }
            }
            # 可以继续添加其他配置段的验证规则
        }
    
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
            
            # 验证配置
            self._validate_config(config_data)
            
            # 更新配置数据
            self._config_data = config_data
            
            self.logger.info(f"配置文件加载成功: {self.config_path}")
            return deepcopy(self._config_data)
            
        except Exception as e:
            error_msg = f"配置文件加载失败: {e}"
            self.logger.error(error_msg)
            raise ConfigLoadError(error_msg)
    
    async def reload_config(self, backup_current: bool = True) -> bool:
        """
        热重载配置文件
        
        Args:
            backup_current: 是否备份当前配置
            
        Returns:
            重载是否成功
        """
        async with self._config_lock:
            try:
                # 备份当前配置
                if backup_current:
                    self._backup_config("before_reload")
                
                # 保存旧配置
                old_config = deepcopy(self._config_data)
                
                # 重新加载配置
                new_config = self.load_config()
                
                # 检测变更并通知
                await self._notify_config_changes(old_config, new_config, "file")
                
                self.logger.info("配置热重载成功")
                return True
                
            except Exception as e:
                self.logger.error(f"配置热重载失败: {e}")
                return False
    
    async def update_config_section(
        self, 
        section: str, 
        new_value: Any, 
        source: str = "api",
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
                
                # 验证更新后的配置
                self._validate_config_section(section.split('.')[0], 
                                            self._config_data.get(section.split('.')[0]))
                
                # 持久化到文件
                if persist:
                    await self._persist_config()
                
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
    
    def _validate_config(self, config_data: Dict[str, Any]):
        """验证整个配置"""
        for section_name, section_data in config_data.items():
            self._validate_config_section(section_name, section_data)
    
    def _validate_config_section(self, section_name: str, section_data: Any):
        """
        验证配置段
        
        Args:
            section_name: 配置段名称
            section_data: 配置段数据
            
        Raises:
            ConfigValidationError: 验证失败
        """
        if section_name not in self._config_schemas:
            # 如果没有定义验证规则，跳过验证
            return
        
        schema = self._config_schemas[section_name]
        
        # 简单的验证逻辑（这里可以使用jsonschema等库进行更完整的验证）
        if "required" in schema:
            for required_field in schema["required"]:
                if required_field not in section_data:
                    raise ConfigValidationError(
                        f"配置段 '{section_name}' 缺少必需字段: {required_field}"
                    )
        
        # 可以添加更多验证逻辑
        self.logger.debug(f"配置段 '{section_name}' 验证通过")
    
    async def _persist_config(self):
        """持久化配置到文件"""
        try:
            # 创建临时文件
            temp_path = self.config_path.with_suffix('.tmp')
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self._config_data, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
                elif self.config_path.suffix.lower() == '.json':
                    json.dump(self._config_data, f, indent=2, ensure_ascii=False)
            
            # 原子性替换
            shutil.move(str(temp_path), str(self.config_path))
            
            self.logger.debug("配置已持久化到文件")
            
        except Exception as e:
            self.logger.error(f"配置持久化失败: {e}")
            # 清理临时文件
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def _backup_config(self, suffix: str = ""):
        """备份当前配置"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"config_backup_{timestamp}"
            if suffix:
                backup_name += f"_{suffix}"
            backup_name += self.config_path.suffix
            
            backup_path = self.backup_dir / backup_name
            shutil.copy2(str(self.config_path), str(backup_path))
            
            self.logger.info(f"配置已备份到: {backup_path}")
            
            # 清理旧备份（保留最近30个）
            self._cleanup_old_backups()
            
        except Exception as e:
            self.logger.error(f"配置备份失败: {e}")
    
    def _cleanup_old_backups(self, keep_count: int = 30):
        """清理旧的备份文件"""
        try:
            backup_files = list(self.backup_dir.glob("config_backup_*"))
            
            if len(backup_files) > keep_count:
                # 按修改时间排序
                backup_files.sort(key=lambda x: x.stat().st_mtime)
                
                # 删除多余的备份
                for backup_file in backup_files[:-keep_count]:
                    backup_file.unlink()
                    self.logger.debug(f"删除旧备份: {backup_file}")
                    
        except Exception as e:
            self.logger.error(f"清理备份文件失败: {e}")
    
    async def restore_config(self, backup_path: Optional[str] = None) -> bool:
        """
        从备份恢复配置
        
        Args:
            backup_path: 备份文件路径，为None时使用最新备份
            
        Returns:
            恢复是否成功
        """
        async with self._config_lock:
            try:
                if backup_path is None:
                    # 找到最新的备份
                    backup_files = list(self.backup_dir.glob("config_backup_*"))
                    if not backup_files:
                        raise ConfigLoadError("没有找到备份文件")
                    
                    backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    backup_path = str(backup_files[0])
                
                # 备份当前配置
                self._backup_config("before_restore")
                
                # 恢复配置
                shutil.copy2(backup_path, str(self.config_path))
                
                # 重新加载
                await self.reload_config(backup_current=False)
                
                self.logger.info(f"配置已从备份恢复: {backup_path}")
                return True
                
            except Exception as e:
                self.logger.error(f"配置恢复失败: {e}")
                return False
    
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
    
    def _start_file_watching(self):
        """启动文件监控"""
        if not WATCHDOG_AVAILABLE:
            self.logger.warning("文件监控不可用，已禁用热重载功能")
            return
        
        try:
            # 获取防抖间隔
            debounce_interval = self.get_config("hot_reload.file_watch.debounce_interval") or 1.0
            
            self._file_watcher = ConfigFileWatcher(self, debounce_interval)
            self._file_observer = Observer()
            self._file_observer.schedule(
                self._file_watcher,
                str(self.config_dir),
                recursive=True
            )
            self._file_observer.start()
            
            self.logger.info("文件监控已启动")
            
        except Exception as e:
            self.logger.error(f"启动文件监控失败: {e}")
    
    async def _handle_file_change(self, file_path: str):
        """处理文件变更事件"""
        try:
            file_path = Path(file_path)
            
            # 检查是否是主配置文件
            if file_path.samefile(self.config_path):
                self.logger.info(f"检测到配置文件变更: {file_path}")
                
                # 检查是否启用了热重载
                if self.get_config("hot_reload.enabled"):
                    await self.reload_config()
                else:
                    self.logger.info("热重载已禁用，跳过自动重载")
            
        except Exception as e:
            self.logger.error(f"处理文件变更失败: {e}")
    
    def stop(self):
        """停止配置管理器"""
        if self._file_observer:
            self._file_observer.stop()
            self._file_observer.join()
            self.logger.info("文件监控已停止")
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "config_file": str(self.config_path),
            "config_dir": str(self.config_dir),
            "backup_dir": str(self.backup_dir),
            "file_watching_enabled": WATCHDOG_AVAILABLE and self._file_observer is not None,
            "last_loaded": datetime.now().isoformat(),
            "sections": list(self._config_data.keys()) if self._config_data else []
        }
    
    def validate_connectivity(self) -> Dict[str, bool]:
        """验证各个组件的连接配置"""
        results = {}
        
        try:
            # 验证相机配置
            camera_config = self.get_config("camera")
            if camera_config and camera_config.get("enabled"):
                # 这里可以添加实际的连接测试
                results["camera"] = True
            
            # 验证机器人配置
            robots_config = self.get_config("robots")
            if robots_config and robots_config.get("enabled"):
                results["robots"] = True
            
            # 验证MQTT配置
            mqtt_config = self.get_config("mqtt")
            if mqtt_config and mqtt_config.get("enabled"):
                results["mqtt"] = True
            
        except Exception as e:
            self.logger.error(f"连接验证失败: {e}")
        
        return results


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
            config_path = "VisionCore/Config/config.yaml"
        _config_manager_instance = ConfigManager(config_path)
    
    return _config_manager_instance


# 便捷函数
def get_config(path: Optional[str] = None) -> Any:
    """获取配置值的便捷函数"""
    return get_config_manager().get_config(path)


async def update_config(section: str, value: Any, source: str = "api") -> bool:
    """更新配置的便捷函数"""
    return await get_config_manager().update_config_section(section, value, source)


if __name__ == "__main__":
    # 测试代码
    import asyncio
    
    async def test_config_manager():
        # 初始化配置管理器
        config_mgr = ConfigManager()
        
        # 测试获取配置
        print("系统配置:", config_mgr.get_config("system"))
        print("相机IP:", config_mgr.get_config("camera.connection.ip"))
        
        # 测试更新配置
        await config_mgr.update_config_section("camera.connection.ip", "192.168.1.100")
        print("更新后的相机IP:", config_mgr.get_config("camera.connection.ip"))
        
        # 获取系统信息
        print("系统信息:", config_mgr.get_system_info())
        
        # 停止配置管理器
        config_mgr.stop()
    
    # 运行测试
    asyncio.run(test_config_manager())
