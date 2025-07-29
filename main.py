#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VisionCore 主程序
负责启动和管理整个视觉系统
"""

import time
import sys
import signal
import os
from typing import Dict, Any

from System.SystemInitializer import SystemInitializer
from utils.decorators import handle_keyboard_interrupt


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
    
    def start(self):
        """启动应用程序主循环"""
        print("启动 VisionCore 系统...")
        
        try:
            while self.running:
                try:
                    # 初始化系统
                    if not self._initialize_system():
                        if self.restart_on_failure:
                            print("系统初始化失败，5秒后重试...")
                            # 使用可中断的睡眠
                            for _ in range(5):
                                time.sleep(1)
                            continue
                        else:
                            break
                    
                    # 运行主循环
                    self._run_main_loop()
                    
                except KeyboardInterrupt:
                    print("\n接收到停止信号，正在优雅关闭...")
                    self.running = False
                    self.restart_on_failure = False
                    break
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
                        # 使用可中断的睡眠
                        for _ in range(10):
                            time.sleep(1)
                    else:
                        break
                finally:
                    # 清理资源
                    if self.initializer:
                        self.initializer.cleanup()
                        self.initializer = None
        
        except KeyboardInterrupt:
            print("\n接收到停止信号，正在优雅关闭...")
            self.running = False
            self.restart_on_failure = False
        
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
    
    def _stop_running(self):
        """停止主循环的清理方法"""
        self.running = False
    
    def _run_main_loop(self):
        """运行主程序循环"""
        logger = self.initializer.logger
        
        tcp_server = self.initializer.get_tcp_server()
        mqtt_client = self.initializer.get_mqtt_client()
        
        # 为TCP服务器设置消息处理回调
        if tcp_server:
            message_handler = self.initializer.create_tcp_message_handler(self.handle_catch)
            tcp_server.set_message_callback(message_handler)
            logger.info("TCP服务器消息处理器已设置")
        
        # 为MQTT客户端设置消息处理回调
        if mqtt_client:
            self.initializer.set_mqtt_message_callback(self.handle_mqtt_message)
            logger.info("MQTT消息处理器已设置")
        
        logger.info("系统运行中，按Ctrl+C停止...")
        
        # 主程序循环
        loop_count = 0
        try:
            while self.running:
                try:
                    # 检查系统健康状态（每60秒）
                    if loop_count % 60 == 0:
                        self._check_system_health(logger)
                    
                    time.sleep(1)
                    loop_count += 1
                    
                    # 防止计数器溢出
                    if loop_count > 86400:  # 24小时重置
                        loop_count = 0
                    
                except Exception as e:
                    logger.error(f"主循环异常: {e}", exc_info=True)
                    # 继续运行，不退出主循环
                    time.sleep(1)
        except KeyboardInterrupt:
            logger.info("接收到停止信号，主程序循环停止")
            self.running = False
    
    def handle_catch(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理catch指令
        
        Args:
            client_id: 客户端ID
            message: 接收到的消息
            
        Returns:
            Dict: 响应消息
        """
        logger = self.initializer.logger
        logger.info(f"收到catch指令来自客户端: {client_id}")
        
        try:
            # 获取系统组件
            camera = self.initializer.get_camera()
            detector = self.initializer.get_detector()
            mqtt_client = self.initializer.get_mqtt_client()
            
            # 检查组件是否可用
            if not camera:
                logger.warning("catch指令执行失败: 相机未初始化或未连接")
                return {
                    "type": "catch_response",
                    "success": False,
                    "message": "相机未初始化或未连接",
                    "timestamp": time.time()
                }
            
            if not detector:
                logger.warning("catch指令执行失败: 检测器未初始化")
                return {
                    "type": "catch_response",
                    "success": False,
                    "message": "检测器未初始化",
                    "timestamp": time.time()
                }
            
            # 执行实际的catch逻辑
            try:
                logger.info("开始执行catch操作...")
                
                # TODO: 在这里添加具体的catch实现
                # 1. 获取相机图像
                # if hasattr(camera, 'get_fresh_frame'):
                #     image = camera.get_fresh_frame()
                #     if image is not None:
                #         logger.info("成功获取相机图像")
                #         
                #         # 2. 进行目标检测
                #         results = detector.detect(image)
                #         logger.info(f"检测完成，发现 {len(results)} 个目标")
                #         
                #         # 3. 可选：通过MQTT发布结果
                #         if mqtt_client and hasattr(mqtt_client, 'publish'):
                #             mqtt_client.publish("sickvision/detection/result", {
                #                 "client_id": client_id,
                #                 "results": results,
                #                 "timestamp": time.time()
                #             })
                #         
                #         return {
                #             "type": "catch_response",
                #             "success": True,
                #             "message": f"catch执行成功，检测到 {len(results)} 个目标",
                #             "results": results,
                #             "timestamp": time.time()
                #         }
                #     else:
                #         logger.error("无法获取相机图像")
                #         return {
                #             "type": "catch_response",
                #             "success": False,
                #             "message": "无法获取相机图像",
                #             "timestamp": time.time()
                #         }
                
                # 临时响应（在实现具体逻辑前）
                logger.info("catch指令已接收，功能正在开发中")
                return {
                    "type": "catch_response",
                    "success": True,
                    "message": "catch指令已接收，功能正在开发中",
                    "timestamp": time.time()
                }
                
            except Exception as e:
                logger.error(f"执行catch时出错: {e}")
                return {
                    "type": "catch_response",
                    "success": False,
                    "message": f"catch执行失败: {str(e)}",
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.error(f"处理catch指令失败: {e}")
            return {
                "type": "catch_response",
                "success": False,
                "message": f"catch指令处理失败: {str(e)}",
                "timestamp": time.time()
            }
    
    def handle_mqtt_message(self, mqtt_message):
        """
        处理MQTT消息 - 主要用于配置更新
        
        Args:
            mqtt_message: MQTT消息对象，包含topic、payload、qos等信息
        """
        logger = self.initializer.logger
        
        try:
            topic = mqtt_message.topic
            payload = mqtt_message.payload
            
            logger.info(f"收到MQTT消息: {topic}")
            logger.debug(f"消息内容: {payload}")
            
            # 根据主题处理不同的消息
            if topic == "sickvision/config/update":
                self._handle_mqtt_config_update(payload)
            elif topic == "sickvision/system/command":
                self._handle_mqtt_system_command(payload)
            else:
                logger.debug(f"未处理的MQTT主题: {topic}")
                
        except Exception as e:
            logger.error(f"处理MQTT消息时出错: {e}")
    
    def _handle_mqtt_config_update(self, payload):
        """
        处理配置更新
        
        Args:
            payload: 配置更新数据，包含要更新的配置项
        """
        logger = self.initializer.logger
        logger.info("收到MQTT配置更新命令")
        
        try:
            if not isinstance(payload, dict):
                error_msg = "配置数据格式错误，必须是字典格式"
                logger.error(error_msg)
                self._send_config_response(False, error_msg)
                return
            
            # 获取更新类型 # partial: 部分更新, full: 完整更新
            config_data = payload
            restart_required = payload.get("restart_required", False)
            
            if not config_data:
                error_msg = "配置数据为空"
                logger.error(error_msg)
                self._send_config_response(False, error_msg)
                return
            
            logger.info(f"开始更新配置")
            
            # 备份当前配置
            backup_success = self._backup_config()
            if not backup_success:
                error_msg = "配置备份失败"
                logger.error(error_msg)
                self._send_config_response(False, error_msg)
                return
            
            # 更新配置文件
            update_success = self._update_config_file(config_data)
            if not update_success:
                error_msg = "配置文件更新失败"
                logger.error(error_msg)
                # 恢复备份
                self._restore_config_backup()
                self._send_config_response(False, error_msg)
                return
            
            # 根据restart_required决定重启还是热重载
            if restart_required:
                logger.info("配置更新完成，准备重启系统...")
                self._send_config_response(True, "配置更新成功，系统将重启")
                # 延迟重启，先发送响应
                import threading
                threading.Timer(2.0, self._restart_system).start()
            else:
                # 尝试热重载
                reload_success = self._hot_reload_config(config_data)
                if reload_success:
                    logger.info("配置热重载成功")
                    self._send_config_response(True, "配置更新成功，已热重载")
                else:
                    logger.warning("配置热重载失败，建议重启系统")
                    self._send_config_response(True, "配置更新成功，但热重载失败，建议重启系统")
                
        except Exception as e:
            logger.error(f"处理配置更新时出错: {e}")
            # 尝试恢复备份
            self._restore_config_backup()
            self._send_config_response(False, f"配置更新失败: {str(e)}")
    
    def _backup_config(self) -> bool:
        """备份当前配置文件（最多保留10个备份）"""
        try:
            import shutil
            import os
            import glob
            from datetime import datetime
            
            config_path = self.config_path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{config_path}.backup_{timestamp}"
            
            # 创建备份
            shutil.copy2(config_path, backup_path)
            
            # 保存最新备份路径
            self.latest_backup = backup_path
            
            self.initializer.logger.info(f"配置文件已备份到: {backup_path}")
            
            # 清理旧备份文件，只保留最新的10个
            self._cleanup_old_backups(config_path, max_backups=10)
            
            return True
            
        except Exception as e:
            self.initializer.logger.error(f"配置备份失败: {e}")
            return False
    
    def _cleanup_old_backups(self, config_path: str, max_backups: int = 10):
        """清理旧的备份文件，只保留最新的指定数量"""
        try:
            import glob
            import os
            
            # 查找所有备份文件
            backup_pattern = f"{config_path}.backup_*"
            backup_files = glob.glob(backup_pattern)
            
            if len(backup_files) <= max_backups:
                return  # 备份文件数量未超过限制
            
            # 按文件修改时间排序（最新的在前）
            backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # 删除超出数量限制的旧备份文件
            files_to_delete = backup_files[max_backups:]
            deleted_count = 0
            
            for old_backup in files_to_delete:
                try:
                    os.remove(old_backup)
                    deleted_count += 1
                    self.initializer.logger.debug(f"已删除旧备份文件: {old_backup}")
                except Exception as e:
                    self.initializer.logger.warning(f"删除备份文件失败: {old_backup}, 错误: {e}")
            
            if deleted_count > 0:
                remaining_count = len(backup_files) - deleted_count
                self.initializer.logger.info(f"已清理 {deleted_count} 个旧备份文件，当前保留 {remaining_count} 个备份")
                
        except Exception as e:
            self.initializer.logger.error(f"清理备份文件时出错: {e}")
    
    def _restore_config_backup(self) -> bool:
        """恢复配置备份"""
        try:
            import os
            import shutil
            if hasattr(self, 'latest_backup') and os.path.exists(self.latest_backup):
                shutil.copy2(self.latest_backup, self.config_path)
                self.initializer.logger.info("配置文件已从备份恢复")
                return True
            return False
        except Exception as e:
            self.initializer.logger.error(f"配置恢复失败: {e}")
            return False
    
    def _update_config_file(self, config_data: dict) -> bool:
        """
        更新配置文件
        
        Args:
            config_data: 要更新的配置数据
            update_type: 更新类型 (partial/full)
        """
        try:
            import yaml
            # 部分更新配置
            # 读取当前配置
            with open(self.config_path, 'r', encoding='utf-8') as f:
                current_config = yaml.safe_load(f) or {}
            
            # 深度合并配置
            new_config = self._deep_merge_dict(current_config, config_data)
            
            # 写入新配置
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(new_config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            self.initializer.logger.info("配置文件更新成功")
            return True
            
        except Exception as e:
            self.initializer.logger.error(f"更新配置文件失败: {e}")
            return False
    
    def _deep_merge_dict(self, base: dict, update: dict) -> dict:
        """深度合并字典（忽略大小写）"""
        result = base.copy()
        
        # 创建基础字典的小写键映射
        base_lower_keys = {k.lower(): k for k in base.keys()}
        
        for key, value in update.items():
            # 查找匹配的键（忽略大小写）
            matching_key = base_lower_keys.get(key.lower(), key)
            
            if matching_key in result and isinstance(result[matching_key], dict) and isinstance(value, dict):
                # 递归合并字典
                result[matching_key] = self._deep_merge_dict(result[matching_key], value)
            else:
                # 如果找到了大小写不同的匹配键，使用原始键名
                if matching_key != key and matching_key in result:
                    # 删除旧键，使用新键
                    del result[matching_key]
                result[key] = value
        
        return result
    
    def _hot_reload_config(self, config_data: dict) -> bool:
        """
        热重载配置（不重启系统）
        
        Args:
            config_data: 新的配置数据
        """
        logger = self.initializer.logger
        
        try:
            logger.info("开始热重载配置...")
            
            # 重新加载配置管理器
            if not self.initializer.initialize_config():
                logger.error("重新加载配置管理器失败")
                return False
            
            success_count = 0
            total_components = 0
            
            # 根据配置变化决定需要重启的组件
            if "detectionServer" in config_data:
                total_components += 1
                if self._reload_tcp_server():
                    success_count += 1
                    logger.info("TCP服务器配置热重载成功")
                else:
                    logger.error("TCP服务器配置热重载失败")
            
            if "mqtt" in config_data:
                total_components += 1
                if self._reload_mqtt_client():
                    success_count += 1
                    logger.info("MQTT客户端配置热重载成功")
                else:
                    logger.error("MQTT客户端配置热重载失败")
            
            if "camera" in config_data:
                total_components += 1
                if self._reload_camera():
                    success_count += 1
                    logger.info("相机配置热重载成功")
                else:
                    logger.error("相机配置热重载失败")
            
            if "logging" in config_data:
                total_components += 1
                if self._reload_logging():
                    success_count += 1
                    logger.info("日志配置热重载成功")
                else:
                    logger.error("日志配置热重载失败")
            
            # 如果没有指定组件，说明可能是系统级配置，不需要重启组件
            if total_components == 0:
                logger.info("系统级配置更新，无需重启组件")
                return True
            
            # 判断热重载是否成功
            if success_count == total_components:
                logger.info(f"所有组件配置热重载成功 ({success_count}/{total_components})")
                return True
            else:
                logger.warning(f"部分组件配置热重载失败 ({success_count}/{total_components})")
                return False
                
        except Exception as e:
            logger.error(f"热重载配置时出错: {e}")
            return False
    
    def _reload_tcp_server(self) -> bool:
        """重启TCP服务器以应用新配置"""
        try:
            if self.initializer._restart_tcp_server():
                # 重新设置回调
                tcp_server = self.initializer.get_tcp_server()
                if tcp_server:
                    message_handler = self.initializer.create_tcp_message_handler(self.handle_catch)
                    tcp_server.set_message_callback(message_handler)
                return True
            return False
        except Exception:
            return False
    
    def _reload_mqtt_client(self) -> bool:
        """重启MQTT客户端以应用新配置"""
        try:
            if self.initializer._restart_mqtt():
                # 消息回调会自动恢复，无需手动设置
                return True
            return False
        except Exception:
            return False
    
    def _reload_camera(self) -> bool:
        """重启相机以应用新配置"""
        try:
            return self.initializer._restart_camera()
        except Exception:
            return False
    
    def _reload_logging(self) -> bool:
        """重新配置日志系统"""
        try:
            return self.initializer.initialize_logging()
        except Exception:
            return False
    
    def _restart_system(self):
        """重启整个系统"""
        logger = self.initializer.logger
        logger.info("开始重启系统...")
        
        # 设置重启标志
        self.running = False
        self.restart_on_failure = True
    
    def _send_config_response(self, success: bool, message: str):
        """发送配置更新响应"""
        try:
            mqtt_client = self.initializer.get_mqtt_client()
            if mqtt_client:
                response = {
                    "success": success,
                    "message": message,
                    "timestamp": time.time()
                }
                mqtt_client.publish("sickvision/config/reload/response", response)
                
        except Exception as e:
            self.initializer.logger.error(f"发送配置响应失败: {e}")
    
    def _handle_mqtt_system_command(self, payload):
        """处理系统命令"""
        logger = self.initializer.logger
        logger.info(f"收到MQTT系统命令: {payload}")
        
        try:
            if not isinstance(payload, dict):
                logger.error("系统命令格式错误")
                return
            
            command = payload.get("command")
            
            if command == "restart":
                logger.info("收到系统重启命令")
                self._restart_system()
                    
            elif command == "get_config":
                # 获取当前配置
                try:
                    import yaml
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        current_config = yaml.safe_load(f)
                    
                    mqtt_client = self.initializer.get_mqtt_client()
                    if mqtt_client:
                        response = {
                            "command": "get_config",
                            "config": current_config,
                            "timestamp": time.time()
                        }
                        mqtt_client.publish("PI/robot/config", response)
                        
                except Exception as e:
                    logger.error(f"获取配置失败: {e}")
                    
            else:
                logger.warning(f"未知系统命令: {command}")
                
        except Exception as e:
            logger.error(f"处理系统命令时出错: {e}")
    
    def _check_system_health(self, logger):
        """检查系统健康状态"""
        try:
            if self.initializer.monitor:
                status = self.initializer.get_system_status()
                
                # 记录系统状态
                healthy_components = sum(1 for comp in status.get("components", {}).values() if comp.get("healthy", False))
                total_components = len(status.get("components", {}))
                
                # 检查是否有组件不健康
                unhealthy = [name for name, comp in status.get("components", {}).items() if not comp.get("healthy", False)]
                
                # 只在有组件异常时记录日志
                if unhealthy:
                    logger.warning(f"系统健康检查: {healthy_components}/{total_components} 组件正常，不健康的组件: {', '.join(unhealthy)}")
                # 如果所有组件正常，不记录日志（静默运行）
                
                # 如果TCP服务器运行正常，向客户端发送系统状态
                tcp_server = self.initializer.get_tcp_server()
                if tcp_server and tcp_server.is_connected and len(tcp_server.clients) > 0:
                    status_msg = {
                        "type": "system_health_report",
                        "healthy_components": healthy_components,
                        "total_components": total_components,
                        "unhealthy_components": unhealthy,
                        "timestamp": time.time()
                    }
                    tcp_server.broadcast_message(status_msg)
                
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
        logger = initializer.get_logger()
        camera = initializer.get_camera()
        mqtt_client = initializer.get_mqtt_client()
        detector = initializer.get_detector()
        
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
    