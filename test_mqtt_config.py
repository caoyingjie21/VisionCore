#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MQTT配置管理测试脚本
用于测试VisionCore的MQTT配置更新功能
"""

import json
import time
import threading
from typing import Dict, Any

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("请先安装paho-mqtt库: pip install paho-mqtt")


class MqttConfigTester:
    """MQTT配置测试器"""
    
    def __init__(self, broker_host="localhost", broker_port=1883, client_id="config_tester"):
        """
        初始化MQTT测试客户端
        
        Args:
            broker_host: MQTT代理地址
            broker_port: MQTT代理端口
            client_id: 客户端ID
        """
        if not MQTT_AVAILABLE:
            raise ImportError("paho-mqtt库未安装")
        
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        
        # 创建MQTT客户端
        self.client = mqtt.Client(client_id=client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        # 状态
        self.connected = False
        self.responses = {}
        
        # 订阅响应主题
        self.response_topics = [
            "sickvision/config/reload/response",
            "sickvision/system/status",
            "sickvision/system/config"
        ]
    
    def connect(self) -> bool:
        """连接到MQTT代理"""
        try:
            print(f"正在连接MQTT代理: {self.broker_host}:{self.broker_port}")
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            
            # 等待连接建立
            timeout = 10
            while not self.connected and timeout > 0:
                time.sleep(0.5)
                timeout -= 0.5
            
            if self.connected:
                print("MQTT连接成功")
                return True
            else:
                print("MQTT连接超时")
                return False
                
        except Exception as e:
            print(f"MQTT连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()
            print("MQTT连接已断开")
    
    def _on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            self.connected = True
            print("MQTT连接建立")
            
            # 订阅响应主题
            for topic in self.response_topics:
                client.subscribe(topic, qos=2)
                print(f"已订阅主题: {topic}")
        else:
            print(f"MQTT连接失败，返回码: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """断开连接回调"""
        self.connected = False
        if rc != 0:
            print(f"MQTT意外断开，返回码: {rc}")
        else:
            print("MQTT正常断开")
    
    def _on_message(self, client, userdata, msg):
        """消息接收回调"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode('utf-8'))
            
            print(f"\n=== 收到响应 ===")
            print(f"主题: {topic}")
            print(f"内容: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            print("=" * 20)
            
            # 保存响应用于验证
            self.responses[topic] = payload
            
        except Exception as e:
            print(f"处理响应消息时出错: {e}")
    
    def send_config_update(self, config_data: Dict[str, Any], update_type: str = "partial", restart_required: bool = False):
        """
        发送配置更新
        
        Args:
            config_data: 配置数据
            update_type: 更新类型 ("partial" 或 "full")
            restart_required: 是否需要重启
        """
        if not self.connected:
            print("未连接到MQTT代理")
            return False
        
        message = {
            "type": update_type,
            "config": config_data,
            "restart_required": restart_required
        }
        
        topic = "sickvision/config/update"
        
        print(f"\n=== 发送配置更新 ===")
        print(f"主题: {topic}")
        print(f"类型: {update_type}")
        print(f"重启: {restart_required}")
        print(f"配置: {json.dumps(config_data, indent=2, ensure_ascii=False)}")
        
        try:
            result = self.client.publish(topic, json.dumps(message), qos=2)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print("配置更新消息发送成功")
                return True
            else:
                print(f"配置更新消息发送失败: {result.rc}")
                return False
        except Exception as e:
            print(f"发送配置更新失败: {e}")
            return False
    
    def send_system_command(self, command: str, **kwargs):
        """
        发送系统命令
        
        Args:
            command: 命令名称
            **kwargs: 命令参数
        """
        if not self.connected:
            print("未连接到MQTT代理")
            return False
        
        message = {
            "command": command,
            **kwargs
        }
        
        topic = "sickvision/system/command"
        
        print(f"\n=== 发送系统命令 ===")
        print(f"主题: {topic}")
        print(f"命令: {json.dumps(message, indent=2, ensure_ascii=False)}")
        
        try:
            result = self.client.publish(topic, json.dumps(message), qos=2)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print("系统命令发送成功")
                return True
            else:
                print(f"系统命令发送失败: {result.rc}")
                return False
        except Exception as e:
            print(f"发送系统命令失败: {e}")
            return False
    
    def wait_for_response(self, topic: str, timeout: int = 10) -> Dict[str, Any]:
        """
        等待特定主题的响应
        
        Args:
            topic: 主题名称
            timeout: 超时时间（秒）
            
        Returns:
            响应消息或None
        """
        print(f"等待响应: {topic} (超时: {timeout}秒)")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if topic in self.responses:
                response = self.responses.pop(topic)
                print(f"收到响应: {topic}")
                return response
            time.sleep(0.1)
        
        print(f"等待响应超时: {topic}")
        return None
    
    # 预定义测试用例
    def test_tcp_config_update(self):
        """测试TCP服务器配置更新"""
        print("\n" + "="*50)
        print("测试TCP服务器配置更新")
        print("="*50)
        
        config = {
            "tcp_server": {
                "port": 9999,
                "max_connections": 20,
                "heartbeat_interval": 60
            }
        }
        
        self.send_config_update(config, "partial", False)
        response = self.wait_for_response("sickvision/config/reload/response", 15)
        
        if response:
            if response.get("success"):
                print("✅ TCP配置更新成功")
            else:
                print(f"❌ TCP配置更新失败: {response.get('message')}")
        else:
            print("❌ 未收到TCP配置更新响应")
    
    def test_camera_config_update(self):
        """测试相机配置更新"""
        print("\n" + "="*50)
        print("测试相机配置更新")
        print("="*50)
        
        config = {
            "camera": {
                "enabled": True,
                "connection": {
                    "ip": "192.168.1.102",
                    "port": 2122,
                    "timeout": 15
                }
            }
        }
        
        self.send_config_update(config, "partial", False)
        response = self.wait_for_response("sickvision/config/reload/response", 15)
        
        if response:
            if response.get("success"):
                print("✅ 相机配置更新成功")
            else:
                print(f"❌ 相机配置更新失败: {response.get('message')}")
        else:
            print("❌ 未收到相机配置更新响应")
    
    def test_logging_config_update(self):
        """测试日志配置更新"""
        print("\n" + "="*50)
        print("测试日志配置更新")
        print("="*50)
        
        config = {
            "logging": {
                "level": "DEBUG",
                "file": {
                    "enabled": True,
                    "backup_count": 50
                }
            }
        }
        
        self.send_config_update(config, "partial", False)
        response = self.wait_for_response("sickvision/config/reload/response", 15)
        
        if response:
            if response.get("success"):
                print("✅ 日志配置更新成功")
            else:
                print(f"❌ 日志配置更新失败: {response.get('message')}")
        else:
            print("❌ 未收到日志配置更新响应")
    
    def test_system_restart(self):
        """测试系统重启命令"""
        print("\n" + "="*50)
        print("测试系统重启命令")
        print("="*50)
        
        config = {
            "system": {
                "debug": True,
                "max_workers": 8
            }
        }
        
        self.send_config_update(config, "partial", True)  # 需要重启
        response = self.wait_for_response("sickvision/config/reload/response", 10)
        
        if response:
            if response.get("success"):
                print("✅ 系统重启配置更新成功")
            else:
                print(f"❌ 系统重启配置更新失败: {response.get('message')}")
        else:
            print("❌ 未收到系统重启配置更新响应")
    
    def test_get_system_status(self):
        """测试获取系统状态"""
        print("\n" + "="*50)
        print("测试获取系统状态")
        print("="*50)
        
        self.send_system_command("get_status")
        response = self.wait_for_response("sickvision/system/status", 10)
        
        if response:
            print("✅ 获取系统状态成功")
            print(f"状态信息: {json.dumps(response, indent=2, ensure_ascii=False)}")
        else:
            print("❌ 未收到系统状态响应")
    
    def test_get_current_config(self):
        """测试获取当前配置"""
        print("\n" + "="*50)
        print("测试获取当前配置")
        print("="*50)
        
        self.send_system_command("get_config")
        response = self.wait_for_response("sickvision/system/config", 10)
        
        if response:
            print("✅ 获取当前配置成功")
            config = response.get("config", {})
            print(f"配置项数量: {len(config)}")
            print(f"主要配置: {list(config.keys())}")
        else:
            print("❌ 未收到当前配置响应")
    
    def test_invalid_config(self):
        """测试无效配置更新"""
        print("\n" + "="*50)
        print("测试无效配置更新")
        print("="*50)
        
        # 发送空配置
        config = {}
        
        self.send_config_update(config, "partial", False)
        response = self.wait_for_response("sickvision/config/reload/response", 10)
        
        if response:
            if not response.get("success"):
                print("✅ 正确拒绝了无效配置")
                print(f"错误信息: {response.get('message')}")
            else:
                print("❌ 错误接受了无效配置")
        else:
            print("❌ 未收到无效配置响应")
    
    def run_all_tests(self):
        """运行所有测试"""
        if not self.connect():
            print("无法连接MQTT代理，测试中止")
            return
        
        try:
            print("开始MQTT配置管理功能测试...")
            time.sleep(2)  # 等待订阅生效
            
            # 运行各项测试
            self.test_get_current_config()
            time.sleep(2)
            
            self.test_get_system_status()
            time.sleep(2)
            
            self.test_tcp_config_update()
            time.sleep(3)
            
            self.test_camera_config_update()
            time.sleep(3)
            
            self.test_logging_config_update()
            time.sleep(3)
            
            self.test_invalid_config()
            time.sleep(3)
            
            # 最后测试重启（可选）
            restart = input("\n是否测试系统重启功能? (y/N): ").lower().strip()
            if restart == 'y':
                self.test_system_restart()
            
            print("\n" + "="*50)
            print("所有测试完成！")
            print("="*50)
            
        finally:
            self.disconnect()
    
    def interactive_mode(self):
        """交互模式"""
        if not self.connect():
            print("无法连接MQTT代理")
            return
        
        print("\n=== MQTT配置管理交互模式 ===")
        print("可用命令:")
        print("  1. tcp       - 更新TCP服务器配置")
        print("  2. camera    - 更新相机配置")
        print("  3. logging   - 更新日志配置")
        print("  4. status    - 获取系统状态")
        print("  5. config    - 获取当前配置")
        print("  6. restart   - 系统重启")
        print("  7. custom    - 自定义配置")
        print("  0. quit      - 退出")
        print()
        
        try:
            while True:
                cmd = input("请选择命令 (0-7): ").strip()
                
                if cmd == "0" or cmd.lower() == "quit":
                    break
                elif cmd == "1":
                    self.test_tcp_config_update()
                elif cmd == "2":
                    self.test_camera_config_update()
                elif cmd == "3":
                    self.test_logging_config_update()
                elif cmd == "4":
                    self.test_get_system_status()
                elif cmd == "5":
                    self.test_get_current_config()
                elif cmd == "6":
                    self.test_system_restart()
                elif cmd == "7":
                    self._custom_config_test()
                else:
                    print("无效命令，请输入0-7")
                
                print()
                
        except KeyboardInterrupt:
            print("\n接收到中断信号")
        finally:
            self.disconnect()
    
    def _custom_config_test(self):
        """自定义配置测试"""
        print("自定义配置更新")
        print("请输入JSON格式的配置数据:")
        
        try:
            config_str = input("配置JSON: ")
            config_data = json.loads(config_str)
            
            update_type = input("更新类型 (partial/full) [partial]: ").strip() or "partial"
            restart_str = input("是否需要重启 (y/N): ").strip().lower()
            restart_required = restart_str == 'y'
            
            self.send_config_update(config_data, update_type, restart_required)
            response = self.wait_for_response("sickvision/config/reload/response", 15)
            
            if response:
                if response.get("success"):
                    print("✅ 自定义配置更新成功")
                else:
                    print(f"❌ 自定义配置更新失败: {response.get('message')}")
            else:
                print("❌ 未收到自定义配置更新响应")
                
        except json.JSONDecodeError:
            print("❌ 无效的JSON格式")
        except Exception as e:
            print(f"❌ 自定义配置测试失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MQTT配置管理测试脚本")
    parser.add_argument("--host", default="localhost", help="MQTT代理地址")
    parser.add_argument("--port", type=int, default=1883, help="MQTT代理端口")
    parser.add_argument("--mode", choices=["test", "interactive"], default="interactive", 
                       help="运行模式: test=自动测试, interactive=交互模式")
    
    args = parser.parse_args()
    
    try:
        tester = MqttConfigTester(args.host, args.port)
        
        if args.mode == "test":
            tester.run_all_tests()
        else:
            tester.interactive_mode()
            
    except KeyboardInterrupt:
        print("\n程序被中断")
    except Exception as e:
        print(f"程序出错: {e}")


if __name__ == "__main__":
    main() 