#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MQTT快速测试脚本
用于快速测试MQTT配置更新功能
"""

import json
import time

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("请先安装paho-mqtt库: pip install paho-mqtt")
    exit(1)


def on_connect(client, userdata, flags, rc):
    """连接回调"""
    if rc == 0:
        print("✅ MQTT连接成功")
        # 订阅响应主题
        client.subscribe("sickvision/config/reload/response", qos=2)
        client.subscribe("sickvision/system/status", qos=2)
        client.subscribe("sickvision/system/config", qos=2)
        print("📡 已订阅响应主题")
    else:
        print(f"❌ MQTT连接失败，返回码: {rc}")


def on_message(client, userdata, msg):
    """消息接收回调"""
    topic = msg.topic
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        print(f"\n📨 收到响应:")
        print(f"   主题: {topic}")
        print(f"   内容: {json.dumps(payload, indent=4, ensure_ascii=False)}")
        print("-" * 50)
    except Exception as e:
        print(f"❌ 解析响应失败: {e}")


def send_config_update(client, config_data, restart_required=False):
    """发送配置更新"""
    message = {
        "type": "partial",
        "config": config_data,
        "restart_required": restart_required
    }
    
    print(f"\n🚀 发送配置更新:")
    print(f"   配置: {json.dumps(config_data, indent=4, ensure_ascii=False)}")
    print(f"   重启: {restart_required}")
    
    result = client.publish("sickvision/config/update", json.dumps(message), qos=2)
    
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print("✅ 配置更新消息发送成功")
    else:
        print(f"❌ 配置更新消息发送失败: {result.rc}")


def send_system_command(client, command):
    """发送系统命令"""
    message = {"command": command}
    
    print(f"\n🚀 发送系统命令: {command}")
    
    result = client.publish("sickvision/system/command", json.dumps(message), qos=2)
    
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print("✅ 系统命令发送成功")
    else:
        print(f"❌ 系统命令发送失败: {result.rc}")


def main():
    """主函数"""
    print("🔧 MQTT配置管理快速测试")
    print("=" * 50)
    
    # 创建MQTT客户端
    client = mqtt.Client(client_id="quick_test")
    client.on_connect = on_connect
    client.on_message = on_message
    
    # 连接到MQTT代理
    try:
        print("🔗 正在连接MQTT代理...")
        client.connect("localhost", 1883, 60)
        client.loop_start()
        
        # 等待连接建立
        time.sleep(2)
        
        # # 测试1: 获取当前配置
        # print("\n" + "="*50)
        # print("📋 测试1: 获取当前配置")
        # send_system_command(client, "get_config")
        # time.sleep(3)
        
        # # 测试2: 获取系统状态
        # print("\n" + "="*50)
        # print("📊 测试2: 获取系统状态")
        # send_system_command(client, "get_status")
        # time.sleep(3)
        
        # 测试3: 更新TCP服务器配置
        print("\n" + "="*50)
        print("🔧 测试3: 更新TCP服务器配置")
        tcp_config = {
            "tcp_server": {
                "port": 9999,
                "max_connections": 15
            }
        }
        send_config_update(client, tcp_config)
        time.sleep(5)
        
        # # 测试4: 更新日志配置
        # print("\n" + "="*50)
        # print("📝 测试4: 更新日志配置")
        # log_config = {
        #     "logging": {
        #         "level": "DEBUG",
        #         "console": {
        #             "enabled": True
        #         }
        #     }
        # }
        # send_config_update(client, log_config)
        # time.sleep(5)
        
        # # 测试5: 测试无效配置
        # print("\n" + "="*50)
        # print("❌ 测试5: 测试无效配置")
        # invalid_config = {}
        # send_config_update(client, invalid_config)
        # time.sleep(3)
        
        # print("\n" + "="*50)
        # print("✅ 快速测试完成！")
        # print("如需更详细的测试，请使用: python test_mqtt_config.py")
        
    except KeyboardInterrupt:
        print("\n⏹️ 测试被中断")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    finally:
        client.loop_stop()
        client.disconnect()
        print("🔌 MQTT连接已断开")


if __name__ == "__main__":
    main() 