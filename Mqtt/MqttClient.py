#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MQTT客户端模块 - MqttClient
提供基本的MQTT连接、订阅、发布功能
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("Warning: paho-mqtt not available, MQTT功能已禁用")


@dataclass
class MqttMessage:
    """MQTT消息数据类"""
    topic: str
    payload: Any
    qos: int
    retain: bool
    timestamp: datetime


class MqttConnectionError(Exception):
    """MQTT连接错误"""
    pass


class MqttClient:
    """MQTT客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化MQTT客户端
        
        Args:
            config: MQTT配置字典
        """
        if not MQTT_AVAILABLE:
            raise ImportError("paho-mqtt库未安装，请先安装: pip install paho-mqtt")
        
        self.config = config
        self.connection_config = config.get("connection", {})
        self.qos_config = config.get("qos", {})
        self.topics_config = config.get("topics", {})
        self.message_config = config.get("message", {})
        
        # MQTT客户端
        self.client: Optional[mqtt.Client] = None
        self.is_connected = False
        
        # 消息回调
        self._message_callbacks: Dict[str, List[Callable[[MqttMessage], None]]] = {}
        self._general_callback: Optional[Callable[[MqttMessage], None]] = None
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化客户端
        self._init_client()
    
    def _init_client(self):
        """初始化MQTT客户端"""
        try:
            # 创建客户端实例
            client_id = self.connection_config.get("client_id", "sickvision_client")
            self.client = mqtt.Client(client_id=client_id)
            
            # 设置认证
            username = self.connection_config.get("username")
            password = self.connection_config.get("password")
            if username and password:
                self.client.username_pw_set(username, password)
            
            # 设置SSL
            if self.connection_config.get("use_ssl", False):
                self.client.tls_set()
            
            # 设置回调函数
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            self.client.on_subscribe = self._on_subscribe
            self.client.on_publish = self._on_publish
            
            self.logger.info("MQTT客户端初始化成功")
            
        except Exception as e:
            self.logger.error(f"MQTT客户端初始化失败: {e}")
            raise MqttConnectionError(f"客户端初始化失败: {e}")
    
    async def connect(self) -> bool:
        """
        连接到MQTT broker
        
        Returns:
            连接是否成功
        """
        try:
            if self.is_connected:
                self.logger.warning("MQTT客户端已连接")
                return True
            
            broker_host = self.connection_config.get("broker_host", "localhost")
            broker_port = self.connection_config.get("broker_port", 1883)
            keepalive = self.connection_config.get("keepalive", 60)
            
            self.logger.info(f"连接到MQTT broker: {broker_host}:{broker_port}")
            
            # 连接到broker
            result = self.client.connect(broker_host, broker_port, keepalive)
            
            if result == mqtt.MQTT_ERR_SUCCESS:
                # 启动网络循环
                self.client.loop_start()
                
                # 等待连接完成
                await self._wait_for_connection()
                
                self.logger.info("MQTT连接成功")
                return True
            else:
                error_msg = f"MQTT连接失败: {mqtt.error_string(result)}"
                self.logger.error(error_msg)
                raise MqttConnectionError(error_msg)
                
        except Exception as e:
            self.logger.error(f"MQTT连接异常: {e}")
            return False
    
    async def disconnect(self):
        """断开MQTT连接"""
        try:
            if self.client and self.is_connected:
                self.client.loop_stop()
                self.client.disconnect()
                self.is_connected = False
                self.logger.info("MQTT连接已断开")
        except Exception as e:
            self.logger.error(f"MQTT断开连接失败: {e}")
    
    async def _wait_for_connection(self, timeout: int = 10):
        """等待连接建立"""
        for _ in range(timeout * 10):  # 100ms间隔检查
            if self.is_connected:
                return
            await asyncio.sleep(0.1)
        
        raise MqttConnectionError("连接超时")
    
    def subscribe(self, topic: str, qos: Optional[int] = None, callback: Optional[Callable[[MqttMessage], None]] = None) -> bool:
        """
        订阅主题
        
        Args:
            topic: 主题名称
            qos: 服务质量等级
            callback: 消息回调函数
            
        Returns:
            订阅是否成功
        """
        try:
            if not self.is_connected:
                self.logger.error("MQTT未连接，无法订阅")
                return False
            
            if qos is None:
                qos = self.qos_config.get("subscribe", 1)
            
            # 执行订阅
            result, _ = self.client.subscribe(topic, qos)
            
            if result == mqtt.MQTT_ERR_SUCCESS:
                # 注册回调
                if callback:
                    self.add_topic_callback(topic, callback)
                
                self.logger.info(f"订阅主题成功: {topic} (QoS: {qos})")
                return True
            else:
                self.logger.error(f"订阅主题失败: {topic}, 错误: {mqtt.error_string(result)}")
                return False
                
        except Exception as e:
            self.logger.error(f"订阅主题异常: {e}")
            return False
    
    def unsubscribe(self, topic: str) -> bool:
        """
        取消订阅主题
        
        Args:
            topic: 主题名称
            
        Returns:
            取消订阅是否成功
        """
        try:
            if not self.is_connected:
                self.logger.error("MQTT未连接，无法取消订阅")
                return False
            
            result, _ = self.client.unsubscribe(topic)
            
            if result == mqtt.MQTT_ERR_SUCCESS:
                # 移除回调
                if topic in self._message_callbacks:
                    del self._message_callbacks[topic]
                
                self.logger.info(f"取消订阅成功: {topic}")
                return True
            else:
                self.logger.error(f"取消订阅失败: {topic}, 错误: {mqtt.error_string(result)}")
                return False
                
        except Exception as e:
            self.logger.error(f"取消订阅异常: {e}")
            return False
    
    def publish(self, topic: str, payload: Any, qos: Optional[int] = None, retain: Optional[bool] = None) -> bool:
        """
        发布消息
        
        Args:
            topic: 主题名称
            payload: 消息内容
            qos: 服务质量等级
            retain: 是否保留消息
            
        Returns:
            发布是否成功
        """
        try:
            if not self.is_connected:
                self.logger.error("MQTT未连接，无法发布消息")
                return False
            
            if qos is None:
                qos = self.qos_config.get("publish", 1)
            
            if retain is None:
                retain = self.message_config.get("retain", False)
            
            # 处理消息内容
            if isinstance(payload, (dict, list)):
                payload = json.dumps(payload, ensure_ascii=False)
            elif not isinstance(payload, (str, bytes)):
                payload = str(payload)
            
            # 检查消息大小
            max_size = self.message_config.get("max_payload_size", 1048576)  # 1MB
            if len(payload.encode('utf-8')) > max_size:
                self.logger.error(f"消息大小超过限制: {len(payload)} > {max_size}")
                return False
            
            # 发布消息
            result = self.client.publish(topic, payload, qos, retain)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.logger.debug(f"消息发布成功: {topic}")
                return True
            else:
                self.logger.error(f"消息发布失败: {topic}, 错误: {mqtt.error_string(result.rc)}")
                return False
                
        except Exception as e:
            self.logger.error(f"消息发布异常: {e}")
            return False
    
    def add_topic_callback(self, topic: str, callback: Callable[[MqttMessage], None]):
        """
        为特定主题添加消息回调
        
        Args:
            topic: 主题名称（支持通配符）
            callback: 回调函数
        """
        if topic not in self._message_callbacks:
            self._message_callbacks[topic] = []
        
        if callback not in self._message_callbacks[topic]:
            self._message_callbacks[topic].append(callback)
            self.logger.debug(f"添加主题回调: {topic}")
    
    def remove_topic_callback(self, topic: str, callback: Callable[[MqttMessage], None]):
        """
        移除特定主题的消息回调
        
        Args:
            topic: 主题名称
            callback: 回调函数
        """
        if topic in self._message_callbacks and callback in self._message_callbacks[topic]:
            self._message_callbacks[topic].remove(callback)
            self.logger.debug(f"移除主题回调: {topic}")
    
    def set_general_callback(self, callback: Callable[[MqttMessage], None]):
        """
        设置通用消息回调（处理所有消息）
        
        Args:
            callback: 回调函数
        """
        self._general_callback = callback
        self.logger.debug("设置通用消息回调")
    
    def _on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            self.is_connected = True
            self.logger.info("MQTT连接建立成功")
            
            # 自动订阅配置中的主题
            self._auto_subscribe()
        else:
            self.is_connected = False
            error_msg = f"MQTT连接失败，返回码: {rc}"
            self.logger.error(error_msg)
    
    def _on_disconnect(self, client, userdata, rc):
        """断开连接回调"""
        self.is_connected = False
        if rc != 0:
            self.logger.warning(f"MQTT意外断开连接，返回码: {rc}")
        else:
            self.logger.info("MQTT正常断开连接")
    
    def _on_message(self, client, userdata, msg):
        """消息接收回调"""
        try:
            # 解析消息
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            # 尝试解析JSON
            try:
                payload = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                # 保持原始字符串格式
                pass
            
            # 创建消息对象
            mqtt_msg = MqttMessage(
                topic=topic,
                payload=payload,
                qos=msg.qos,
                retain=msg.retain,
                timestamp=datetime.now()
            )
            
            self.logger.debug(f"收到消息: {topic}")
            
            # 调用特定主题回调
            message_handled = False
            for topic_pattern, callbacks in self._message_callbacks.items():
                if self._topic_matches(topic, topic_pattern):
                    for callback in callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                asyncio.create_task(callback(mqtt_msg))
                            else:
                                callback(mqtt_msg)
                            message_handled = True
                        except Exception as e:
                            self.logger.error(f"主题回调执行失败: {e}")
            
            # 调用通用回调
            if self._general_callback and not message_handled:
                try:
                    if asyncio.iscoroutinefunction(self._general_callback):
                        asyncio.create_task(self._general_callback(mqtt_msg))
                    else:
                        self._general_callback(mqtt_msg)
                except Exception as e:
                    self.logger.error(f"通用回调执行失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"消息处理异常: {e}")
    
    def _on_subscribe(self, client, userdata, mid, granted_qos):
        """订阅成功回调"""
        self.logger.debug(f"订阅确认，消息ID: {mid}, QoS: {granted_qos}")
    
    def _on_publish(self, client, userdata, mid):
        """发布成功回调"""
        self.logger.debug(f"发布确认，消息ID: {mid}")
    
    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """检查主题是否匹配模式（支持通配符）"""
        # 简单的通配符匹配实现
        if pattern == topic:
            return True
        
        # 支持 + 和 # 通配符
        pattern_parts = pattern.split('/')
        topic_parts = topic.split('/')
        
        i = 0
        j = 0
        
        while i < len(pattern_parts) and j < len(topic_parts):
            if pattern_parts[i] == '#':
                return True  # # 匹配剩余所有层级
            elif pattern_parts[i] == '+':
                # + 匹配一个层级
                i += 1
                j += 1
            elif pattern_parts[i] == topic_parts[j]:
                i += 1
                j += 1
            else:
                return False
        
        return i == len(pattern_parts) and j == len(topic_parts)
    
    def _auto_subscribe(self):
        """自动订阅配置中的主题"""
        try:
            subscribe_topics = self.topics_config.get("subscribe", {})
            
            for topic_name, topic_pattern in subscribe_topics.items():
                self.subscribe(topic_pattern)
                self.logger.info(f"自动订阅: {topic_name} -> {topic_pattern}")
                
        except Exception as e:
            self.logger.error(f"自动订阅失败: {e}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """获取连接状态信息"""
        return {
            "connected": self.is_connected,
            "broker_host": self.connection_config.get("broker_host"),
            "broker_port": self.connection_config.get("broker_port"),
            "client_id": self.connection_config.get("client_id"),
            "subscribed_topics": list(self._message_callbacks.keys()),
            "timestamp": datetime.now().isoformat()
        }


# 便捷函数
def create_mqtt_client(config: Dict[str, Any]) -> MqttClient:
    """
    创建MQTT客户端的便捷函数
    
    Args:
        config: MQTT配置
        
    Returns:
        MQTT客户端实例
    """
    return MqttClient(config)


if __name__ == "__main__":
    # 测试代码
    import asyncio
    
    async def test_mqtt_client():
        # 测试配置
        test_config = {
            "connection": {
                "broker_host": "localhost",
                "broker_port": 1883,
                "keepalive": 60,
                "client_id": "test_client"
            },
            "qos": {
                "subscribe": 1,
                "publish": 1
            },
            "topics": {
                "subscribe": {
                    "test_topic": "test/topic"
                }
            }
        }
        
        # 创建客户端
        mqtt_client = MqttClient(test_config)
        
        # 设置消息回调
        def on_message(msg: MqttMessage):
            print(f"收到消息: {msg.topic} -> {msg.payload}")
        
        mqtt_client.set_general_callback(on_message)
        
        try:
            # 连接
            if await mqtt_client.connect():
                print("MQTT连接成功")
                
                # 订阅主题
                mqtt_client.subscribe("test/topic")
                
                # 发布消息
                mqtt_client.publish("test/topic", {"message": "Hello MQTT!"})
                
                # 等待一段时间
                await asyncio.sleep(2)
                
                # 断开连接
                await mqtt_client.disconnect()
                print("测试完成")
            else:
                print("MQTT连接失败")
                
        except Exception as e:
            print(f"测试异常: {e}")
    
    # 运行测试
    if MQTT_AVAILABLE:
        asyncio.run(test_mqtt_client())
    else:
        print("MQTT库不可用，跳过测试")
