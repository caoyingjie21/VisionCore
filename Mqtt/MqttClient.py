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

# 导入MQTTResponse和VisionCoreCommands
try:
    from ClassModel.MqttResponse import MQTTResponse
    from SystemEnums.VisionCoreCommands import VisionCoreCommands, MessageType
    MQTT_RESPONSE_AVAILABLE = True
except ImportError:
    MQTT_RESPONSE_AVAILABLE = False
    print("Warning: MQTTResponse not available, response methods will be disabled")


@dataclass
class MqttMessage:
    """MQTT消息数据类"""
    topic: str
    payload: Any
    qos: int
    retain: bool
    timestamp: Optional[datetime] = None


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
            # 如果已有客户端，先清理
            if self.client:
                try:
                    self.client.loop_stop()
                    self.client.disconnect()
                except:
                    pass
                self.client = None
            
            # 创建客户端实例，使用唯一的客户端ID
            base_client_id = self.connection_config.get("client_id", "sickvision_client")
            import time
            unique_client_id = f"{base_client_id}_{int(time.time() * 1000) % 100000}"
            
            self.client = mqtt.Client(client_id=unique_client_id)
            self.logger.info(f"创建MQTT客户端，ID: {unique_client_id}")
            
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
            
            # 重置连接状态
            self.is_connected = False
            
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
            # 如果已经连接，直接返回
            if self.is_connected and self.client:
                self.logger.info("MQTT客户端已连接")
                return True
            
            # 如果连接状态不一致，重新初始化客户端
            if not self.client or self.is_connected != getattr(self.client, '_state', None):
                self.logger.info("重新初始化MQTT客户端...")
                self._init_client()
            
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
                return False
                
        except Exception as e:
            self.logger.error(f"MQTT连接异常: {e}")
            return False
    
    async def disconnect(self):
        """断开MQTT连接"""
        try:
            if self.client:
                # 先设置连接状态为False，防止其他操作
                self.is_connected = False
                
                # 停止网络循环
                if hasattr(self.client, '_thread') and self.client._thread and self.client._thread.is_alive():
                    self.client.loop_stop()
                    # 等待循环线程结束
                    await asyncio.sleep(0.5)
                
                # 断开连接
                try:
                    self.client.disconnect()
                    # 等待断开完成
                    await asyncio.sleep(0.2)
                except Exception as e:
                    self.logger.warning(f"断开MQTT连接时出现异常: {e}")
                
                # 清理回调
                self.client.on_connect = None
                self.client.on_disconnect = None
                self.client.on_message = None
                self.client.on_subscribe = None
                self.client.on_publish = None
                
                self.logger.info("MQTT连接已断开并清理完成")
            else:
                self.logger.debug("MQTT客户端未初始化，无需断开")
                
        except Exception as e:
            self.logger.error(f"MQTT断开连接失败: {e}")
        finally:
            # 确保状态被重置
            self.is_connected = False
    
    async def _wait_for_connection(self, timeout: int = 10):
        """等待连接建立"""
        for _ in range(timeout * 10):  # 100ms间隔检查
            if self.is_connected:
                return
            await asyncio.sleep(0.1)
        
        raise MqttConnectionError("连接超时")
    
    async def force_reinit(self):
        """强制重新初始化MQTT客户端（用于解决连接冲突）"""
        try:
            self.logger.info("强制重新初始化MQTT客户端...")
            
            # 完全断开并清理
            await self.disconnect()
            
            # 额外等待，确保资源完全释放
            await asyncio.sleep(1.0)
            
            # 清除所有回调
            self._message_callbacks.clear()
            self._general_callback = None
            
            # 重新初始化
            self._init_client()
            
            self.logger.info("MQTT客户端强制重新初始化完成")
            
        except Exception as e:
            self.logger.error(f"强制重新初始化MQTT客户端失败: {e}")
            raise
    
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
            
            # 处理消息内容 - 确保payload是字符串或字节类型
            original_payload = payload
            if isinstance(payload, (dict, list)):
                try:
                    payload = json.dumps(payload, ensure_ascii=False, default=str)
                except (TypeError, ValueError) as e:
                    self.logger.error(f"JSON序列化失败: {e}")
                    # 使用更安全的字符串转换
                    try:
                        payload = json.dumps(str(payload), ensure_ascii=False)
                    except:
                        payload = "{}"
            elif not isinstance(payload, (str, bytes)):
                payload = str(payload)
            
            # 确保payload是字符串类型（paho-mqtt期望字符串或字节）
            if not isinstance(payload, (str, bytes)):
                payload = str(payload)
            
            # 检查消息大小
            max_size = self.message_config.get("max_payload_size", 1048576)  # 1MB
            
            # 安全地计算payload大小
            try:
                if isinstance(payload, str):
                    payload_size = len(payload.encode('utf-8'))
                elif isinstance(payload, bytes):
                    payload_size = len(payload)
                else:
                    payload_size = len(str(payload).encode('utf-8'))
            except Exception as e:
                self.logger.error(f"计算消息大小失败: {e}")
                payload_size = 0
            
            if payload_size > max_size:
                self.logger.error(f"消息大小超过限制: {payload_size} > {max_size}")
                return False
            
            # 发布消息
            # 确保所有参数都是正确的类型
            if not isinstance(topic, str):
                self.logger.error(f"topic必须是字符串类型，当前类型: {type(topic)}")
                return False
            
            if not isinstance(payload, (str, bytes)):
                self.logger.error(f"payload必须是字符串或字节类型，当前类型: {type(payload)}")
                return False
            
            if not isinstance(qos, int):
                self.logger.error(f"qos必须是整数类型，当前类型: {type(qos)}")
                return False
            
            if not isinstance(retain, bool):
                self.logger.error(f"retain必须是布尔类型，当前类型: {type(retain)}")
                return False
            
            result = self.client.publish(topic, payload, qos, retain)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.logger.debug(f"消息发布成功: {topic}")
                return True
            else:
                self.logger.error(f"消息发布失败: {topic}, 错误: {mqtt.error_string(result.rc)}")
                return False
                
        except Exception as e:
            self.logger.error(f"消息发布异常: {e}, payload类型: {type(payload)}, payload值: {payload}")
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
        was_connected = self.is_connected
        self.is_connected = False
        
        if rc != 0:
            # 意外断开连接
            if was_connected:
                self.logger.warning(f"MQTT意外断开连接，返回码: {rc}")
                # 记录错误码含义
                error_meanings = {
                    1: "连接被拒绝 - 协议版本不正确",
                    2: "连接被拒绝 - 客户端标识符被拒绝", 
                    3: "连接被拒绝 - 服务器不可用",
                    4: "连接被拒绝 - 用户名或密码错误",
                    5: "连接被拒绝 - 未授权",
                    7: "连接被拒绝 - 网络错误或客户端ID冲突"
                }
                if rc in error_meanings:
                    self.logger.warning(f"错误详情: {error_meanings[rc]}")
            else:
                # 如果之前就没连接成功，可能是重复的断开事件
                self.logger.debug(f"MQTT断开连接事件(重复): {rc}")
        else:
            # 正常断开连接
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
    
    def get_mqtt_publish_topic(self) -> str:
        """
        获取 MQTT 发布主题
        
        Returns:
            发布主题字符串
        """
        try:
            topics_config = self.topics_config
            if "publish" in topics_config:
                publish_config = topics_config["publish"]
                # 如果publish是字典，获取message字段
                if isinstance(publish_config, dict):
                    return publish_config.get("message", "PI/robot/message")
                # 如果publish是字符串，直接返回
                elif isinstance(publish_config, str):
                    return publish_config
                else:
                    self.logger.warning(f"MQTT publish配置格式错误: {publish_config}")
                    return "PI/robot/message"
            return "PI/robot/message"  # 默认主题
        except Exception as e:
            self.logger.warning(f"获取 MQTT 发布主题失败，使用默认主题: {e}")
            return "PI/robot/message"
    
    def send_mqtt_response(self, response: 'MQTTResponse') -> bool:
        """
        统一发送 MQTT 响应
        
        Args:
            response: MQTT响应对象
            
        Returns:
            bool: 发送是否成功
        """
        if not MQTT_RESPONSE_AVAILABLE:
            self.logger.error("MQTTResponse不可用，无法发送响应")
            return False
            
        try:
            if not self.is_connected:
                self.logger.warning("MQTT 客户端未连接，无法发送响应")
                return False
            
            # 获取发布主题
            topic = self.get_mqtt_publish_topic()
            # 直接传递字典，让 MQTT 客户端自己处理 JSON 序列化
            payload = response.to_dict()
            success = self.publish(topic, payload)
            
            if success:
                self.logger.debug(f"MQTT 响应已发送到主题: {topic}")
                return True
            else:
                self.logger.warning(f"MQTT 响应发送失败: {topic}")
                return False
                
        except Exception as e:
            self.logger.error(f"发送 MQTT 响应失败: {e}")
            return False


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
