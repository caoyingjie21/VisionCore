# 类型化资源获取方法使用指南

## 📋 概述

为了提供更好的IDE智能提示和开发体验，`SystemInitializer` 现在提供了类型化的资源获取方法。这些方法返回具体的类型而不是通用的 `Any` 类型，让IDE能够提供完整的代码自动补全和方法提示。

## 🔄 新旧对比

### ❌ 旧方式 - 通用方法
```python
# 返回 Optional[Any]，IDE无法提供智能提示
mqtt_client = initializer.get_resource('mqtt_client')
tcp_server = initializer.get_resource('tcp_server')

# IDE不知道这些对象的类型，无法提示可用方法
mqtt_client.  # <-- 没有智能提示
tcp_server.   # <-- 没有智能提示
```

### ✅ 新方式 - 类型化方法
```python
# 返回具体类型，IDE提供完整智能提示
mqtt_client = initializer.get_mqtt_client()      # 返回 Optional[MqttClient]
tcp_server = initializer.get_tcp_server()        # 返回 Optional[TcpServer]

# IDE会显示所有可用方法和参数
mqtt_client.publish()      # <-- 完整的方法提示
tcp_server.broadcast_message()  # <-- 完整的方法提示
```

## 📝 可用的类型化方法

### 🌐 网络组件
```python
# MQTT客户端
mqtt_client: Optional[MqttClient] = initializer.get_mqtt_client()
if mqtt_client:
    mqtt_client.connect()
    mqtt_client.publish(topic, payload, qos)
    mqtt_client.subscribe(topic, qos)
    mqtt_client.set_general_callback(callback)

# TCP服务器
tcp_server: Optional[TcpServer] = initializer.get_tcp_server()
if tcp_server:
    tcp_server.start()
    tcp_server.stop()
    tcp_server.set_message_callback(callback)
    tcp_server.broadcast_message(message)
```

### 🎥 硬件组件
```python
# 相机 (返回Any，因为相机类型可能多样)
camera = initializer.get_camera()
if camera:
    # 需要根据具体的相机实现来调用方法
    if hasattr(camera, 'get_fresh_frame'):
        frame = camera.get_fresh_frame()

# 检测器 (返回Any，因为检测器类型可能多样)
detector = initializer.get_detector()
if detector:
    # 需要根据具体的检测器实现来调用方法
    if hasattr(detector, 'detect'):
        results = detector.detect(image)
```

### ⚙️ 系统组件
```python
# 配置管理器
config_mgr: Optional[ConfigManager] = initializer.get_config_manager()
if config_mgr:
    config_mgr.get("section.key")
    config_mgr.set("section.key", value)
    config_mgr.save()

# 日志记录器
logger: Optional[logging.Logger] = initializer.get_logger()
if logger:
    logger.info("消息")
    logger.warning("警告")
    logger.error("错误")

# 系统监控器
monitor: Optional[SystemMonitor] = initializer.get_monitor()
if monitor:
    status = monitor.get_system_status()
    monitor.start()
    monitor.stop()
```

## 🎯 使用场景示例

### 1. 主程序中的组件获取
```python
class VisionCoreApp:
    def _run_main_loop(self):
        # 使用类型化方法获取组件
        tcp_server = self.initializer.get_tcp_server()
        mqtt_client = self.initializer.get_mqtt_client()
        
        # IDE会提供完整的方法提示
        if tcp_server:
            tcp_server.set_message_callback(self.handle_tcp_message)
        
        if mqtt_client:
            mqtt_client.set_general_callback(self.handle_mqtt_message)
```

### 2. 业务逻辑中的组件使用
```python
def handle_catch(self, client_id: str, message: Dict[str, Any]):
    # 获取组件
    camera = self.initializer.get_camera()
    detector = self.initializer.get_detector()
    mqtt_client = self.initializer.get_mqtt_client()
    logger = self.initializer.get_logger()
    
    # IDE提供智能提示
    if logger:
        logger.info(f"处理catch请求: {client_id}")
    
    if mqtt_client:
        mqtt_client.publish("sickvision/catch/result", result)
```

### 3. 配置更新后的组件重载
```python
def _reload_tcp_server(self) -> bool:
    if self.initializer._restart_tcp_server():
        # 获取重启后的服务器实例
        tcp_server = self.initializer.get_tcp_server()
        if tcp_server:
            # IDE会提示set_message_callback的参数类型
            tcp_server.set_message_callback(self.message_handler)
        return True
    return False
```

## 🛠️ 开发优势

### 1. **智能代码补全**
- IDE会显示所有可用的方法和属性
- 自动提示方法参数和返回类型
- 减少查文档的需要

### 2. **类型安全**
- 编译时可以检测类型错误
- 减少运行时错误
- 更好的代码质量

### 3. **开发效率**
- 更快的开发速度
- 减少拼写错误
- 更容易重构代码

### 4. **代码可读性**
- 明确的类型注解
- 更容易理解代码意图
- 更好的团队协作

## 📚 完整示例

参考 `example_typed_usage.py` 文件，其中包含了完整的使用示例：

```bash
python example_typed_usage.py
```

## ⚠️ 注意事项

1. **向后兼容**: 旧的 `get_resource()` 方法仍然可用，不会破坏现有代码
2. **相机和检测器**: 由于可能有多种实现，返回 `Any` 类型，需要使用 `hasattr()` 检查方法存在性
3. **空值检查**: 所有方法都返回 `Optional` 类型，使用前需要检查是否为 `None`

## 🔄 迁移建议

1. **逐步迁移**: 可以逐步将 `get_resource()` 替换为对应的类型化方法
2. **新代码**: 建议在新代码中直接使用类型化方法
3. **IDE配置**: 确保IDE支持Python类型提示以获得最佳体验

---

通过使用这些类型化方法，你将获得更好的开发体验和代码质量！ 