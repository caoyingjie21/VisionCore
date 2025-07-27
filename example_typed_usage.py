#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
类型化资源获取示例
展示如何使用新的类型化方法来获得更好的IDE智能提示
"""

from System.SystemInitializer import SystemInitializer


def example_typed_resource_usage():
    """展示类型化资源获取的使用方式"""
    
    # 初始化系统
    initializer = SystemInitializer("./Config/config.yaml")
    
    if not initializer.initialize_config():
        print("配置初始化失败")
        return
    
    if not initializer.initialize_all_components():
        print("组件初始化失败")
        return
    
    # === 使用类型化方法获取资源 - 有完整的IDE智能提示 ===
    
    # 获取MQTT客户端 - IDE会显示MqttClient的所有方法
    mqtt_client = initializer.get_mqtt_client()
    if mqtt_client:
        # IDE会提示所有MqttClient的方法，如：
        # mqtt_client.connect()
        # mqtt_client.disconnect()
        # mqtt_client.publish()
        # mqtt_client.subscribe()
        # mqtt_client.set_general_callback()
        print(f"MQTT客户端状态: {mqtt_client.is_connected}")
        
        # 发布消息示例
        mqtt_client.publish("test/topic", {"message": "Hello from typed example"})
    
    # 获取TCP服务器 - IDE会显示TcpServer的所有方法
    tcp_server = initializer.get_tcp_server()
    if tcp_server:
        # IDE会提示所有TcpServer的方法，如：
        # tcp_server.start()
        # tcp_server.stop()
        # tcp_server.set_message_callback()
        # tcp_server.broadcast_message()
        print(f"TCP服务器运行状态: {tcp_server.is_running}")
        print(f"连接的客户端数量: {len(tcp_server.clients)}")
        
        # 广播消息示例
        tcp_server.broadcast_message({
            "type": "example_message",
            "content": "Hello from typed example",
            "timestamp": 1234567890
        })
    
    # 获取相机 - 虽然是Any类型，但有明确的注释说明
    camera = initializer.get_camera()
    if camera:
        # 根据相机的具体实现，IDE可能会有部分提示
        print(f"相机实例: {type(camera).__name__}")
        # 如果知道具体的相机类型，可以进行类型转换：
        # if hasattr(camera, 'get_fresh_frame'):
        #     frame = camera.get_fresh_frame()
    
    # 获取检测器 - 虽然是Any类型，但有明确的注释说明
    detector = initializer.get_detector()
    if detector:
        print(f"检测器实例: {type(detector).__name__}")
        # 如果知道具体的检测器类型，可以进行类型转换：
        # if hasattr(detector, 'detect'):
        #     results = detector.detect(image)
    
    # 获取配置管理器 - IDE会显示ConfigManager的所有方法
    config_mgr = initializer.get_config_manager()
    if config_mgr:
        # IDE会提示所有ConfigManager的方法，如：
        # config_mgr.get()
        # config_mgr.set()
        # config_mgr.save()
        # config_mgr.reload()
        print(f"配置文件路径: {config_mgr.config_path}")
        
        # 获取配置示例
        tcp_config = config_mgr.get("tcp_server")
        if tcp_config:
            print(f"TCP端口: {tcp_config.get('port', 'N/A')}")
    
    # 获取日志记录器 - IDE会显示标准logging.Logger的所有方法
    logger = initializer.get_logger()
    if logger:
        # IDE会提示所有标准Logger的方法，如：
        # logger.info()
        # logger.debug()
        # logger.warning()
        # logger.error()
        # logger.critical()
        logger.info("这是一个来自类型化示例的日志消息")
    
    # 获取系统监控器 - IDE会显示SystemMonitor的所有方法
    monitor = initializer.get_monitor()
    if monitor:
        # IDE会提示所有SystemMonitor的方法，如：
        # monitor.start()
        # monitor.stop()
        # monitor.get_system_status()
        status = monitor.get_system_status()
        print(f"系统状态: {status}")
    
    # === 对比：使用通用方法 - 没有智能提示 ===
    
    # 通用方法返回Any类型，IDE无法提供智能提示
    generic_mqtt = initializer.get_resource('mqtt_client')
    if generic_mqtt:
        # IDE不知道这是什么类型，无法提供方法提示
        # 需要开发者自己记住或查文档
        pass
    
    # 清理资源
    initializer.cleanup()


def example_business_logic_with_types():
    """展示在业务逻辑中使用类型化方法的优势"""
    
    initializer = SystemInitializer("./Config/config.yaml")
    
    # 初始化系统
    if not (initializer.initialize_config() and initializer.initialize_all_components()):
        return
    
    # 获取组件
    logger = initializer.get_logger()
    mqtt_client = initializer.get_mqtt_client()
    tcp_server = initializer.get_tcp_server()
    camera = initializer.get_camera()
    detector = initializer.get_detector()
    
    # 业务逻辑示例
    if logger and mqtt_client and tcp_server:
        logger.info("开始执行业务逻辑")
        
        # MQTT发布 - IDE会提示所有参数
        mqtt_client.publish(
            topic="business/status",
            payload={"status": "running", "timestamp": 1234567890},
            qos=1
        )
        
        # TCP广播 - IDE会提示所有参数
        tcp_server.broadcast_message({
            "type": "business_update",
            "data": "系统正在运行",
            "timestamp": 1234567890
        })
        
        # 如果有相机和检测器
        if camera and detector:
            logger.info("准备执行检测任务")
            # 在这里添加具体的检测逻辑
            # frame = camera.get_fresh_frame()  # IDE会根据相机类型提示
            # results = detector.detect(frame)  # IDE会根据检测器类型提示
    
    # 清理
    initializer.cleanup()


if __name__ == "__main__":
    print("=== 类型化资源获取示例 ===")
    example_typed_resource_usage()
    
    print("\n=== 业务逻辑中的类型化使用示例 ===")
    example_business_logic_with_types() 