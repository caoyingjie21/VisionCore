# VisionCore 无界面视觉系统

VisionCore 是一个专为开发板设计的无界面视觉处理系统，支持自动重启、健康监控和MQTT通信。

## 🚀 主要特性

- **🔄 自动重启**: 组件故障时自动重启，确保系统持续运行
- **📊 健康监控**: 实时监控各组件状态，自动故障恢复
- **📡 MQTT通信**: 支持远程配置和状态监控
- **🎥 相机支持**: 支持SICK视觉相机
- **🧠 AI检测**: 集成RKNN推理引擎和YOLOv8
- **📝 完整日志**: 详细的日志记录和轮转
- **⚙️ 配置热重载**: 支持运行时配置更新

## 📁 项目结构

```
VisionCore/
├── main.py                 # 主程序入口
├── Config/
│   └── config.yaml         # 系统配置文件
├── Managers/
│   ├── ConfigManager.py    # 配置管理器
│   └── LogManager.py       # 日志管理器
├── Mqtt/
│   └── MqttClient.py      # MQTT客户端
├── Rknn/
│   └── RknnYolo.py        # RKNN推理模块
├── SickVision/            # SICK相机SDK
├── utils/
│   └── initializers.py    # 系统初始化器
├── visioncore.service     # systemd服务文件
├── install.sh             # 自动安装脚本
└── requirements.txt       # Python依赖
```

## 🛠 安装部署

### 方法1: 自动安装（推荐）

```bash
# 下载项目
git clone <repository-url>
cd VisionCore

# 运行自动安装脚本
sudo ./install.sh
```

### 方法2: 手动安装

```bash
# 1. 安装系统依赖
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-dev
sudo apt-get install -y libopencv-dev python3-opencv

# 2. 安装Python依赖
pip3 install -r requirements.txt

# 3. 复制到系统目录
sudo cp -r . /opt/VisionCore

# 4. 配置systemd服务
sudo cp visioncore.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable visioncore
```

## ⚙️ 配置

### 主配置文件: `/opt/VisionCore/Config/config.yaml`

```yaml
# 系统基本配置
system:
  name: SickVision-Headless
  version: 1.0.0
  debug: false

# 相机配置
camera:
  enabled: true
  type: SICK
  connection:
    ip: 192.168.1.101
    port: 2122
    timeout: 10

# MQTT配置
mqtt:
  enabled: true
  connection:
    broker_host: localhost
    broker_port: 1883
    client_id: sickvision_headless

# 日志配置
logging:
  enabled: true
  level: INFO
  file:
    enabled: true
    path: logs
```

## 🎮 运行管理

### 服务管理命令

```bash
# 启动服务
visioncore-start

# 停止服务
visioncore-stop

# 重启服务
visioncore-restart

# 查看状态
visioncore-status

# 查看日志
visioncore-logs

# 实时日志
visioncore-logs -f
```

### 手动运行

```bash
# 开发模式
cd /opt/VisionCore
python3 main.py

# 后台运行
nohup python3 main.py > /dev/null 2>&1 &
```

## 📊 系统监控

### 健康检查功能

系统会自动监控以下组件：
- **相机连接状态**: 每30秒检查一次
- **MQTT连接状态**: 自动重连机制
- **模型推理状态**: 检测器可用性

### 自动重启策略

- **组件级重启**: 单个组件故障时自动重启
- **系统级重启**: 多次失败后重启整个系统
- **限制重启次数**: 防止无限重启循环

## 📡 MQTT接口

### 订阅主题

```
sickvision/config/update      # 配置更新
sickvision/control/start      # 启动控制
sickvision/control/stop       # 停止控制
sickvision/detection/trigger  # 检测触发
sickvision/system/command     # 系统命令
```

### 发布主题

```
sickvision/detection/result   # 检测结果
sickvision/system/status      # 系统状态
sickvision/system/heartbeat   # 心跳信号
sickvision/system/error       # 错误报告
```

## 🔧 开发指南

### 添加新组件

1. 在 `utils/initializers.py` 中添加初始化函数
2. 在 `SystemInitializer` 类中注册组件
3. 在配置文件中添加相应配置项

### 自定义业务逻辑

在 `main.py` 的 `_execute_business_logic` 方法中添加：

```python
def _execute_business_logic(self, camera, detector, mqtt_client, logger):
    # 1. 获取图像
    if camera:
        success, depth_data, image, params = camera.get_fresh_frame()
        if success:
            # 2. 进行检测
            results = detector.detect(image)
            
            # 3. 发布结果
            mqtt_client.publish("sickvision/detection/result", {
                "detections": len(results),
                "timestamp": time.time()
            })
```

## 📋 日志管理

### 日志位置

- **系统日志**: `/var/log/visioncore/`
- **应用日志**: `/opt/VisionCore/logs/`
- **systemd日志**: `journalctl -u visioncore`

### 日志级别

- `DEBUG`: 详细调试信息
- `INFO`: 一般信息（默认）
- `WARNING`: 警告信息
- `ERROR`: 错误信息
- `CRITICAL`: 严重错误

## 🚨 故障排除

### 常见问题

1. **相机连接失败**
   ```bash
   # 检查网络连接
   ping 192.168.1.101
   
   # 检查端口
   telnet 192.168.1.101 2122
   ```

2. **MQTT连接失败**
   ```bash
   # 检查MQTT服务器
   mosquitto_pub -h localhost -t test -m "hello"
   ```

3. **服务启动失败**
   ```bash
   # 查看详细错误
   journalctl -u visioncore -f
   
   # 检查配置文件
   python3 -c "import yaml; yaml.safe_load(open('/opt/VisionCore/Config/config.yaml'))"
   ```

### 重置系统

```bash
# 停止服务
sudo systemctl stop visioncore

# 清理日志
sudo rm -rf /var/log/visioncore/*
sudo rm -rf /opt/VisionCore/logs/*

# 重新启动
sudo systemctl start visioncore
```

## 📝 开发计划

- [ ] Web管理界面
- [ ] 更多相机支持
- [ ] 数据库存储
- [ ] 性能优化
- [ ] Docker容器化

## 📄 许可证

[MIT License](LICENSE)

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📞 支持

如有问题请联系技术支持或提交Issue。 