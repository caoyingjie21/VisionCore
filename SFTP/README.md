# QtSFTP 模块

`QtSFTP` 是一个完整的 SFTP 客户端类，提供连接管理和文件上传功能。该类整合了 `SystemInitializer` 中的 SFTP 连接逻辑和 `main.py` 中的上传功能，提供了可复用的 SFTP 操作接口。

## 功能特性

- ✅ **连接管理**: 自动连接、断开、重连
- ✅ **文件上传**: 支持普通文件和图像文件上传
- ✅ **路径管理**: 支持自定义远程路径和根路径上传
- ✅ **错误处理**: 完整的错误处理和日志记录
- ✅ **连接测试**: 内置连接测试功能
- ✅ **上下文管理**: 支持 `with` 语句
- ✅ **异步连接**: 非阻塞连接，支持超时控制
- ✅ **多种认证**: 支持密码和私钥认证

## 安装依赖

```bash
pip install paramiko
pip install opencv-python  # 可选，用于图像处理
pip install pillow         # 可选，用于图像处理
```

## 基本使用

### 1. 创建 SFTP 客户端

```python
from SFTP.QtSFTP import QtSFTP
import logging

# 配置日志
logger = logging.getLogger(__name__)

# SFTP 配置
sftp_config = {
    "host": "your-sftp-server.com",
    "port": 22,
    "username": "your-username",
    "password": "your-password",
    "remote_path": "/uploads",
    "connection_timeout": 15.0,
    "ssh_timeout": 10
}

# 创建客户端
sftp_client = QtSFTP(sftp_config, logger)
```

### 2. 连接和上传文件

```python
# 连接到服务器
if sftp_client.connect():
    print("连接成功")
    
    # 上传文件
    result = sftp_client.upload_file("./local_file.txt", "remote_file.txt")
    if result["success"]:
        print(f"上传成功: {result['message']}")
    else:
        print(f"上传失败: {result['message']}")
    
    # 断开连接
    sftp_client.disconnect()
```

### 3. 使用上下文管理器

```python
# 自动连接和断开
with QtSFTP(sftp_config, logger) as sftp:
    result = sftp.upload_file("./test.png")
    print(f"上传结果: {result}")
```

## 高级功能

### 1. 图像上传

```python
import cv2
import numpy as np

# 读取图像
image = cv2.imread("./test.jpg")

# 上传图像（自动生成时间戳文件名）
result = sftp_client.upload_image(
    image_data=image,
    image_format="jpg",
    prefix="detection"
)

if result["success"]:
    print(f"图像上传成功: {result['filename']}")
```

### 2. 根路径上传

```python
# 上传到服务器根路径
result = sftp_client.upload_to_root("./test.png", "test.png")
```

### 3. 连接测试

```python
# 测试连接（上传 test.png 到根路径）
result = sftp_client.test_connection("./test.png")
if result["success"]:
    print("SFTP 连接测试成功")
```

### 4. 错误回调

```python
def error_callback(component, error):
    print(f"组件 {component} 出现错误: {error}")

sftp_client.set_error_callback(error_callback)
```

## 配置选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `host` | str | "localhost" | SFTP 服务器地址 |
| `port` | int | 22 | SFTP 端口 |
| `username` | str | "anonymous" | 用户名 |
| `password` | str | "" | 密码 |
| `private_key_path` | str | None | 私钥文件路径 |
| `remote_path` | str | "/uploads" | 默认远程路径 |
| `connection_timeout` | float | 15.0 | 连接超时时间（秒） |
| `ssh_timeout` | int | 10 | SSH 连接超时时间（秒） |

## 方法说明

### 连接管理

- `connect(timeout=None)`: 连接到 SFTP 服务器
- `disconnect()`: 断开连接
- `get_connection_info()`: 获取连接信息

### 文件上传

- `upload_file(local_path, remote_filename=None, remote_path=None, verify_upload=True)`: 上传文件
- `upload_image(image_data, image_format="jpg", prefix="detection", remote_path=None, verify_upload=True)`: 上传图像
- `upload_to_root(local_path, remote_filename=None, verify_upload=True)`: 上传到根路径

### 测试功能

- `test_connection(test_file_path="./test.png")`: 测试连接

## 返回结果格式

所有上传方法都返回统一的结果格式：

```python
{
    "success": bool,           # 是否成功
    "message": str,           # 结果消息
    "filename": str,          # 文件名
    "remote_path": str,       # 远程路径
    "file_size": int,         # 文件大小（字节）
    "timestamp": float        # 时间戳
}
```

## 错误处理

类提供了完整的错误处理机制：

1. **连接错误**: 自动重试和超时控制
2. **认证错误**: 支持密码和私钥两种方式
3. **文件错误**: 检查文件存在性和权限
4. **网络错误**: 超时和重连机制
5. **回调通知**: 可设置错误回调函数

## 集成到现有系统

### 在 SystemInitializer 中使用

```python
from SFTP.QtSFTP import QtSFTP

def initialize_sftp_client(self):
    sftp_config = self.config_mgr.get_config("sftp")
    if not sftp_config or not sftp_config.get("enabled", False):
        return True
    
    sftp_client = QtSFTP(sftp_config, self.logger)
    if sftp_client.connect():
        self.resources['sftp_client'] = sftp_client
        return True
    return False
```

### 在 main.py 中使用

```python
def handle_sftp_test(self):
    sftp_client = self.initializer.get_sftp_client()
    if sftp_client:
        result = sftp_client.test_connection("./test.png")
        return result
    else:
        return {"success": False, "message": "SFTP 客户端未初始化"}
```

## 注意事项

1. **依赖库**: 确保安装了 `paramiko` 库
2. **网络连接**: 确保网络连接正常
3. **权限**: 确保有足够的文件读写权限
4. **超时设置**: 根据网络情况调整超时时间
5. **资源清理**: 使用完毕后及时断开连接

## 示例代码

完整的使用示例请参考 `test_qtsftp.py` 文件。 