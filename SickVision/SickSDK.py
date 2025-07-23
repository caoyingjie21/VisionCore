"""
@Description :   this moudle is used to control the sick vision device and get the data.
                 The common module in the folder is required
@Author      :   Cao Yingjie
@Time        :   2025/04/23 08:47:44
"""

from common.Control import Control
from common.Streaming import Data
from common.Stream import Streaming
from common.Streaming.BlobServerConfiguration import BlobClientConfig
from utils.decorators import retry, require_connection, safe_disconnect
import cv2
import numpy as np
import time
import socket

class QtVisionSick:
    """
    西克相机控制类
    用于获取相机的强度图数据
    该类获取的流默认为TCP流,如果需要UDP流,请参考sick_visionary_python_samples/visionary_StreamingDemo.py
    """
    
    def __init__(self, ipAddr="192.168.10.5", port=2122, protocol="Cola2"):
        """
        初始化西克相机
        
        Args:
            ipAddr (str): 相机IP地址
            port (int): 相机控制端口
            protocol (str): 通信协议
        """
        self.ipAddr = ipAddr
        self.control_port = port  # 控制端口
        self.streaming_port = 2114  # 数据流端口
        self.protocol = protocol
        self.deviceControl = None
        self.streaming_device = None
        self.is_connected = False
        self.camera_params = None  # 存储相机参数
        
    def _check_camera_available(self):
        """
        检查相机是否可访问
        
        Returns:
            bool: 相机是否可访问
        """
        try:
            # 创建socket连接测试
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)  # 设置超时时间为2秒
            result = sock.connect_ex((self.ipAddr, self.control_port))
            sock.close()
            return result == 0
        except Exception as e:
            return False
    
    @retry(max_retries=3, delay=1.0, logger_name=__name__)
    def connect(self):
        """
        连接相机并初始化流
            
        Returns:
            bool: 连接是否成功
            
        Raises:
            Exception: 连接过程中的任何异常
        """
        if not self._check_camera_available():
            raise ConnectionError(f"Camera at {self.ipAddr}:{self.control_port} is not accessible")
        
        # 创建设备控制实例
        self.deviceControl = Control(self.ipAddr, self.protocol, self.control_port)
        
        # 打开连接
        self.deviceControl.open()
        
        # 尝试登录 - 在连接时登录，保持登录状态
        try:
            self.deviceControl.login(Control.USERLEVEL_SERVICE, '123456')
        except Exception as e:
            self.deviceControl.login(Control.USERLEVEL_AUTH_CLIENT, 'CLIENT')
        
        # 获取设备信息
        name, version = self.deviceControl.getIdent()
        
        # 尝试设置较低的帧速率以减少延迟
        try:
            # 获取当前帧周期 (微秒)
            current_frame_period = self.deviceControl.getFramePeriodUs()
            
            # 设置较低的帧率 (例如 30 fps = 33333 微秒)
            self.deviceControl.setFramePeriodUs(33333)
            new_frame_period = self.deviceControl.getFramePeriodUs()
        except Exception as e:
            pass
        
        # 配置流设置
        streamingSettings = BlobClientConfig()
        streamingSettings.setTransportProtocol(self.deviceControl, streamingSettings.PROTOCOL_TCP)
        streamingSettings.setBlobTcpPort(self.deviceControl, self.streaming_port)
        
        # 初始化流
        self.streaming_device = Streaming(self.ipAddr, self.streaming_port)
        self.streaming_device.openStream()
        
        # 启动连续流
        self.deviceControl.startStream()
        
        self.is_connected = True
        return True

    @require_connection
    def get_latest_frame(self, flush_count=2):
        """
        获取最新的帧数据，清空缓冲区确保获取真正最新的帧
        
        Args:
            flush_count (int): 清空缓冲区的帧数，默认为2
            
        Returns:
            tuple: (success, depth_data, intensity_image, camera_params)
                success (bool): 是否成功获取数据
                depth_data (list): 深度图数据
                intensity_image (numpy.ndarray): 强度图
                camera_params: 相机参数对象
        """
        # 连续流模式下，使用安全的缓冲区清空策略
        try:
            # 备份当前socket超时设置
            old_timeout = self.streaming_device.sock_stream.gettimeout()
            
            # 设置较短的超时时间来检测缓冲区状态
            self.streaming_device.sock_stream.settimeout(0.05)  # 50ms超时
            
            frames_flushed = 0
            # 安全地读取并丢弃旧帧
            for i in range(flush_count):
                try:
                    # 完整读取一帧来保持数据包边界
                    self.streaming_device.getFrame()
                    frames_flushed += 1
                    
                except socket.timeout:
                    # 超时表示缓冲区已空，这是正常情况
                    break
                except Exception as e:
                    # 其他异常，记录并退出
                    break
            
            # 恢复原始超时设置
            self.streaming_device.sock_stream.settimeout(old_timeout)
            
            # 现在获取最新的帧
            return self._get_frame_data()
            
        except Exception as e:
            # 确保恢复超时设置
            try:
                self.streaming_device.sock_stream.settimeout(old_timeout)
            except:
                pass
                
            # 降级到普通获取方法
            return self.get_frame()

    @require_connection
    def get_frame(self):
        """
        获取当前帧数据
        
        Returns:
            tuple: (success, depth_data, intensity_image, camera_params)
                success (bool): 是否成功获取数据
                depth_data (list): 深度图数据
                intensity_image (numpy.ndarray): 强度图
                camera_params: 相机参数对象
        """
        # 获取帧数据
        return self._get_frame_data()
    
    def _get_frame_data(self):
        """
        内部方法：获取并处理帧数据
        
        Returns:
            tuple: (success, depth_data, intensity_image, camera_params)
        """
        self.streaming_device.getFrame()
        wholeFrame = self.streaming_device.frame
        # 解析数据
        myData = Data.Data()
        myData.read(wholeFrame)
        if not myData.hasDepthMap:
            raise ValueError("No depth map data available")
        # 获取深度数据
        distance_data = list(myData.depthmap.distance)
        # 获取强度数据
        intensityData = list(myData.depthmap.intensity)
        numCols = myData.cameraParams.width
        numRows = myData.cameraParams.height
        # 重塑数据为图像
        image = np.array(intensityData).reshape((numRows, numCols))
        # 直接调整对比度，不进行归一化
        adjusted_image = cv2.convertScaleAbs(image, alpha=0.05, beta=1)
        # 保存相机参数
        self.camera_params = myData.cameraParams
        return True, distance_data, adjusted_image, self.camera_params
    
    def get_camera_params(self):
        """
        获取相机参数
        
        Returns:
            camera_params: 相机参数对象，如果未获取过帧数据则返回None
        """
        return getattr(self, 'camera_params', None)
            
    @safe_disconnect  
    def disconnect(self):
        """断开相机连接并释放资源"""
        if self.is_connected:
            if self.deviceControl:
                # 先停止流
                try:
                    # 确保在停止流前先登录
                    try:
                        self.deviceControl.login(Control.USERLEVEL_AUTH_CLIENT, 'CLIENT')
                    except Exception as e:
                        pass
                        
                    # 停止数据流
                    self.deviceControl.stopStream()
                    time.sleep(0.2)  # 等待相机处理命令
                except Exception as e:
                    pass
                    
                # 关闭流设备
                if self.streaming_device:
                    try:
                        self.streaming_device.closeStream()
                    except Exception as e:
                        pass
                    
                # 登出设备
                try:
                    self.deviceControl.logout()
                except Exception as e:
                    pass
                    
                # 关闭控制连接
                try:
                    self.deviceControl.close()
                except Exception as e:
                    pass
                    
            self.is_connected = False
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.disconnect()
        
    def __del__(self):
        """确保在销毁时断开连接"""
        self.disconnect()

    

  
  

