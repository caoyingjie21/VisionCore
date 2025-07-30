from enum import Enum

class VisionCoreCommands(Enum):
    """VisionCore系统支持的MQTT命令枚举"""
    
    # 系统控制命令
    RESTART = "restart"
    
    # 配置相关命令
    GET_CONFIG = "get_config"
    
    # 图像获取命令
    GET_CALIBRAT_IMAGE = "get_calibrat_image"
    GET_IMAGE = "get_image"

    # SFTP测试
    SFTP_TEST = "sftp_test"
    
    # 系统状态命令
    GET_SYSTEM_STATUS = "get_system_status"

    SAVE_CONFIG = "save_config"

    ERROR_TIP = "error_tip"
    
    # 模型测试命令
    MODEL_TEST = "model_test"
    
    def __eq__(self, other):
        """允许枚举直接与字符串比较"""
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)
    
    def __hash__(self):
        """保持哈希一致性"""
        return hash(self.value)
    
    @classmethod
    def get_all_commands(cls):
        """获取所有命令值的列表"""
        return [cmd.value for cmd in cls]
    
    @classmethod
    def is_valid_command(cls, command: str) -> bool:
        """检查命令是否有效"""
        return command in cls.get_all_commands()
    
    @classmethod
    def get_command_description(cls, command: str) -> str:
        """获取命令描述"""
        descriptions = {
            cls.RESTART.value: "重启整个系统",
            cls.GET_CONFIG.value: "获取当前系统配置，包括可用模型列表",
            cls.GET_CALIBRAT_IMAGE.value: "获取相机标定图像",
            cls.GET_IMAGE.value: "获取相机图像并上传到SFTP服务器",
            cls.SFTP_TEST.value: "SFTP连接测试，上传test.png文件到服务器",
            cls.GET_SYSTEM_STATUS.value: "获取系统健康状态和组件信息",
            cls.MODEL_TEST.value: "执行模型测试",
            cls.ERROR_TIP.value: "获取错误提示"
        }
        return descriptions.get(command, "未知命令")


class MessageType(Enum):
    """消息类型枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"
    CRITICAL = "critical"
    FATAL = "fatal"
    SUCCESS = "success"
    FAILURE = "failure"



