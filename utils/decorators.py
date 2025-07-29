"""
@Description :   通用装饰器模块
@Author      :   Cao Yingjie
@Time        :   2025/05/12
"""

import time
import functools
import signal
import sys
from typing import Type, Union, Tuple, Callable, Optional


def _interruptible_sleep(duration: float):
    """
    可中断的睡眠函数，每0.1秒检查一次中断信号
    
    Args:
        duration: 睡眠时长（秒）
    """
    if duration <= 0:
        return
    
    start_time = time.time()
    end_time = start_time + duration
    
    while time.time() < end_time:
        remaining = end_time - time.time()
        sleep_interval = min(0.1, remaining)
        if sleep_interval > 0:
            time.sleep(sleep_interval)


def handle_keyboard_interrupt(
    return_value=False,
    exit_on_interrupt: bool = False,
    log_message: str = None,
    cleanup_callback: Optional[Callable] = None
):
    """
    键盘中断处理装饰器，用于优雅处理Ctrl+C信号
    
    Args:
        return_value: 中断时返回的值（默认False）
        exit_on_interrupt (bool): 是否在中断时退出程序（默认False）
        log_message (str): 自定义日志消息，None时使用默认消息
        cleanup_callback (callable): 中断时调用的清理函数
        
    Returns:
        function: 装饰过的函数
        
    Example:
        @handle_keyboard_interrupt(return_value=False, log_message="停止初始化")
        def initialize_component(self):
            # 可能需要长时间运行的初始化代码
            while True:
                time.sleep(1)
                
        @handle_keyboard_interrupt(exit_on_interrupt=True, cleanup_callback=lambda: print("清理资源"))
        def main_loop(self):
            # 主循环代码
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                # 获取对象实例的logger（如果存在）
                logger = None
                if args and hasattr(args[0], 'logger'):
                    logger = args[0].logger
                elif args and hasattr(args[0], 'initializer') and hasattr(args[0].initializer, 'logger'):
                    logger = args[0].initializer.logger
                
                # 生成日志消息
                if log_message:
                    message = f"收到中断信号，{log_message}"
                else:
                    func_name = func.__name__
                    message = f"收到中断信号，停止{func_name}"
                
                # 记录日志
                if logger:
                    logger.info(message)
                else:
                    print(message)
                
                # 执行清理回调
                if cleanup_callback:
                    try:
                        cleanup_callback()
                    except Exception as e:
                        if logger:
                            logger.error(f"清理回调执行失败: {e}")
                        else:
                            print(f"清理回调执行失败: {e}")
                
                # 根据配置决定返回值或退出
                if exit_on_interrupt:
                    sys.exit(0)
                else:
                    return return_value
                    
        return wrapper
    
    return decorator


def interruptible_retry(
    max_retries: int = None,
    retry_delay: float = 5.0,
    board_mode: bool = True,
    log_prefix: str = "操作"
):
    """
    可中断的重试装饰器，结合了重试逻辑和中断处理
    
    Args:
        max_retries (int): 最大重试次数，None表示无限重试
        retry_delay (float): 重试延迟时间（秒）
        board_mode (bool): 是否为板端模式，影响重试策略
        log_prefix (str): 日志前缀
        
    Returns:
        function: 装饰过的函数
        
    Example:
        @interruptible_retry(max_retries=None, retry_delay=5.0, log_prefix="MQTT连接")
        def connect_mqtt(self):
            # 连接逻辑，失败时会自动重试
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取实例对象
            instance = args[0] if args else None
            
            # 获取logger
            logger = None
            if instance and hasattr(instance, 'logger'):
                logger = instance.logger
            elif instance and hasattr(instance, 'initializer') and hasattr(instance.initializer, 'logger'):
                logger = instance.initializer.logger
            
            # 确定重试次数和延迟
            if max_retries is None:
                # 根据实例的infinite_retry属性决定
                if instance and hasattr(instance, 'infinite_retry') and instance.infinite_retry:
                    retries = float('inf')
                else:
                    retries = 3  # 默认重试3次
            else:
                retries = max_retries
            
            # 获取实际的重试延迟
            actual_delay = retry_delay
            if instance and hasattr(instance, 'retry_delay'):
                actual_delay = instance.retry_delay
            
            attempt = 0
            
            try:
                while attempt < retries:
                    try:
                        # 记录尝试日志
                        if retries == float('inf'):
                            if logger:
                                logger.info(f"正在{log_prefix}... (尝试 {attempt + 1}, 无限重试模式)")
                            else:
                                print(f"正在{log_prefix}... (尝试 {attempt + 1}, 无限重试模式)")
                        else:
                            if logger:
                                logger.info(f"正在{log_prefix}... (尝试 {attempt + 1}/{retries})")
                            else:
                                print(f"正在{log_prefix}... (尝试 {attempt + 1}/{retries})")
                        
                        # 执行原函数
                        result = func(*args, **kwargs)
                        
                        # 成功则返回结果
                        if result:
                            return result
                        else:
                            # 失败记录日志
                            if retries == float('inf'):
                                if logger:
                                    logger.error(f"{log_prefix}失败 (尝试 {attempt + 1}), {actual_delay}秒后重试...")
                                else:
                                    print(f"{log_prefix}失败 (尝试 {attempt + 1}), {actual_delay}秒后重试...")
                            else:
                                if logger:
                                    logger.error(f"{log_prefix}失败 (尝试 {attempt + 1}/{retries})")
                                else:
                                    print(f"{log_prefix}失败 (尝试 {attempt + 1}/{retries})")
                    
                    except Exception as e:
                        # 异常记录日志
                        if retries == float('inf'):
                            if logger:
                                logger.error(f"{log_prefix}异常 (尝试 {attempt + 1}): {e}, {actual_delay}秒后重试...")
                            else:
                                print(f"{log_prefix}异常 (尝试 {attempt + 1}): {e}, {actual_delay}秒后重试...")
                        else:
                            if logger:
                                logger.error(f"{log_prefix}异常 (尝试 {attempt + 1}/{retries}): {e}")
                            else:
                                print(f"{log_prefix}异常 (尝试 {attempt + 1}/{retries}): {e}")
                    
                    attempt += 1
                    
                    # 如果不是最后一次尝试，则等待后重试
                    if attempt < retries:
                        if board_mode and retries == float('inf'):
                            # 板端模式：使用固定重试间隔（可中断的睡眠）
                            _interruptible_sleep(actual_delay)
                        else:
                            # 非板端模式：使用递增等待时间（可中断的睡眠）
                            wait_time = attempt * 2
                            if logger:
                                logger.info(f"等待 {wait_time} 秒后重试...")
                            _interruptible_sleep(wait_time)
                    elif retries != float('inf'):
                        if logger:
                            logger.error(f"{log_prefix}最终失败")
                        else:
                            print(f"{log_prefix}最终失败")
                
                return False
                
            except KeyboardInterrupt:
                if logger:
                    logger.info(f"收到中断信号，停止{log_prefix}")
                else:
                    print(f"收到中断信号，停止{log_prefix}")
                # 重新抛出KeyboardInterrupt，让外层处理
                raise
                
        return wrapper
    
    return decorator


def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    logger_name: Optional[str] = None,
    on_retry: Optional[Callable] = None
):
    """
    重试装饰器，用于自动重试可能失败的函数
    
    Args:
        max_retries (int): 最大重试次数
        delay (float): 重试之间的延迟时间（秒）
        exceptions (Exception or tuple): 需要捕获的异常类型
        logger_name (str, optional): 保留兼容性，但不使用
        on_retry (callable, optional): 重试前调用的函数，接收(重试次数, 异常, 参数字典)作为参数
        
    Returns:
        function: 装饰过的函数
        
    Example:
        @retry(max_retries=3, delay=2.0)
        def connect_to_device(self, device_id):
            # 连接代码...
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            # 第一次尝试 + 最大重试次数
            for attempt in range(max_retries + 1):
                try:
                    # 第一次尝试或重试
                    if attempt > 0:
                        if on_retry:
                            on_retry(attempt, last_exception, dict(args=args, kwargs=kwargs))
                        if delay > 0:
                            time.sleep(delay)
                    
                    # 执行被装饰的函数
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    # 如果是最后一次尝试，则抛出异常
                    if attempt >= max_retries:
                        raise
            
            # 理论上不会执行到这里
            return None
            
        return wrapper
    
    return decorator


def require_connection(func):
    """
    检查相机是否已连接的装饰器
    
    为相机操作类的方法添加此装饰器，会在执行方法前检查相机是否已连接。
    如果未连接，将抛出ConnectionError异常。
    
    注意：此装饰器假设被装饰的方法属于一个具有is_connected属性的类
    
    Args:
        func: 需要装饰的方法
        
    Returns:
        function: 装饰过的函数
        
    Example:
        @require_connection
        def get_frame(self):
            # 获取帧的代码...
            pass
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 检查对象是否有is_connected属性且为True
        if not getattr(self, 'is_connected', False):
            method_name = func.__name__
            class_name = self.__class__.__name__
            error_msg = f"设备未连接，无法执行 {class_name}.{method_name}() 操作"
            
            # 抛出连接错误异常
            raise ConnectionError(error_msg)
            
        # 设备已连接，执行原方法
        return func(self, *args, **kwargs)
        
    return wrapper


def safe_disconnect(func):
    """
    安全断开连接的装饰器
    
    为相机类的disconnect方法添加此装饰器，可以确保无论发生什么异常，
    设备的连接状态都会被正确重置，资源会被释放。
    
    Args:
        func: 断开连接的方法
        
    Returns:
        function: 装饰过的函数
        
    Example:
        @safe_disconnect
        def disconnect(self):
            # 断开连接的代码...
            pass
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            # 执行原始的断开连接方法
            return func(self, *args, **kwargs)
        except Exception as e:
            # 忽略异常但不重新抛出
            pass
        finally:
            # 无论发生什么，确保连接状态被重置
            self.is_connected = False
            # 确保资源被释放
            if hasattr(self, 'deviceControl'):
                self.deviceControl = None
            if hasattr(self, 'streaming_device'):
                self.streaming_device = None
                
    return wrapper 