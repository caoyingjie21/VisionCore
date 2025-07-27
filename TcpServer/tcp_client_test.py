#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TCP客户端测试脚本
用于测试VisionCore TCP服务器的catch指令功能
"""

import socket
import json
import time
import threading
from typing import Dict, Any


class TcpClientTest:
    """TCP客户端测试类"""
    
    def __init__(self, host: str = "localhost", port: int = 8888):
        """
        初始化TCP客户端
        
        Args:
            host: 服务器地址
            port: 服务器端口
        """
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.receive_thread = None
    
    def connect(self) -> bool:
        """连接到TCP服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.running = True
            
            # 启动接收线程
            self.receive_thread = threading.Thread(target=self._receive_messages, daemon=True)
            self.receive_thread.start()
            
            print(f"已连接到TCP服务器: {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self.running = False
        if self.socket:
            self.socket.close()
            self.socket = None
        print("已断开连接")
    
    def send_message(self, message: Dict[str, Any]) -> bool:
        """发送消息"""
        try:
            if not self.socket:
                print("未连接到服务器")
                return False
            
            data = json.dumps(message, ensure_ascii=False).encode('utf-8')
            self.socket.send(data)
            print(f"已发送: {message}")
            return True
            
        except Exception as e:
            print(f"发送消息失败: {e}")
            return False
    
    def _receive_messages(self):
        """接收消息的线程"""
        while self.running:
            try:
                if not self.socket:
                    break
                
                data = self.socket.recv(4096)
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode('utf-8'))
                    print(f"接收到: {json.dumps(message, indent=2, ensure_ascii=False)}")
                except json.JSONDecodeError:
                    print(f"接收到无效JSON: {data}")
                    
            except Exception as e:
                if self.running:
                    print(f"接收消息时出错: {e}")
                break
    
    def test_catch(self):
        """测试catch指令"""
        print("\n=== 测试 Catch 指令 ===")
        self.send_message({"type": "catch"})
    
    def test_invalid_message(self):
        """测试无效消息"""
        print("\n=== 测试无效消息 ===")
        self.send_message({"type": "invalid_type"})
    
    def run_catch_test(self):
        """运行catch测试"""
        if not self.connect():
            return
        
        try:
            # 等待欢迎消息
            time.sleep(1)
            
            # 测试catch指令
            self.test_catch()
            time.sleep(2)
            
            # 测试无效消息
            self.test_invalid_message()
            time.sleep(2)
            
            print("\n=== 测试完成 ===")
            
        finally:
            # 等待一段时间以接收所有响应
            time.sleep(3)
            self.disconnect()
    
    def interactive_mode(self):
        """交互模式"""
        if not self.connect():
            return
        
        print("\n=== TCP客户端交互模式 ===")
        print("可用命令:")
        print("  catch         - 发送catch指令")
        print("  quit          - 退出")
        print()
        
        try:
            while True:
                cmd = input("请输入命令: ").strip().lower()
                
                if cmd == "quit":
                    break
                elif cmd == "catch":
                    self.test_catch()
                else:
                    print("无效命令，支持: catch, quit")
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n接收到中断信号")
        finally:
            self.disconnect()


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # catch测试模式
            client = TcpClientTest()
            client.run_catch_test()
        elif sys.argv[1] == "interactive":
            # 交互模式
            client = TcpClientTest()
            client.interactive_mode()
        else:
            print("用法:")
            print("  python tcp_client_test.py test        # 运行catch指令测试")
            print("  python tcp_client_test.py interactive # 进入交互模式")
    else:
        print("TCP客户端测试脚本")
        print("用法:")
        print("  python tcp_client_test.py test        # 运行catch指令测试")
        print("  python tcp_client_test.py interactive # 进入交互模式")


if __name__ == "__main__":
    main() 