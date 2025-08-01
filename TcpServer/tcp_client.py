#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TCP客户端
连接到VisionCore服务器，发送指令并接收响应
"""

import socket
import threading
import time
import sys

class TcpClient:
    def __init__(self, host='127.0.0.1', port=8888):
        """
        初始化TCP客户端
        
        Args:
            host: 服务器地址
            port: 服务器端口
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.running = True
        
    def connect(self):
        """连接到服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)  # 5秒超时
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"已连接到服务器 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self.connected = False
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        print("已断开连接")
    
    def send_message(self, message):
        """发送消息到服务器"""
        if not self.connected:
            print("未连接到服务器")
            return False
        
        try:
            # 确保消息以\r\n结尾，符合服务器期望的格式
            if not message.endswith('\r\n'):
                message += '\r\n'
            
            # 发送消息
            self.socket.send(message.encode('utf-8'))
            print(f"已发送: {message.strip()}")
            return True
        except Exception as e:
            print(f"发送失败: {e}")
            self.connected = False
            return False
    
    def receive_message(self):
        """接收服务器消息"""
        if not self.connected:
            return None
        
        try:
            # 接收响应
            data = self.socket.recv(4096)
            if data:
                response = data.decode('utf-8').strip()
                print(f"收到响应: {response}")
                return response
            else:
                print("服务器断开连接")
                self.connected = False
                return None
        except socket.timeout:
            return None
        except Exception as e:
            print(f"接收失败: {e}")
            self.connected = False
            return None
    
    def send_and_receive(self, message):
        """发送消息并接收响应"""
        if self.send_message(message):
            return self.receive_message()
        return None
    
    def run_interactive(self):
        """运行交互式客户端"""
        print("=== VisionCore TCP客户端 ===")
        print(f"连接到 {self.host}:{self.port}")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'catch' 执行检测")
        print("输入其他指令发送给服务器")
        print("-" * 30)
        
        if not self.connect():
            return
        
        try:
            while self.running and self.connected:
                try:
                    # 获取用户输入
                    user_input = input("请输入指令: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # 检查退出命令
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("正在退出...")
                        break
                    
                    # 发送消息并接收响应
                    response = self.send_and_receive(user_input)
                    
                    if response is None and not self.connected:
                        print("连接已断开，尝试重新连接...")
                        if not self.connect():
                            break
                    
                except KeyboardInterrupt:
                    print("\n正在退出...")
                    break
                except EOFError:
                    print("\n正在退出...")
                    break
                except Exception as e:
                    print(f"输入错误: {e}")
                    
        finally:
            self.disconnect()

def main():
    """主函数"""
    # 创建客户端
    client = TcpClient("192.168.2.100", 8888)
    
    # 运行交互式客户端
    client.run_interactive()

if __name__ == "__main__":
    main() 