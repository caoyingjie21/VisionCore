#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VisionCore 主程序
负责启动和管理整个视觉系统
"""

import time
import sys
import signal
import os
from typing import Dict, Any
import json
import cv2
import numpy as np
from Rknn.RknnYolo import RKNN_YOLO
from System.SystemInitializer import SystemInitializer
from ClassModel.MqttResponse import MQTTResponse
from utils.decorators import handle_keyboard_interrupt
# 直接导入所有命令常量
from SystemEnums.VisionCoreCommands import MessageType, VisionCoreCommands


class VisionCoreApp:
    """VisionCore应用程序类，支持自动重启和监控"""
    
    def __init__(self, config_path: str = "./Config/config.yaml"):
        """
        初始化应用程序
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.initializer = None
        self.running = True
        self.restart_on_failure = True
    
        # 防抖和性能监控（默认值，将从配置文件中读取）
        self.last_tcp_command_time = {}  # 记录每个客户端的最后命令时间
        self.tcp_processing_flags = {}   # 记录每个客户端的处理状态
        self.tcp_debounce_time = 2.0     # 防抖时间间隔（秒）
        self.z_offset = 0.0              # Z坐标补偿值
        self.pixel_threshold = 5.0       # 稳定性检测像素阈值
    
    def start(self):
        """启动应用程序主循环"""
        print("启动 VisionCore 系统...")
        
        try:
            while self.running:
                try:
                    # 初始化系统
                    if not self._initialize_system():
                        if self.restart_on_failure:
                            print("系统初始化失败，5秒后重试...")
                            # 使用可中断的睡眠
                            for _ in range(5):
                                time.sleep(1)
                            continue
                        else:
                            break
                    
                    # 运行主循环
                    self._run_main_loop()
                    
                except KeyboardInterrupt:
                    print("\n接收到停止信号，正在优雅关闭...")
                    self.running = False
                    self.restart_on_failure = False
                    break
                except Exception as e:
                    if self.initializer and self.initializer.logger:
                        self.initializer.logger.error(f"系统运行异常: {e}", exc_info=True)
                    else:
                        print(f"系统运行异常: {e}")
                    
                    if self.restart_on_failure and self.running:
                        if self.initializer and self.initializer.logger:
                            self.initializer.logger.warning("系统将在10秒后自动重启...")
                        else:
                            print("系统将在10秒后自动重启...")
                        # 使用可中断的睡眠
                        for _ in range(10):
                            time.sleep(1)
                    else:
                        break
                finally:
                    # 清理资源
                    if self.initializer:
                        self.initializer.cleanup()
                        self.initializer = None
        
        except KeyboardInterrupt:
            print("\n接收到停止信号，正在优雅关闭...")
            self.running = False
            self.restart_on_failure = False
        
        print("VisionCore 系统已关闭")
    
    def _initialize_system(self) -> bool:
        """初始化系统组件"""
        try:
            # 创建系统初始化器
            self.initializer = SystemInitializer(self.config_path)
            
            # 初始化配置
            if not self.initializer.initialize_config():
                print("配置初始化失败")
                return False
            
            # 从配置文件读取防抖和性能监控参数
            config_mgr = self.initializer.get_config_manager()
            if config_mgr:
                debounce_time = config_mgr.get_config("stability.debounceTime")
                z_offset = config_mgr.get_config("stability.zOffset")
                pixel_threshold = config_mgr.get_config("stability.pixelThreshold")
                
                self.tcp_debounce_time = debounce_time if debounce_time is not None else 2.0
                self.z_offset = z_offset if z_offset is not None else 0.0
                self.pixel_threshold = pixel_threshold if pixel_threshold is not None else 5.0
                print(f"已加载配置: 防抖时间={self.tcp_debounce_time}s, Z补偿={self.z_offset}, 像素阈值={self.pixel_threshold}")
            
            # 初始化所有组件
            if not self.initializer.initialize_all_components():
                return False
            
            return True
            
        except Exception as e:
            print(f"系统初始化异常: {e}")
            return False
    
    def _stop_running(self):
        """停止主循环的清理方法"""
        self.running = False
    
    def _run_main_loop(self):
        """运行主程序循环"""
        logger = self.initializer.logger
        
        tcp_server = self.initializer.get_tcp_server()
        mqtt_client = self.initializer.get_mqtt_client()
        
        # 为TCP服务器设置消息处理回调
        if tcp_server:
            message_handler = self.initializer.create_tcp_message_handler(self.handle_catch)
            tcp_server.set_message_callback(message_handler)
            
            # 设置客户端断开连接的回调
            tcp_server.set_disconnect_callback(self.handle_tcp_client_disconnected)
            logger.info("TCP服务器消息处理器已设置")
        
        # 为MQTT客户端设置消息处理回调
        if mqtt_client:
            self.initializer.set_mqtt_message_callback(self.handle_mqtt_message)
            logger.info("MQTT消息处理器已设置")
        
        logger.info("系统运行中，按Ctrl+C停止...")
        
        # 主程序循环
        loop_count = 0
        try:
            while self.running:
                try:
                    # 检查系统健康状态（每60秒）
                    if loop_count % 60 == 0:
                        self._check_system_health(logger)
                    
                    time.sleep(1)
                    loop_count += 1
                    
                    # 防止计数器溢出
                    if loop_count > 86400:  # 24小时重置
                        loop_count = 0
                    
                except Exception as e:
                    logger.error(f"主循环异常: {e}", exc_info=True)
                    # 继续运行，不退出主循环
                    time.sleep(1)
        except KeyboardInterrupt:
            logger.info("接收到停止信号，主程序循环停止")
            self.running = False
    
    def handle_catch(self, client_id: str, message: str):
        """
        处理catch指令 - 单步模式下进行快速单次检测（优化版）
        
        Args:
            client_id: 客户端ID
            message: 接收到的消息字符串
            
        Returns:
            str: 响应消息字符串
        """
        # 记录整个流程的开始时间
        process_start_time = time.time()
        current_time = time.time()
        
        logger = self.initializer.logger
        logger.info(f"收到catch指令来自客户端: {client_id}")
        
        try:
            # 防抖检查：检查是否在防抖时间内收到重复命令
            if client_id in self.last_tcp_command_time:
                time_since_last_command = current_time - self.last_tcp_command_time[client_id]
                if time_since_last_command < self.tcp_debounce_time:
                    logger.debug(f"防抖: 忽略来自 {client_id} 的重复catch命令 (距离上次: {time_since_last_command*1000:.1f}ms)")
                    return "0,0.000,0.000,0.000,0.000"
            
            # 检查是否正在处理该客户端的命令
            if self.tcp_processing_flags.get(client_id, False):
                logger.debug(f"防抖: 客户端 {client_id} 的命令正在处理中，忽略新命令")
                return "0,0.000,0.000,0.000,0.000"
            
            # 记录命令时间和处理状态
            self.last_tcp_command_time[client_id] = current_time
            self.tcp_processing_flags[client_id] = True
            
            # 获取系统组件
            camera = self.initializer.get_camera()
            detector = self.initializer.get_detector()
            configManager = self.initializer.get_config_manager()
            
            # 检查组件是否可用
            if not camera:
                elapsed_time = (time.time() - process_start_time) * 1000
                logger.warning(f"相机未就绪，向 {client_id} 发送默认数据 (耗时: {elapsed_time:.1f}ms)")
                self.initializer._notify_component_failure("camera", f"相机未初始化或未连接 (客户端: {client_id})")
                return "0,0.000,0.000,0.000,0.000"
            
            if not detector:
                elapsed_time = (time.time() - process_start_time) * 1000
                logger.warning(f"检测器未初始化，向 {client_id} 发送默认数据 (耗时: {elapsed_time:.1f}ms)")
                self.initializer._notify_component_failure("detector", f"检测器未初始化 (客户端: {client_id})")
                return "0,0.000,0.000,0.000,0.000"
            
            if not configManager:
                elapsed_time = (time.time() - process_start_time) * 1000
                logger.warning(f"配置管理器未初始化，向 {client_id} 发送默认数据 (耗时: {elapsed_time:.1f}ms)")
                self.initializer._notify_component_failure("configManager", f"配置管理器未初始化 (客户端: {client_id})")
                return "0,0.000,0.000,0.000,0.000"

            # 获取配置
            enable_roi = configManager.get_config("roi.enable")
            enable_stability = configManager.get_config("stability.isEnabled")
            
            # 处理配置值的默认值
            enable_roi = enable_roi if enable_roi is not None else False
            enable_stability = enable_stability if enable_stability is not None else False
            
            # 记录相机模式信息
            camera_mode = "单步模式" if camera.use_single_step else "连续流模式"
            logger.info(f"相机模式: {camera_mode}")
            
            # 执行检测（单步模式推荐使用单次检测）
            detection_start_time = time.time()
            if enable_stability and not camera.use_single_step:
                # 仅在连续流模式下才建议使用稳定性检测
                logger.info("连续流模式下执行稳定性检测")
                final_result = self._perform_stable_detection(client_id, camera, detector, configManager)
            else:
                # 单步模式下或禁用稳定性检测时，执行单次检测
                if camera.use_single_step:
                    logger.info("单步模式: 执行快速单次检测（推荐）")
                else:
                    logger.info("稳定性检测已禁用: 执行单次检测")
                final_result = self._perform_single_detection(camera, detector, configManager)
            
            detection_time = (time.time() - detection_start_time) * 1000  # 转换为毫秒
            
            # 构造并发送响应数据
            response_start_time = time.time()
            if final_result:
                detection_count = final_result.get('detection_count', 0)
                best_target = final_result.get('best_target')
                
                if best_target is not None and best_target.get('robot_3d') is not None:
                    # 有有效目标，发送完整坐标数据
                    x, y, z = best_target['robot_3d']
                    angle = best_target.get('angle', 0.0)
                    
                    # 应用Z坐标补偿
                    z_compensated = z + self.z_offset
                    
                    # 构造数据格式：检测个数,x,y,z,angle
                    response_data = f"{detection_count},{x:.3f},{y:.3f},{z_compensated:.3f},{angle:.3f}"
                    
                    # 计算总耗时和响应构造时间
                    total_time = (time.time() - process_start_time) * 1000
                    response_time = (time.time() - response_start_time) * 1000
                    
                    # 获取检测过程中的详细耗时信息
                    timing_info = final_result.get('timing', {})
                    frame_time = timing_info.get('frame_time', 0)
                    roi_time = timing_info.get('roi_time', 0)
                    model_detect_time = timing_info.get('detect_time', 0)
                    coord_time = timing_info.get('coord_time', 0)
                    save_time = timing_info.get('save_time', 0)
                    
                    logger.info(f"检测完成，发送坐标给 {client_id}: 检测个数={detection_count}, "
                               f"X={x:.3f}, Y={y:.3f}, Z={z_compensated:.3f}, A={angle:.3f}")
                    logger.info(f"详细耗时分析:")
                    logger.info(f"  1. 获取图像: {frame_time:.1f}ms")
                    logger.info(f"  2. ROI计算: {roi_time:.1f}ms")
                    logger.info(f"  3. 模型检测: {model_detect_time:.1f}ms")
                    logger.info(f"  4. 坐标转换: {coord_time:.1f}ms")
                    logger.info(f"  5. 保存图像: {save_time:.1f}ms")
                    logger.info(f"  6. 响应构造: {response_time:.1f}ms")
                    logger.info(f"  总耗时: {total_time:.1f}ms")
                    
                    return response_data
                else:
                    # 没有有效目标或坐标转换失败，发送检测个数为0
                    response_data = "0,0.000,0.000,0.000,0.000"
                        
                    # 计算总耗时和响应构造时间
                    total_time = (time.time() - process_start_time) * 1000
                    response_time = (time.time() - response_start_time) * 1000
                    
                    # 获取检测过程中的详细耗时信息
                    timing_info = final_result.get('timing', {})
                    frame_time = timing_info.get('frame_time', 0)
                    roi_time = timing_info.get('roi_time', 0)
                    model_detect_time = timing_info.get('detect_time', 0)
                    coord_time = timing_info.get('coord_time', 0)
                    save_time = timing_info.get('save_time', 0)
                        
                    logger.info(f"检测完成，但没有有效目标，发送默认数据给 {client_id}")
                    logger.info(f"详细耗时分析:")
                    logger.info(f"  1. 获取图像: {frame_time:.1f}ms")
                    logger.info(f"  2. ROI计算: {roi_time:.1f}ms")
                    logger.info(f"  3. 模型检测: {model_detect_time:.1f}ms")
                    logger.info(f"  4. 坐标转换: {coord_time:.1f}ms")
                    logger.info(f"  5. 保存图像: {save_time:.1f}ms")
                    logger.info(f"  6. 响应构造: {response_time:.1f}ms")
                    logger.info(f"  总耗时: {total_time:.1f}ms")
                    
                    return response_data
            else:
                # 检测失败，发送默认数据
                response_data = "0,0.000,0.000,0.000,0.000"
                    
                # 计算总耗时和响应构造时间
                total_time = (time.time() - process_start_time) * 1000
                response_time = (time.time() - response_start_time) * 1000
                    
                logger.error(f"检测失败，发送默认数据给 {client_id}")
                logger.error(f"详细耗时分析:")
                logger.error(f"  检测耗时: {detection_time:.1f}ms")
                logger.error(f"  响应构造: {response_time:.1f}ms")
                logger.error(f"  总耗时: {total_time:.1f}ms")
                
                return response_data
                        
        except Exception as e:
            total_time = (time.time() - process_start_time) * 1000
            logger.error(f"处理 {client_id} 的catch命令时出错: {str(e)} (总耗时: {total_time:.1f}ms)", exc_info=True)
            # 通过MQTT发送错误通知
            self.initializer._notify_component_failure("catch_execution", f"catch执行失败: {str(e)} (客户端: {client_id})")
            return "0,0.000,0.000,0.000,0.000"
        finally:
            # 清除处理标志，允许后续命令处理
            if client_id in self.tcp_processing_flags:
                self.tcp_processing_flags[client_id] = False
    
    def _perform_stable_detection(self, client_id: str, camera, detector, configManager):
        """
        执行稳定性检测，直到连续两次结果稳定或达到最大检测次数
        
        Args:
            client_id: 客户端ID
            camera: 相机实例
            detector: 检测器实例
            configManager: 配置管理器
            
        Returns:
            dict: 最终检测结果
        """
        logger = self.initializer.logger
        detection_results = []
        last_stable_result = None
        
        stability_num = configManager.get_config("stability.detectionCount")
        stability_num = stability_num if stability_num is not None else 5
        
        # 从配置获取稳定性检测延迟
        stabilize_delay = configManager.get_config("camera.buffer.stabilizeDelay")
        stabilize_delay = stabilize_delay if stabilize_delay is not None else 0.1
        
        logger.info(f"开始稳定性检测，最大尝试次数: {stability_num}")
        
        for attempt in range(stability_num):
            try:
                # 在检测之间添加配置的延迟，确保相机缓冲区稳定
                if attempt > 0:
                    logger.debug(f"等待相机稳定，准备第{attempt + 1}次检测")
                    time.sleep(stabilize_delay)
                
                # 执行2D检测（带重试机制）
                detection_result = self._perform_2d_detection_with_retry(camera, detector, configManager, attempt + 1)
                
                if detection_result is None:
                    logger.warning(f"第{attempt + 1}次检测失败")
                    continue
                        
                detection_results.append(detection_result)
                
                # 如果是第一次检测，记录结果并继续
                if attempt == 0:
                    logger.debug(f"第{attempt + 1}次检测完成，检测到{detection_result.get('detection_count', 0)}个目标")
                    continue
                
                # 从第二次开始检查稳定性
                previous_result = detection_results[attempt - 1]
                current_result = detection_result
                
                # 检查稳定性（基于2D坐标）
                is_stable = self._check_detection_stability(current_result, previous_result)
                
                if is_stable:
                    logger.info(f"第{attempt + 1}次检测：结果稳定，开始3D坐标计算")
                    # 检测稳定后，进行3D坐标转换
                    final_result = self._perform_3d_coordinate_calculation(current_result, camera, detector, configManager)
                    last_stable_result = final_result
                    break
                else:
                    logger.debug(f"第{attempt + 1}次检测：结果不稳定，继续检测")
                    
            except Exception as e:
                logger.error(f"第{attempt + 1}次稳定性检测时出错: {str(e)}")
                continue
        
        # 如果没有达到稳定状态，使用最后一次检测结果
        if last_stable_result is None and detection_results:
            logger.warning("未达到稳定状态，使用最后一次检测结果")
            last_result = detection_results[-1]
            last_stable_result = self._perform_3d_coordinate_calculation(last_result, camera, detector, configManager)
            
        return last_stable_result

    def _perform_2d_detection_with_retry(self, camera, detector, configManager, attempt_num):
        """
        执行2D检测，带重试机制处理缓冲区错误
        
        Args:
            camera: 相机实例
            detector: 检测器实例
            configManager: 配置管理器
            attempt_num: 检测尝试次数
        
        Returns:
            dict: 2D检测结果字典
        """
        logger = self.initializer.logger
        
        # 从配置获取重试参数
        max_retries = configManager.get_config("camera.buffer.retryAttempts")
        max_retries = max_retries if max_retries is not None else 3
        
        retry_delay = configManager.get_config("camera.buffer.retryDelay")
        retry_delay = retry_delay if retry_delay is not None else 0.2
        
        for retry in range(max_retries):
            try:
                # 记录检测流程的开始时间
                total_start_time = time.time()
                
                # 获取相机帧数据，带错误处理
                frame_start_time = time.time()
                success, depth_data, frame, camera_params = self._get_camera_frame_safe(camera, retry, configManager)
                frame_time = (time.time() - frame_start_time) * 1000  # 转换为毫秒
                
                if not success or frame is None:
                    if retry < max_retries - 1:
                        logger.warning(f"第{attempt_num}次检测第{retry + 1}次取图失败，重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"第{attempt_num}次检测所有重试均失败")
                        return None
                
                # 如果ROI启用，对图像进行ROI裁剪
                enable_roi = configManager.get_config("roi.enable")
                enable_roi = enable_roi if enable_roi is not None else False
                if enable_roi:
                    roi_config = self._get_roi_config()
                frame = self._apply_roi_to_image(frame, roi_config)
                        
                        # 执行检测
                detect_start_time = time.time()
                results = detector.detect(frame)
                detect_time = (time.time() - detect_start_time) * 1000  # 转换为毫秒
                
                # ROI过滤和2D结果处理
                roi_start_time = time.time()
                detection_count = 0
                filtered_results_for_display = []
                best_target_2d = None
                
                if results:
                    # 获取图像尺寸
                    image_height, image_width = frame.shape[:2]
                    
                    for result in results:
                        if not (hasattr(result, 'pt1x') and hasattr(result, 'pt1y')):
                            continue
                        
                        # 计算检测框中心点
                        center_x = (result.pt1x + result.pt2x + result.pt3x + result.pt4x) / 4
                        center_y = (result.pt1y + result.pt2y + result.pt3y + result.pt4y) / 4
                        
                        # ROI过滤（如果启用）
                        if enable_roi:
                            if not self._is_detection_in_roi_fast(center_x, center_y, image_width, image_height):
                                continue
                        
                        detection_count += 1
                        filtered_results_for_display.append(result)
                        
                        # 记录第一个有效目标作为最佳目标（简化版）
                        if best_target_2d is None:
                            best_target_2d = {
                                'center': [center_x, center_y],
                                'corners': [
                                    [result.pt1x, result.pt1y],
                                    [result.pt2x, result.pt2y],
                                    [result.pt3x, result.pt3y],
                                    [result.pt4x, result.pt4y]
                                ],
                                'result': result
                            }
                
                roi_time = (time.time() - roi_start_time) * 1000  # 转换为毫秒
                total_time = (time.time() - total_start_time) * 1000
                
                # 构造2D检测结果
                detection_result = {
                    'detection_count': detection_count,
                    'best_target_2d': best_target_2d,
                    'timing': {
                        'frame_time': frame_time,
                        'detect_time': detect_time,
                        'roi_time': roi_time,
                        'total_time': total_time
                    },
                    'raw_data': {
                        'results': results,
                        'depth_data': depth_data,
                        'camera_params': camera_params,
                        'frame': frame,
                        'filtered_results': filtered_results_for_display
                    }
                }
                
                return detection_result
                
            except Exception as e:
                if "buffer" in str(e).lower() or "magic word" in str(e).lower():
                    logger.warning(f"第{attempt_num}次检测第{retry + 1}次遇到缓冲区错误: {str(e)}")
                    if retry < max_retries - 1:
                        # 缓冲区错误时，尝试清理缓冲区
                        self._clear_camera_buffer(camera, configManager)
                        time.sleep(retry_delay * 1.5)  # 缓冲区错误时等待更长时间
                        continue
                    else:
                        logger.error(f"第{attempt_num}次检测第{retry + 1}次其他错误: {str(e)}")
                    if retry < max_retries - 1:
                        time.sleep(retry_delay / 2)  # 其他错误时等待较短时间
                        continue
                
        return None

    def _get_camera_frame_safe(self, camera, retry_count, configManager=None):
        """
        安全获取相机帧数据，处理缓冲区错误
        
        Args:
            camera: 相机实例
            retry_count: 重试次数
            configManager: 配置管理器（可选）
            
        Returns:
            tuple: (success, depth_data, frame, camera_params)
        """
        logger = self.initializer.logger
        
        try:
            if retry_count == 0:
                # 第一次尝试使用get_fresh_frame
                return camera.get_fresh_frame()
            else:
                # 重试时使用普通的get_frame，避免缓冲区操作
                return camera.get_frame()
                
        except Exception as e:
            if "buffer" in str(e).lower() or "magic word" in str(e).lower():
                logger.warning(f"相机缓冲区错误: {str(e)}")
                # 尝试清理缓冲区
                self._clear_camera_buffer(camera, configManager)
                return False, None, None, None
            else:
                logger.error(f"相机获取帧时出错: {str(e)}")
                return False, None, None, None

    def _clear_camera_buffer(self, camera, configManager=None):
        """
        清理相机缓冲区，尝试恢复正常状态
        
        Args:
            camera: 相机实例
            configManager: 配置管理器（可选）
        """
        logger = self.initializer.logger
        
        # 从配置获取清理参数
        clear_attempts = 5
        if configManager:
            clear_attempts = configManager.get_config("camera.buffer.clearAttempts")
            clear_attempts = clear_attempts if clear_attempts is not None else 5
        
        try:
            # 尝试重新连接相机以清理缓冲区
            if hasattr(camera, 'streaming_device') and camera.streaming_device:
                old_timeout = camera.streaming_device.sock_stream.gettimeout()
                # 设置很短的超时时间，快速清理缓冲区
                camera.streaming_device.sock_stream.settimeout(0.01)
                
                # 尝试读取并丢弃缓冲区中的数据
                for i in range(clear_attempts):
                    try:
                        camera.streaming_device.getFrame()
                        logger.debug(f"清理缓冲区第{i+1}次")
                    except:
                        break
                
                # 恢复原始超时设置
                camera.streaming_device.sock_stream.settimeout(old_timeout)
                logger.debug("相机缓冲区清理完成")
                
        except Exception as e:
            logger.debug(f"清理相机缓冲区时出错: {str(e)}")

    def _perform_single_detection(self, camera, detector: RKNN_YOLO, configManager):
        """
        执行单次检测 - 优化版本，专注于核心检测性能
        
        Args:
            camera: 相机实例
            detector: 检测器实例
            configManager: 配置管理器
        
        Returns:
            dict: 检测结果字典
        """
        logger = self.initializer.logger
        try:
            # 记录整个检测流程的开始时间
            total_start_time = time.time()
            
            # 步骤1: 获取相机帧数据 - 高性能版本
            frame_start_time = time.time()
            success, depth_data, frame, camera_params = camera.get_frame()
            frame_time = (time.time() - frame_start_time) * 1000
            
            if not success or frame is None:
                logger.error("无法获取有效的相机帧数据")
                return None
            
            # 步骤2: 检查ROI配置（快速检查，不进行复杂计算）
            roi_start_time = time.time()
            enable_roi = configManager.get_config("roi.enable")
            roi_config = None
            
            if enable_roi:
                # 简化的ROI配置获取，只计算检测需要的参数
                roi_settings = self._get_roi_config()
                if roi_settings and roi_settings.get("enable"):
                    img_height, img_width = frame.shape[:2]
                    roi_width = roi_settings.get("width", 0)
                    roi_height = roi_settings.get("height", 0)
                    center_x_offset = roi_settings.get("centerXOffset", 0)
                    center_y_offset = roi_settings.get("centerYOffset", 0)
                    
                    center_x = img_width // 2 + center_x_offset
                    center_y = img_height // 2 + center_y_offset
                    
                    x1 = max(0, center_x - roi_width // 2)
                    y1 = max(0, center_y - roi_height // 2)
                    x2 = min(img_width, x1 + roi_width)
                    y2 = min(img_height, y1 + roi_height)
                    
                    roi_config = {
                        'enabled': True,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                    }
            
            roi_time = (time.time() - roi_start_time) * 1000
            
            # 步骤3: 执行纯检测（不包含绘制和保存）
            detect_start_time = time.time()
            
            # 使用优化的检测方法，只做检测和坐标转换
            detection_results = detector.detect(frame)
            detect_time = (time.time() - detect_start_time) * 1000
            
            # 步骤4: 坐标转换（如果有检测结果）
            coord_start_time = time.time()
            best_target = None
            detection_count = 0
            
            if detection_results and depth_data and camera_params:
                # 使用高性能坐标转换
                best_target = detector.process_detection_to_coordinates_fast(
                    detection_results, depth_data, camera_params, roi_config
                )
                
                # 计算最终检测数量（考虑ROI过滤）
                if roi_config and roi_config.get('enabled'):
                    roi_x1, roi_y1 = roi_config['x1'], roi_config['y1']
                    roi_x2, roi_y2 = roi_config['x2'], roi_config['y2']
                    
                    for result in detection_results:
                        center_x = (result.pt1x + result.pt2x + result.pt3x + result.pt4x) * 0.25
                        center_y = (result.pt1y + result.pt2y + result.pt3y + result.pt4y) * 0.25
                        if roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2:
                            detection_count += 1
                else:
                    detection_count = len(detection_results)
            
            coord_time = (time.time() - coord_start_time) * 1000
            
            # 计算总耗时（只包含核心检测部分）
            core_total_time = (time.time() - total_start_time) * 1000
            
            # 步骤5: 异步保存图像（如果启用）- 不计入检测耗时
            save_start_time = time.time()
            annotated_image = None
            
            # 检查是否需要绘制和保存图像
            save_config = configManager.get_config("detection.saveImage")
            if save_config and save_config.get("enabled", False):
                try:
                    # 创建带注释的图像用于保存，但不计入检测耗时
                    annotated_image = detector._draw_detection_annotations(
                        frame.copy(), detection_results, best_target, 
                        detect_time, coord_time, core_total_time, roi_config
                    )
                    
                    # 异步保存图像（不阻塞检测流程）
                    import threading
                    save_data = {
                        'detection_count': detection_count,
                        'best_target': best_target,
                        'annotated_image': annotated_image
                    }
                    threading.Thread(
                        target=self._save_detection_image_async,
                        args=(save_data, "single_detection"),
                        daemon=True
                    ).start()
                    
                except Exception as e:
                    logger.debug(f"图像处理失败: {e}")
            
            save_time = (time.time() - save_start_time) * 1000
            
            # 构造结果（只包含核心检测数据）
            result = {
                'detection_count': detection_count,
                'best_target': best_target,
                'annotated_image': annotated_image,  # 可能为None
                'original_image': frame,
                'timing': {
                    'frame_time': frame_time,
                    'roi_time': roi_time,
                    'detect_time': detect_time,
                    'coord_time': coord_time,
                    'save_time': save_time,
                    'total_time': core_total_time  # 只包含核心检测时间
                }
            }
            
            # 输出简化的耗时统计
            logger.info(f"检测到{detection_count}个目标, 总耗时: {core_total_time:.1f}ms")
            logger.info(f"详细耗时: 获取图像={frame_time:.1f}ms, 检测={detect_time:.1f}ms, "
                       f"坐标转换={coord_time:.1f}ms, ROI计算={roi_time:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"单次检测时出错: {str(e)}")
            return None

    def _save_detection_image_async(self, save_data: dict, client_id: str):
        """
        异步保存检测结果图像到SFTP，不阻塞主检测流程
        
        Args:
            save_data: 包含检测结果和图像的字典
            client_id: 客户端ID，用于文件命名
        """
        logger = self.initializer.logger
        
        try:
            # 获取配置管理器
            configManager = self.initializer.get_config_manager()
            if not configManager:
                return
            
            # 获取绘制好的图像
            annotated_image = save_data.get('annotated_image')
            if annotated_image is None:
                return
            
            # 获取SFTP客户端
            sftp_client = self.initializer.get_sftp_client()
            if not sftp_client or not sftp_client.connected:
                logger.debug("SFTP客户端未连接，跳过图像上传")
                return
            
            # 生成文件前缀
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            client_safe = client_id.replace(":", "_").replace(".", "_")
            
            # 根据检测结果添加标识
            best_target = save_data.get('best_target')
            if best_target and best_target.get('robot_3d'):
                result_tag = "SUCCESS"
                x, y, z = best_target['robot_3d']
                angle = best_target.get('angle', 0.0)
                coord_info = f"_x{x:.1f}_y{y:.1f}_z{z:.1f}_a{angle:.1f}"
            else:
                result_tag = "NO_TARGET"
                coord_info = ""
            
            # 构造文件前缀
            prefix = f"detection_{timestamp}_{client_safe}_{result_tag}{coord_info}"
            
            # 获取远程路径配置
            save_config = configManager.get_config("detection.saveImage")
            remote_path = save_config.get("remote_path")
            
            # 使用SFTP客户端上传图像
            result = sftp_client.upload_image(
                image_data=annotated_image,
                image_format="jpg",
                prefix=prefix,
                remote_path=remote_path
            )
            
            if result["success"]:
                logger.debug(f"检测图像异步上传成功: {result['filename']}")
                
                # 通过MQTT通知工控机图像更新
                mqtt_client = self.initializer.get_mqtt_client()
                if mqtt_client:
                    from ClassModel.MqttResponse import MQTTResponse
                    from SystemEnums.VisionCoreCommands import MessageType
                    
                    # 构造检测结果数据
                    detection_data = {
                        "filename": result["filename"],
                        "remote_path": result["remote_path"],
                        "file_size": result["file_size"],
                        "detection_count": save_data.get('detection_count', 0),
                        "client_id": client_id,
                        "timestamp": timestamp,
                        "image_type": "detection_result"
                    }
                    
                    # 如果有检测目标，添加坐标信息
                    if best_target and best_target.get('robot_3d'):
                        x, y, z = best_target['robot_3d']
                        angle = best_target.get('angle', 0.0)
                        detection_data.update({
                            "target_found": True,
                            "coordinates": {
                                "x": round(x, 3),
                                "y": round(y, 3),
                                "z": round(z, 3),
                                "angle": round(angle, 3)
                            }
                        })
                    else:
                        detection_data["target_found"] = False
                    
                    # 发送MQTT通知
                    response = MQTTResponse(
                        command="detection_image_upload",
                        component="detection",
                        messageType=MessageType.SUCCESS,
                        message=f"检测图像上传成功: {result['filename']}",
                        data=detection_data
                    )
                    
                    mqtt_client.send_mqtt_response(response)
                    
            else:
                logger.warning(f"检测图像异步上传失败: {result['message']}")
                
        except Exception as e:
            logger.warning(f"异步上传检测图像异常: {e}")

    def _cleanup_old_images(self, save_dir: str, max_files: int):
        """
        清理旧的图像文件
        
        Args:
            save_dir: 图像保存目录
            max_files: 最大文件数量
        """
        try:
            import os
            import glob
            
            pattern = os.path.join(save_dir, "*.jpg")
            files = glob.glob(pattern)
            
            if len(files) <= max_files:
                return
            
            files.sort(key=os.path.getmtime)
            files_to_delete = files[:-max_files]
            
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                except Exception as e:
                    self.initializer.logger.warning(f"删除文件失败 {file_path}: {e}")
                    
            if files_to_delete:
                self.initializer.logger.info(f"清理了 {len(files_to_delete)} 个旧图像文件")
                
        except Exception as e:
            self.initializer.logger.error(f"清理旧图像文件异常: {e}")

    def _perform_2d_detection(self, camera, detector, configManager):
        """
        执行2D检测 - 已废弃，请使用_perform_single_detection
        """
        return self._perform_single_detection(camera, detector, configManager)

    def _perform_3d_coordinate_calculation(self, detection_2d_result, camera, detector, configManager):
        """
        执行3D坐标计算 - 已废弃，请使用detector.detect_with_coordinates
        """
        logger = self.initializer.logger
        logger.warning("_perform_3d_coordinate_calculation方法已废弃，请使用detector.detect_with_coordinates")
        return None

    def _check_detection_stability(self, current_result, previous_result):
        """
        检查检测稳定性 - 已废弃，单步模式下不需要
        """
        logger = self.initializer.logger
        logger.warning("_check_detection_stability方法已废弃，单步模式下不需要稳定性检测")
        return False

    def _process_detection_to_coordinates_fast(self, results, depth_data, camera_params, detector):
        """
        处理检测结果到坐标 - 已废弃，请使用detector.process_detection_to_coordinates_fast
        """
        logger = self.initializer.logger
        logger.warning("_process_detection_to_coordinates_fast方法已废弃，请使用detector.process_detection_to_coordinates_fast")
        return None

    def _is_detection_in_roi_fast(self, image_x: float, image_y: float, image_width: int, image_height: int) -> bool:
        """
        检查检测是否在ROI内 - 已废弃，ROI处理已集成到detector中
        """
        logger = self.initializer.logger
        logger.warning("_is_detection_in_roi_fast方法已废弃，ROI处理已集成到detector中")
        return True
    
    def handle_mqtt_message(self, mqtt_message):
        """
        处理MQTT消息 - 主要用于配置更新
        
        Args:
            mqtt_message: MQTT消息对象，包含topic、payload、qos等信息
        """
        logger = self.initializer.logger
        
        try:
            topic = mqtt_message.topic
            payload = mqtt_message.payload
            
            logger.info(f"收到MQTT消息: {topic}")
            logger.debug(f"消息内容: {payload}")

            
            
            # # 根据主题处理不同的消息
            # if topic == "sickvision/config/update":
            #     self._handle_mqtt_config_update(payload)
            if topic == "sickvision/system/command":
                self._handle_mqtt_system_command(payload)
            else:
                logger.debug(f"未处理的MQTT主题: {topic}")
                
        except Exception as e:
            logger.error(f"处理MQTT消息时出错: {e}")
    
    def _handle_mqtt_config_update(self, payload):
        """
        处理配置更新
        
        Args:
            payload: 配置更新数据，包含要更新的配置项
        """
        logger = self.initializer.logger
        logger.info(f"收到MQTT系统命令: {payload}")
        payload = json.loads(payload)
        
        try:
            if not isinstance(payload, dict):
                error_msg = "配置数据格式错误，必须是字典格式"
                logger.error(error_msg)
                self._send_config_response(False, error_msg)
                return
            
            # 获取更新类型 # partial: 部分更新, full: 完整更新
            config_data = payload
            restart_required = payload.get("restart_required", False)
            
            if not config_data:
                error_msg = "配置数据为空"
                logger.error(error_msg)
                self._send_config_response(False, error_msg)
                return
            
            logger.info(f"开始更新配置")
            
            # 备份当前配置
            backup_success = self._backup_config()
            if not backup_success:
                error_msg = "配置备份失败"
                logger.error(error_msg)
                self._send_config_response(False, error_msg)
                return
            
            # 更新配置文件
            update_success = self._update_config_file(config_data)
            if not update_success:
                error_msg = "配置文件更新失败"
                logger.error(error_msg)
                # 恢复备份
                self._restore_config_backup()
                self._send_config_response(False, error_msg)
                return
            
            # 根据restart_required决定重启还是热重载
            if restart_required:
                logger.info("配置更新完成，准备重启系统...")
                self._send_config_response(True, "配置更新成功，系统将重启")
                # 延迟重启，先发送响应
                import threading
                threading.Timer(2.0, self._restart_system).start()
            else:
                # 尝试热重载
                reload_success = self._hot_reload_config(config_data)
                if reload_success:
                    logger.info("配置热重载成功")
                    self._send_config_response(True, "配置更新成功，已热重载")
                else:
                    logger.warning("配置热重载失败，建议重启系统")
                    self._send_config_response(True, "配置更新成功，但热重载失败，建议重启系统")
                
        except Exception as e:
            logger.error(f"处理配置更新时出错: {e}")
            # 尝试恢复备份
            self._restore_config_backup()
            self._send_config_response(False, f"配置更新失败: {str(e)}")
    
    def _backup_config(self) -> bool:
        """备份当前配置文件（最多保留10个备份）"""
        try:
            import shutil
            import os
            import glob
            from datetime import datetime
            
            config_path = self.config_path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{config_path}.backup_{timestamp}"
            
            # 创建备份
            shutil.copy2(config_path, backup_path)
            
            # 保存最新备份路径
            self.latest_backup = backup_path
            
            self.initializer.logger.info(f"配置文件已备份到: {backup_path}")
            
            # 清理旧备份文件，只保留最新的10个
            self._cleanup_old_backups(config_path, max_backups=10)
            
            return True
            
        except Exception as e:
            self.initializer.logger.error(f"配置备份失败: {e}")
            return False
    
    def _cleanup_old_backups(self, config_path: str, max_backups: int = 10):
        """清理旧的备份文件，只保留最新的指定数量"""
        try:
            import glob
            import os
            
            # 查找所有备份文件
            backup_pattern = f"{config_path}.backup_*"
            backup_files = glob.glob(backup_pattern)
            
            if len(backup_files) <= max_backups:
                return  # 备份文件数量未超过限制
            
            # 按文件修改时间排序（最新的在前）
            backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # 删除超出数量限制的旧备份文件
            files_to_delete = backup_files[max_backups:]
            deleted_count = 0
            
            for old_backup in files_to_delete:
                try:
                    os.remove(old_backup)
                    deleted_count += 1
                    self.initializer.logger.debug(f"已删除旧备份文件: {old_backup}")
                except Exception as e:
                    self.initializer.logger.warning(f"删除备份文件失败: {old_backup}, 错误: {e}")
            
            if deleted_count > 0:
                remaining_count = len(backup_files) - deleted_count
                self.initializer.logger.info(f"已清理 {deleted_count} 个旧备份文件，当前保留 {remaining_count} 个备份")
                
        except Exception as e:
            self.initializer.logger.error(f"清理备份文件时出错: {e}")
    
    
    def _restore_config_backup(self) -> bool:
        """恢复配置备份"""
        try:
            import os
            import shutil
            if os.path.exists(self.latest_backup):
                shutil.copy2(self.latest_backup, self.config_path)
                self.initializer.logger.info("配置文件已从备份恢复")
                return True
            return False
        except Exception as e:
            self.initializer.logger.error(f"配置恢复失败: {e}")
            return False
    
    def _update_config_file(self, config_data: dict) -> bool:
        """
        更新配置文件
        
        Args:
            config_data: 要更新的配置数据
            update_type: 更新类型 (partial/full)
        """
        try:
            import yaml
            # 部分更新配置
            # 读取当前配置
            with open(self.config_path, 'r', encoding='utf-8') as f:
                current_config = yaml.safe_load(f) or {}
            
            # 深度合并配置
            new_config = self._deep_merge_dict(current_config, config_data)
            
            # 写入新配置
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(new_config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            self.initializer.logger.info("配置文件更新成功")
            return True
            
        except Exception as e:
            self.initializer.logger.error(f"更新配置文件失败: {e}")
            return False
    
    def _deep_merge_dict(self, base: dict, update: dict) -> dict:
        """深度合并字典（忽略大小写）"""
        result = base.copy()
        
        # 创建基础字典的小写键映射
        base_lower_keys = {k.lower(): k for k in base.keys()}
        
        for key, value in update.items():
            # 查找匹配的键（忽略大小写）
            matching_key = base_lower_keys.get(key.lower(), key)
            
            if matching_key in result and isinstance(result[matching_key], dict) and isinstance(value, dict):
                # 递归合并字典
                result[matching_key] = self._deep_merge_dict(result[matching_key], value)
            else:
                # 如果找到了大小写不同的匹配键，使用原始键名
                if matching_key != key and matching_key in result:
                    # 删除旧键，使用新键
                    del result[matching_key]
                result[key] = value
        
        return result
    
    def _hot_reload_config(self, config_data: dict) -> bool:
        """
        热重载配置（不重启系统）
        
        Args:
            config_data: 新的配置数据
        """
        logger = self.initializer.logger
        try:
            logger.info("开始热重载配置...")
            
            # 重新加载配置管理器
            if not self.initializer.initialize_config():
                logger.error("重新加载配置管理器失败")
                return False
            
            success_count = 0
            total_components = 0
            
            # 根据配置变化决定需要重启的组件
            if "detectionServer" in config_data:
                total_components += 1
                if self._reload_tcp_server():
                    success_count += 1
                    logger.info("TCP服务器配置热重载成功")
                else:
                    logger.error("TCP服务器配置热重载失败")
            
            if "mqtt" in config_data:
                total_components += 1
                if self._reload_mqtt_client():
                    success_count += 1
                    logger.info("MQTT客户端配置热重载成功")
                else:
                    logger.error("MQTT客户端配置热重载失败")
            
            if "camera" in config_data:
                total_components += 1
                if self._reload_camera():
                    success_count += 1
                    logger.info("相机配置热重载成功")
                else:
                    logger.error("相机配置热重载失败")
            
            if "logging" in config_data:
                total_components += 1
                if self._reload_logging():
                    success_count += 1
                    logger.info("日志配置热重载成功")
                else:
                    logger.error("日志配置热重载失败")
            
            # 检查模型相关配置更新
            if "model" in config_data:
                total_components += 1
                if self._reload_detector():
                    success_count += 1
                    logger.info("检测器配置热重载成功")
                else:
                    logger.error("检测器配置热重载失败")
            
            # 如果没有指定组件，说明可能是系统级配置，不需要重启组件
            if total_components == 0:
                logger.info("系统级配置更新，无需重启组件")
                return True
            
            # 判断热重载是否成功
            if success_count == total_components:
                logger.info(f"所有组件配置热重载成功 ({success_count}/{total_components})")
                return True
            else:
                logger.warning(f"部分组件配置热重载失败 ({success_count}/{total_components})")
                return False
                
        except Exception as e:
            logger.error(f"热重载配置时出错: {e}")
            return False
    
    def _reload_tcp_server(self) -> bool:
        """重启TCP服务器以应用新配置"""
        try:
            if self.initializer._restart_tcp_server():
                # 重新设置回调
                tcp_server = self.initializer.get_tcp_server()
                if tcp_server:
                    message_handler = self.initializer.create_tcp_message_handler(self.handle_catch)
                    tcp_server.set_message_callback(message_handler)
                return True
            return False
        except Exception:
            return False
    
    def _reload_mqtt_client(self) -> bool:
        """重启MQTT客户端以应用新配置"""
        try:
            if self.initializer._restart_mqtt():
                # 消息回调会自动恢复，无需手动设置
                return True
            return False
        except Exception:
            return False
    
    def _reload_camera(self) -> bool:
        """重启相机以应用新配置"""
        try:
            return self.initializer._restart_camera()
        except Exception:
            return False
    
    def _reload_logging(self) -> bool:
        """重新配置日志系统"""
        try:
            return self.initializer.initialize_logging()
        except Exception:
            return False
    
    def _reload_detector(self) -> bool:
        """重启检测器以应用新配置"""
        try:
            return self.initializer._restart_detector()
        except Exception:
            return False
    
    def _restart_system(self):
        """重启整个系统"""
        logger = self.initializer.logger
        logger.info("开始重启系统...")
        
        # 设置重启标志
        self.running = False
        self.restart_on_failure = True
    
    def _send_config_response(self, success: bool, message: str):
        """发送配置更新响应"""
        try:
            response = MQTTResponse(
                command="save_config",
                component="config_manager",
                messageType=MessageType.SUCCESS if success else MessageType.ERROR,
                message=message,
                data={"success": success}
            )
            mqtt_client = self.initializer.get_mqtt_client()
            if mqtt_client:
                mqtt_client.send_mqtt_response(response)
        except Exception as e:
            self.initializer.logger.error(f"发送配置响应失败: {e}")
    
    def _handle_mqtt_system_command(self, payload):
        """处理系统命令"""
        logger = self.initializer.logger
        logger.info(f"收到MQTT系统命令: {payload}")
        
        try:
            if not isinstance(payload, dict):
                logger.error("系统命令格式错误")
                return
            
            command = payload.get("command")
            
            if command == VisionCoreCommands.RESTART:
                logger.info("收到系统重启命令")
                response = MQTTResponse(
                    command=VisionCoreCommands.RESTART.value,
                    component="system",
                    messageType=MessageType.SUCCESS,
                    message="系统重启命令已接收，正在重启...",
                    data={}
                )
                mqtt_client = self.initializer.get_mqtt_client()
                if mqtt_client:
                    mqtt_client.send_mqtt_response(response)
                self._restart_system()
            # 获取配置
            elif command == VisionCoreCommands.GET_CONFIG:
                try:
                    import yaml
                    import os
                    import glob
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        current_config = yaml.safe_load(f)
                    models_dir = "./Models"
                    model_files = []
                    if os.path.exists(models_dir):
                        rknn_pattern = os.path.join(models_dir, "*.rknn")
                        rknn_paths = glob.glob(rknn_pattern)
                        pt_pattern = os.path.join(models_dir, "*.pt")
                        pt_paths = glob.glob(pt_pattern)
                        all_paths = rknn_paths + pt_paths
                        model_files = [os.path.basename(path) for path in all_paths]
                    current_config["models"] = model_files
                    
                    response = MQTTResponse(
                        command=VisionCoreCommands.GET_CONFIG.value,
                        component="config_manager",
                        messageType=MessageType.SUCCESS,
                        message="配置获取成功",
                        data=current_config
                    )
                    mqtt_client = self.initializer.get_mqtt_client()
                    if mqtt_client:
                        mqtt_client.send_mqtt_response(response)
                    
                except Exception as e:
                    logger.error(f"获取配置失败: {e}")
                    response = MQTTResponse(
                        command=VisionCoreCommands.GET_CONFIG.value,
                        component="config_manager",
                        messageType=MessageType.ERROR,
                        message=f"获取配置失败: {e}",
                        data={}
                    )
                    mqtt_client = self.initializer.get_mqtt_client()
                    if mqtt_client:
                        mqtt_client.send_mqtt_response(response)
            # 保存配置
            elif command == VisionCoreCommands.SAVE_CONFIG:
                self._handle_mqtt_config_update(payload.get("Data"))
            # 获取标定图像
            elif command == VisionCoreCommands.GET_CALIBRAT_IMAGE:
                try:
                    # 获取相机
                    camera = self.initializer.get_camera()
                    if not camera:
                        error_msg = "相机未连接或未初始化"
                        logger.error(error_msg)
                        response = MQTTResponse(
                            command=VisionCoreCommands.GET_CALIBRAT_IMAGE.value,
                            component="camera",
                            messageType=MessageType.ERROR,
                            message=error_msg,
                            data={}
                        )
                        mqtt_client = self.initializer.get_mqtt_client()
                        if mqtt_client:
                            mqtt_client.send_mqtt_response(response)
                        return
                    
                    # 获取标定图像
                    try:
                        _, _, image, _ = camera.get_fresh_frame()
                        if image is None:
                            error_msg = "无法获取标定图像"
                            logger.error(error_msg)
                            response = MQTTResponse(
                                command=VisionCoreCommands.GET_CALIBRAT_IMAGE.value,
                                component="camera",
                                messageType=MessageType.ERROR,
                                message=error_msg,
                                data={}
                            )
                            mqtt_client = self.initializer.get_mqtt_client()
                            if mqtt_client:
                                mqtt_client.send_mqtt_response(response)
                            return
                        
                        success_msg = f"标定图像获取成功，图像尺寸: {image.shape}"
                        logger.info(success_msg)
                        
                        response = MQTTResponse(
                            command=VisionCoreCommands.GET_CALIBRAT_IMAGE.value,
                            component="camera",
                            messageType=MessageType.SUCCESS,
                            message=success_msg,
                            data={
                                "image_shape": list(image.shape),
                                "image_type": str(type(image))
                            }
                        )
                        mqtt_client = self.initializer.get_mqtt_client()
                        if mqtt_client:
                            mqtt_client.send_mqtt_response(response)
                        
                    except Exception as image_error:
                        error_msg = f"获取标定图像失败: {image_error}"
                        logger.error(error_msg)
                        
                        # 检查是否是缓冲区错误或其他需要重启相机的错误
                        error_str = str(image_error).lower()
                        if ("buffer" in error_str or 
                            "something is wrong" in error_str or
                            "connection" in error_str or
                            "timeout" in error_str):
                            
                            logger.warning("检测到相机缓冲区或连接错误，尝试重启相机组件...")
                            
                            try:
                                # 尝试重启相机
                                restart_success = self.initializer._restart_camera()
                                if restart_success:
                                    logger.info("相机组件重启成功")
                                    # 发送重启成功的响应
                                    response = MQTTResponse(
                                        command=VisionCoreCommands.GET_CALIBRAT_IMAGE.value,
                                        component="camera",
                                        messageType=MessageType.ERROR,
                                        message=f"相机出现错误已自动重启: {error_msg}",
                                        data={
                                            "auto_restart": True,
                                            "restart_success": True,
                                            "original_error": error_msg
                                        }
                                    )
                                else:
                                    logger.error("相机组件重启失败")
                                    response = MQTTResponse(
                                        command=VisionCoreCommands.GET_CALIBRAT_IMAGE.value,
                                        component="camera",
                                        messageType=MessageType.ERROR,
                                        message=f"相机错误且重启失败: {error_msg}",
                                        data={
                                            "auto_restart": True,
                                            "restart_success": False,
                                            "original_error": error_msg
                                        }
                                    )
                            except Exception as restart_error:
                                logger.error(f"重启相机时出现异常: {restart_error}")
                                response = MQTTResponse(
                                    command=VisionCoreCommands.GET_CALIBRAT_IMAGE.value,
                                    component="camera",
                                    messageType=MessageType.ERROR,
                                    message=f"相机错误且重启异常: {error_msg}",
                                    data={
                                        "auto_restart": True,
                                        "restart_success": False,
                                        "restart_error": str(restart_error),
                                        "original_error": error_msg
                                    }
                                )
                        else:
                            # 其他类型的错误，不重启相机
                            response = MQTTResponse(
                                command=VisionCoreCommands.GET_CALIBRAT_IMAGE.value,
                                component="camera",
                                messageType=MessageType.ERROR,
                                message=error_msg,
                                data={}
                            )
                        
                        mqtt_client = self.initializer.get_mqtt_client()
                        if mqtt_client:
                            mqtt_client.send_mqtt_response(response)
                
                except Exception as e:
                    error_msg = f"执行标定图像获取命令失败: {e}"
                    logger.error(error_msg)
                    response = MQTTResponse(
                        command=VisionCoreCommands.GET_CALIBRAT_IMAGE.value,
                        component="camera",
                        messageType=MessageType.ERROR,
                        message=error_msg,
                        data={}
                    )
                    mqtt_client = self.initializer.get_mqtt_client()
                    if mqtt_client:
                        mqtt_client.send_mqtt_response(response)
            # 相机获取图像
            elif command == VisionCoreCommands.GET_IMAGE:
                # 获取相机图像并上传到SFTP
                logger.info("收到获取图像命令")
                
                try:
                    camera = self.initializer.get_camera()
                    sftp_client = self.initializer.get_sftp_client()
                    
                    # 检查相机是否可用
                    if not camera:
                        error_msg = "相机未连接或未初始化"
                        logger.error(error_msg)
                        response = MQTTResponse(
                            command=VisionCoreCommands.GET_IMAGE.value,
                            component="camera",
                            messageType=MessageType.ERROR,
                            message=error_msg,
                            data={}
                        )
                        mqtt_client = self.initializer.get_mqtt_client()
                        if mqtt_client:
                            mqtt_client.send_mqtt_response(response)
                        return
                    
                    # 检查SFTP是否可用
                    if not sftp_client or not sftp_client.connected:
                        error_msg = "SFTP客户端未连接或未初始化"
                        logger.error(error_msg)
                        response = MQTTResponse(
                            command=VisionCoreCommands.GET_IMAGE.value,
                            component="sftp",
                            messageType=MessageType.ERROR,
                            message=error_msg,
                            data={}
                        )
                        mqtt_client = self.initializer.get_mqtt_client()
                        if mqtt_client:
                            mqtt_client.send_mqtt_response(response)
                        return
                    
                    logger.info("正在获取相机图像...")
                    
                    # 获取相机图像
                    try:
                        _, _, image, _ = camera.get_fresh_frame()
                        
                        if image is None:
                            error_msg = "无法获取相机图像"
                            logger.error(error_msg)
                            response = MQTTResponse(
                                command=VisionCoreCommands.GET_IMAGE.value,
                                component="camera",
                                messageType=MessageType.ERROR,
                                message=error_msg,
                                data={}
                            )
                            mqtt_client = self.initializer.get_mqtt_client()
                            if mqtt_client:
                                mqtt_client.send_mqtt_response(response)
                            return
                        
                        logger.info(f"成功获取相机图像，尺寸: {image.shape}")
                        
                        # 使用QtSFTP类的upload_image方法上传图像
                        logger.info("正在上传图像到SFTP服务器...")
                        
                        result = sftp_client.upload_image(
                            image_data=image,
                            image_format="jpg",
                            prefix="calibration",
                            remote_path="/"  # 上传到根路径
                        )
                        
                        if result["success"]:
                            success_msg = f"图像上传成功: {result['filename']} ({result['file_size']} bytes) -> 根路径"
                            logger.info(success_msg)
                            
                            response = MQTTResponse(
                                command=VisionCoreCommands.GET_IMAGE.value,
                                component="sftp",
                                messageType=MessageType.SUCCESS,
                                message=success_msg,
                                data={
                                    "filename": result["filename"],
                                    "remote_path": result["remote_path"],
                                    "file_size": result["file_size"],
                                    "image_shape": list(image.shape)
                                }
                            )
                            mqtt_client = self.initializer.get_mqtt_client()
                            if mqtt_client:
                                mqtt_client.send_mqtt_response(response)
                        else:
                            error_msg = f"图像上传失败: {result['message']}"
                            logger.error(error_msg)
                            response = MQTTResponse(
                                command=VisionCoreCommands.GET_IMAGE.value,
                                component="sftp",
                                messageType=MessageType.ERROR,
                                message=error_msg,
                                data={}
                            )
                            mqtt_client = self.initializer.get_mqtt_client()
                            if mqtt_client:
                                mqtt_client.send_mqtt_response(response)
                        
                    except Exception as camera_error:
                        error_msg = f"获取相机图像失败: {camera_error}"
                        logger.error(error_msg)
                        
                        # 检查是否是缓冲区错误或其他需要重启相机的错误
                        error_str = str(camera_error).lower()
                        if ("buffer" in error_str or 
                            "something is wrong" in error_str or
                            "connection" in error_str or
                            "timeout" in error_str):
                            
                            logger.warning("检测到相机缓冲区或连接错误，尝试重启相机组件...")
                            
                            try:
                                # 尝试重启相机
                                restart_success = self.initializer._restart_camera()
                                if restart_success:
                                    logger.info("相机组件重启成功")
                                    # 发送重启成功的响应
                                    response = MQTTResponse(
                                        command=VisionCoreCommands.GET_IMAGE.value,
                                        component="camera",
                                        messageType=MessageType.ERROR,
                                        message=f"相机出现错误已自动重启: {error_msg}",
                                        data={
                                            "auto_restart": True,
                                            "restart_success": True,
                                            "original_error": error_msg
                                        }
                                    )
                                else:
                                    logger.error("相机组件重启失败")
                                    response = MQTTResponse(
                                        command=VisionCoreCommands.GET_IMAGE.value,
                                        component="camera",
                                        messageType=MessageType.ERROR,
                                        message=f"相机错误且重启失败: {error_msg}",
                                        data={
                                            "auto_restart": True,
                                            "restart_success": False,
                                            "original_error": error_msg
                                        }
                                    )
                            except Exception as restart_error:
                                logger.error(f"重启相机时出现异常: {restart_error}")
                                response = MQTTResponse(
                                    command=VisionCoreCommands.GET_IMAGE.value,
                                    component="camera",
                                    messageType=MessageType.ERROR,
                                    message=f"相机错误且重启异常: {error_msg}",
                                    data={
                                        "auto_restart": True,
                                        "restart_success": False,
                                        "restart_error": str(restart_error),
                                        "original_error": error_msg
                                    }
                                )
                        else:
                            # 其他类型的错误，不重启相机
                            response = MQTTResponse(
                                command=VisionCoreCommands.GET_IMAGE.value,
                                component="camera",
                                messageType=MessageType.ERROR,
                                message=error_msg,
                                data={}
                            )
                        
                        mqtt_client = self.initializer.get_mqtt_client()
                        if mqtt_client:
                            mqtt_client.send_mqtt_response(response)
                
                except Exception as e:
                    error_msg = f"执行get_image命令失败: {e}"
                    logger.error(error_msg)
                    response = MQTTResponse(
                        command=VisionCoreCommands.GET_IMAGE.value,
                        component="camera",
                        messageType=MessageType.ERROR,
                        message=error_msg,
                        data={}
                    )
                    mqtt_client = self.initializer.get_mqtt_client()
                    if mqtt_client:
                        mqtt_client.send_mqtt_response(response)
            # SFTP测试
            elif command == VisionCoreCommands.SFTP_TEST:
                # SFTP测试 - 上传test.png文件
                logger.info("收到SFTP测试命令")
                
                try:
                    import os
                    sftp_client = self.initializer.get_sftp_client()
                    
                    # 检查SFTP是否可用
                    if not sftp_client or not sftp_client.connected:
                        error_msg = "SFTP客户端未连接或未初始化"
                        logger.error(error_msg)
                        response = MQTTResponse(
                            command=VisionCoreCommands.SFTP_TEST.value,
                            component="sftp",
                            messageType=MessageType.ERROR,
                            message=error_msg,
                            data={}
                        )
                        mqtt_client = self.initializer.get_mqtt_client()
                        if mqtt_client:
                            mqtt_client.send_mqtt_response(response)
                        return
                    
                    # 使用QtSFTP类的test_connection方法
                    logger.info("正在测试SFTP连接...")
                    result = sftp_client.test_connection("./test.png")
                    
                    # 构建响应
                    if result["success"]:
                        response = MQTTResponse(
                            command=VisionCoreCommands.SFTP_TEST.value,
                            component="sftp",
                            messageType=MessageType.SUCCESS,
                            message=result["message"],
                            data={
                                "filename": result["filename"],
                                "remote_path": result["remote_path"],
                                "file_size": result["file_size"]
                            }
                        )
                        logger.info(f"SFTP测试成功: {result['message']}")
                    else:
                        response = MQTTResponse(
                            command=VisionCoreCommands.SFTP_TEST.value,
                            component="sftp",
                            messageType=MessageType.ERROR,
                            message=result["message"],
                            data={}
                        )
                        logger.error(f"SFTP测试失败: {result['message']}")
                    
                    # 发送响应
                    mqtt_client = self.initializer.get_mqtt_client()
                    if mqtt_client:
                        mqtt_client.send_mqtt_response(response)
                
                except Exception as e:
                    error_msg = f"执行SFTP测试命令失败: {e}"
                    logger.error(error_msg)
                    response = MQTTResponse(
                        command=VisionCoreCommands.SFTP_TEST.value,
                        component="sftp",
                        messageType=MessageType.ERROR,
                        message=error_msg,
                        data={}
                    )
                    mqtt_client = self.initializer.get_mqtt_client()
                    if mqtt_client:
                        mqtt_client.send_mqtt_response(response)
            # 获取系统状态
            elif command == VisionCoreCommands.GET_SYSTEM_STATUS:
                try:
                    # 获取系统状态
                    system_status = self.initializer.get_system_status()
                    
                    response = MQTTResponse(
                        command=VisionCoreCommands.GET_SYSTEM_STATUS.value,
                        component="system",
                        messageType=MessageType.SUCCESS,
                        message="系统状态获取成功",
                        data=system_status
                    )
                    mqtt_client = self.initializer.get_mqtt_client()
                    if mqtt_client:
                        mqtt_client.send_mqtt_response(response)
                    
                except Exception as e:
                    logger.error(f"获取系统状态失败: {e}")
                    response = MQTTResponse(
                        command=VisionCoreCommands.GET_SYSTEM_STATUS.value,
                        component="system",
                        messageType=MessageType.ERROR,
                        message=f"获取系统状态失败: {e}",
                        data={}
                    )
                    mqtt_client = self.initializer.get_mqtt_client()
                    if mqtt_client:
                        mqtt_client.send_mqtt_response(response)
            # 模型测试
            elif command == VisionCoreCommands.MODEL_TEST:
                try:
                    # 获取检测器
                    detector = self.initializer.get_detector()
                    if not detector:
                        error_msg = "检测器未初始化"
                        logger.error(error_msg)
                        response = MQTTResponse(
                            command=VisionCoreCommands.MODEL_TEST.value,
                            component="detector",
                            messageType=MessageType.ERROR,
                            message=error_msg,
                            data={}
                        )
                        mqtt_client = self.initializer.get_mqtt_client()
                        if mqtt_client:
                            mqtt_client.send_mqtt_response(response)
                        return
                    
                    # 获取相机
                    camera = self.initializer.get_camera()
                    if not camera:
                        error_msg = "相机未连接或未初始化"
                        logger.error(error_msg)
                        response = MQTTResponse(
                            command=VisionCoreCommands.MODEL_TEST.value,
                            component="camera",
                            messageType=MessageType.ERROR,
                            message=error_msg,
                            data={}
                        )
                        mqtt_client = self.initializer.get_mqtt_client()
                        if mqtt_client:
                            mqtt_client.send_mqtt_response(response)
                        return
                    
                    # 获取测试图像
                    try:
                        _, _, image, _ = camera.get_fresh_frame()
                        if image is None:
                            error_msg = "无法获取测试图像"
                            logger.error(error_msg)
                            response = MQTTResponse(
                                command=VisionCoreCommands.MODEL_TEST.value,
                                component="camera",
                                messageType=MessageType.ERROR,
                                message=error_msg,
                                data={}
                            )
                            mqtt_client = self.initializer.get_mqtt_client()
                            if mqtt_client:
                                mqtt_client.send_mqtt_response(response)
                            return
                        
                        # 执行检测
                        logger.info("正在执行模型测试...")
                        results = detector.detect(image)
                        
                        # 解析结果
                        detection_count = 0
                        if results:
                            if isinstance(results, (list, tuple)):
                                detection_count = len(results)
                            elif hasattr(results, '__len__'):
                                detection_count = len(results)
                            else:
                                detection_count = 1 if results else 0
                        
                        success_msg = f"模型测试完成，检测到 {detection_count} 个目标"
                        logger.info(success_msg)
                        
                        response = MQTTResponse(
                            command=VisionCoreCommands.MODEL_TEST.value,
                            component="detector",
                            messageType=MessageType.SUCCESS,
                            message=success_msg,
                            data={
                                "detection_count": detection_count,
                                "image_shape": list(image.shape),
                                "model_type": type(detector).__name__
                            }
                        )
                        mqtt_client = self.initializer.get_mqtt_client()
                        if mqtt_client:
                            mqtt_client.send_mqtt_response(response)
                        
                    except Exception as test_error:
                        error_msg = f"模型测试执行失败: {test_error}"
                        logger.error(error_msg)
                        
                        # 检查是否是相机相关的错误
                        error_str = str(test_error).lower()
                        if ("buffer" in error_str or 
                            "something is wrong" in error_str or
                            "connection" in error_str or
                            "timeout" in error_str or
                            "camera" in error_str):
                            
                            logger.warning("检测到相机缓冲区或连接错误，尝试重启相机组件...")
                            
                            try:
                                # 尝试重启相机
                                restart_success = self.initializer._restart_camera()
                                if restart_success:
                                    logger.info("相机组件重启成功")
                                    # 发送重启成功的响应
                                    response = MQTTResponse(
                                        command=VisionCoreCommands.MODEL_TEST.value,
                                        component="camera",
                                        messageType=MessageType.ERROR,
                                        message=f"相机出现错误已自动重启: {error_msg}",
                                        data={
                                            "auto_restart": True,
                                            "restart_success": True,
                                            "original_error": error_msg
                                        }
                                    )
                                else:
                                    logger.error("相机组件重启失败")
                                    response = MQTTResponse(
                                        command=VisionCoreCommands.MODEL_TEST.value,
                                        component="camera",
                                        messageType=MessageType.ERROR,
                                        message=f"相机错误且重启失败: {error_msg}",
                                        data={
                                            "auto_restart": True,
                                            "restart_success": False,
                                            "original_error": error_msg
                                        }
                                    )
                            except Exception as restart_error:
                                logger.error(f"重启相机时出现异常: {restart_error}")
                                response = MQTTResponse(
                                    command=VisionCoreCommands.MODEL_TEST.value,
                                    component="camera",
                                    messageType=MessageType.ERROR,
                                    message=f"相机错误且重启异常: {error_msg}",
                                    data={
                                        "auto_restart": True,
                                        "restart_success": False,
                                        "restart_error": str(restart_error),
                                        "original_error": error_msg
                                    }
                                )
                        else:
                            # 其他类型的错误，不重启相机
                            response = MQTTResponse(
                                command=VisionCoreCommands.MODEL_TEST.value,
                                component="detector",
                                messageType=MessageType.ERROR,
                                message=error_msg,
                                data={}
                            )
                        
                        mqtt_client = self.initializer.get_mqtt_client()
                        if mqtt_client:
                            mqtt_client.send_mqtt_response(response)
                    
                except Exception as e:
                    error_msg = f"执行模型测试命令失败: {e}"
                    logger.error(error_msg)
                    response = MQTTResponse(
                        command=VisionCoreCommands.MODEL_TEST.value,
                        component="detector",
                        messageType=MessageType.ERROR,
                        message=error_msg,
                        data={}
                    )
                    mqtt_client = self.initializer.get_mqtt_client()
                    if mqtt_client:
                        mqtt_client.send_mqtt_response(response)            
            else:
                logger.warning(f"未知系统命令: {command}")
                # 可以添加支持命令的提示
                supported_commands = VisionCoreCommands.get_all_commands()
                logger.info(f"支持的命令: {supported_commands}")
                
                response = MQTTResponse(
                    command=command,
                    component="system",
                    messageType=MessageType.ERROR,
                    message=f"未知系统命令: {command}",
                    data={
                        "supported_commands": supported_commands
                    }
                )
                mqtt_client = self.initializer.get_mqtt_client()
                if mqtt_client:
                    mqtt_client.send_mqtt_response(response)
                
        except Exception as e:
            logger.error(f"处理系统命令时出错: {e}")
    
    def _check_system_health(self, logger):
        """检查系统健康状态"""
        try:
            if self.initializer.monitor:
                status = self.initializer.get_system_status()
                
                # 记录系统状态
                healthy_components = sum(1 for comp in status.get("components", {}).values() if comp.get("healthy", False))
                total_components = len(status.get("components", {}))
                
                # 检查是否有组件不健康
                unhealthy = [name for name, comp in status.get("components", {}).items() if not comp.get("healthy", False)]
                
                # 只在有组件异常时记录日志
                if unhealthy:
                    logger.warning(f"系统健康检查: {healthy_components}/{total_components} 组件正常，不健康的组件: {', '.join(unhealthy)}")
                # 如果所有组件正常，不记录日志（静默运行）
                
                # 注释掉TCP服务端主动广播健康状态的代码
                # TCP服务端只应该在收到catch指令时回传检测数据，不能发送其他任何数据
                # if tcp_server and tcp_server.is_connected and len(tcp_server.clients) > 0:
                #     status_msg = f"系统健康报告: {healthy_components}/{total_components} 组件正常"
                #     if unhealthy:
                #         status_msg += f", 不健康的组件: {', '.join(unhealthy)}"
                #     tcp_server.broadcast_message(status_msg)
                
        except Exception as e:
            logger.error(f"健康检查异常: {e}")
    
    def _get_roi_config(self) -> dict:
        """
        获取ROI配置
        
        Returns:
            dict: ROI配置信息
        """
        configManager = self.initializer.get_config_manager()
        if not configManager:
            return {}
        
        roi_config = {
            "enable": configManager.get_config("roi.enable"),
            "centerXOffset": configManager.get_config("roi.centerXOffset"),
            "centerYOffset": configManager.get_config("roi.centerYOffset"),
            "width": configManager.get_config("roi.width"),
            "height": configManager.get_config("roi.height")
        }
        
        # 确保ROI配置有效
        if roi_config["enable"]:
            if roi_config["centerXOffset"] < 0: roi_config["centerXOffset"] = 0
            if roi_config["centerYOffset"] < 0: roi_config["centerYOffset"] = 0
            if roi_config["width"] <= 0: roi_config["width"] = 1
            if roi_config["height"] <= 0: roi_config["height"] = 1
        
        return roi_config
    
    def _apply_roi_to_image(self, image, roi_config: dict):
        """
        对图像应用ROI裁剪
        
        Args:
            image: 输入图像
            roi_config: ROI配置字典
            
        Returns:
            np.ndarray: 裁剪后的图像
        """
        if not roi_config["enable"]:
            return image
        
        # 获取图像尺寸
        img_height, img_width = image.shape[:2]
        
        # 获取ROI参数
        roi_width = roi_config.get("width", 0)
        roi_height = roi_config.get("height", 0)
        center_x_offset = roi_config.get("centerXOffset", 0)
        center_y_offset = roi_config.get("centerYOffset", 0)
        
        # 计算ROI区域
        center_x = img_width // 2 + center_x_offset
        center_y = img_height // 2 + center_y_offset
        
        # 计算ROI边界
        x1 = max(0, center_x - roi_width // 2)
        y1 = max(0, center_y - roi_height // 2)
        x2 = min(img_width, x1 + roi_width)
        y2 = min(img_height, y1 + roi_height)
        
        # 确保ROI区域有效
        if x2 <= x1 or y2 <= y1:
            self.initializer.logger.warning("ROI区域无效，使用完整图像")
            return image
        
        # 裁剪图像
        cropped_image = image[y1:y2, x1:x2]
        
        self.initializer.logger.info(f"ROI裁剪完成: 原图尺寸 {img_width}x{img_height} -> ROI尺寸 {x2-x1}x{y2-y1}")
        return cropped_image

    def handle_tcp_client_disconnected(self, client_id: str, reason: str = ""):
        """
        处理TCP客户端断开连接，清理防抖数据
        
        Args:
            client_id: 断开连接的客户端ID
            reason: 断开连接的原因
        """
        logger = self.initializer.logger
        
        # 清理防抖相关数据
        if client_id in self.last_tcp_command_time:
            del self.last_tcp_command_time[client_id]
        if client_id in self.tcp_processing_flags:
            del self.tcp_processing_flags[client_id]
        
        logger.debug(f"已清理客户端 {client_id} 的防抖数据 (断开原因: {reason})")


def main():
    """
    主函数 - 支持自动重启的版本
    """
    # 创建并启动应用程序
    app = VisionCoreApp("./Config/config.yaml")
    app.start()


def main_simple():
    """
    简化版主函数 - 单次运行版本
    """
    # 创建系统初始化器
    initializer = SystemInitializer("./Config/config.yaml")
    
    try:
        # 初始化配置
        if not initializer.initialize_config():
            sys.exit(1)
        
        # 初始化所有组件
        if not initializer.initialize_all_components():
            sys.exit(1)
        
        # 获取关键组件
        logger = initializer.get_logger()
        camera = initializer.get_camera()
        mqtt_client = initializer.get_mqtt_client()
        detector = initializer.get_detector()
        
        # 主程序运行逻辑
        logger.info("系统运行中，按Ctrl+C停止...")
        
        # 简单的运行循环
        while True:
            time.sleep(1)
            # TODO: 添加具体的业务逻辑
            
    except KeyboardInterrupt:
        if initializer.logger:
            initializer.logger.info("接收到停止信号，正在关闭系统...")
        else:
            print("接收到停止信号，正在关闭系统...")
    except Exception as e:
        if initializer.logger:
            initializer.logger.error(f"系统运行出错: {e}", exc_info=True)
        else:
            print(f"系统运行出错: {e}")
    finally:
        # 清理所有资源
        initializer.cleanup()


if __name__ == "__main__":
    # 使用自动重启的应用程序类（推荐用于生产环境）
    main()
    
    # 或者使用简单版本（用于开发和测试）
    # main_simple()
    