"""
@Description :   Define the model loading class, to export rknn model and using.
                 the file must run on linux and install rknn-toolkit2 with python.
                 more information refer to https://github.com/airockchip/rknn-toolkit2/tree/master
@Author      :   Cao Yingjie
@Time        :   2025/04/23 08:47:48
"""

import os
import sys
import urllib
import urllib.request
import time
import numpy as np
import argparse
import cv2, math
import platform
import json
from math import ceil
from itertools import product as product
from pathlib import Path
from typing import List, Tuple, Optional, Any, Union
# -------- 平台适配导入 --------
USING_PC = sys.platform.startswith("win") or platform.system().lower().startswith("windows")

try:
    if USING_PC:
        from ultralytics import YOLO as UltralyticsYOLO
        RKNN = None  # 占位，PC 上不用
    else:
        from rknn.api import RKNN
        UltralyticsYOLO = None
except ImportError:
    # 如果对应包缺失
    RKNN = None
    UltralyticsYOLO = None

class RKNN_YOLO:
    """
    RKNN YOLO模型封装类
    用于加载和运行RKNN模型进行目标检测，并集成坐标转换功能
    """
    
    def __init__(self, model_path, target='rk3588', device_id=None, conf_threshold=0.5, nms_threshold=0.45, matrix_file_path=None):
        """
        初始化RKNN YOLO模型
        
        Args:
            model_path (str): RKNN模型路径
            target (str, optional): 目标RKNPU平台. 默认为 'rk3588'
            device_id (str, optional): 设备ID. 默认为 None
            conf_threshold (float, optional): 置信度阈值. 默认为 0.5
            nms_threshold (float, optional): NMS阈值. 默认为 0.45
            matrix_file_path (str, optional): 变换矩阵文件路径. 默认为 None
        """
        self.CLASSES = ['seasoning']
        self.meshgrid = []
        self.class_num = len(self.CLASSES)
        self.head_num = 3
        self.strides = [8, 16, 32]
        self.map_size = [[32, 32], [16, 16], [8, 8]]  # 256x256输入对应的特征图尺寸
        self.reg_num = 16
        self.input_height = 256
        self.input_width = 256
        self.nms_thresh = nms_threshold
        self.object_thresh = conf_threshold
        self.conf_threshold = conf_threshold  # 添加置信度阈值属性
        self.nms_threshold = nms_threshold    # 添加NMS阈值属性
        self.rknn = None
        self.pc_yolo = None  # Windows 平台使用
        
        # 坐标转换相关属性
        self.transformation_matrix: Optional[np.ndarray] = None
        self.matrix_metadata: dict = {}
        
        # 设置变换矩阵文件路径
        if matrix_file_path:
            self.matrix_file_path = Path(matrix_file_path)
        else:
            # 默认路径：Config/transformation_matrix.json
            config_dir = Path(__file__).parent.parent / "Config"
            self.matrix_file_path = config_dir / "transformation_matrix.json"
        
        try:
            if USING_PC:
                if UltralyticsYOLO is None:
                    raise RuntimeError("未安装 ultralytics 库，无法在 Windows 平台加载 YOLO 模型")
                self.pc_yolo = UltralyticsYOLO(model_path)
            else:
                # 初始化RKNN
                self.rknn = RKNN(verbose=True)
                ret = self.rknn.load_rknn(model_path)
                if ret != 0:
                    raise RuntimeError(f'Load RKNN model "{model_path}" failed!')
                ret = self.rknn.init_runtime(
                    target=target,
                    device_id=device_id,
                    core_mask=RKNN.NPU_CORE_0 | RKNN.NPU_CORE_1 | RKNN.NPU_CORE_2)
                if ret != 0:
                    raise RuntimeError('Init runtime environment failed!')
                self._generate_meshgrid()
            
            # 加载变换矩阵
            self.load_transformation_matrix()
            
        except Exception as e:
            if self.rknn is not None:
                try:
                    self.rknn.release()
                except:
                    pass
                self.rknn = None
            raise RuntimeError(f"初始化模型时出错: {str(e)}")
    
    def load_transformation_matrix(self) -> bool:
        """
        加载4x4变换矩阵
        
        Returns:
            bool: 是否成功加载
        """
        if not self.matrix_file_path.exists():
            # 变换矩阵文件不存在，但不影响基本功能
            return False
        
        try:
            with open(self.matrix_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查数据格式
            if 'matrix' not in data:
                return False
            
            matrix_data = data['matrix']
            
            # 验证矩阵尺寸
            if (not isinstance(matrix_data, list) or 
                len(matrix_data) != 4 or 
                not all(len(row) == 4 for row in matrix_data)):
                return False
            
            # 加载矩阵
            self.transformation_matrix = np.array(matrix_data, dtype=np.float64)
            
            # 保存元数据
            self.matrix_metadata = {
                'calibration_points_count': data.get('calibration_points_count', 0),
                'calibration_rmse': data.get('calibration_rmse', 0.0),
                'transformation_type': data.get('transformation_type', 'unknown'),
                'matrix_size': data.get('matrix_size', '4x4')
            }
            
            return True
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return False
        except Exception as e:
            return False
    
    def perform_2d_detection(self, frame):
        """
        执行2D检测 - 只进行目标检测，返回所有检测框的坐标
        用于图像稳定性检测，不进行ROI过滤
        
        Args:
            frame: 图像帧数据
        
        Returns:
            dict: 2D检测结果字典，包含所有检测框的坐标
        """
        try:
            # 记录检测流程的开始时间
            total_start_time = time.time()
            
            if frame is None:
                return None
            
            # 执行检测
            detect_start = time.time()
            results = self.detect(frame)
            detect_time = (time.time() - detect_start) * 1000  # 转换为毫秒
            
            # 处理检测结果
            process_start_time = time.time()
            detection_count = 0
            detection_boxes = []  # 存储所有检测框的坐标
            
            if results:
                for result in results:
                    if not (hasattr(result, 'pt1x') and hasattr(result, 'pt1y')):
                        continue
                    
                    # 计算检测框中心点
                    center_x = (result.pt1x + result.pt2x + result.pt3x + result.pt4x) / 4
                    center_y = (result.pt1y + result.pt2y + result.pt3y + result.pt4y) / 4
                    
                    # 添加到检测框列表
                    detection_count += 1
                    detection_boxes.append({
                        'center': [center_x, center_y],
                        'corners': [
                            [result.pt1x, result.pt1y],
                            [result.pt2x, result.pt2y],
                            [result.pt3x, result.pt3y],
                            [result.pt4x, result.pt4y]
                        ],
                        'result': result
                    })
            
            process_time = (time.time() - process_start_time) * 1000  # 转换为毫秒
            total_time = (time.time() - total_start_time) * 1000
            
            # 构造2D检测结果
            detection_result = {
                'detection_count': detection_count,
                'detection_boxes': detection_boxes,  # 所有检测框的坐标信息
                'timing': {
                    'detect_time': detect_time,
                    'process_time': process_time,
                    'total_time': total_time
                }
            }
            
            return detection_result
            
        except Exception as e:
            return None
    
    def process_detection_to_coordinates_fast(self, results, depth_data, camera_params, roi_config=None):
        """
        高性能版本的坐标计算 - 目标10ms内完成
        
        Args:
            results: 检测结果列表（DetectBox对象列表）
            depth_data: 深度数据
            camera_params: 相机参数
            roi_config: ROI配置信息（可选），如果提供则只处理ROI内的目标
            
        Returns:
            dict or None: 最优目标的坐标信息
        """
        if not results or not depth_data or not camera_params:
            return None
        
        # 预计算常用值，避免重复计算
        width = camera_params.width
        height = camera_params.height
        cx, cy = camera_params.cx, camera_params.cy
        fx, fy = camera_params.fx, camera_params.fy
        k1, k2 = camera_params.k1, camera_params.k2
        f2rc = camera_params.f2rc
        
        # 预计算世界坐标变换矩阵（如果需要）
        m_c2w = None
        if hasattr(camera_params, 'cam2worldMatrix') and len(camera_params.cam2worldMatrix) == 16:
            m_c2w = np.array(camera_params.cam2worldMatrix).reshape(4, 4)
        
        # ROI过滤（如果提供ROI配置）
        filtered_results = results
        if roi_config and roi_config.get('enabled') and roi_config.get('x1') is not None:
            roi_x1, roi_y1 = roi_config['x1'], roi_config['y1']
            roi_x2, roi_y2 = roi_config['x2'], roi_config['y2']
            
            filtered_results = []
            for result in results:
                # 快速中心点计算
                center_x = (result.pt1x + result.pt2x + result.pt3x + result.pt4x) * 0.25
                center_y = (result.pt1y + result.pt2y + result.pt3y + result.pt4y) * 0.25
                
                # 检查是否在ROI内
                if roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2:
                    filtered_results.append(result)
            
            if not filtered_results:
                # ROI内没有检测目标
                return None
        
        best_target = None
        min_camera_z = float('inf')  # 追踪相机坐标系中的最小Z值
        
        # 批量处理所有过滤后的检测结果
        for i, result in enumerate(filtered_results):
            # 快速边界框计算
            center_x = int((result.pt1x + result.pt2x + result.pt3x + result.pt4x) * 0.25)
            center_y = int((result.pt1y + result.pt2y + result.pt3y + result.pt4y) * 0.25)
            
            # 边界检查
            if not (0 <= center_x < width and 0 <= center_y < height):
                continue
                
            # 获取深度值
            depth_index = center_y * width + center_x
            if depth_index >= len(depth_data):
                continue
                
            depth = depth_data[depth_index]
            if depth <= 0:
                continue
            
            # 简化的角度计算（基于主要边向量）
            angle = self._calculate_angle_fast(result)
            
            # 计算3D坐标
            success, camera_3d = self._calculate_3d_fast(center_x, center_y, depth, 
                                                       cx, cy, fx, fy, k1, k2, f2rc, m_c2w)
            
            if success and camera_3d[2] < min_camera_z:  # 比较相机坐标系中的Z值
                # 坐标系转换
                robot_3d = None
                if self.transformation_matrix is not None:
                    robot_3d = self._transform_point_fast(camera_3d, self.transformation_matrix)
                
                if robot_3d is not None:
                    min_camera_z = camera_3d[2]  # 更新最小Z值
                    best_target = {
                        'target_id': i + 1,
                        'center': [center_x, center_y],
                        'camera_3d': camera_3d,
                        'robot_3d': robot_3d,
                        'original_depth': depth,
                        'angle': angle,
                        'original_result': result
                    }
        
        return best_target
    
    def get_transformation_matrix_status(self) -> dict:
        """
        获取变换矩阵的加载状态
        
        Returns:
            dict: 包含变换矩阵状态信息的字典
        """
        status = {
            'loaded': False,
            'matrix_path': str(self.matrix_file_path),
            'metadata': {},
            'error': None
        }
        
        if self.transformation_matrix is not None:
            status['loaded'] = True
            status['metadata'] = self.matrix_metadata.copy()
        else:
            status['error'] = "变换矩阵未加载"
        
        return status
    
    def _generate_meshgrid(self):
        """生成网格坐标"""
        for index in range(self.head_num):
            for i in range(self.map_size[index][0]):
                for j in range(self.map_size[index][1]):
                    self.meshgrid.append(j + 0.5)
                    self.meshgrid.append(i + 0.5)
                    
    def _get_covariance_matrix(self, boxes):
        """计算协方差矩阵"""
        a, b, c = boxes.w, boxes.h, boxes.angle
        cos = math.cos(c)
        sin = math.sin(c)
        cos2 = math.pow(cos, 2)
        sin2 = math.pow(sin, 2)
        return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin
        
    def _probiou(self, obb1, obb2, eps=1e-7):
        """计算旋转框IOU"""
        x1, y1 = obb1.x, obb1.y
        x2, y2 = obb2.x, obb2.y
        a1, b1, c1 = self._get_covariance_matrix(obb1)
        a2, b2, c2 = self._get_covariance_matrix(obb2)

        t1 = (((a1 + a2) * math.pow((y1 - y2), 2) + (b1 + b2) * math.pow((x1 - x2), 2)) / ((a1 + a2) * (b1 + b2) - math.pow((c1 + c2), 2) + eps)) * 0.25
        t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - math.pow((c1 + c2), 2) + eps)) * 0.5

        temp1 = (a1 * b1 - math.pow(c1, 2)) if (a1 * b1 - math.pow(c1, 2)) > 0 else 0
        temp2 = (a2 * b2 - math.pow(c2, 2)) if (a2 * b2 - math.pow(c2, 2)) > 0 else 0
        t3 = math.log((((a1 + a2) * (b1 + b2) - math.pow((c1 + c2), 2)) / (4 * math.sqrt((temp1 * temp2)) + eps)+ eps)) * 0.5

        if (t1 + t2 + t3) > 100:
            bd = 100
        elif (t1 + t2 + t3) < eps:
            bd = eps
        else:
            bd = t1 + t2 + t3
        hd = math.sqrt((1.0 - math.exp(-bd) + eps))
        return 1 - hd
        
    def _nms_rotated(self, boxes, nms_thresh):
        """旋转框NMS"""
        pred_boxes = []
        sort_boxes = sorted(boxes, key=lambda x: x.score, reverse=True)
        for i in range(len(sort_boxes)):
            if sort_boxes[i].classId != -1:
                pred_boxes.append(sort_boxes[i])
                for j in range(i + 1, len(sort_boxes), 1):
                    ious = self._probiou(sort_boxes[i], sort_boxes[j])
                    if ious > nms_thresh:
                        sort_boxes[j].classId = -1
        return pred_boxes
        
    def _sigmoid(self, x):
        """Sigmoid函数"""
        return 1 / (1 + math.exp(-x))
        
    def _xywhr2xyxyxyxy(self, x, y, w, h, angle):
        """
        转换中心点格式 (x, y, w, h, angle) 到四点格式 (x1, y1, x2, y2, x3, y3, x4, y4)
        
        Args:
            x, y: 中心点坐标
            w, h: 宽度和高度
            angle: 旋转角度（弧度）
            
        Returns:
            四个角点的坐标
        """
        try:
            # 计算旋转角度的正弦和余弦
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # 计算未旋转状态下的四个角点偏移量
            w2, h2 = w / 2.0, h / 2.0
            
            # 生成四个角点（顺时针方向）
            pt1x = x - w2 * cos_a + h2 * sin_a
            pt1y = y - w2 * sin_a - h2 * cos_a
            
            pt2x = x + w2 * cos_a + h2 * sin_a
            pt2y = y + w2 * sin_a - h2 * cos_a
            
            pt3x = x + w2 * cos_a - h2 * sin_a
            pt3y = y + w2 * sin_a + h2 * cos_a
            
            pt4x = x - w2 * cos_a - h2 * sin_a
            pt4y = y - w2 * sin_a + h2 * cos_a
            
            # 确保返回的坐标是有效的
            if (np.isnan(pt1x) or np.isnan(pt1y) or np.isnan(pt2x) or np.isnan(pt2y) or 
                np.isnan(pt3x) or np.isnan(pt3y) or np.isnan(pt4x) or np.isnan(pt4y)):
                # 如果有NaN值，退回到非旋转框
                pt1x, pt1y = x - w2, y - h2
                pt2x, pt2y = x + w2, y - h2
                pt3x, pt3y = x + w2, y + h2
                pt4x, pt4y = x - w2, y + h2
                print(f"警告: 检测到NaN坐标，退回到非旋转框。输入: x={x}, y={y}, w={w}, h={h}, angle={angle}")
            
            return pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y
        
        except Exception as e:
            # 出现异常时，退回到非旋转框
            w2, h2 = w / 2.0, h / 2.0
            pt1x, pt1y = x - w2, y - h2
            pt2x, pt2y = x + w2, y - h2
            pt3x, pt3y = x + w2, y + h2
            pt4x, pt4y = x - w2, y + h2
            print(f"旋转边界框计算异常，退回到非旋转框: {str(e)}")
            return pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y
        
    def _postprocess(self, out):
        """后处理函数"""
        detect_result = []
        output = []
        for i in range(len(out)):
            output.append(out[i].reshape((-1)))

        gridIndex = -2
        cls_index = 0
        cls_max = 0

        for index in range(self.head_num):
            reg = output[index * 2 + 0]
            cls = output[index * 2 + 1]
            ang = output[self.head_num * 2 + index]

            for h in range(self.map_size[index][0]):
                for w in range(self.map_size[index][1]):
                    gridIndex += 2

                    if 1 == self.class_num:
                        cls_max = self._sigmoid(cls[0 * self.map_size[index][0] * self.map_size[index][1] + h * self.map_size[index][1] + w])
                        cls_index = 0
                    else:
                        for cl in range(self.class_num):
                            cls_val = cls[cl * self.map_size[index][0] * self.map_size[index][1] + h * self.map_size[index][1] + w]
                            if 0 == cl:
                                cls_max = cls_val
                                cls_index = cl
                            else:
                                if cls_val > cls_max:
                                    cls_max = cls_val
                                    cls_index = cl
                        cls_max = self._sigmoid(cls_max)

                    if cls_max > self.object_thresh:
                        regdfl = []
                        for lc in range(4):
                            sfsum = 0
                            locval = 0
                            for df in range(self.reg_num):
                                temp = math.exp(reg[((lc * self.reg_num) + df) * self.map_size[index][0] * self.map_size[index][1] + h * self.map_size[index][1] + w])
                                reg[((lc * self.reg_num) + df) * self.map_size[index][0] * self.map_size[index][1] + h * self.map_size[index][1] + w] = temp
                                sfsum += temp

                            for df in range(self.reg_num):
                                sfval = reg[((lc * self.reg_num) + df) * self.map_size[index][0] * self.map_size[index][1] + h * self.map_size[index][1] + w] / sfsum
                                locval += sfval * df
                            regdfl.append(locval)

                        angle = (self._sigmoid(ang[h * self.map_size[index][1] + w]) - 0.25) * math.pi

                        left, top, right, bottom = regdfl[0], regdfl[1], regdfl[2], regdfl[3]
                        cos, sin = math.cos(angle), math.sin(angle)
                        fx = (right - left) / 2
                        fy = (bottom - top) / 2

                        cx = ((fx * cos - fy * sin) + self.meshgrid[gridIndex + 0]) * self.strides[index]
                        cy = ((fx * sin + fy * cos) + self.meshgrid[gridIndex + 1])* self.strides[index]
                        cw = (left + right) * self.strides[index]
                        ch = (top + bottom) * self.strides[index]

                        box = CSXYWHR(cls_index, cls_max, cx, cy, cw, ch, angle)
                        detect_result.append(box)

        pred_boxes = self._nms_rotated(detect_result, self.nms_thresh)
        result = []
        
        for i in range(len(pred_boxes)):
            classid = pred_boxes[i].classId
            score = pred_boxes[i].score
            cx = pred_boxes[i].x
            cy = pred_boxes[i].y
            cw = pred_boxes[i].w
            ch = pred_boxes[i].h
            angle = pred_boxes[i].angle

            bw_ = cw if cw > ch else ch
            bh_ = ch if cw > ch else cw
            bt = angle % math.pi if cw > ch else (angle + math.pi / 2) % math.pi

            pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y = self._xywhr2xyxyxyxy(cx, cy, bw_, bh_, bt)
            bbox = DetectBox(classid, score, pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y, angle)
            result.append(bbox)
        return result
        
    def detect(self, image):
        """
        对输入图像进行目标检测
        
        Args:
            image (numpy.ndarray): 输入图像，BGR格式
            
        Returns:
            list: 检测结果列表，每个元素为DetectBox对象
        """
        # Windows / PC: 使用 UltralyticsYOLO
        if self.pc_yolo is not None:
            image_h, image_w = image.shape[:2]
            img_resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
            img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else img_resized
            
            # UltralyticsYOLO 期待 BGR 或路径输入，直接传递 ndarray
            m = self.pc_yolo.predict(img_bgr)
            
            # 检查是否有OBB检测结果
            if m[0].obb is None or len(m[0].obb) == 0:
                return []
            
            obb_results = m[0].obb
            detect_boxes = []
            
            # 获取所有必要的数据
            xyxyxyxy = obb_results.xyxyxyxy.cpu().numpy()  # 四个角点坐标
            xywhr = obb_results.xywhr.cpu().numpy()        # 中心点+宽高+角度
            conf = obb_results.conf.cpu().numpy()          # 置信度
            cls = obb_results.cls.cpu().numpy()            # 类别
            
            # 遍历每个检测结果
            for i in range(len(xyxyxyxy)):
                # 提取四个角点坐标
                points = xyxyxyxy[i]  # shape: (4, 2)
                pt1x, pt1y = float(points[0][0]), float(points[0][1])
                pt2x, pt2y = float(points[1][0]), float(points[1][1])
                pt3x, pt3y = float(points[2][0]), float(points[2][1])
                pt4x, pt4y = float(points[3][0]), float(points[3][1])
                
                # 提取其他信息
                confidence = float(conf[i])
                class_id = int(cls[i])
                angle = float(xywhr[i][4])  # 角度信息（弧度）
                
                # 过滤低置信度的检测结果
                if confidence < self.conf_threshold:
                    continue
                
                # 将坐标缩放回原始图像尺寸
                scale_x = image_w / 256.0
                scale_y = image_h / 256.0
                
                pt1x = int(pt1x * scale_x)
                pt1y = int(pt1y * scale_y)
                pt2x = int(pt2x * scale_x)
                pt2y = int(pt2y * scale_y)
                pt3x = int(pt3x * scale_x)
                pt3y = int(pt3y * scale_y)
                pt4x = int(pt4x * scale_x)
                pt4y = int(pt4y * scale_y)
                
                # 创建DetectBox对象
                detect_box = DetectBox(
                    classId=class_id,
                    score=confidence,
                    pt1x=pt1x, pt1y=pt1y,
                    pt2x=pt2x, pt2y=pt2y,
                    pt3x=pt3x, pt3y=pt3y,
                    pt4x=pt4x, pt4y=pt4y,
                    angle=angle
                )
                
                detect_boxes.append(detect_box)
            
            return detect_boxes

        # AARCH: 使用 RKNN 推理
        if self.rknn is None:
            raise RuntimeError("当前实例未加载任何模型")

        # 预处理
        image_h, image_w = image.shape[:2]
        img_resized = cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_rgb = np.expand_dims(img_rgb, 0)
        
        # 推理
        results = self.rknn.inference(inputs=[img_rgb], data_format='nhwc')
        # 后处理
        pred_boxes = self._postprocess(results)

        # 转换回原始图像尺寸
        for box in pred_boxes:
            box.pt1x = int(box.pt1x / self.input_width * image_w)
            box.pt1y = int(box.pt1y / self.input_height * image_h)
            box.pt2x = int(box.pt2x / self.input_width * image_w)
            box.pt2y = int(box.pt2y / self.input_height * image_h)
            box.pt3x = int(box.pt3x / self.input_width * image_w)
            box.pt3y = int(box.pt3y / self.input_height * image_h)
            box.pt4x = int(box.pt4x / self.input_width * image_w)
            box.pt4y = int(box.pt4y / self.input_height * image_h)

        return pred_boxes

    def detect_and_coordinat_transform(self, image, coordinat_transform):
        """
        检测后使用坐标变换
        
        Args:
            image: 输入图像
            coordinat_transform: 坐标变换矩阵

        Returns:
            list: 检测结果列表，每个元素为DetectBox对象
        """
        detect_boxes = self.detect(image)
        return detect_boxes

    def release(self):
        """
        释放RKNN资源
        在不再使用检测器时调用此方法
        """
        if hasattr(self, 'rknn') and self.rknn is not None:
            try:
                self.rknn.release()
            except Exception as e:
                print(f"释放RKNN资源时出错: {str(e)}")
            finally:
                self.rknn = None

    def __del__(self):
        """析构函数，确保资源被释放"""
        try:
            self.release()
        except AttributeError:
            # 如果self.rknn已经是None，忽略AttributeError错误
            pass

    def _calculate_angle_fast(self, result):
        """快速角度计算 - 简化版本"""
        try:
            # 使用对角线向量计算主要方向
            dx1 = result.pt3x - result.pt1x  # 对角线1
            dy1 = result.pt3y - result.pt1y
            dx2 = result.pt4x - result.pt2x  # 对角线2  
            dy2 = result.pt4y - result.pt2y
            
            # 选择较长的对角线作为主方向
            len1_sq = dx1*dx1 + dy1*dy1
            len2_sq = dx2*dx2 + dy2*dy2
            
            if len1_sq > len2_sq:
                angle_rad = math.atan2(dy1, dx1)
            else:
                angle_rad = math.atan2(dy2, dx2)
            
            # 转换为角度并规范化到[0, 180)
            angle_deg = math.degrees(angle_rad) % 180
            return angle_deg
            
        except:
            return 0.0
    
    def _calculate_3d_fast(self, x, y, depth, cx, cy, fx, fy, k1, k2, f2rc, m_c2w):
        """快速3D坐标计算"""
        try:
            # 计算相机坐标系下的坐标
            xp = (cx - x) / fx
            yp = (cy - y) / fy
            
            # 径向畸变校正
            r2 = xp*xp + yp*yp
            k = 1 + k1*r2 + k2*r2*r2
            
            xd = xp * k
            yd = yp * k
            
            # 3D坐标计算
            s0_inv = 1.0 / math.sqrt(xd*xd + yd*yd + 1)
            x_cam = xd * depth * s0_inv
            y_cam = yd * depth * s0_inv
            z_cam = depth * s0_inv - f2rc
            
            # 世界坐标系转换（如果需要）
            if m_c2w is not None:
                x_world = m_c2w[0,3] + z_cam*m_c2w[0,2] + y_cam*m_c2w[0,1] + x_cam*m_c2w[0,0]
                y_world = m_c2w[1,3] + z_cam*m_c2w[1,2] + y_cam*m_c2w[1,1] + x_cam*m_c2w[1,0]
                z_world = m_c2w[2,3] + z_cam*m_c2w[2,2] + y_cam*m_c2w[2,1] + x_cam*m_c2w[2,0]
                return True, [x_world, y_world, z_world]
            else:
                return True, [x_cam, y_cam, z_cam]
                
        except:
            return False, [0, 0, 0]
    
    def _transform_point_fast(self, camera_point, transformation_matrix):
        """快速坐标变换"""
        try:
            # 直接矩阵乘法，避免numpy数组创建开销
            x, y, z = camera_point
            T = transformation_matrix
            
            # 齐次坐标变换
            x_robot = T[0,0]*x + T[0,1]*y + T[0,2]*z + T[0,3]
            y_robot = T[1,0]*x + T[1,1]*y + T[1,2]*z + T[1,3]
            z_robot = T[2,0]*x + T[2,1]*y + T[2,2]*z + T[2,3]
            w = T[3,0]*x + T[3,1]*y + T[3,2]*z + T[3,3]
            
            if w != 0:
                return [x_robot/w, y_robot/w, z_robot/w]
            else:
                return [x_robot, y_robot, z_robot]
                
        except:
            return None
    
    def detect_with_coordinates(self, image, depth_data=None, camera_params=None, draw_annotations=True, roi_config=None):
        """
        检测并返回带坐标转换的结果，可选绘制检测框和标注
        
        Args:
            image: 输入图像
            depth_data: 深度数据（可选）
            camera_params: 相机参数（可选）
            draw_annotations: 是否在图像上绘制检测框和标注
            roi_config: ROI配置信息（可选）
            
        Returns:
            dict: 包含检测结果、坐标信息和绘制图像的字典
        """
        import cv2
        import time
        from datetime import datetime
        
        # 记录开始时间
        start_time = time.time()
        
        # 确保输入图像是彩色格式
        original_image = image.copy()
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 执行基础检测 - 只计算detect方法本身的时间
        detect_start = time.time()
        detection_results = self.detect(image)
        detect_end = time.time()
        
        # 计算检测耗时（只有detect方法的时间）
        detect_time = (detect_end - detect_start) * 1000
        
        if not detection_results:
            # 没有检测到目标，返回空结果
            result_image = original_image.copy() if draw_annotations else None
            
            if draw_annotations:
                # 在图像上显示"无检测目标"，同时绘制ROI
                self._draw_no_detection_info(result_image, detect_time, roi_config)
            
            total_time = (time.time() - start_time) * 1000
            
            return {
                'detection_count': 0,
                'detection_boxes': [],
                'coordinates': None,
                'best_target': None,
                'annotated_image': result_image,
                'original_image': original_image,
                'timing': {
                    'detect_time': detect_time,
                    'coord_time': 0.0,
                    'total_time': total_time
                }
            }
        
        # 如果有深度数据和相机参数，进行坐标转换
        coord_start_time = time.time()
        best_target = None
        
        if depth_data is not None and camera_params is not None:
            best_target = self.process_detection_to_coordinates_fast(
                detection_results, depth_data, camera_params, roi_config
            )
        
        coord_time = (time.time() - coord_start_time) * 1000
        total_time = (time.time() - start_time) * 1000
        
        # 绘制检测结果
        result_image = None
        if draw_annotations:
            # 使用原始图像进行绘制，包括ROI
            result_image = self._draw_detection_annotations(
                original_image.copy(), detection_results, best_target, detect_time, coord_time, total_time, roi_config
            )
        
        # 构造返回结果
        result = {
            'detection_count': len(detection_results),
            'detection_boxes': detection_results,
            'coordinates': best_target,
            'best_target': best_target,
            'annotated_image': result_image,
            'original_image': original_image,
            'timing': {
                'detect_time': detect_time,
                'coord_time': coord_time,
                'total_time': total_time
            }
        }
        
        return result

    def _draw_detection_annotations(self, image, detection_results, best_target, detect_time, coord_time, total_time, roi_config):
        """
        在图像上绘制检测框和标注信息
        
        Args:
            image: 要绘制的图像
            detection_results: 检测结果列表
            best_target: 最佳目标信息
            detect_time: 检测耗时（只计算detect方法的时间）
            coord_time: 坐标计算耗时
            total_time: 总耗时
            roi_config: ROI配置信息
            
        Returns:
            numpy.ndarray: 绘制了标注的图像
        """
        import cv2
        
        # 确保图像是彩色的
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 确定最佳目标的索引
        best_target_index = -1
        if best_target and best_target.get('target_id'):
            best_target_index = best_target['target_id'] - 1  # target_id从1开始，索引从0开始
        
        # 绘制所有检测框
        for i, detection in enumerate(detection_results):
            # 判断是否为最佳目标
            is_best = (i == best_target_index)
            
            # 设置颜色和线条粗细：最佳目标用绿色，其他用红色
            color = (0, 255, 0) if is_best else (0, 0, 255)  # BGR格式
            line_thickness = 3 if is_best else 2  # 最佳目标用更粗的线
            
            # 绘制检测框（四个角点连线）
            points = np.array([
                [detection.pt1x, detection.pt1y],
                [detection.pt2x, detection.pt2y],
                [detection.pt3x, detection.pt3y],
                [detection.pt4x, detection.pt4y]
            ], np.int32)
            
            cv2.polylines(image, [points], True, color, line_thickness)
        
        # 在左上角显示检测耗时和坐标信息（小字体绿色，竖着排列）
        info_lines = []
        info_lines.append(f"Detect: {detect_time:.1f}ms")
        
        # 如果有最佳目标，显示坐标信息
        if best_target and best_target.get('robot_3d'):
            x, y, z = best_target['robot_3d']
            angle = best_target.get('angle', 0.0)
            info_lines.append(f"X: {x:.1f}")
            info_lines.append(f"Y: {y:.1f}")
            info_lines.append(f"Z: {z:.1f}")
            info_lines.append(f"A: {angle:.1f}")
        
        # 绘制左上角信息（小字体绿色，竖着排列）
        for i, text in enumerate(info_lines):
            y_pos = 20 + i * 18  # 行间距18像素
            
            # 绘制阴影
            cv2.putText(image, text, (12, y_pos + 1), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            # 绘制主文本（绿色）
            cv2.putText(image, text, (10, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 绘制ROI（如果提供）
        if roi_config and roi_config.get('enabled') and roi_config.get('x1') is not None and roi_config.get('y1') is not None and roi_config.get('x2') is not None and roi_config.get('y2') is not None:
            x1, y1, x2, y2 = roi_config['x1'], roi_config['y1'], roi_config['x2'], roi_config['y2']
            
            # 绘制ROI矩形框（黄色，较粗的线）
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # 绘制半透明的ROI区域背景
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
            cv2.addWeighted(overlay, 0.1, image, 0.9, 0, image)
            
            # 绘制ROI标签
            roi_text = "ROI"
            roi_text_size = cv2.getTextSize(roi_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            roi_text_x = x1
            roi_text_y = y1 - 10 if y1 > 30 else y1 + roi_text_size[1] + 10
            
            # 绘制ROI文本背景
            cv2.rectangle(image, (roi_text_x - 2, roi_text_y - roi_text_size[1] - 2), 
                         (roi_text_x + roi_text_size[0] + 2, roi_text_y + 2), (0, 255, 255), -1)
            
            # 绘制ROI文本（黑色）
            cv2.putText(image, roi_text, (roi_text_x, roi_text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return image

    def _draw_no_detection_info(self, image, detect_time, roi_config):
        """
        在图像上绘制无检测目标的信息
        
        Args:
            image: 要绘制的图像
            detect_time: 检测耗时（只计算detect方法的时间）
            roi_config: ROI配置信息
        """
        import cv2
        
        # 确保图像是彩色的
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 在图像中央显示"无检测目标"
        text = "No Detection Target"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        img_height, img_width = image.shape[:2]
        text_x = (img_width - text_size[0]) // 2
        text_y = (img_height + text_size[1]) // 2
        
        # 绘制文本（带阴影效果，无背景框）
        # 先绘制阴影（黑色）
        cv2.putText(image, text, (text_x + 3, text_y + 3), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
        # 再绘制主文本（红色）
        cv2.putText(image, text, (text_x, text_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # 在左上角显示检测耗时（小字体绿色）
        info_text = f"Detect: {detect_time:.1f}ms (No Target)"
        
        # 绘制阴影
        cv2.putText(image, info_text, (12, 21), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        # 绘制主文本（绿色）
        cv2.putText(image, info_text, (10, 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 绘制ROI（如果提供）
        if roi_config and roi_config.get('enabled') and roi_config.get('x1') is not None and roi_config.get('y1') is not None and roi_config.get('x2') is not None and roi_config.get('y2') is not None:
            x1, y1, x2, y2 = roi_config['x1'], roi_config['y1'], roi_config['x2'], roi_config['y2']
            
            # 绘制ROI矩形框（黄色，较粗的线）
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # 绘制半透明的ROI区域背景
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
            cv2.addWeighted(overlay, 0.1, image, 0.9, 0, image)
            
            # 绘制ROI标签
            roi_text = "ROI"
            roi_text_size = cv2.getTextSize(roi_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            roi_text_x = x1
            roi_text_y = y1 - 10 if y1 > 30 else y1 + roi_text_size[1] + 10
            
            # 绘制ROI文本背景
            cv2.rectangle(image, (roi_text_x - 2, roi_text_y - roi_text_size[1] - 2), 
                         (roi_text_x + roi_text_size[0] + 2, roi_text_y + 2), (0, 255, 255), -1)
            
            # 绘制ROI文本（黑色）
            cv2.putText(image, roi_text, (roi_text_x, roi_text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# 辅助类定义
class CSXYWHR:
    def __init__(self, classId, score, x, y, w, h, angle):
        self.classId = classId
        self.score = score
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle

class DetectBox:
    def __init__(self, classId, score, pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y, angle):
        self.classId = classId
        self.score = score
        self.pt1x = pt1x
        self.pt1y = pt1y
        self.pt2x = pt2x
        self.pt2y = pt2y
        self.pt3x = pt3x
        self.pt3y = pt3y
        self.pt4x = pt4x
        self.pt4y = pt4y
        self.angle = angle