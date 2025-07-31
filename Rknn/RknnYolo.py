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
from math import ceil
from itertools import product as product
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
    用于加载和运行RKNN模型进行目标检测
    """
    
    def __init__(self, model_path, target='rk3588', device_id=None, conf_threshold=0.5, nms_threshold=0.45):
        """
        初始化RKNN YOLO模型
        
        Args:
            model_path (str): RKNN模型路径
            target (str, optional): 目标RKNPU平台. 默认为 'rk3588'
            device_id (str, optional): 设备ID. 默认为 None
            conf_threshold (float, optional): 置信度阈值. 默认为 0.5
            nms_threshold (float, optional): NMS阈值. 默认为 0.45
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
        except Exception as e:
            if self.rknn is not None:
                try:
                    self.rknn.release()
                except:
                    pass
                self.rknn = None
            raise RuntimeError(f"初始化模型时出错: {str(e)}")
        
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


    def detect_and_coordinat_transform_without(self, image, coordinat_transform):
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