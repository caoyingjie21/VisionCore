# VisionCore Utils 工具模块

基于SickVision项目分析，重新设计的坐标系统工具模块，遵循单一职责原则。

## 📁 模块结构

```
utils/
├── calculator.py    # 坐标转换计算器
├── calibrator.py    # 坐标系标定器  
├── decorators.py    # 通用装饰器
└── README.md        # 本文档
```

## 🎯 设计原则

### 职责分离
- **calculator.py**: 专门负责从相机坐标系向机器人坐标系的转换
- **calibrator.py**: 专门负责坐标系标定和变换矩阵计算

### 符合VisionCore架构
- 使用统一的日志系统
- 支持配置文件管理
- 完善的错误处理
- 支持热重载

## 📋 核心类

### CoordinateCalculator（坐标转换计算器）

负责相机坐标系到机器人坐标系的转换。

**主要功能：**
- 从深度数据计算3D相机坐标
- 相机坐标到机器人坐标的转换
- 角度转换和补偿计算
- 变换矩阵的加载和使用
- 支持热重载

**使用示例：**
```python
from utils.calculator import CoordinateCalculator

# 初始化
calculator = CoordinateCalculator()

# 检查标定状态
if calculator.is_calibrated():
    # 坐标转换
    robot_point = calculator.transform_point_to_robot_coordinates([100, 50, 600])
    
    # 角度转换
    robot_angle = calculator.transform_angle_to_robot_coordinates(45.0)
    
    # 角度补偿
    compensation = calculator.calculate_angle_compensation(45.0, target_angle_deg=0.0)
```

### CoordinateCalibrator（坐标系标定器）

负责计算相机坐标系到机器人坐标系的变换矩阵。

**主要功能：**
- 收集标定点数据
- 计算4x4齐次变换矩阵
- 验证标定精度
- 保存和管理标定结果
- 标定质量评估

**使用示例：**
```python
from utils.calibrator import CoordinateCalibrator

# 初始化
calibrator = CoordinateCalibrator()

# 添加标定点
calibrator.add_calibration_point([100, 50, 600], [300, 400, 100])
# ... 添加更多标定点

# 计算变换矩阵
result = calibrator.calculate_transformation_matrix()

# 保存结果
calibrator.save_transformation_matrix()
calibrator.save_calibration_data()

# 查看标定报告
calibrator.print_calibration_report()
```

## 🚀 快速开始

运行完整示例：
```bash
cd VisionCore
python example_coordinate_system.py
```

示例演示：
1. 坐标系标定流程
2. 坐标转换使用
3. 热重载功能
4. 标定质量评估

## 🔧 配置说明

### 变换矩阵文件格式

默认位置：`Config/transformation_matrix.json`

```json
{
    "matrix": [
        [0.993, 0.004, -0.020, 332.13],
        [-0.032, -0.989, 0.058, 68.44], 
        [-0.052, -0.021, -0.777, 586.69],
        [0.0, 0.0, 0.0, 1.0]
    ],
    "calibration_points_count": 6,
    "calibration_rmse": 2.845,
    "transformation_type": "complete_3d",
    "matrix_size": "4x4",
    "calibration_datetime": "2025-01-27T10:30:00"
}
```

### 标定数据文件格式

默认位置：`Config/calibration_data.json`

```json
{
    "camera_points": [[100, 50, 600], [200, 100, 650], ...],
    "robot_points": [[300, 400, 100], [350, 450, 100], ...],
    "points_count": 6,
    "calibration_datetime": "2025-01-27T10:30:00"
}
```

## 📊 质量标准

### 标定精度评估
- **优秀**: RMSE < 2.0mm
- **良好**: RMSE < 5.0mm  
- **可接受**: RMSE < 10.0mm
- **较差**: RMSE ≥ 10.0mm

### 标定点数量建议
- **最少**: 4个点（数学要求）
- **基本**: 4-6个点
- **良好**: 6-10个点
- **充足**: 10个以上点

## 🔄 与VisionCore集成

### 1. 在系统初始化时加载
```python
from utils.calculator import CoordinateCalculator

class VisionSystem:
    def __init__(self):
        self.coordinate_calculator = CoordinateCalculator()
```

### 2. 在检测流程中使用
```python
# 计算3D坐标
success, camera_3d = self.coordinate_calculator.calculate_3d_coordinates_from_depth(
    pixel_x, pixel_y, depth_data, camera_params
)

# 转换到机器人坐标系
if success:
    robot_3d = self.coordinate_calculator.transform_point_to_robot_coordinates(camera_3d)
```

### 3. 支持配置热重载
```python
# MQTT配置更新时重载变换矩阵
self.coordinate_calculator.reload_transformation_matrix()
```

## ⚡ 性能特点

- **高效矩阵运算**: 使用NumPy优化
- **内存友好**: 避免不必要的数据复制
- **错误恢复**: 完善的异常处理
- **热重载**: 支持运行时更新配置

## 🔗 兼容性

为保持向后兼容，提供了类别名：
```python
# 兼容旧代码
from utils.calculator import Calculator  # = CoordinateCalculator
from utils.calibrator import Calibrator  # = CoordinateCalibrator
```

---

**设计目标**: 职责明确、易于使用、高度可靠 