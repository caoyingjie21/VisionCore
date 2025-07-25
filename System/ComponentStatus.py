#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
组件状态数据类模块
用于跟踪系统组件的健康状态和错误信息
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ComponentStatus:
    """组件状态数据类"""
    name: str                           # 组件名称
    is_healthy: bool                    # 是否健康
    last_check: datetime               # 最后检查时间
    error_count: int                   # 错误计数
    last_error: Optional[str] = None   # 最后的错误信息 