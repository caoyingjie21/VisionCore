from datetime import datetime
import json
from dataclasses import dataclass
from typing import Any, Dict, Union

from SystemEnums.VisionCoreCommands import MessageType

@dataclass
class MQTTResponse:
    command: str
    component: str
    messageType: MessageType
    message: str
    data: dict
    timestamp: float

    def __init__(self, command: str, component: str, messageType: MessageType, message: str, data: dict, timestamp: float = None):
        self.component = component
        self.command = command
        self.messageType = messageType
        self.message = message
        self.data = data
        self.timestamp = timestamp or datetime.now().timestamp()

    def to_dict(self):
        return {
            "command": self.command,
            "component": self.component,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
            "messageType": self.messageType
        }


