"""
Copyright (c) FRC Team 3756 RamFerno.
Open Source Software; you can modify and/or share it under the terms of
the license viewable in the root directory of this project.
"""

import math
from typing import Optional, TypeVar, List, Type, Dict, Union


################################################

class Object:
    """The abstract class for all vision tracked objects"""

    def __init__(self) -> None:
        self.x: Optional[int] = None
        self.y: Optional[int] = None
        self.scale: Optional[float] = None
        self.ratio: Optional[float] = None
        self.confidence: Optional[float] = None
        self.relative_distance_mm: Optional[float] = None
        self.relative_angle_deg: Optional[float] = None
        self.absolute_field_x: Optional[float] = None
        self.absolute_field_y: Optional[float] = None
        self.timestamp = None

    def update_frame_location(self, x: int, y: int, scale: float, ratio: float, timestamp: float) -> None:
        self.x = x
        self.y = y
        self.scale = scale
        self.ratio = ratio
        self.timestamp = timestamp

    def update_confidence(self, confidence: float) -> None:
        self.confidence = confidence

    def update_relative_location(self, relative_distance_mm: float, relative_angle_deg: float) -> None:
        self.relative_distance_mm = relative_distance_mm
        self.relative_angle_deg = relative_angle_deg

    def update_absolute_location(self, absolute_field_x: float, absolute_field_y: float) -> None:
        self.absolute_field_x = absolute_field_x
        self.absolute_field_y = absolute_field_y

    def update_timestamp(self, timestamp: float) -> None:
        self.timestamp = timestamp


################################################

class Algae(Object):
    """The class that holds all the characteristics of an Algae"""

    def __init__(self) -> None:
        super().__init__()


################################################

class Cage(Object):
    """The class that holds all the characteristics of a Cage"""

    def __init__(self) -> None:
        super().__init__()


################################################

class Coral(Object):
    """The class that holds all the characteristics of a Coral"""

    def __init__(self) -> None:
        super().__init__()


################################################

class Robot(Object):
    """The class that holds all the characteristics of a Robot"""

    def __init__(self) -> None:
        super().__init__()
        self.travel_angle = None
        self.travel_speed = None
        self.velocity_x = None
        self.velocity_y = None
        self.acceleration_x = None
        self.acceleration_y = None

    def set_velocity(self, x, y, timestamp):
        if self.timestamp is not None and self.is_data_recent(timestamp):
            time_diff = timestamp - self.timestamp
            new_velocity_x = (x - self.x) / time_diff
            new_velocity_y = (y - self.y) / time_diff
            if self.velocity_x is not None and self.velocity_y is not None:
                self.acceleration_x = (
                    new_velocity_x - self.velocity_x) / time_diff
                self.acceleration_y = (
                    new_velocity_y - self.velocity_y) / time_diff
            self.velocity_x, self.velocity_y = new_velocity_x, new_velocity_y

    def calculate_speed(self) -> Optional[float]:
        if self.velocity_x is not None and self.velocity_y is not None:
            return math.sqrt(self.velocity_x ** 2 + self.velocity_y ** 2)
        return None

    def estimate_position(self, time_step):
        """Predicts future position based on current velocity and travel angle."""
        if self.velocity_x is not None and self.velocity_y is not None:
            pred_x = self.x + self.velocity_x * time_step
            pred_y = self.y + self.velocity_y * time_step
            return pred_x, pred_y
        elif self.travel_angle is not None and self.travel_speed is not None:
            angle_rad = math.radians(self.travel_angle)
            pred_x = self.x + self.travel_speed * \
                math.cos(angle_rad) * time_step
            pred_y = self.y + self.travel_speed * \
                math.sin(angle_rad) * time_step
            return pred_x, pred_y
        return None, None

    def is_data_recent(self, current_time):
        return (current_time - self.timestamp) <= 1 if self.timestamp is not None else False


T = TypeVar("T", Algae, Cage, Coral, Robot)


class GamePieces:
    def __init__(self) -> None:
        self._data: Dict[Type[Union[Algae, Cage, Coral, Robot]], List[Union[Algae, Cage, Coral, Robot]]] = {
            Algae: [],
            Cage: [],
            Coral: [],
            Robot: []
        }

    def add(self, cls_or_gp: Union[Type[T], 'GamePieces'], obj: Optional[T] = None) -> None:
        if isinstance(cls_or_gp, GamePieces):
            for key, obj_list in cls_or_gp._data.items():
                self._data[key].extend(obj_list)
        elif obj is not None:
            cls = cls_or_gp
            self._data[cls].append(obj)
        else:
            raise ValueError("Invalid arguments: must provide either (cls, obj) or another GamePieces instance.")

    def get_algae(self) -> List[Algae]:
        return self._data[Algae]  # type: ignore

    def get_cage(self) -> List[Cage]:
        return self._data[Cage]  # type: ignore

    def get_coral(self) -> List[Coral]:
        return self._data[Coral]  # type: ignore

    def get_robot(self) -> List[Robot]:
        return self._data[Robot]  # type: ignore

    def get_all(self) -> List[Object]:
        return [obj for objs in self._data.values() for obj in objs]

    def clear(self) -> None:
        for key in self._data:
            self._data[key] = []
