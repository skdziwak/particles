import numpy as np

class Params:
    def __init__(self, speed, width, height, turn_speed, sensor_angle, sensor_length, decay, wrapping_borders):
        self.speed = speed
        self.width = width
        self.height = height
        self.turn_speed = turn_speed
        self.sensor_angle = sensor_angle
        self.sensor_length = sensor_length
        self.decay = decay
        self.wrapping_borders = wrapping_borders
        self._data = None
        self.update()
    
    def update(self):
        self._data = np.array([
            self.speed,
            self.width,
            self.height,
            self.turn_speed,
            self.sensor_angle,
            self.sensor_length,
            self.decay,
            1 if self.wrapping_borders else 0
        ], dtype=np.float32)