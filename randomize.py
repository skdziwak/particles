import random
import numpy as np
from parameters import Params
from matplotlib import pyplot as plt
from perlin_noise import PerlinNoise

def generate_noise(length, octaves=(6, 12, 24, 48)):
    noises = [PerlinNoise(octaves=a) for a in octaves]
    result = np.zeros((length,))
    for i in range(length):
        result[i] = sum(n(i / length) for n in noises)
    result -= np.min(result)
    result /= np.max(result)
    return result

class Randomizer: # Randomizes speed, turn_speed, sensor_angle, sensor_length, decay
    def __init__(self, length, averages=[0.001, 1, 45, 0.05, 0.01], deviations=[0.0005, 0.9, 40, 0.045, 0.009]):
        self.noises = [generate_noise(length) for i in range(5)]
        self.averages = averages
        self.deviations = deviations
        self.length = length

    def update(self, params : Params, i):
        params.speed = (self.noises[0][i] * 2 - 1) * self.deviations[0] + self.averages[0]
        params.turn_speed = (self.noises[1][i] * 2 - 1) * self.deviations[1] + self.averages[1]
        params.sensor_angle = (self.noises[2][i] * 2 - 1) * self.deviations[2] + self.averages[2]
        params.sensor_length = (self.noises[3][i] * 2 - 1) * self.deviations[3] + self.averages[3]
        params.decay = (self.noises[4][i] * 2 - 1) * self.deviations[4] + self.averages[4]
        params.update()
