# Parse arguments
import argparse
parser = argparse.ArgumentParser(description='GPU Slime Simulation')
parser.add_argument('-gs', '--grid-size', dest='GRID_SIZE', action='store', type=int, default=32)
parser.add_argument('-bs', '--block-size', dest='BLOCK_SIZE', action='store', type=int, default=32)
args = parser.parse_args()

# Set constants
GRID = (args.GRID_SIZE, args.GRID_SIZE)
BLOCK = (args.BLOCK_SIZE, args.BLOCK_SIZE, 1)
SIZE = args.GRID_SIZE * args.BLOCK_SIZE

# Imports
import os
import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from matplotlib import pyplot as plt
from PIL import Image
from pycuda.compiler import SourceModule
from pathlib import Path

# Compile CUDA Kernels
module = SourceModule(Path('simulation.cpp').read_text(), include_dirs=[os.path.join(os.getcwd(), 'include')], no_extern_c=True)
filter2D = module.get_function('filter2D')
update = module.get_function('update')

# Bluf filter
def create_blur(shape):
    fltr = np.zeros(shape=shape, dtype=np.float32)
    a = shape[0] // 2
    b = shape[1] // 2
    for x in range(shape[0]):
        for y in range(shape[1]):
            fltr[x,y] = (a - x) ** 2 + (b - y) ** 2
    return np.sqrt(fltr)

agents = np.random.rand(20000, 3)
agents = np.array(agents, dtype=np.float32)
matrix = np.zeros(shape=(1024, 1024), dtype=np.float32)

matrix_driver = drv.InOut(agents)
agents_driver = drv.InOut(agents)

t = time.time()
for i in range(1000):
    update(agents_driver, grid=(200, 1), block=(100, 1, 1))

print(time.time() - t)