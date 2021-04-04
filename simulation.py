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

# Load Image
matrix = np.array(Image.open('matrix.png'), dtype=np.float32)
matrix = matrix[:,:,0].copy()
matrix /= 255

# Prepare buffers
matrix2 = np.zeros_like(matrix, dtype=np.float32)
def create_blur(shape):
    fltr = np.zeros(shape=shape, dtype=np.float32)
    a = shape[0] // 2
    b = shape[1] // 2
    for x in range(shape[0]):
        for y in range(shape[1]):
            fltr[x,y] = (a - x) ** 2 + (b - y) ** 2
    return np.sqrt(fltr)
fltr = create_blur((24, 24))
t = time.time()
filter2D(drv.In(matrix), drv.In(np.array(fltr.shape, dtype=np.int32)), drv.In(fltr), drv.Out(matrix2), grid=GRID, block=BLOCK)
print(time.time() - t)

plt.set_cmap('magma')
fig, axs = plt.subplots(1, 2)
axs[0].text = 'Original'
axs[0].imshow(matrix)
axs[1].text = 'Blurry'
axs[1].imshow(matrix2)
fig.show()
plt.show()