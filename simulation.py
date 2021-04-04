# Parse arguments
import argparse
parser = argparse.ArgumentParser(description='GPU Slime Simulation')
parser.add_argument('--width', dest='WIDTH', action='store', type=int, default=1024)
parser.add_argument('--height', dest='HEIGHT', action='store', type=int, default=1024)
parser.add_argument('-b', '--blocks', dest='BLOCKS', action='store', type=int, default=20)
parser.add_argument('-bs', '--block-size', dest='BLOCK_SIZE', action='store', type=int, default=128)
parser.add_argument('-s', '--speed', dest='SPEED', action='store', type=float, default=0.005)
parser.add_argument('-d', '--decay', dest='DECAY', action='store', type=float, default=0.97)
args = parser.parse_args()


# Imports
import os
import time
import shutil
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from matplotlib import pyplot as plt
from PIL import Image
from pycuda.compiler import SourceModule
from pathlib import Path

if os.path.exists('tmp'):
    shutil.rmtree('tmp')
os.mkdir('tmp')

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

blur = create_blur((3, 3))

agents = np.random.rand(args.BLOCKS * args.BLOCK_SIZE, 3)
agents = np.array(agents, dtype=np.float32)
matrix = np.zeros(shape=(args.HEIGHT, args.WIDTH), dtype=np.float32)
matrix2 = np.zeros_like(matrix, dtype=np.float32)
params = np.array([args.SPEED, args.WIDTH, args.HEIGHT], dtype=np.float32)

cmap = plt.get_cmap('magma')

for i in range(5000):
    print(i)
    matrix *= args.DECAY
    
    update(drv.InOut(agents), drv.InOut(matrix), drv.In(params), grid=(args.BLOCKS, 1), block=(args.BLOCK_SIZE, 1, 1))
    filter2D(drv.In(matrix), drv.In(np.array(blur.shape, dtype=np.int32)), drv.In(blur), drv.Out(matrix2), grid=(32, 32), block=(32, 32, 1))
    np.copyto(matrix, matrix2)

    if i > 100: 
        img = cmap(matrix)
        img *= 255
        img = Image.fromarray(img.astype('uint8'), 'RGBA')
        img.save('tmp/f{}.png'.format(i))
