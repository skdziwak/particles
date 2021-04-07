# Parse arguments
import argparse
parser = argparse.ArgumentParser(description='GPU Slime Simulation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('SECONDS', type=int)
parser.add_argument('OUTPUT', type=str)
parser.add_argument('-f', '--fps', dest='FPS', action='store', type=int, default=25, help='Frames per second')
parser.add_argument('-u', '--upf', dest='UPF', action='store', type=int, default=30, help='Updates per frame')
parser.add_argument('-g', '--grid-size', dest='GRID', action='store', type=int, default=32, help='Matrix grid size')
parser.add_argument('-b', '--block-size', dest='BLOCK', action='store', type=int, default=32, help='Matrix block size')
parser.add_argument('-ab', '--agent-blocks', dest='AGENT_BLOCKS', action='store', type=int, default=100, help='Agents blocks')
parser.add_argument('-abs', '--agent-block-size', dest='AGENT_BLOCK_SIZE', action='store', type=int, default=1024, help='Agents in each block')
parser.add_argument('-s', '--speed', dest='SPEED', action='store', type=float, default=0.001, help='Agent\'s speed')
parser.add_argument('-d', '--decay', dest='DECAY', action='store', type=float, default=0.002, help='Decay in each update')
parser.add_argument('-bl', '--blur-size', dest='BLUR', action='store', type=int, default=7, help='Blur filter size')
parser.add_argument('-t', '--turn-speed', dest='TURN_SPEED', action='store', type=float, default=0.21, help='Agent\'s turning speed')
parser.add_argument('-sa', '--sensor-angle', dest='SENSOR_ANGLE', action='store', type=float, default=30, help='Agent\'s angle of sensor')
parser.add_argument('-sl', '--sensor-length', dest='SENSOR_LENGTH', action='store', type=float, default=0.03, help='Agent\'s length of sensor')
parser.add_argument('-c', '--codec', dest='CODEC', action='store', type=str, default='H264', help='Video codec')
parser.add_argument('-cm', '--colormap', dest='CM', action='store', type=str, default='magma', help='Matplotlib colormap')
parser.add_argument('-w' '--wrapping', dest='WRAPPING_BORDERS', action='store_true', help='Wrapping borders')

args = parser.parse_args()

PREVIEW = False

# Imports
import os, sys
import time
import shutil
import random
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
from pycuda.compiler import SourceModule
from pathlib import Path
import cv2

# Compile CUDA Kernels
print('Compiling kernels')
module = SourceModule(Path('simulation.cpp').read_text(), include_dirs=[os.path.join(os.getcwd(), 'include')], no_extern_c=True)
update = module.get_function('update')
decay = module.get_function('decay')
filter2D = module.get_function('filter2D')
apply_colormap = module.get_function('apply_colormap')

# Bluf filter
def create_blur(a):
    fltr = np.zeros(shape=(a, a), dtype=np.float32)
    c = a // 2
    max_c = (a - 1) / 2
    for x in range(a):
        for y in range(a):
            fltr[x,y] = 1 - np.sqrt((c - x) ** 2 + (c - y) ** 2) / max_c
    fltr *= fltr > 0
    return fltr / np.sum(fltr)
print('Creating blur filter')
blur = create_blur(args.BLUR)
blur_shape = np.array(blur.shape, dtype=np.int32)

# Creating colormap
print('Creating colormap')
CMAP_SIZE = 16384
cmap = np.ndarray(shape=(CMAP_SIZE), dtype=np.float32)
for i in range(cmap.shape[0]):
    cmap[i] = i / CMAP_SIZE
cmap = plt.get_cmap(args.CM)(cmap)[...,[2,1,0]]
cmap *= 255
cmap = cmap.astype(np.uint8).flatten()

print('Allocating memory')

# Alloc memory in RAM
agents = np.random.rand(args.AGENT_BLOCKS * args.AGENT_BLOCK_SIZE, 4)
agents = np.array(agents, dtype=np.float32)
matrix = np.zeros(shape=(args.GRID * args.BLOCK, args.GRID * args.BLOCK), dtype=np.float32)
image = np.zeros(shape=(args.GRID * args.BLOCK, args.GRID * args.BLOCK, 3), dtype=np.uint8)
params = np.array(
    [
        args.SPEED, 
        args.GRID * args.BLOCK, 
        args.GRID * args.BLOCK, 
        args.TURN_SPEED, 
        args.SENSOR_ANGLE, 
        args.SENSOR_LENGTH, 
        args.DECAY,
        1 if args.WRAPPING_BORDERS else 0
    ], dtype=np.float32)

# Alloc memory in VRAM
agents_gpu = cuda.mem_alloc_like(agents)
matrix_gpu = cuda.mem_alloc_like(matrix)
matrix2_gpu = cuda.mem_alloc_like(matrix)
params_gpu = cuda.mem_alloc_like(params)
blur_shape_gpu = cuda.mem_alloc_like(blur_shape)
blur_gpu = cuda.mem_alloc_like(blur)
cmap_gpu = cuda.mem_alloc_like(cmap)
image_gpu = cuda.mem_alloc_like(image)

# Copy initial values
cuda.memcpy_htod(agents_gpu, agents)
cuda.memcpy_htod(matrix_gpu, matrix)
cuda.memcpy_htod(params_gpu, params)
cuda.memcpy_htod(blur_gpu, blur)
cuda.memcpy_htod(blur_shape_gpu, blur_shape)
cuda.memcpy_htod(cmap_gpu, cmap)

MATRIX_GRID = (args.GRID, args.GRID)
MATRIX_BLOCK = (args.BLOCK, args.BLOCK, 1)

out = cv2.VideoWriter(args.OUTPUT,cv2.VideoWriter_fourcc(*args.CODEC), args.FPS, (args.GRID * args.BLOCK, args.GRID * args.BLOCK))

# Print progress bar
def progress_bar(f, width):
    s = '[{}>{}]'.format('=' * int(f * width), ' ' * int((1-f) * width))
    sys.stdout.write('\r' + s + ' {:.2f}%'.format(f * 100))
    sys.stdout.flush()

TIMEFRAMES = args.SECONDS * args.FPS * args.UPF
UPS = args.FPS * args.UPF

print('Rendering')

t = time.time()

for i in range(args.SECONDS * args.FPS * args.UPF):
    progress = float(i) / TIMEFRAMES

    update(agents_gpu, matrix_gpu, params_gpu, grid=(args.AGENT_BLOCKS, 1), block=(args.AGENT_BLOCK_SIZE, 1, 1))
    decay(matrix_gpu, params_gpu, grid=MATRIX_GRID, block=MATRIX_BLOCK)

    if i % args.UPF == 0: 
        filter2D(matrix_gpu, blur_shape_gpu, blur_gpu, matrix2_gpu, grid=MATRIX_GRID, block=MATRIX_BLOCK)
        apply_colormap(matrix2_gpu, cmap_gpu, image_gpu, grid=MATRIX_GRID, block=MATRIX_BLOCK)
        cuda.memcpy_dtoh(image, image_gpu)

        out.write(image)
        progress_bar(progress, 100)
out.release()
progress_bar(1, 100)

print('\nFinished in {:.2f} s'.format(time.time() - t))