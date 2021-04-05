# Parse arguments
import argparse
parser = argparse.ArgumentParser(description='GPU Slime Simulation')
parser.add_argument('SECONDS', type=int)
parser.add_argument('OUTPUT', type=str)
parser.add_argument('-f', '--fps', dest='FPS', action='store', type=int, default=25)
parser.add_argument('-u', '--upf', dest='UPF', action='store', type=int, default=300)
parser.add_argument('-g', '--grid-size', dest='GRID', action='store', type=int, default=32)
parser.add_argument('-b', '--block-size', dest='BLOCK', action='store', type=int, default=32)
parser.add_argument('-ab', '--agent-blocks', dest='AGENT_BLOCKS', action='store', type=int, default=100)
parser.add_argument('-abs', '--agent-block-size', dest='AGENT_BLOCK_SIZE', action='store', type=int, default=1024)
parser.add_argument('-s', '--speed', dest='SPEED', action='store', type=float, default=0.0002)
parser.add_argument('-d', '--decay', dest='DECAY', action='store', type=float, default=0.002)
parser.add_argument('-bl', '--blur-size', dest='BLUR', action='store', type=int, default=7)
parser.add_argument('-t', '--turn-speed', dest='TURN_SPEED', action='store', type=float, default=0.21)
parser.add_argument('-sa', '--sensor-angle', dest='SENSOR_ANGLE', action='store', type=float, default=30)
parser.add_argument('-sl', '--sensor-length', dest='SENSOR_LENGTH', action='store', type=float, default=0.03)
parser.add_argument('-c', '--codec', dest='CODEC', action='store', type=str, default='H264')
parser.add_argument('-cm', '--colormap', dest='CM', action='store', type=str, default=None)

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

if os.path.exists('tmp'):
    shutil.rmtree('tmp')
os.mkdir('tmp')

# Compile CUDA Kernels
module = SourceModule(Path('simulation.cpp').read_text(), include_dirs=[os.path.join(os.getcwd(), 'include')], no_extern_c=True)
update = module.get_function('update')
decay = module.get_function('decay')
filter2D = module.get_function('filter2D')

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

blur = create_blur(args.BLUR)
blur_shape = np.array(blur.shape, dtype=np.int32)

agents = np.random.rand(args.AGENT_BLOCKS * args.AGENT_BLOCK_SIZE, 4)
agents = np.array(agents, dtype=np.float32)
matrix = np.zeros(shape=(args.GRID * args.BLOCK, args.GRID * args.BLOCK), dtype=np.float32)
params = np.array([args.SPEED, args.GRID * args.BLOCK, args.GRID * args.BLOCK, args.TURN_SPEED, args.SENSOR_ANGLE, args.SENSOR_LENGTH, args.DECAY], dtype=np.float32)

# Alloc memory on GPU
agents_gpu = cuda.mem_alloc_like(agents)
matrix_gpu = cuda.mem_alloc_like(matrix)
matrix2_gpu = cuda.mem_alloc_like(matrix)
params_gpu = cuda.mem_alloc_like(params)
blur_shape_gpu = cuda.mem_alloc_like(blur_shape)
blur_gpu = cuda.mem_alloc_like(blur)

# Copy initial values
cuda.memcpy_htod(agents_gpu, agents)
cuda.memcpy_htod(matrix_gpu, matrix)
cuda.memcpy_htod(matrix_gpu, matrix)
cuda.memcpy_htod(params_gpu, params)
cuda.memcpy_htod(blur_gpu, blur)
cuda.memcpy_htod(blur_shape_gpu, blur_shape)

MATRIX_GRID = (args.GRID, args.GRID)
MATRIX_BLOCK = (args.BLOCK, args.BLOCK, 1)

out = cv2.VideoWriter(args.OUTPUT,cv2.VideoWriter_fourcc(*args.CODEC), args.FPS, (args.GRID * args.BLOCK, args.GRID * args.BLOCK))

if args.CM:
    cmap = plt.get_cmap(args.CM)
else:
    cmap = None

def animate_params(p):
    pass

def progress_bar(f, width):
    s = '[{}>{}]'.format('=' * int(f * width), ' ' * int((1-f) * width))
    sys.stdout.write('\r' + s + ' {:.2f}%'.format(f * 100))
    sys.stdout.flush()

TIMEFRAMES = args.SECONDS * args.FPS * args.UPF
UPS = args.FPS * args.UPF
t = time.time()

for i in range(args.SECONDS * args.FPS * args.UPF):
    progress = float(i) / TIMEFRAMES
    animate_params(progress)
    cuda.memcpy_htod(params_gpu, params)

    update(agents_gpu, matrix_gpu, params_gpu, grid=(args.AGENT_BLOCKS, 1), block=(args.AGENT_BLOCK_SIZE, 1, 1))
    decay(matrix_gpu, params_gpu, grid=MATRIX_GRID, block=MATRIX_BLOCK)

    if i % args.UPF == 0: 
        filter2D(matrix_gpu, blur_shape_gpu, blur_gpu, matrix2_gpu, grid=MATRIX_GRID, block=MATRIX_BLOCK)
        cuda.memcpy_dtoh(matrix, matrix2_gpu)
        if cmap:
            img = (cmap(matrix, ) * 255).astype(np.uint8)
            img = img[...,:3]
            img = img[...,::-1]
        else:
            img = (matrix * 255).astype(np.uint8)
            img = np.repeat(img[:,:,np.newaxis], 3, axis=2)
        out.write(img)
        progress_bar(progress, 100)
out.release()
progress_bar(1, 100)

print('\nFinished in {:.2f} s'.format(time.time() - t))