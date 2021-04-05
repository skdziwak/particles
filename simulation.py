# Parse arguments
import argparse
parser = argparse.ArgumentParser(description='GPU Slime Simulation')
parser.add_argument('FRAMES', type=int)
parser.add_argument('OUTPUT', type=str)
parser.add_argument('-f', '--fps', dest='FPS', action='store', type=int, default=25)
parser.add_argument('-g', '--grid-size', dest='GRID', action='store', type=int, default=32)
parser.add_argument('-b', '--blocks', dest='BLOCKS', action='store', type=int, default=20)
parser.add_argument('-bs', '--block-size', dest='BLOCK_SIZE', action='store', type=int, default=128)
parser.add_argument('-s', '--speed', dest='SPEED', action='store', type=float, default=0.005)
parser.add_argument('-d', '--decay', dest='DECAY', action='store', type=float, default=0.03)
parser.add_argument('-t', '--turn-speed', dest='TURN_SPEED', action='store', type=float, default=0.21)
parser.add_argument('-sa', '--sensor-angle', dest='SENSOR_ANGLE', action='store', type=float, default=45)
parser.add_argument('-sl', '--sensor-length', dest='SENSOR_LENGTH', action='store', type=float, default=0.05)
parser.add_argument('-c', '--codec', dest='CODEC', action='store', type=str, default='H264')

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
import tkinter
from scipy.signal import gaussian

if PREVIEW:
    window = tkinter.Tk()
    window.geometry('{}x{}+10+20'.format(args.GRID * 32, args.GRID * 32))
    canvas = tkinter.Canvas(window, bg="white", height=args.GRID * 32, width=args.GRID * 32)
    canvas.pack()


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

blur = create_blur(7)
plt.imshow(blur)
plt.show()
blur_shape = np.array(blur.shape, dtype=np.int32)

agents = np.random.rand(args.BLOCKS * args.BLOCK_SIZE, 4)
for i in range(agents.shape[0]):
    agents[i,0] = random.random() / 4 - 0.5
    agents[i,1] = random.random() / 4 - 0.5
agents = np.array(agents, dtype=np.float32)
matrix = np.zeros(shape=(args.GRID * 32, args.GRID * 32), dtype=np.float32)
params = np.array([args.SPEED, args.GRID * 32, args.GRID * 32, args.TURN_SPEED, args.SENSOR_ANGLE, args.SENSOR_LENGTH], dtype=np.float32)

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

BLUR_GRID = (args.GRID, args.GRID)
BLUR_BLOCK = (32, 32, 1)

out = cv2.VideoWriter(args.OUTPUT,cv2.VideoWriter_fourcc(*args.CODEC), args.FPS, (args.GRID * 32, args.GRID * 32))

t = time.time()
for i in range(args.FRAMES):
    if i % 100 == 0:
        sys.stdout.write('\r{}/{} '.format(i, args.FRAMES))
        sys.stdout.flush()
    update(agents_gpu, matrix_gpu, params_gpu, grid=(args.BLOCKS, 1), block=(args.BLOCK_SIZE, 1, 1))
    decay(matrix_gpu, grid=BLUR_GRID, block=BLUR_BLOCK)
    #blur(matrix_gpu, matrix2_gpu, grid=BLUR_GRID, block=BLUR_BLOCK)
    #cuda.memcpy_dtod(matrix_gpu, matrix2_gpu, matrix.nbytes)

    if i % 10 == 0: 
        filter2D(matrix_gpu, blur_shape_gpu, blur_gpu, matrix2_gpu, grid=BLUR_GRID, block=BLUR_BLOCK)
        cuda.memcpy_dtoh(matrix, matrix2_gpu)
        matrix /= np.max(matrix)
        img = (matrix * 255).astype(np.uint8)
        img = np.repeat(img[:,:,np.newaxis], 3, axis=2)
        out.write(img)
    
print(time.time() - t)

out.release()