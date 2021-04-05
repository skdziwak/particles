# Parse arguments
import argparse
parser = argparse.ArgumentParser(description='GPU Slime Simulation')
parser.add_argument('-g', '--grid-size', dest='GRID', action='store', type=int, default=32)
parser.add_argument('-b', '--blocks', dest='BLOCKS', action='store', type=int, default=20)
parser.add_argument('-bs', '--block-size', dest='BLOCK_SIZE', action='store', type=int, default=128)
parser.add_argument('-s', '--speed', dest='SPEED', action='store', type=float, default=0.005)
parser.add_argument('-d', '--decay', dest='DECAY', action='store', type=float, default=0.03)
parser.add_argument('-t', '--turn-speed', dest='TURN_SPEED', action='store', type=float, default=0.21)
parser.add_argument('-sa', '--sensor-angle', dest='SENSOR_ANGLE', action='store', type=float, default=45)
parser.add_argument('-sl', '--sensor-length', dest='SENSOR_LENGTH', action='store', type=float, default=0.05)

args = parser.parse_args()

PREVIEW = False

# Imports
import os
import time
import shutil
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
from pycuda.compiler import SourceModule
from pathlib import Path
import cv2
import tkinter
from scipy.ndimage.filters import gaussian_filter

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

# blur = create_blur((3,3))

agents = np.random.rand(args.BLOCKS * args.BLOCK_SIZE, 4)
agents = np.array(agents, dtype=np.float32)
matrix = np.zeros(shape=(args.GRID * 32, args.GRID * 32), dtype=np.float32)
matrix2 = np.zeros_like(matrix, dtype=np.float32)
params = np.array([args.SPEED, args.GRID * 32, args.GRID * 32, args.TURN_SPEED, args.SENSOR_ANGLE, args.SENSOR_LENGTH], dtype=np.float32)

cmap = plt.get_cmap('magma')

out = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'H264'), 25, (args.GRID * 32, args.GRID * 32), False)

try:
    for i in range(25 * 30):
        if i % 25 == 0:
            print(i // 25)
        matrix = matrix - args.DECAY
        matrix *= matrix > 0
        
        update(drv.InOut(agents), drv.InOut(matrix), drv.In(params), grid=(args.BLOCKS, 1), block=(args.BLOCK_SIZE, 1, 1))
        matrix = gaussian_filter(matrix, sigma=0.5)

        img = matrix * 255
        img = img.astype('uint8')
        img = np.repeat(img[:,:,np.newaxis], 3, axis=2)
        out.write(img)

        if PREVIEW:
            img = ImageTk.PhotoImage(image=Image.fromarray(img, 'RGBA'))
            canvas.create_image(0, 0, anchor='nw', image=img)
            window.update()
except KeyboardInterrupt:
    pass

out.release()