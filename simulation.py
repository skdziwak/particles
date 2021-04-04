# Parsing arguments
import argparse

parser = argparse.ArgumentParser(description='GPU Slime Simulation')
parser.add_argument('a', type=int, nargs='?', default=43)

args = parser.parse_args()
print(args)

# Imports
import os
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from matplotlib import pyplot as plt
from PIL import Image
from pycuda.compiler import SourceModule
from pathlib import Path


module = SourceModule(Path('simulation.cpp').read_text(), include_dirs=[os.path.join(os.getcwd(), 'include')], no_extern_c=True)
