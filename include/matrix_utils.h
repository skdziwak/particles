#ifndef _MATRIX_UTILS_H
#define _MATRIX_UTILS_H

extern "C" {
    __device__ int getX() {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

    __device__ int getY() {
        return blockIdx.y * blockDim.y + threadIdx.y;
    }

    __device__ int getIndex2D(int x, int y) {
        if(x < 0 || x >= gridDim.x * blockDim.x || y < 0 || y >= gridDim.y * blockDim.y) return -1;
        return x + y * gridDim.x * blockDim.x;
    }

    __device__ int getWidth() {
        return gridDim.x * blockDim.x;
    }

    __device__ int getHeight() {
        return gridDim.y * blockDim.y;
    }
}

#endif