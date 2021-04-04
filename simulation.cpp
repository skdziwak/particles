#include "matrix_utils.h"
#define SPEED 0.05
#define PI 3.141592654f

extern "C" {

    struct Agent {
        float x;
        float y;
        float a;
    };

    __global__ void update(Agent *ptr, float matrix[1024][1024]) {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        ptr[i].x += cos(ptr[i].a * 2 * PI) * SPEED;
        ptr[i].y += sin(ptr[i].a * 2 * PI) * SPEED;
        const int x = ptr[i].x * 1024;
        const int y = ptr[i].y * 1024;

        if (x >= 0 && x < 1024 && y >= 0 && y < 1024/) {
            matrix[y][x] = 1.0;
        }
    }

    __global__ void filter2D(float *input, int *shape, float *filter, float *output) {
        const int x = getX();
        const int y = getY();
        const int fw = shape[0];
        const int fh = shape[1];
        const int ox = fw / 2;
        const int oy = fh / 2;

        const int w = getWidth();
        const int h = getHeight();
        const int i = getIndex2D(x, y);

        int c = 0;
        float value = 0.0;
        for(int a = 0 ; a < fw ; a++) {
            for(int b = 0 ; b < fh ; b++) {
                int j = getIndex2D(x - ox + a, y - oy + b);
                if(j != -1) {
                    value += input[j] * filter[a + b * fw];
                    c++;
                }   
            }
        }
        output[i] = value / c;
    }
    
}