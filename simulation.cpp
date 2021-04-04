#include "matrix_utils.h"
#define PI 3.141592654f

extern "C" {

    __device__ int hash(int key) {
        int hash = 0;
        int b0 = (key & 255);
        int b1 = (key & 65280) >> 8;
        int b2 = (key & 16711680) >> 16;
        int b3 = (key & -16777216) >> 24;
        hash += b0;
        hash += (hash << 10);
        hash = hash ^ (hash >> 6);
        hash += b1;
        hash += (hash << 10);
        hash = hash ^ (hash >> 6);
        hash += b2;
        hash += (hash << 10);
        hash = hash ^ (hash >> 6);
        hash += b3;
        hash += (hash << 10);
        hash = hash ^ (hash >> 6);
        hash += (hash << 3);
        hash = hash ^ (hash >> 11);
        hash += (hash << 15);
        return hash;
    }


    struct Agent {
        float x;
        float y;
        float angle;
    };

    struct Params {
        float speed;
        float width;
        float height;
    };

    __global__ void update(Agent *agents, float *matrix, Params *params) {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        const int w = params->width, h = params->height;
        Agent *a = agents + i;

        // Move the agent
        a->x += cos(2 * PI * a->angle) * params->speed;
        a->y += sin(2 * PI * a->angle) * params->speed;

        // Check if the agent is over the border
        if(a->x >= 1) a->x -= 1.0;
        if(a->y >= 1) a->y -= 1.0;
        if(a->x < 0) a->x += 1.0;
        if(a->y < 0) a->y += 1.0;

        // Draw the agent
        const int x = a->x * w;
        const int y = a->y * h;
        if (x >= 0 && x < w && y >= 0 && y < h) {
            matrix[x * h + y] = 1;
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