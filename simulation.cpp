#include "matrix_utils.h"

extern "C" {

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