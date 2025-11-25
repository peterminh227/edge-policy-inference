// Samurice: Hanoi 29 Oct 2025
// How to compile:
//   gcc -O3 -std=c99 policy_test.c -lm -o policy_test
// How to run:
//   ./policy_test

#include <stdio.h>
#include <string.h>
#define WEIGHTS_HEADER "policy_random_weights.h" 
#include "src/policy_infer.h"

int main(){
    printf("POL_INPUT=%d POL_HIDDEN=%d POL_LAYERS=%d OUT=%d\n",
           POL_INPUT, POL_HIDDEN, POL_LAYERS, POL_LIN_OUT[POL_LAYERS-1]);

    float h[POL_HIDDEN], c[POL_HIDDEN];
    pol_reset(h, c);

    float x[POL_INPUT];           
    for(int i=0;i<POL_INPUT;++i) x[i] = (float)i*0.01f; // example input
    
    float y[256] = {0};           
    pol_step(x, h, c, y);

    int outdim = POL_LIN_OUT[POL_LAYERS-1];
    printf("action (with dim %d):", outdim);
    for(int i=0;i<outdim && i<12; ++i) printf(" %.6g", y[i]);
    printf("\n");
    return 0;
}
