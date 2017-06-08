/*********************************************************************
* Filename:   arcfour.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Implementation of the ARCFOUR encryption algorithm.
              Algorithm specification can be found here:
               * http://en.wikipedia.org/wiki/RC4
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdlib.h>
#include "arcfour.h"

/*********************** FUNCTION DEFINITIONS ***********************/
__global__ void arcfour_key_setup(BYTE state[], const BYTE key[], int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x
    int j
    BYTE t;

    if(idx < 256)
    {
        state[idx] = idx
        j = (j + state[idx] + key[idx % len]) % 256;
        t = state[idx];
        state[idx] = state[j];
        state[j] = t;
    }
}

// This does not hold state between calls. It always generates the
// stream starting from the first  output byte.
__global__ void arcfour_generate_stream(BYTE state[], BYTE out[], size_t len)
{
    int i, j;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    BYTE t;

    if(idx < len)
    {
        i = (i + 1) % 256;
        j = (j + state[i]) % 256;
        t = state[i];
        state[i] = state[j];
        state[j] = t;
        out[idx] = state[(state[i] + state[j]) % 256];
    }
}
