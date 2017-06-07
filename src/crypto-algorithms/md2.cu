/*********************************************************************
* Filename:   md2.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Implementation of the MD2 hashing algorithm.
                  Algorithm specification can be found here:
                   * http://tools.ietf.org/html/rfc1319 .
              Input is  little endian byte order.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include "md2cu.h"

/**************************** VARIABLES *****************************/
__constant__ static const BYTE s[256] = {
    41, 46, 67, 201, 162, 216, 124, 1, 61, 54, 84, 161, 236, 240, 6,
    19, 98, 167, 5, 243, 192, 199, 115, 140, 152, 147, 43, 217, 188,
    76, 130, 202, 30, 155, 87, 60, 253, 212, 224, 22, 103, 66, 111, 24,
    138, 23, 229, 18, 190, 78, 196, 214, 218, 158, 222, 73, 160, 251,
    245, 142, 187, 47, 238, 122, 169, 104, 121, 145, 21, 178, 7, 63,
    148, 194, 16, 137, 11, 34, 95, 33, 128, 127, 93, 154, 90, 144, 50,
    39, 53, 62, 204, 231, 191, 247, 151, 3, 255, 25, 48, 179, 72, 165,
    181, 209, 215, 94, 146, 42, 172, 86, 170, 198, 79, 184, 56, 210,
    150, 164, 125, 182, 118, 252, 107, 226, 156, 116, 4, 241, 69, 157,
    112, 89, 100, 113, 135, 32, 134, 91, 207, 101, 230, 45, 168, 2, 27,
    96, 37, 173, 174, 176, 185, 246, 28, 70, 97, 105, 52, 64, 126, 15,
    85, 71, 163, 35, 221, 81, 175, 58, 195, 92, 249, 206, 186, 197,
    234, 38, 44, 83, 13, 110, 133, 40, 132, 9, 211, 223, 205, 244, 65,
    129, 77, 82, 106, 220, 55, 200, 108, 193, 171, 250, 36, 225, 123,
    8, 12, 189, 177, 74, 120, 136, 149, 139, 227, 99, 232, 109, 233,
    203, 213, 254, 59, 0, 29, 57, 242, 239, 183, 14, 102, 88, 208, 228,
    166, 119, 114, 248, 235, 117, 75, 10, 49, 68, 80, 180, 143, 237,
    31, 26, 219, 153, 141, 51, 159, 17, 131, 20
};

/*********************** FUNCTION DEFINITIONS ***********************/
__device__ void md2_transform(BYTE ctx[], BYTE data[])
{
    int j,k,t;

    //memcpy(&ctx->state[16], data);
    for (j=0; j < 16; ++j) {
        ctx[MD2_CTX_STATE_OFFSET + j + 16] = data[j];
        ctx[MD2_CTX_STATE_OFFSET + j + 32] = (ctx[MD2_CTX_STATE_OFFSET + j+16] ^ ctx[MD2_CTX_STATE_OFFSET + j]);
    }

    t = 0;
    for (j = 0; j < 18; ++j) {
        for (k = 0; k < 48; ++k) {
            ctx[MD2_CTX_STATE_OFFSET + k] ^= s[t];
            t = ctx[MD2_CTX_STATE_OFFSET + k];
        }
        t = (t+j) & 0xFF;
    }

    t = ctx[MD2_CTX_CHECKSUM_OFFSET + 15];
    for (j=0; j < 16; ++j) {
        ctx[MD2_CTX_CHECKSUM_OFFSET + j] ^= s[data[j] ^ t];
        t = ctx[MD2_CTX_CHECKSUM_OFFSET + j];
    }
}

__global__ void md2_init(BYTE ctx[], int *ctx_len)
{
    int i;
    printf("Called md2_init\n");

    for (i=0; i < 48; ++i) {
        // printf("Setting state bit %d (context array position %d)\n", i, i + MD2_CTX_STATE_OFFSET);
        ctx[i + MD2_CTX_STATE_OFFSET] = 0;
    }
    for (i=0; i < 16; ++i) {
        // printf("Setting checksum bit %d (context array position %d)\n", i, i + MD2_CTX_CHECKSUM_OFFSET);
        ctx[i + MD2_CTX_CHECKSUM_OFFSET] = 0;
    }
    printf("Setting ctx_len\n");
    *ctx_len = 0;
}

__global__ void md2_update(BYTE ctx[], int *ctx_len, const BYTE data[], size_t len)
{
    size_t i;
    printf("Called md2_update\n");

    for (i = 0; i < len; ++i) {
        ctx[*ctx_len] = data[i];
        *ctx_len = *ctx_len + 1;
        if (*ctx_len == MD2_BLOCK_SIZE) {
            
            md2_transform(ctx, ctx);
            *ctx_len = 0;
        }
    }
}

__global__ void md2_final(BYTE ctx[], int *ctx_len, BYTE hash[])
{
    int to_pad;

    printf("Called md2_final\n");
    to_pad = MD2_BLOCK_SIZE - *ctx_len;
    printf("to_pad: %d\n", to_pad);

    while (*ctx_len < MD2_BLOCK_SIZE) {
        ctx[*ctx_len] = to_pad;
        *ctx_len = *ctx_len + 1;
    }

    md2_transform(ctx, ctx);
    md2_transform(ctx, &ctx[MD2_CTX_CHECKSUM_OFFSET]);

    memcpy(hash, &ctx[MD2_CTX_STATE_OFFSET], MD2_BLOCK_SIZE);
}
