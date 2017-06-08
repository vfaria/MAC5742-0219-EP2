/*********************************************************************
* Filename:   md2_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding MD2
              implementation. These tests do not encompass the full
              range of available test vectors, however, if the tests
              pass it is very, very likely that the code is correct
              and was compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include "md2cu.h"

/*********************** FUNCTION DEFINITIONS ***********************/
int md2_test()
{
    BYTE text1[] = {"abc"};
    BYTE text2[] = {"abcdefghijklmnopqrstuvwxyz"};
    BYTE text3_1[] = {"ABCDEFGHIJKLMNOPQRSTUVWXYZabcde"};
    BYTE text3_2[] = {"fghijklmnopqrstuvwxyz0123456789"};

    BYTE *d_text;

    BYTE hash1[MD2_BLOCK_SIZE] = {0xda,0x85,0x3b,0x0d,0x3f,0x88,0xd9,0x9b,0x30,0x28,0x3a,0x69,0xe6,0xde,0xd6,0xbb};
    BYTE hash2[MD2_BLOCK_SIZE] = {0x4e,0x8d,0xdf,0xf3,0x65,0x02,0x92,0xab,0x5a,0x41,0x08,0xc3,0xaa,0x47,0x94,0x0b};
    BYTE hash3[MD2_BLOCK_SIZE] = {0xda,0x33,0xde,0xf2,0xa4,0x2d,0xf1,0x39,0x75,0x35,0x28,0x46,0xc3,0x03,0x38,0xcd};

    BYTE buf[16];
    BYTE *d_buf;

    BYTE *d_ctx;
    int *d_ctx_len;  
    
    int pass = 1;

    // Device structures used in all tests:
    cudaMalloc(&d_ctx, 80 * sizeof(BYTE));
    cudaMalloc(&d_ctx_len, sizeof(int));
    cudaMalloc(&d_buf, 16 * sizeof(BYTE));

    // Test for text1
    cudaMalloc(&d_text, strlen((char *) text1) * sizeof(BYTE));
    cudaMemcpy(d_text, &text1, strlen((char *) text1) * sizeof(BYTE), cudaMemcpyHostToDevice);

    md2_init<<<1, 1>>>(d_ctx, d_ctx_len);
    md2_update<<<1, 1>>>(d_ctx, d_ctx_len, d_text, strlen((char *) text1));
    md2_final<<<1, 1>>>(d_ctx, d_ctx_len, d_buf);

    cudaMemcpy(&buf, d_buf, 16 * sizeof(BYTE), cudaMemcpyDeviceToHost);
    cudaFree(d_text);

    pass = pass && !memcmp(hash1, buf, MD2_BLOCK_SIZE);

    // Test for text2
    cudaMalloc(&d_text, strlen((char *) text2) * sizeof(BYTE));
    cudaMemcpy(d_text, &text2, strlen((char *) text2) * sizeof(BYTE), cudaMemcpyHostToDevice);

    md2_init<<<1, 1>>>(d_ctx, d_ctx_len);
    md2_update<<<1, 1>>>(d_ctx, d_ctx_len, d_text, strlen((char *) text2));
    md2_final<<<1, 1>>>(d_ctx, d_ctx_len, d_buf);

    cudaMemcpy(&buf, d_buf, 16 * sizeof(BYTE), cudaMemcpyDeviceToHost);
    cudaFree(d_text);

    pass = pass && !memcmp(hash2, buf, MD2_BLOCK_SIZE);

    // Test for text3
    cudaMalloc(&d_text, strlen((char *) text3_1) * sizeof(BYTE));
    cudaMemcpy(d_text, &text3_1, strlen((char *) text3_1) * sizeof(BYTE), cudaMemcpyHostToDevice);

    md2_init<<<1, 1>>>(d_ctx, d_ctx_len);
    md2_update<<<1, 1>>>(d_ctx, d_ctx_len, d_text, strlen((char *) text3_1));
    cudaFree(d_text);

    cudaMalloc(&d_text, strlen((char *) text3_2) * sizeof(BYTE));
    cudaMemcpy(d_text, &text3_2, strlen((char *) text3_2) * sizeof(BYTE), cudaMemcpyHostToDevice);
    md2_update<<<1, 1>>>(d_ctx, d_ctx_len, d_text, strlen((char *) text3_2));
    md2_final<<<1, 1>>>(d_ctx, d_ctx_len, d_buf);

    cudaMemcpy(&buf, d_buf, 16 * sizeof(BYTE), cudaMemcpyDeviceToHost);
    cudaFree(d_text);

    pass = pass && !memcmp(hash3, buf, MD2_BLOCK_SIZE);

    cudaFree(d_ctx);
    cudaFree(d_ctx_len);

    return(pass);
}

int main()
{
    printf("MD2 tests: %s\n", md2_test() ? "SUCCEEDED" : "FAILED");
}
