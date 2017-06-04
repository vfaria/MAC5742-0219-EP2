/*********************************************************************
* Filename:   rot-13_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding ROT-13
              implementation. These tests do not encompass the full
              range of available test vectors, however, if the tests
              pass it is very, very likely that the code is correct
              and was compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <string.h>
#include "rot-13cu.h"

/*********************** FUNCTION DEFINITIONS ***********************/
int rot13_test()
{
    char text[] = {"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"};
    char code[] = {"NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"};
    char buf[1024];
    int pass = 1;
    int len;

    char* d_buf;
    int* d_len;

    if(cudaMalloc(&d_buf, sizeof(char) * 1024) != cudaSuccess)
    {
        return 0;
    }

    if(cudaMalloc(&d_len, sizeof(int)) != cudaSuccess)
    {
        return 0;
    }

    strcpy(buf, text);
    len = strlen(text);

    if(cudaMemcpy(d_buf, buf, sizeof(char) * 1024, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cudaFree(d_buf);
        cudaFree(d_len);
        return 0;
    }

    if(cudaMemcpy(d_len, &len, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cudaFree(d_buf);
        cudaFree(d_len);
        return 0;
    }

    // To encode, just apply ROT-13.
    rot13<<<4, 256>>>(d_buf, len);

    if(cudaMemcpy(buf, d_buf, sizeof(char) * 1024, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        cudaFree(d_buf);
        return 0;
    }
    
    pass = pass && !strcmp(code, buf);

    // To decode, just re-apply ROT-13.
    rot13<<<4, 256>>>(d_buf, len);

    if(cudaMemcpy(buf, d_buf, sizeof(char) * 1024, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        cudaFree(d_buf);
        return 0;
    }

    pass = pass && !strcmp(text, buf);

    cudaFree(d_buf);
    cudaFree(d_len);

    return(pass);
}

int main()
{
    printf("ROT-13 tests with CUDA: %s\n", rot13_test() ? "SUCCEEDED" : "FAILED");

    return(0);
}
