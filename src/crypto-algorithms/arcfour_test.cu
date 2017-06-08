/*********************************************************************
* Filename:   arcfour_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding ARCFOUR
              implementation. These tests do not encompass the full
              range of available test vectors, however, if the tests
              pass it is very, very likely that the code is correct
              and was compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <memory.h>
#include "arcfour.h"

/*********************** FUNCTION DEFINITIONS ***********************/
int rc4_test()
{
    BYTE state[256];
    BYTE key[3][10] = {{"Key"}, {"Wiki"}, {"Secret"}};
    BYTE stream[3][10] = {{0xEB,0x9F,0x77,0x81,0xB7,0x34,0xCA,0x72,0xA7,0x19},
                          {0x60,0x44,0xdb,0x6d,0x41,0xb7},
                          {0x04,0xd4,0x6b,0x05,0x3c,0xa8,0x7b,0x59}};
    int stream_len[3] = {10,6,8};
    BYTE buf[1024];
    int idx;
    int pass = 1;

    BYTE *d_state;
    BYTE *d_key;
    BYTE *d_stream;
    BYTE *d_buf;

    cudaMalloc(&d_state,  sizeof(BYTE) * 256);
    cudaMalloc(&d_key,    sizeof(BYTE) * 3 * 10);
    cudaMalloc(&d_stream, sizeof(BYTE) * 3 * 10);
    cudaMalloc(&d_buf,    sizeof(BYTE) * 1024);
    cudaMalloc(&d_stream_len, sizeof(BYTE) * 3);

    cudaMemcpy(d_state, state,           sizeof(BYTE) * 256,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key,               sizeof(BYTE) * 3 * 10, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stream, stream,         sizeof(BYTE) * 3 * 10, cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf, buf,               sizeof(BYTE) * 1024,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_stream_len, stream_len, sizeof(BYTE) * 3,      cudaMemcpyHostToDevice);

    // Only test the output stream. Note that the state can be reused.
    for (idx = 0; idx < 3; idx++) {
        arcfour_key_setup<<<4,256>>>(d_state, d_key[idx], d_strlen(d_key[idx]));
        arcfour_generate_stream(d_state,d_ buf, d_stream_len[idx]);

        cudaMemcpy(stream, d_stream,         sizeof(BYTE) * 3 * 10, cudaMemcpyDeviceToHost);
        cudaMemcpy(buf, d_buf,               sizeof(BYTE) * 1024,   cudaMemcpyDeviceToHost);
        cudaMemcpy(stream_len, d_stream_len, sizeof(BYTE) * 3,      cudaMemcpyDeviceToHost);

        pass = pass && !memcmp(stream[idx], buf, stream_len[idx]);
    }


    cudaFree(d_state);
    cudaFree(d_key);
    cudaFree(d_stream);
    cudaFree(d_buf);
    cudaFree(d_stream_len);

    return(pass);
}

int main()
{
    printf("ARCFOUR tests: %s\n", rc4_test() ? "SUCCEEDED" : "FAILED");

    return(0);
}
