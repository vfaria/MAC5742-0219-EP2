/*********************************************************************
* Filename:   blowfish_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding Base64
              implementation. These tests do not encompass the full
              range of available test vectors, however, if the tests
              pass it is very, very likely that the code is correct
              and was compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <memory.h>
#include "base64cu.h"

/*********************** FUNCTION DEFINITIONS ***********************/
int base64_test()
{
    BYTE text[1024] = {"Man is distinguished, not only by his reason, but by this singular passion from other animals, which is a lust of the mind, that by a perseverance of delight in the continued and indefatigable generation of knowledge, exceeds the short vehemence of any carnal pleasure."};
    BYTE code[1024] = {"TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlz\nIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2Yg\ndGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGlu\ndWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRo\nZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4="};
    BYTE buf[1024];
    size_t buf_len;
    size_t* d_buf_len;
    size_t* d_buf_len_null;
    int pass = 1;
    BYTE* d_text;

    BYTE* d_buf;

    cudaMalloc(&d_buf_len, sizeof(int*) * 1024);
    cudaMalloc(&d_buf_len_null, sizeof(int*) * 1024);
    cudaMalloc(&d_text, sizeof(char) * 1024);
    cudaMalloc(&d_buf, sizeof(char) * 1024);
/*    cudaMalloc(&d_charset, sizeof(char) * strlen((char*)charset));

    cudaMemcpy(d_charset, &charset, sizeof(char) * strlen((char*)charset), cudaMemcpyHostToDevice);
*/    cudaMemcpy(d_text, text, sizeof(char) * 1024, cudaMemcpyHostToDevice);

    base64_encode<<<1, 512>>>(d_text, d_buf, strlen((char*)text), 1, d_buf_len);
    //base64_encode<<<4, 256>>>(d_text, NULL, strlen((char*)text), 1, d_buf_len_null);

    cudaMemcpy(&buf_len, d_buf_len, sizeof(size_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(&buf_len_null, d_buf_len_null, sizeof(size_t), cudaMemcpyDeviceToHost);

    cudaMemcpy(buf, d_buf, sizeof(char) * 1024, cudaMemcpyDeviceToHost);

    printf("%s\n\n", buf);
    printf("%s\n\n", code);

    pass = pass && ((buf_len == strlen((char*)code)));// &&
     //                (buf_len == buf_len_null));

    pass = pass && !strcmp((char*)code, (char*)buf);
    
/*
    memset(buf, 0, sizeof(buf));
    buf_len = base64_decode(code, buf, strlen((char*)code));
    pass = pass && ((buf_len == strlen((char*)text)) &&
                    (buf_len == base64_decode(code, NULL, strlen((char*)code))));
    pass = pass && !strcmp((char*)text, (char*)buf);
*/
    return(pass);
}

int main()
{
    printf("Base64 tests: %s\n", base64_test() ? "PASSED" : "FAILED");

    return 0;
}
