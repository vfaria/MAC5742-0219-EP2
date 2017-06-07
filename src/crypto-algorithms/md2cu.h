/*********************************************************************
* Filename:   md2.h
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Defines the API for the corresponding MD2 implementation.
*********************************************************************/

#ifndef MD2_H
#define MD2_H

/*************************** HEADER FILES ***************************/
#include <stddef.h>

/****************************** MACROS ******************************/
#define MD2_BLOCK_SIZE 16

/**************************** DATA TYPES ****************************/
#define MD2_CTX_STATE_OFFSET 16
#define MD2_CTX_CHECKSUM_OFFSET 64
typedef unsigned char BYTE;             // 8-bit byte

/* 
typedef struct {
   BYTE data[16];
   BYTE state[48];
   BYTE checksum[16];
   int len;
} MD2_CTX;
*/

/*********************** FUNCTION DECLARATIONS **********************/
__global__ void md2_init(BYTE ctx[], int *ctx_len);
__global__ void md2_update(BYTE ctx[], int *ctx_len, const BYTE data[], size_t len);
__global__ void md2_final(BYTE ctx[], int *ctx_len, BYTE hash[]);   // size of hash must be MD2_BLOCK_SIZE

#endif   // MD2_H
