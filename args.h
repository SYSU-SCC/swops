#ifndef _ARGS_H
#define _ARGS_H

typedef struct swptex_mmPara {
  float* A;
  float* B;
  float* C;
  size_t M;
  size_t N;
  size_t K;
} swptex_mmPara, *swptex_mmPara_t;

typedef struct sw_bmmPara {
  float* A;
  float* B;
  float* C;
  size_t M;
  size_t N;
  size_t K;
  size_t blkM;
  size_t blkN;
  size_t blkK;
  size_t counts;
} sw_bmmPara, *sw_bmmPara_t;

#endif
