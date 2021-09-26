#ifndef _ARGS_H
#define _ARGS_H

typedef struct swptex_mmPara{
    float *A;
    float *B;
    float *C;
    int M;
    int N;
    int K;
} swptex_mmPara, *swptex_mmPara_t;

typedef struct sw_bmmPara{
    float *A;
    float *B;
    float *C;

    float *Ap;
    float *Bp;
    float *Cp;

    int M;
    int Ms;
    int Me;

    int N;
    int Ns;
    int Ne;

    int K;
    int Ks;
    int Ke;

    int transposeA;
    int transposeB;

    int blk_M;
    int blk_N;
    int blk_K;

    int counts;
} sw_bmmPara, *sw_bmmPara_t;

typedef struct sw_gemmPara
{
    int blk_M;
    int blk_N;
    int blk_K;

    int sli_M[6];
    int sli_N[6];
    int sli_K[6];

    float *A;
    float *Ap;
    float *A_sli[6];
    float *Ap_sli[6];

    float *B;
    float *Bp;
    float *B_sli[6];
    float *Bp_sli[6];

    float *C;
    float *Cp;
    float *C_sli[6];
    float *Cp_sli[6];

    int sli_C;

    int M;
    int Ms;
    int Me;

    int N;
    int Ns;
    int Ne;

    int K;
    int Ks;
    int Ke;
} sw_gemmPara, *sw_gemmPara_t;

typedef struct sw_addmPara{
    float* A;
    float* B;
    float* C;
    int MN_size;
} sw_addmPara, *sw_addmPara_t;

#endif
