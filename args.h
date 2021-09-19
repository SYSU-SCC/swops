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

  size_t blk_M;
  size_t blk_N;
  size_t blk_K;
  
  size_t counts;
} sw_bmmPara, *sw_bmmPara_t;

typedef struct sw_gemmPara {
  enum CBLAS_ORDER
    {
	CblasRowMajor = 101,
	CblasColMajor = 102
    }Atype, Btype, Ctype;

  size_t counts;
  size_t slice;

  size_t blk_M;
  size_t blk_N;
  size_t blk_K;

  size_t Mp;
  size_t Np;
  size_t Kp;

  size_t sli_M[6];
  size_t sli_N[6];
  size_t sli_K[6];
  size_t rem_M;
  size_t rem_Ms;
  size_t rem_Me;
  
  float* A;
  float* Ap;
  float* A_sli[6];
  float* Ap_sli[6];
  size_t copy_all_A;

  float* B;
  float* Bp;
  float* B_sli[6];
  float* Bp_sli[6];
  size_t copy_all_B;

  float* C;
  float* Cp;
  float* C_sli[6];
  float* Cp_sli[6];
  size_t copy_all_C;

  float* T;
  float* Tp;

  size_t M;
  size_t Ms;
  size_t Me;

  size_t N;
  size_t Ns;
  size_t Ne;

  size_t K;
  size_t Ks;
  size_t Ke;
} sw_gemmPara, *sw_gemmPara_t;

#endif
