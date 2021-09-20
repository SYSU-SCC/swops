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

  int counts;
  int slice;

  int blk_M;
  int blk_N;
  int blk_K;

  int Mp;
  int Np;
  int Kp;

  int sli_M[6];
  int sli_N[6];
  int sli_K[6];
  int rem_M;
  int rem_Ms;
  int rem_Me;
  
  float* A;
  float* Ap;
  float* A_sli[6];
  float* Ap_sli[6];
  int copy_all_A;

  float* B;
  float* Bp;
  float* B_sli[6];
  float* Bp_sli[6];
  int copy_all_B;

  float* C;
  float* Cp;
  float* C_sli[6];
  float* Cp_sli[6];
  int sli_C;
  int copy_all_C;

  float* T;
  float* Tp;

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

#endif
