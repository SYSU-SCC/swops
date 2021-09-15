#include <crts.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "args.h"
extern SLAVE_FUN(sw_slave_mm_AB)(swptex_mmPara_t);
extern SLAVE_FUN(sw_slave_mm_ATB)(swptex_mmPara_t);
extern SLAVE_FUN(sw_slave_mm_ABT)(swptex_mmPara_t);

extern SLAVE_FUN(sw_slave_gemm_rrr_f32)(sw_gemmPara_t);

extern SLAVE_FUN(sw_slave_gemm_rrr_all_cgn_f32)(sw_gemmPara_t);
extern SLAVE_FUN(sw_slave_gemm_rcr_all_cgn_f32)(sw_gemmPara_t);
extern SLAVE_FUN(sw_slave_gemm_trans_f32)(sw_gemmPara_t);
extern SLAVE_FUN(sw_slave_padding_only_f32)(sw_gemmPara_t);
extern SLAVE_FUN(sw_slave_bmm_rrr)(sw_bmmPara_t);
extern SLAVE_FUN(swapNBHW_f)(swTransPara_t);

extern void *para_cross; // param on cross seg

int swptex_mm(const void *A, const void *B, void *C, size_t M, size_t N,
              size_t K, int transposeA, int transposeB)
{
    swptex_mmPara para;
    para.A = A;
    para.B = B;
    para.C = C;
    para.M = M;
    para.N = N;
    para.K = K;

    para_cross = &para; // cross seg variable to pass param

    int ret = athread_init_cgs();
    if (!transposeA && transposeB)
    {
        ret = athread_spawn_cgs(sw_slave_mm_ABT, &para);
    }
    else if (transposeA && !transposeB)
    {
        ret = athread_spawn_cgs(sw_slave_mm_ATB, &para);
    }
    else if (!transposeA && !transposeB)
    {
        ret = athread_spawn_cgs(sw_slave_mm_AB, &para);
    }
    else
    {
        printf("not supported\n");
        return 0;
    }
    athread_join_cgs();
}

void check_A_B_f32(const float* A, const float* Ap,
                   const float* B, const float* Bp,
                   const int M, const int Ms, const int Me,
                   const int K, const int Ks, const int Ke){
    for(int m = 0; m < M; m++){
        for(int k = 0; k < K; k++){
            if(A[m * K + k] != B[m * K + k]){
                printf("check A B error m %d k %d A %f B %f",m,k,A[m * K + k],B[m * K + k]);
                return;
            }
        }
    }
    for(int m = 0; m < Me; m++){
        for(int k = 0; k < Ke; k++){
            if(Ap[m * Ke + k] != Bp[m * Ke + k]){
                printf("check A B error m %d k %d Ap %f Bp %f",m,k,A[m * Ke + k],B[m * Ke + k]);
                return;
            }
        }
    }
    printf("checking A B passed\n");
}

void check_copy_border_f32(const float* A, const float* Ap,
                           const int M, const int Ms, const int Me, const int blk_M,
                           const int K, const int Ks, const int Ke, const int blk_K){
    //check Ap zero P
    for(int m = 0; m < Ms; m++){
        for(int k = 0; k < Ke; k++){
            if((k < Ks || k > K) && Ap[m * Ke + k]!= 0){
                printf("Ap zero error! at %d %d value: %f\n", m, k, Ap[m * Ke + k]);
                return;
            }
        }
    }
    //check Ap value P
    for(int m = 0; m < Ms; m++){
        for(int k = Ks; k < K; k++){
            if(Ap[m * Ke + k] != A[m * K + k]){
                printf("Ap value error! at %d %d A value: %f Ap value %f\n", m, k, A[m * K + k], Ap[m * Ke + k]);
                return;
            }
        }
    }
    //check Ap zero Q
    for(int m = Ms; m < Me; m++){
        for(int k = K; k < Ke; k++){
            if(Ap[m * Ke + k]!= 0){
                printf("Ap zero error! at %d %d value: %f\n", m, k, Ap[m * Ke + k]);
                return;
            }
        }
    }
    for(int m = M; m < Me; m++){
        for(int k = 0; k < Ke; k++){
            if(Ap[m * Ke + k]!= 0){
                printf("Ap zero error! at %d %d value: %f\n", m, k, Ap[m * Ke + k]);
                return;
            }
        }
    }
    //check Ap value Q
    for(int m = Ms; m < M; m++){
        for(int k = 0; k < K; k++){
            if(Ap[m * Ke + k] != A[m * K + k]){
                printf("Ap value error! at %d %d A value: %f\n", m, k, A[m * K + k], Ap[m * Ke + k]);
                return;
            }
        }
    }
    printf("checking copy border passed\n");
}

void check_C_all_f32(float* C, float* check_C, 
                        const int M,
                        const int N){
    printf("checking C all\n");
    for(int m = 0; m < M; m++){
        for(int n = 0; n < N; n++){
            if(fabs(C[m * N + n] - check_C[m * N + n]) > 1e-5){
                printf("checking C all error m %d n %d C %f check_C %f\n", m, n, C[m * N + n], check_C[m * N + n]);
                return ;
            }
        }
    }
    printf("checking C all passed\n");
}


void set_dma_check(float* A, float* B, int M, int N, int K, int blk_M, int blk_N, int blk_K){
    const int num_M = (M + blk_M - 1) / blk_M;
    const int num_N = (N + blk_N - 1) / blk_N;
    const int num_K = (K + blk_K - 1) / blk_K;
    for(int c_M = 0; c_M < num_M; c_M++){
        for(int c_K = 0; c_K < num_K; c_K++){
            
        }
    }
}

void test_gemm_rrr(){
    struct timeval tv1, tv2;
    int M = 1777;
    int N = 1777;
    int K = 1777;
    int blk_M = 512;
    int blk_N = 512;
    int blk_K = 512;
    int bn = 1;// six gemm
    float *A = malloc(sizeof(float) * bn * M * K);
    float *B = malloc(sizeof(float) * bn * K * N);
    float *C = malloc(sizeof(float) * bn * M * N);
    float *check_C = malloc(sizeof(float) * bn * M * N);
    for (int i = 0; i < bn * M * K; i++){
        A[i] = rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < bn * K * N; i++){
        B[i] = rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < bn * M * N; i++){
        C[i] = 0;
    }
    for (int i = 0; i < bn * M * N; i++){
        check_C[i] = 0;
    }

    int Ms = (M / blk_M) * blk_M;
    int Ns = (N / blk_N) * blk_N;
    int Ks = (K / blk_K) * blk_K;
    int Me = M % blk_M != 0 ? Ms + blk_M : Ms;
    int Ne = N % blk_N != 0 ? Ns + blk_N : Ns;
    int Ke = K % blk_K != 0 ? Ks + blk_K : Ks;

    printf("M %d Ms %d Me %d blk_M %d\nN %d Ns %d Ne %d blk_N %d\nK %d Ks %d Ke %d blk_K %d\n",
            M,Ms,Me,blk_M,N,Ns,Ne,blk_N,K,Ks,Ke,blk_K);
    printf("GEMM size: M %d N %d K %d\n", M, N, K);
    printf("Testing GEMM RRR F32 triple buffer asm no\n");
    gettimeofday(&tv1, NULL);

    float* Ap = malloc(sizeof(float) * bn * Me * Ke);
    //memset(Ap,0,sizeof(float) * bn * Me * Ke);//initialized on many-cores
    float* Bp = malloc(sizeof(float) * bn * Ke * Ne);
    //memset(Bp,0,sizeof(float) * bn * Ke * Ne);//initialized on many-cores
    float* Cp = malloc(sizeof(float) * bn * Me * Ne);
    //memset(Cp,0,sizeof(float) * bn * Me * Ne);//initialized on many-cores

    sw_gemmPara para;
    para.counts = bn;
    para.blk_M = blk_M;
    para.blk_N = blk_N;
    para.blk_K = blk_K;
    para.A = A;
    para.Ap = Ap;
    para.B = B;
    para.Bp = Bp;
    para.C = C;
    para.Cp = Cp;
    para.M = M;
    para.Ms = Ms;
    para.Me = Me;
    para.N = N;
    para.Ns = Ns;
    para.Ne = Ne;
    para.K = K;
    para.Ks = Ks;
    para.Ke = Ke;
    para_cross = &para;

    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_slave_gemm_rrr_f32, &para);
    athread_join_cgs();

    gettimeofday(&tv2, NULL);

    double optimized_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("Result of GEMM RRR F32 triple buffer asm no: %lf\n", optimized_seconds);

    //check_copy_border_f32(A, Ap, M, Ms, Me, blk_M, K, Ks, Ke, blk_K);
    //check_copy_border_f32(B, Bp, K, Ks, Ke, blk_K, N, Ns, Ne, blk_N);
    //check_A_B_f32(A, Ap, B, Bp, M, Ms, Me, K, Ks, Ke);
    printf("Testing GEMM RRR F32 hardware cache \n");
    gettimeofday(&tv1, NULL);
    swptex_mm(A,B,check_C,M,N,K,0,0);
    gettimeofday(&tv2, NULL);
    double origin_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("Result of GEMM RRR F32 hardware cache: %lf\n", origin_seconds);

    //check_copy_border_f32(check_C, Cp, M, Ms, Me, blk_M, N, Ns, Ne, blk_N);
    check_C_all_f32(C, check_C, M, N);

    free(A);
    free(Ap);
    free(B);
    free(Bp);
    free(C);
    free(Cp);
    free(check_C);
}

void test_gemm_rrr_all_cgn(){
    struct timeval tv1, tv2;
    int M = 12288;
    int N = 768;
    int K = 64;
    int blk_M = 512;
    int blk_N = 768;
    int blk_K = 64;
    int bn = 1;// six gemm
    float *A = malloc(sizeof(float) * bn * M * K);
    float *B = malloc(sizeof(float) * bn * K * N);
    float *C = malloc(sizeof(float) * bn * M * N);
    float *check_C = malloc(sizeof(float) * bn * M * N);
    for (int i = 0; i < bn * M * K; i++){
        A[i] = rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < bn * K * N; i++){
        B[i] = rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < bn * M * N; i++){
        C[i] = 0;
    }
    for (int i = 0; i < bn * M * N; i++){
        check_C[i] = 0;
    }
    const int M_tot = M;
    const int Ms_tot = (M_tot / blk_M) * blk_M;
    const int Me_tot = M_tot % blk_M != 0 ? Ms_tot + blk_M : Ms_tot;



    M = (M + 6 - 1) / 6;
    const int Ms = (M / blk_M) * blk_M;
    const int Ns = (N / blk_N) * blk_N;
    const int Ks = (K / blk_K) * blk_K;
    const int Me = M % blk_M != 0 ? Ms + blk_M : Ms;
    const int Ne = N % blk_N != 0 ? Ns + blk_N : Ns;
    const int Ke = K % blk_K != 0 ? Ks + blk_K : Ks;
    const int slice = 6;

    /* printf("M %d Ms %d Me %d blk_M %d\nN %d Ns %d Ne %d blk_N %d\nK %d Ks %d Ke %d blk_K %d\n",
            M,Ms,Me,blk_M,N,Ns,Ne,blk_N,K,Ks,Ke,blk_K); */
    printf("GEMM size: M %d N %d K %d\n", M_tot, N, K);
    printf("Testing GEMM RRR F32 triple buffer asm no\n");
    gettimeofday(&tv1, NULL);

    float* Ap = malloc(sizeof(float) * slice * Me * Ke);
    float* Bp = malloc(sizeof(float) * Ke * Ne);
    float* Cp = malloc(sizeof(float) * slice * Me * Ne);
    //memset(Ap,0,sizeof(float) * slice * Me * Ke);//initialized on many-cores
    //memset(Bp,0,sizeof(float) * Ke * Ne);//initialized on many-cores
    //memset(Cp,0,sizeof(float) * slice * Me * Ne);//initialized on many-cores

    sw_gemmPara para;
    para.counts = bn;
    para.slice = slice;
    para.blk_M = blk_M;
    para.blk_N = blk_N;
    para.blk_K = blk_K;
    para.A = A;
    para.Ap = Ap;
    para.B = B;
    para.Bp = Bp;
    para.C = C;
    para.Cp = Cp;
    para.M = M;
    para.Ms = Ms;
    para.Me = Me;
    para.N = N;
    para.Ns = Ns;
    para.Ne = Ne;
    para.K = K;
    para.Ks = Ks;
    para.Ke = Ke;
    para_cross = &para;

    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_slave_gemm_rrr_all_cgn_f32, &para);
    athread_join_cgs();

    gettimeofday(&tv2, NULL);

    double optimized_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("Result of GEMM RRR F32 triple buffer asm no: %lf\n", optimized_seconds);

    //check_copy_border_f32(A, Ap, M, Ms, Me, blk_M, K, Ks, Ke, blk_K);
    //check_copy_border_f32(B, Bp, K, Ks, Ke, blk_K, N, Ns, Ne, blk_N);
    //check_A_B_f32(A, Ap, B, Bp, M, Ms, Me, K, Ks, Ke);
    printf("Testing GEMM RRR F32 hardware cache \n");
    gettimeofday(&tv1, NULL);
    swptex_mm(A ,B ,check_C ,M_tot ,N ,K ,0,0);
    gettimeofday(&tv2, NULL);
    double origin_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("Result of GEMM RRR F32 hardware cache: %lf\n", origin_seconds);

    //check_copy_border_f32(check_C, Cp, M, Ms, Me, blk_M, N, Ns, Ne, blk_N);
    check_C_all_f32(C, check_C, M_tot, N);

    free(A);
    free(Ap);
    free(B);
    free(Bp);
    free(C);
    free(Cp);
    free(check_C);

}

void gemm_rcr_all_cgn(float* A, float* B, float* C,int M,int N,int K){

    int blk_M = 512;
    int blk_N = 512;
    int blk_K = 512;

    int local_A_LDM_size = blk_M * blk_K / 64;
    int local_B_LDM_size = blk_K * blk_N / 64;
    int local_C_LDM_size = blk_M * blk_N / 64;

    int Total_LDM_size = 3 * sizeof(float) * (local_A_LDM_size + local_B_LDM_size + local_C_LDM_size);
    if(Total_LDM_size > 210 * 1024){
        printf("gemm_rcr_all_cgn Total_LDM_size > 210 * 1024 \n");
        return;
    }

    const int M_tot = M;
    const int Ms_tot = (M_tot / blk_M) * blk_M;
    const int Me_tot = M_tot % blk_M != 0 ? Ms_tot + blk_M : Ms_tot;
    M = (M + 6 - 1) / 6;
    const int Ms = (M / blk_M) * blk_M;
    const int Ns = (N / blk_N) * blk_N;
    const int Ks = (K / blk_K) * blk_K;
    const int Me = M % blk_M != 0 ? Ms + blk_M : Ms;
    const int Ne = N % blk_N != 0 ? Ns + blk_N : Ns;
    const int Ke = K % blk_K != 0 ? Ks + blk_K : Ks;
    const int slice = 6;

    float* BT = malloc(sizeof(float) * K * N);
    float* BTp = malloc(sizeof(float) * Ke * Ne);

    float* Ap = malloc(sizeof(float) * slice * Me * Ke);
    float* Bp = malloc(sizeof(float) * Ke * Ne);
    float* Cp = malloc(sizeof(float) * slice * Me * Ne);

    sw_gemmPara para;
    para.counts = 1;
    para.slice = slice;
    para.blk_M = blk_M;
    para.blk_N = blk_N;
    para.blk_K = blk_K;
    para.A = A;
    para.Ap = Ap;
    para.B = B;
    para.Bp = Bp;
    para.C = C;
    para.Cp = Cp;
    para.T = BT;
    para.Tp = BTp;
    para.M = M;
    para.Ms = Ms;
    para.Me = Me;
    para.N = N;
    para.Ns = Ns;
    para.Ne = Ne;
    para.K = K;
    para.Ks = Ks;
    para.Ke = Ke;
    para_cross = &para;

    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_slave_gemm_rcr_all_cgn_f32, &para);
    athread_join_cgs();

    free(Ap);
    free(Bp);
    free(Cp);
    free(BT);
    free(BTp);
}

void test_gemm_real_rcr(){
    int M = 3333;
    int N = 1333;
    int K = 1333;
    float *A = malloc(sizeof(float) * M * K);
    float *B = malloc(sizeof(float) * K * N);
    float *C = malloc(sizeof(float) * M * N);
    float *check_C = malloc(sizeof(float) * M * N);
    for (int i = 0; i < M * K; i++){
        A[i] = rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < K * N; i++){
        B[i] = rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < M * N; i++){
        C[i] = 0;
    }
    for (int i = 0; i < M * N; i++){
        check_C[i] = 0;
    }
    gemm_rcr_all_cgn(A,B,C,M,N,K);
    swptex_mm(A ,B ,check_C ,M ,N ,K , 0, 1);
    check_C_all_f32(C, check_C, M, N);
    free(A);
    free(B);
    free(C);
    free(check_C);

}

void test_gemm_rcr_all_cgn(){
    struct timeval tv1, tv2;
    int M = 1333;
    int N = 1333;
    int K = 1333;
    int blk_M = 512;
    int blk_N = 512;
    int blk_K = 512;
    int bn = 1;// six gemm
    float *A = malloc(sizeof(float) * bn * M * K);
    float *B = malloc(sizeof(float) * bn * K * N);
    float *C = malloc(sizeof(float) * bn * M * N);
    float *check_C = malloc(sizeof(float) * bn * M * N);
    for (int i = 0; i < bn * M * K; i++){
        A[i] = rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < bn * K * N; i++){
        B[i] = rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < bn * M * N; i++){
        C[i] = 0;
    }
    for (int i = 0; i < bn * M * N; i++){
        check_C[i] = 0;
    }
    const int M_tot = M;
    const int Ms_tot = (M_tot / blk_M) * blk_M;
    const int Me_tot = M_tot % blk_M != 0 ? Ms_tot + blk_M : Ms_tot;

    M = (M + 6 - 1) / 6;
    const int Ms = (M / blk_M) * blk_M;
    const int Ns = (N / blk_N) * blk_N;
    const int Ks = (K / blk_K) * blk_K;
    const int Me = M % blk_M != 0 ? Ms + blk_M : Ms;
    const int Ne = N % blk_N != 0 ? Ns + blk_N : Ns;
    const int Ke = K % blk_K != 0 ? Ks + blk_K : Ks;
    const int slice = 6;

    /* printf("M %d Ms %d Me %d blk_M %d\nN %d Ns %d Ne %d blk_N %d\nK %d Ks %d Ke %d blk_K %d\n",
            M,Ms,Me,blk_M,N,Ns,Ne,blk_N,K,Ks,Ke,blk_K); */
    printf("GEMM size: M %d N %d K %d\n", M_tot, N, K);
    printf("Testing GEMM RCR F32 triple buffer asm no\n");
    gettimeofday(&tv1, NULL);

    float* BT = malloc(sizeof(float) * bn * K * N);
    for(int n = 0; n < N; n++){
        for(int k = 0; k < K; k++){
            BT[k * N + n] = B[n * K + k];
        }
    }

    float* Ap = malloc(sizeof(float) * slice * Me * Ke);
    float* Bp = malloc(sizeof(float) * Ke * Ne);
    float* BTp = malloc(sizeof(float) * Ke * Ne);
    float* Cp = malloc(sizeof(float) * slice * Me * Ne);
    //memset(Ap,0,sizeof(float) * slice * Me * Ke);//initialized on many-cores
    //memset(Bp,0,sizeof(float) * Ke * Ne);//initialized on many-cores
    //memset(Cp,0,sizeof(float) * slice * Me * Ne);//initialized on many-cores

    sw_gemmPara para;
    para.counts = bn;
    para.slice = slice;
    para.blk_M = blk_M;
    para.blk_N = blk_N;
    para.blk_K = blk_K;
    para.A = A;
    para.Ap = Ap;
    para.B = B;
    para.Bp = Bp;
    para.C = C;
    para.Cp = Cp;
    para.T = BT;
    para.Tp = BTp;
    para.M = M;
    para.Ms = Ms;
    para.Me = Me;
    para.N = N;
    para.Ns = Ns;
    para.Ne = Ne;
    para.K = K;
    para.Ks = Ks;
    para.Ke = Ke;
    para_cross = &para;

    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_slave_gemm_rcr_all_cgn_f32, &para);
    athread_join_cgs();

    gettimeofday(&tv2, NULL);

    double optimized_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("Result of GEMM RCR F32 triple buffer asm no: %lf\n", optimized_seconds);

    //check_copy_border_f32(A, Ap, M, Ms, Me, blk_M, K, Ks, Ke, blk_K);
    //check_copy_border_f32(B, Bp, K, Ks, Ke, blk_K, N, Ns, Ne, blk_N);
    //check_A_B_f32(A, Ap, B, Bp, M, Ms, Me, K, Ks, Ke);
    printf("Testing GEMM RCR F32 hardware cache \n");
    gettimeofday(&tv1, NULL);
    swptex_mm(A ,B ,check_C ,M_tot ,N ,K , 0, 1);
    gettimeofday(&tv2, NULL);
    double origin_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("Result of GEMM RCR F32 hardware cache: %lf\n", origin_seconds);

    //check_copy_border_f32(check_C, Cp, M, Ms, Me, blk_M, N, Ns, Ne, blk_N);
    check_C_all_f32(C, check_C, M_tot, N);

    free(A);
    free(Ap);
    free(B);
    free(Bp);
    free(C);
    free(Cp);
    free(BT);
    free(BTp);
    free(check_C);

}

void test_trans(){
    printf("testing trans\n");
    int M = 1777;
    int N = 1777;
    int K = 1777;
    int blk_M = 512;
    int blk_N = 512;
    int blk_K = 512;
    int bn = 1;// six gemm
    float *C = malloc(sizeof(float) * bn * M * N);
    float *check_C = malloc(sizeof(float) * bn * M * N);
    float *T = malloc(sizeof(float) * bn * M * N);

    for(int m = 0; m < M; m++){
        for(int n = 0; n < N; n++){
            check_C[n * M + m] = C[m * N + n] = rand()*1.0/RAND_MAX;
        }
    }

    int Ms = (M / blk_M) * blk_M;
    int Ns = (N / blk_N) * blk_N;
    int Ks = (K / blk_K) * blk_K;
    int Me = M % blk_M != 0 ? Ms + blk_M : Ms;
    int Ne = N % blk_N != 0 ? Ns + blk_N : Ns;
    int Ke = K % blk_K != 0 ? Ks + blk_K : Ks;

    float* Cp = malloc(sizeof(float) * bn * Me * Ne);
    float* Tp = malloc(sizeof(float) * bn * Me * Ne);

    sw_gemmPara para;
    para.counts = bn;
    para.blk_M = blk_M;
    para.blk_N = blk_N;
    para.blk_K = blk_K;
    para.C = C;
    para.Cp = Cp;
    para.M = M;
    para.Ms = Ms;
    para.Me = Me;
    para.N = N;
    para.Ns = Ns;
    para.Ne = Ne;
    para.K = K;
    para.Ks = Ks;
    para.Ke = Ke;
    para.T = T;
    para.Tp = Tp;
    para_cross = &para;

    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_slave_gemm_trans_f32, &para);
    athread_join_cgs();

    for(int m = 0; m < Ms; m++){
        for(int n = 0; n < Ns; n++){
            if(check_C[n * M + m] != T[n * M + m]){
                printf("trans error m %d n %d check_C %f T %f\n", m, n , check_C[n * M + m], T[n * M + m]);
                //return;
            }
        }
    }
    printf("checking trans done\n");

    free(C);
    free(Cp);
    free(T);
    free(Tp);
    free(check_C);
}

void test_gemm_rcr(){
    struct timeval tv1, tv2;
    int M = 2048;
    int N = 2048;
    int K = 2048;
    int blk_M = 512;
    int blk_N = 512;
    int blk_K = 512;
    int bn = 1;// six gemm
    float *A = malloc(sizeof(float) * bn * M * K);
    float *B = malloc(sizeof(float) * bn * K * N);
    float *C = malloc(sizeof(float) * bn * M * N);
    float *check_C = malloc(sizeof(float) * bn * M * N);
    for (int i = 0; i < bn * M * K; i++){
        A[i] = rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < bn * K * N; i++){
        B[i] = rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < bn * M * N; i++){
        C[i] = 0;
    }
    for (int i = 0; i < bn * M * N; i++){
        check_C[i] = 0;
    }

    int Ms = (M / blk_M) * blk_M;
    int Ns = (N / blk_N) * blk_N;
    int Ks = (K / blk_K) * blk_K;
    int Me = M % blk_M != 0 ? Ms + blk_M : Ms;
    int Ne = N % blk_N != 0 ? Ns + blk_N : Ns;
    int Ke = K % blk_K != 0 ? Ks + blk_K : Ks;

    printf("M %d Ms %d Me %d blk_M %d\nN %d Ns %d Ne %d blk_N %d\nK %d Ks %d Ke %d blk_K %d\n",
            M,Ms,Me,blk_M,N,Ns,Ne,blk_N,K,Ks,Ke,blk_K);
    printf("GEMM size: M %d N %d K %d\n", M, N, K);
    printf("Testing GEMM RRR F32 triple buffer asm no\n");
    gettimeofday(&tv1, NULL);

    float* Ap = malloc(sizeof(float) * bn * Me * Ke);
    memset(Ap,0,sizeof(float) * bn * Me * Ke);//initialized on many-cores
    float* Bp = malloc(sizeof(float) * bn * Ke * Ne);
    memset(Bp,0,sizeof(float) * bn * Ke * Ne);//initialized on many-cores
    float* Cp = malloc(sizeof(float) * bn * Me * Ne);
    memset(Cp,0,sizeof(float) * bn * Me * Ne);//initialized on many-cores

    sw_gemmPara para;
    para.counts = bn;
    para.blk_M = blk_M;
    para.blk_N = blk_N;
    para.blk_K = blk_K;
    para.A = A;
    para.Ap = Ap;
    para.B = B;
    para.Bp = Bp;
    para.C = C;
    para.Cp = Cp;
    para.M = M;
    para.Ms = Ms;
    para.Me = Me;
    para.N = N;
    para.Ns = Ns;
    para.Ne = Ne;
    para.K = K;
    para.Ks = Ks;
    para.Ke = Ke;
    para_cross = &para;

    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_slave_gemm_rrr_f32, &para);
    athread_join_cgs();

    gettimeofday(&tv2, NULL);

    double optimized_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("Result of GEMM RRR F32 triple buffer asm no: %lf\n", optimized_seconds);

    check_copy_border_f32(A, Ap, M, Ms, Me, blk_M, K, Ks, Ke, blk_K);
    check_copy_border_f32(B, Bp, K, Ks, Ke, blk_K, N, Ns, Ne, blk_N);
    //check_A_B_f32(A, Ap, B, Bp, M, Ms, Me, K, Ks, Ke);
    printf("Testing GEMM RRR F32 hardware cache \n");
    gettimeofday(&tv1, NULL);
    swptex_mm(A,B,check_C,M,N,K,0,0);
    gettimeofday(&tv2, NULL);
    double origin_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("Result of GEMM RRR F32 hardware cache: %lf\n", origin_seconds);

    check_copy_border_f32(check_C, Cp, M, Ms, Me, blk_M, N, Ns, Ne, blk_N);
    check_C_all_f32(C, check_C, M, N);

    free(A);
    free(Ap);
    free(B);
    free(Bp);
    free(C);
    free(Cp);
    free(check_C);
}

void test_bmm(void){
    printf("test batch mm\n");
    int M = 96;
    int N = 96;
    int K = 96;
    int blk_M = 96;
    int blk_N = 96;
    int blk_K = 96;
    int bn = 768;//384
    float *A = malloc(sizeof(float) * bn * M * K);
    float *B = malloc(sizeof(float) * bn * K * N);
    float *C = malloc(sizeof(float) * bn * M * N);
    float *check_C = malloc(sizeof(float) * bn * M * N);
    for (int i = 0; i < bn * M * K; i++){
        A[i] = rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < bn * K * N; i++){
        B[i] = rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < bn * M * N; i++){
        C[i] = 0;
    }
    for (int i = 0; i < bn * M * N; i++){
        check_C[i] = 0;
    }
    struct timeval tv1, tv2;

    gettimeofday(&tv1, NULL);
    swptex_bmm(A,B,check_C,bn,M,N,K,0,0);
    gettimeofday(&tv2, NULL);

    double origin_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;

    gettimeofday(&tv1, NULL);
    sw_bmmPara para;
    para.A = A;
    para.B = B;
    para.C = C;
    para.M = M;
    para.N = N;
    para.K = K;
    para.blk_M = blk_M;
    para.blk_N = blk_N;
    para.blk_K = blk_K;
    para.counts = bn;
    para_cross = &para;
    int LDM_size_Ax2 = 2 * para.blk_M * para.blk_K *sizeof(float);
    int LDM_size_Bx2 = 2 * para.blk_K * para.blk_N *sizeof(float);
    int LDM_size_Cx2 = 2 * para.blk_M * para.blk_N *sizeof(float);
    if(LDM_size_Ax2 + LDM_size_Bx2 + LDM_size_Cx2 >= 256 * 1024){
        printf("ldm_malloc error, size of blk: %d, max size: 262144\n",LDM_size_Ax2 + LDM_size_Bx2 + LDM_size_Cx2);
        return;
    }
    else{
        printf("ldm_malloc size of blk: %d, max size: 262144\n",LDM_size_Ax2 + LDM_size_Bx2 + LDM_size_Cx2);
        printf("M %d N %d K %d blk_M %d blk_N %d blk_K %d batch: %d\n", M, N, K, blk_M, blk_N, blk_K, bn);
        int ret = athread_init_cgs();
        ret = athread_spawn_cgs(sw_slave_bmm_rrr, &para);
        athread_join_cgs();
        gettimeofday(&tv2, NULL);

        double optimized_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
        //check results
        for(int i = 0; i < bn * M * N; i++){
            if(fabs(check_C[i] - C[i])>0.0001){
                printf("error at %d check_C: %f C %f\n", i, check_C[i], C[i]);
                break;
            }
        }
        printf("bmm original: %lf\n", origin_seconds);
        printf("bmm optimized: %lf\n", optimized_seconds);
    }
    free(A);
    free(B);
    free(C);
    free(check_C);
}

int swptex_bmm(const void *A, const void *B, void *C, size_t batch, size_t M,
               size_t N, size_t K, int transposeA, int transposeB)
{
    int bn;
    for (bn = 0; bn < batch; ++bn)
    {
        swptex_mm((float *)A + bn * M * K, (float *)B + bn * N * K,
                  (float *)C + bn * M * N, M, N, K, transposeA, transposeB);
    }
}

int swptex_softmax(void *x_, size_t M, size_t N)
{
    // inplace
    float *x = (float *)x_;
    int i, j;
    float tmp, sum;
    for (i = 0; i < M; ++i)
    {
        tmp = x[i * N];
        for (j = 1; j < N; ++j)
        {
            if (x[i * N + j] > tmp)
            {
                tmp = x[i * N + j];
            }
        }
        sum = 0.f;
        for (j = 0; j < N; ++j)
        {
            x[i * N + j] = exp(x[i * N + j] - tmp);
            sum += x[i * N + j];
        }
        for (j = 0; j < N; ++j)
        {
            x[i * N + j] /= sum;
        }
    }
}

int swptex_dsoftmax(void *dy_, const void *y_, size_t M, size_t N)
{
    // inplace
    float *dy = (float *)dy_;
    float *y = (float *)y_;
    int i, j;
    float tmp;
    for (i = 0; i < M; ++i)
    {
        tmp = 0.f;
        for (j = 0; j < N; ++j)
        {
            tmp += dy[i * N + j] * y[i * N + j];
        }
        for (j = 0; j < N; ++j)
        {
            dy[i * N + j] = (dy[i * N + j] - tmp) * y[i * N + j];
        }
    }
}

int swptex_split_and_transpose(void *QKV_, void *Q_, void *K_, void *V_,
                               size_t B, size_t N, size_t S, size_t D)
{
    float *QKV = (float *)QKV_;
    float *Q = (float *)Q_;
    float *K = (float *)K_;
    float *V = (float *)V_;
    int b, n, s;
    for (b = 0; b < B; ++b)
    {
        for (n = 0; n < N; ++n)
        {
            for (s = 0; s < S; ++s)
            {
                memcpy(Q + b * N * S * D + n * S * D + s * D,
                       QKV + n * D + s * N * D * 3 + b * S * N * D * 3,
                       D * sizeof(float));
                memcpy(K + b * N * S * D + n * S * D + s * D,
                       QKV + N * D + n * D + s * N * D * 3 + b * S * N * D * 3,
                       D * sizeof(float));
                memcpy(V + b * N * S * D + n * S * D + s * D,
                       QKV + N * D * 2 + n * D + s * N * D * 3 + b * S * N * D * 3,
                       D * sizeof(float));
            }
        }
    }
}

int swptex_transpose_and_merge(void *QKV_, void *Q_, void *K_, void *V_,
                               size_t B, size_t N, size_t S, size_t D)
{
    float *QKV = (float *)QKV_;
    float *Q = (float *)Q_;
    float *K = (float *)K_;
    float *V = (float *)V_;
    int b, n, s;
    for (b = 0; b < B; ++b)
    {
        for (n = 0; n < N; ++n)
        {
            for (s = 0; s < S; ++s)
            {
                memcpy(QKV + n * D + s * N * D * 3 + b * S * N * D * 3,
                       Q + b * N * S * D + n * S * D + s * D, D * sizeof(float));
                memcpy(QKV + N * D + n * D + s * N * D * 3 + b * S * N * D * 3,
                       K + b * N * S * D + n * S * D + s * D, D * sizeof(float));
                memcpy(QKV + N * D * 2 + n * D + s * N * D * 3 + b * S * N * D * 3,
                       V + b * N * S * D + n * S * D + s * D, D * sizeof(float));
            }
        }
    }
}

int swptex_split(const void *QKV_, void *QKVT_, size_t B, size_t N, size_t S,
                 size_t D)
{
    float *QKV = (float *)QKV_;
    float *QKVT = (float *)QKVT_;
    int b, n, s;
    for (b = 0; b < B; ++b)
    {
        for (n = 0; n < N; ++n)
        {
            for (s = 0; s < S; ++s)
            {
                memcpy(QKVT + b * N * S * D + n * S * D + s * D,
                       QKV + n * D + s * N * D + b * S * N * D, D * sizeof(float));
            }
        }
    }
}

int swptex_merge(const void *QKV_, void *QKVT_, size_t B, size_t N, size_t S,
                 size_t D)
{
    float *QKV = (float *)QKV_;
    float *QKVT = (float *)QKVT_;
    int b, n, s;
    for (b = 0; b < B; ++b)
    {
        for (n = 0; n < N; ++n)
        {
            for (s = 0; s < S; ++s)
            {
                memcpy(QKVT + n * D + s * N * D + b * S * N * D,
                       QKV + b * N * S * D + n * S * D + s * D, D * sizeof(float));
            }
        }
    }
}

int swptex_scale(void *x_, size_t len, float scaling)
{
    float *x = (float *)x_;
    int i;
    for (i = 0; i < len; ++i)
    {
        x[i] *= scaling;
    }
}
