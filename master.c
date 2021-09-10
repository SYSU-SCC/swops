#include <crts.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "args.h"
extern SLAVE_FUN(sw_slave_gemm_rrr)(sw_bmmPara_t);
extern SLAVE_FUN(sw_slave_gemm_rcr)(sw_bmmPara_t);
extern SLAVE_FUN(sw_slave_gemm_rcr_f32)(sw_bmmPara_t);
extern SLAVE_FUN(sw_slave_bmm_rrr)(sw_bmmPara_t);
extern SLAVE_FUN(sw_slave_mm_AB)(swptex_mmPara_t);
extern SLAVE_FUN(sw_slave_mm_ATB)(swptex_mmPara_t);
extern SLAVE_FUN(sw_slave_mm_ABT)(swptex_mmPara_t);

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

void check_copy_border_f32(const float* A, const float* Ap, const float* Aq,
                           const int M, const int Ms, const int Me, const int blk_M,
                           const int K, const int Ks, const int Ke, const int blk_K){
    //check Ap using 0
    printf("checking copy border \n");
    for(int m = 0; m < Ms; m++){
        for(int k = 0; k < blk_K; k++){
            if(k >= K - Ks && Ap[m * blk_K + k] != 0){
                printf("Ap check fail m %d k %d value: %f\n", m, k, Ap[m * blk_K + k]);
                printf("checking copy border error\n");
                return;
            }
        }
    }
    for(int m = 0; m < M - Ms; m++){
        for(int k = 0; k < Ke; k++){
            if(k >= K && Aq[m * Ke + k] != 0){
                printf("Aq check fail m %d k %d value: %f\n", m, k, Aq[m * Ke +k]);
                printf("checking copy border error\n");
                return;
            }
        }
    }
    for(int m = M - Ms; m < Me - Ms; m++){
        for(int k = 0; k < Ke; k++){
            if(Aq[m * Ke + k] != 0){
                printf("Aq check at master fail m %d k %d value: %f\n", m, k, Aq[m * Ke +k]);
                printf("checking copy border error\n");
                return;
            }
        }
    }
    //check Ap using general value
    for(int m = 0; m < Ms; m++){
        for(int k = Ks; k < K; k++){
            if(A[m * K + k] != Ap[m * blk_K + k - Ks]){
                printf("check Ap general error at m %d k %d A: %f Ap: %f\n", m, k, A[m * K + k], Ap[m * blk_K + k - Ks]);
                printf("checking copy border error\n");
                return;
            }
        }
    }
    //check Aq using general value
    for(int m = Ms; m < M; m++){
        for(int k = 0; k < K; k++){
            if(A[m * K + k] != Aq[(m - Ms) * Ke + k]){
                printf("check Aq general error at m %d k %d A: %f Aq: %f\n", m, k, A[m * K + k], Aq[(m - Ms) * Ke + k]);
                printf("checking copy border error\n");
                return;
            }
        }
    }
    printf("checking copy border passed\n");
}

void test_gemm_rcr(){
    printf("test gemm\n");
    int M = 4333;
    int N = 600;
    int K = 600;
    int blk_M = 512;
    int blk_N = 512;
    int blk_K = 512;
    int bn = 6;// six gemm
    float *A = malloc(sizeof(float) * bn * M * K);
    float *B = malloc(sizeof(float) * bn * K * N);
    float *C = malloc(sizeof(float) * bn * M * N);
    float *check_C = malloc(sizeof(float) * bn * M * N);
    for (int i = 0; i < bn * M * K; i++){
        A[i] = rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < bn * K * N; i++){
        B[i] = 1.0;
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

    printf("blk_M %d blk_N %d blk_K %d\nM %d Ms %d Me %d\nN %d Ns %d Ne %d\nK %d Ks %d Ke %d\nbatch: %d\n", 
            blk_M, blk_N, blk_K, M, Ms, Me, N, Ns, Ne, K, Ks, Ke, bn);
    
    float* Ap = malloc(sizeof(float) * bn * Ms * blk_K);
    float* Aq = malloc(sizeof(float) * bn * blk_M * Ke);
    float* Bp = malloc(sizeof(float) * bn * Ns * blk_K);
    float* Bq = malloc(sizeof(float) * bn * blk_N * Ke);
    float* Cp = malloc(sizeof(float) * bn * Ms * blk_N);
    float* Cq = malloc(sizeof(float) * bn * blk_M * Ne);

    sw_gemmPara para;
    para.counts = bn;
    para.blk_M = blk_M;
    para.blk_N = blk_N;
    para.blk_K = blk_K;
    para.A = A;
    para.Ap = Ap;
    para.Aq = Aq;
    para.B = B;
    para.Bp = Bp;
    para.Bq = Bq;
    para.C = C;
    para.Cp = Cp;
    para.Cq = Cq;
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
    ret = athread_spawn_cgs(sw_slave_gemm_rcr_f32, &para);
    athread_join_cgs();

    check_copy_border_f32(A, Ap, Aq, M, Ms, Me, blk_M, K, Ks, Ke, blk_K);

    check_copy_border_f32(B, Bp, Bq, N, Ns, Ne, blk_N, K, Ks, Ke, blk_K);

    check_copy_border_f32(C, Cp, Cq, M, Ms, Me, blk_M, N, Ns, Ne, blk_N);

    if(((blk_M * blk_K) % 64) != 0 || ((blk_K * blk_N) % 64) != 0 || ((blk_M * blk_N) % 64) != 0){
        printf("error: blk_M %d blk_N %d blk_K %d\n", blk_M, blk_N, blk_K);
    }
    else{
        printf("M %d N %d K %d blk_M %d blk_N %d blk_K %d batch: %d\n", M, N, K, blk_M, blk_N, blk_K, bn);
        ret = athread_init_cgs();
        ret = athread_spawn_cgs(sw_slave_gemm_rcr, &para);
        athread_join_cgs();
    }
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    //swptex_bmm(A,B,check_C,bn,M,N,K,0,0);
    gettimeofday(&tv2, NULL);
    double origin_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("gemm original: %lf\n", origin_seconds);
    free(A);
    free(Ap);
    free(Aq);
    free(B);
    free(Bp);
    free(Bq);
    free(C);
    free(Cp);
    free(Cq);
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
