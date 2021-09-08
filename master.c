#include <crts.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "args.h"
extern SLAVE_FUN(sysu_slave_rrr_bmm)(sw_bmmPara_t);
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

void sw_bmm()
{
}

void test_bmm(void)
{
    printf("test batch mm\n");
    int M = 96;
    int N = 96;
    int K = 96;
    int blkM = 64;
    int blkN = 64;
    int blkK = 64;
    int bn = 1024;//384
    float *A = malloc(sizeof(float) * bn * M * K);
    float *B = malloc(sizeof(float) * bn * K * N);
    float *C = malloc(sizeof(float) * bn * M * N);
    float *check_C = malloc(sizeof(float) * bn * M * N);
    for (int i = 0; i < bn * M * K; i++)
    {
        A[i] = rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < bn * K * N; i++)
    {
        B[i] = rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < bn * M * N; i++)
    {
        C[i] = 0;
    }
    for (int i = 0; i < bn * M * N; i++)
    {
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
    para.blkM = blkM;
    para.blkN = blkN;
    para.blkK = blkK;
    para.counts = bn;
    para_cross = &para;
    int LDM_size_Ax2 = 2 * para.blkM * para.blkK *sizeof(float);
    int LDM_size_Bx2 = 2 * para.blkK * para.blkN *sizeof(float);
    int LDM_size_Cx2 = 2 * para.blkM * para.blkN *sizeof(float);
    if(LDM_size_Ax2 + LDM_size_Bx2 + LDM_size_Cx2 >= 256 * 1024){
        printf("ldm_malloc error, size of blk: %d, max size: 262144\n",LDM_size_Ax2 + LDM_size_Bx2 + LDM_size_Cx2);
        return;
    }
    else{
        printf("ldm_malloc size of blk: %d\n",LDM_size_Ax2 + LDM_size_Bx2 + LDM_size_Cx2);
        printf("batch: %d\n",bn);
    }
    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sysu_slave_rrr_bmm, &para);
    athread_join_cgs();
    gettimeofday(&tv2, NULL);

    double optimized_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    //check results
    for(int i = 0; i < bn * M * N; i++){
        if(fabs(check_C[i] - C[i])>0.001){
            printf("error at %d check_C: %f C %f\n", i, check_C[i], C[i]);
            break;
        }
    }
    printf("bmm original: %lf\n", origin_seconds);
    printf("bmm optimized: %lf\n", optimized_seconds);
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
