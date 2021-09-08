#include <crts.h>
#include <simd.h>
#include <slave.h>
#include <stdio.h>
#include <stdlib.h>

#include "args.h"

__cross void *para_cross = NULL;

#define thread_num (64 * 6)

void sysu_slave_rrr_bmm(sw_bmmPara *_){
    sw_bmmPara *para = (sw_bmmPara *)para_cross;
    const float *src_A = para->A;
    const float *src_B = para->B;
    const float *src_C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    size_t blk_M = para->blkM;
    size_t blk_N = para->blkN;
    size_t blk_K = para->blkK;
    size_t counts = para->counts;
    const int local_count = (counts + 383) / 384;
    const int local_start = _MYID * local_count;
    const int local_end = ((local_start + local_count > counts) ? counts : (local_start + local_count));
    if (local_start >= counts){
        return;
    }
    const size_t local_A_size = blk_M * blk_K;
    const size_t local_B_size = blk_K * blk_N;
    const size_t local_C_size = blk_M * blk_N;
    const size_t MK_size = M * K;
    const size_t KN_size = K * N;
    const size_t MN_size = M * N;
    const size_t num_M = (M + blk_M - 1) / blk_M;
    const size_t num_N = (N + blk_N - 1) / blk_N; //这一定能够被整除
    const size_t num_K = (K + blk_K - 1) / blk_K;
    const size_t rem_blk_M = num_M * blk_M - M == 0 ? blk_M : M - (num_M-1) * blk_M;
    const size_t rem_blk_N = num_N * blk_N - N == 0 ? blk_N : N - (num_N-1) * blk_N;
    const size_t rem_blk_K = num_K * blk_K - K == 0 ? blk_K : K - (num_K-1) * blk_K;
    size_t curr_blk_M = blk_M;
    size_t curr_blk_N = blk_N;
    size_t curr_blk_K = blk_K;
    size_t next_blk_M = blk_M;
    size_t next_blk_N = blk_N;
    size_t next_blk_K = blk_K;
    float *start_A = src_A + MK_size * local_start;
    float *start_B = src_B + KN_size * local_start;
    float *start_C = src_C + MN_size * local_start;
    float *local_A = (float *)ldm_malloc(sizeof(float) * blk_M * blk_K * 2);
    float *local_B = (float *)ldm_malloc(sizeof(float) * blk_K * blk_N * 2);
    float *local_C = (float *)ldm_malloc(sizeof(float) * blk_M * blk_N * 2);
    volatile int double_buffer_flag_AB = 0;
    volatile int double_buffer_flag_C = 0;
    volatile athread_rply_t reply_get_A = 0, reply_get_B = 0, reply_put_C = 1;//start 1
    int local_now = local_start;
    athread_dma_iget_stride(local_A + (1 - double_buffer_flag_AB) * local_A_size, 
                            start_A, 
                            sizeof(float) * blk_M * blk_K, 
                            sizeof(float) * blk_K, 
                            sizeof(float) * (K - blk_K),
                            &reply_get_A);
    athread_dma_iget_stride(local_B + (1 - double_buffer_flag_AB) * local_B_size, 
                            start_B, 
                            sizeof(float) * blk_K * blk_N, 
                            sizeof(float) * blk_N, 
                            sizeof(float) * (N - blk_N),
                            &reply_get_B);
    athread_dma_wait_value(&reply_get_A, 1);
    athread_dma_wait_value(&reply_get_B, 1);
    /* if(_MYID == 99){
                    for(int i = 0; i < curr_blk_K * curr_blk_N; i++){
                        if(local_B[(1 - double_buffer_flag_AB) * local_B_size + i]!=1.0){
                            printf("Slave %d compute curr_M %d curr_N %d curr_K %d error at %d local_B value %lf\n", 
                                    _MYID, 0, 0, 0, i, local_B[(1 - double_buffer_flag_AB) * local_B_size + i]);
                            break;
                        }
                    }
                    } */
    reply_get_A = 0;
    reply_get_B = 0;
    if (_MYID == 0){
        /* printf("local_start %d\n",local_start);
        printf("replygetA %d replygetB %d\n",replygetA,replygetB);
        printf("local_A address: %d\n",local_A);
        printf("local_B address: %d\n",local_B);
        printf("local_C address: %d\n",local_C);
        for(int i = 0; i < local_A_size; i++){
            printf("local_A %d value %f\n",i,local_A[i]);
        }
        for(int i = 0; i < local_A_size; i++){
            printf("local_A double buffer %d value %f\n",i,local_A[i + (1 - double_buffer_flag_AB) * local_A_size]);
        } */
        /* printf("blk_M %d blk_N %d blk_K %d num_M %d num_N %d num_K %d rem_M %d rem_N %d rem_K %d\n", 
                blk_M, blk_N, blk_K, num_M, num_N, num_K, rem_M, rem_N, rem_K); */
        /* for(int i = 0; i < 2 * local_C_size; i++){
            local_C[i] = 1.0;
        }
        memset(local_C + (1 - double_buffer_flag_AB) * local_C_size, 0, local_C_size * sizeof(float));
        for(int i = 0; i < 2 * local_C_size; i++){
            printf("local_C  %f\n",local_C[i]);
        } */
    }
    for(int local_now = local_start; local_now < local_end; ++local_now){
        start_A = src_A + MK_size * local_now;
        start_B = src_B + KN_size * local_now;
        start_C = src_C + MN_size * local_now;
        for(int curr_M = 0; curr_M < num_M; curr_M++){//K N M order
            curr_blk_M = curr_M < num_M - 1 ? blk_M : rem_blk_M;
            for(int curr_N = 0; curr_N < num_N; curr_N++){
                curr_blk_N = curr_N < num_N - 1 ? blk_N : rem_blk_N;
                memset(local_C + (1 - double_buffer_flag_C) * local_C_size, 0, local_C_size * sizeof(float));
                for(int curr_K = 0; curr_K < num_K; curr_K++){
                    curr_blk_K = curr_K < num_K - 1 ? blk_K : rem_blk_K;
                    if(curr_N * num_M * num_K + curr_M * num_K + curr_K +1 < num_M * num_N * num_K){//still in local_now
                        next_blk_M = curr_blk_M;
                        next_blk_N = curr_blk_N;
                        next_blk_K = curr_blk_K;
                        if(curr_K == num_K - 2){
                            next_blk_K = rem_blk_K;
                        }
                        if(curr_K == num_K - 1){
                            next_blk_K = blk_K;
                            if(curr_N == num_N - 2){
                                next_blk_N = rem_blk_N;
                            }
                            if(curr_N == num_N - 1){
                                next_blk_N = blk_N;
                                if(curr_M == num_M - 2){
                                    next_blk_M = rem_blk_M;
                                }
                                if(curr_M == num_M - 1){//to local_now + 1
                                    next_blk_M = blk_M;
                                }
                            }
                        }
                        if(curr_K == num_K - 1){
                            if(curr_N == num_N - 1){
                                athread_dma_iget_stride(local_A + (double_buffer_flag_AB * local_A_size),
                                                        start_A + (curr_M + 1) * blk_M * K + 0 * blk_K,
                                                        sizeof(float) * next_blk_M * next_blk_K,
                                                        sizeof(float) * next_blk_K,
                                                        sizeof(float) * (K - next_blk_K),
                                                        &reply_get_A);
                                athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_B_size),
                                                        start_B,
                                                        sizeof(float) * next_blk_K * next_blk_N,
                                                        sizeof(float) * next_blk_N,
                                                        sizeof(float) * (N - next_blk_N),
                                                        &reply_get_B);
                                /* if(_MYID == 3){
                                    printf("curr_M %d curr_N %d curr_K %d Get A: %d Get B: %d next_blk_M %d next_blk_N %d next_blk_K %d\n",
                                            curr_M, curr_N, curr_K, (curr_M + 1) * blk_M * K + 0 * blk_K, 0, next_blk_M, next_blk_N, next_blk_K);
                                } */
                            }
                            else{
                                athread_dma_iget_stride(local_A + (double_buffer_flag_AB * local_A_size),
                                                        start_A + (curr_M * blk_M * K) + 0 * blk_K,
                                                        sizeof(float) * next_blk_M * next_blk_K,
                                                        sizeof(float) * next_blk_K,
                                                        sizeof(float) * (K - next_blk_K),
                                                        &reply_get_A);
                                athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_B_size),
                                                        start_B + 0 * blk_K * N + (curr_N + 1) * blk_N,
                                                        sizeof(float) * next_blk_K * next_blk_N,
                                                        sizeof(float) * next_blk_N,
                                                        sizeof(float) * (N - next_blk_N),
                                                        &reply_get_B);
                                /* if(_MYID == 3){
                                    printf("curr_M %d curr_N %d curr_K %d Get A: %d Get B: %d next_blk_M %d next_blk_N %d next_blk_K %d\n",
                                            curr_M, curr_N, curr_K, (curr_M * blk_M * K) + 0 * blk_K, 0 * blk_K * N + (curr_N + 1) * blk_N, next_blk_M, next_blk_N, next_blk_K);
                                } */
                            }
                        }
                        else{
                            athread_dma_iget_stride(local_A + (double_buffer_flag_AB * local_A_size),
                                                    start_A + (curr_M * blk_M * K) + (curr_K +1) * blk_K,
                                                    sizeof(float) * next_blk_M * next_blk_K,
                                                    sizeof(float) * next_blk_K,
                                                    sizeof(float) * (K - next_blk_K),
                                                    &reply_get_A);
                            athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_B_size),
                                                    start_B + (curr_K + 1) * blk_K * N + (curr_N * blk_N),
                                                    sizeof(float) * next_blk_K * next_blk_N,
                                                    sizeof(float) * next_blk_N,
                                                    sizeof(float) * (N - next_blk_N),
                                                    &reply_get_B);
                            /* if(_MYID == 3){
                                printf("curr_M %d curr_N %d curr_K %d Get A: %d Get B: %d next_blk_M %d next_blk_N %d next_blk_K %d\n",
                                        curr_M, curr_N, curr_K, (curr_M * blk_M * K) + (curr_K +1) * blk_K, (curr_K + 1) * blk_K * N + (curr_N * blk_N), next_blk_M, next_blk_N, next_blk_K);
                            } */
                        }
                    }
                    else if(local_now < local_end - 1){
                        athread_dma_iget_stride(local_A + (double_buffer_flag_AB * local_A_size), 
                                                start_A + MK_size, 
                                                sizeof(float) * blk_M * blk_K, 
                                                sizeof(float) * blk_K, 
                                                sizeof(float) * (K - blk_K),
                                                &reply_get_A);
                        athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_A_size), 
                                                start_B + KN_size, 
                                                sizeof(float) * blk_K * blk_N, 
                                                sizeof(float) * blk_N, 
                                                sizeof(float) * (N - blk_N),
                                                &reply_get_B);
                        /* if(_MYID == 3){
                            printf("local_now < local_end - 1\n");
                            printf("curr_M %d curr_N %d curr_K %d Get A: %d Get B: %d next_blk_M %d next_blk_N %d next_blk_K %d\n",
                                    curr_M, curr_N, curr_K, MK_size, KN_size, next_blk_M, next_blk_N, next_blk_K);
                        } */
                    }
                    /* for(int i = 0; i < curr_blk_M * curr_blk_K; i++){
                        if(local_A[(1 - double_buffer_flag_AB) * local_A_size + i]!=1.0){
                            printf("Slave %d compute curr_M %d curr_N %d curr_K %d error at %d local_A value %lf\n", 
                                    _MYID, curr_M, curr_N, curr_K, i, local_A[(1 - double_buffer_flag_AB) * local_A_size + i]);
                            break;
                        }
                    } */
                    /* if(_MYID == 99){
                    for(int i = 0; i < curr_blk_K * curr_blk_N; i++){
                        if(local_B[(1 - double_buffer_flag_AB) * local_B_size + i]!=1.0){
                            printf("Slave %d compute curr_M %d curr_N %d curr_K %d error at %d local_B value %lf\n", 
                                    _MYID, curr_M, curr_N, curr_K, i, local_B[(1 - double_buffer_flag_AB) * local_B_size + i]);
                            //break;
                        }
                    }
                    } */
                    for(int m = 0; m < curr_blk_M; m++)
                        for(int n = 0; n < curr_blk_N; n++)
                            for(int k = 0; k < curr_blk_K; k++){
                                local_C[(1 - double_buffer_flag_C) * local_C_size + m * curr_blk_N + n]
                             += local_A[(1 - double_buffer_flag_AB) * local_A_size + m * curr_blk_K + k]
                              * local_B[(1 - double_buffer_flag_AB) * local_B_size + k * curr_blk_N + n];
                            }
                    //gemm
                    //(1 - double_buffer_flag_AB) 
                    //(1 - double_buffer_flag_C)
                    /* if(_MYID == 0){
                        printf("curr_M %d curr_N %d curr_K %d curr_blk_M %d curr_blk_N %d curr_blk_K %d next_blk_M %d next_blk_N %d next_blk_K %d\n",
                                curr_M, curr_N, curr_K, curr_blk_M, curr_blk_N, curr_blk_K, next_blk_M, next_blk_N, next_blk_K);
                    } */
                    if(curr_N * num_M * num_K + curr_M * num_K + curr_K +1 < num_M * num_N * num_K || local_now < local_end - 1){
                        athread_dma_wait_value(&reply_get_A, 1);
                        athread_dma_wait_value(&reply_get_B, 1);
                        reply_get_A = 0;
                        reply_get_B = 0;
                        double_buffer_flag_AB = 1 - double_buffer_flag_AB;
                    }
                }
                /* athread_dma_wait_value(&reply_put_C, 0);
                reply_put_C = 0; */
                /* for(int i = 0; i < MN_size; i++)
                    if(local_C[(1 - double_buffer_flag_C) * local_C_size + i]!= 96.0){
                        printf("Slave %d compute error at %d value %lf\n", _MYID, i, local_C[(1 - double_buffer_flag_C) * local_C_size + i]);
                        break;
                    } */
                athread_dma_put_stride(start_C + curr_M * blk_M * N + curr_N * blk_N,
                                        local_C + (1 - double_buffer_flag_C) * local_C_size,
                                        sizeof(float) * curr_blk_M * curr_blk_N,
                                        sizeof(float) * curr_blk_N,
                                        sizeof(float) * (N - curr_blk_N));
                //double_buffer_flag_C = 1 - double_buffer_flag_C;
            }
        }
    }
    ldm_free(local_A,sizeof(float) * blk_M * blk_K * 2);
    ldm_free(local_B,sizeof(float) * blk_K * blk_N * 2);
    ldm_free(local_C,sizeof(float) * blk_M * blk_N * 2);
}
void sw_slave_mm_ABT(swptex_mmPara *_){
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int i, j, k;
    float temp;
    int M_blk = M / thread_num;
    int M_load = M_blk + (myid < M % thread_num ? 1 : 0);
    int M_st = myid * M_blk + (myid < M % thread_num ? myid : M % thread_num);
    float *local_A = A + M_st * K;
    float *local_C = C + M_st * N;
    for (i = 0; i < M_load; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            temp = 0.f;
            for (k = 0; k < K; ++k)
            {
                temp += local_A[i * K + k] * B[j * K + k];
            }
            local_C[i * N + j] = temp;
        }
    }
}

void sw_slave_mm_AB(swptex_mmPara *_)
{
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int i, j, k;
    float temp;
    int M_blk = M / thread_num;
    int M_load = M_blk + (myid < M % thread_num ? 1 : 0);
    int M_st = myid * M_blk + (myid < M % thread_num ? myid : M % thread_num);
    float *local_A = A + M_st * K;
    float *local_C = C + M_st * N;
    for (i = 0; i < M_load; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            temp = 0.f;
            for (k = 0; k < K; ++k)
            {
                temp += local_A[i * K + k] * B[k * N + j];
            }
            local_C[i * N + j] = temp;
        }
    }
}

void sw_slave_mm_ATB(swptex_mmPara *_)
{
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    int i, j, k;
    float temp;
    int M_blk = M / thread_num;
    int M_load = M_blk + (myid < M % thread_num ? 1 : 0);
    int M_st = myid * M_blk + (myid < M % thread_num ? myid : M % thread_num);
    for (i = 0; i < M_load; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            temp = 0.f;
            for (k = 0; k < K; ++k)
            {
                temp += A[M_st + k * M + i] * B[k * N + j];
            }
            C[i * N + M_st * N + j] = temp;
        }
    }
}
