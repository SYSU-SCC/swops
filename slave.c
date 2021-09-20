#include <crts.h>
#include <simd.h>
#include <slave.h>
#include <stdio.h>
#include <stdlib.h>

#include "args.h"

__cross void *para_cross = NULL;

#define thread_num (64 * 6)

inline void sw_slave_gemm_copy_all_back_add(const int ThreadsStart, const int ThreadsEnd,
                                            const int sli_A,
                                            const float* A, const float* Ap,
                                            const int H, const int He,
                                            const int W, const int We){

    const int ThreadsNum = ThreadsEnd - ThreadsStart + 1;
    const int myid = CRTS_cgn * 64 + CRTS_tid - ThreadsStart;

    if(ThreadsNum <= 0){
        printf("sw_slave_gemm_copy_all ThreadsNum Error!\n");
        return;
    }
    /* if(H == He && W == We){
        printf("sw_slave_gemm_copy_all Don't Need Copy All!\n");
        return;
    } */
    if(sli_A == 0){
        printf("sw_slave_gemm_copy_all_back_add sli_A == 0!\n Please use sw_slave_gemm_copy_all_back");
        return;
    }
    if(myid < 0){
        //printf("_MYID %d sw_slave_gemm_copy_all myid %d return!\n", CRTS_cgn * 64 + CRTS_tid, myid);
        return;
    }

    const int counts_Q = H;
    
    const int local_count = (counts_Q + ThreadsNum - 1) / ThreadsNum;
    const int local_start = myid * local_count;
    const int local_end = ((local_start + local_count > counts_Q) ? counts_Q : (local_start + local_count));
    const int local_rows = local_end - local_start;

    if(local_start >= counts_Q || myid >= ThreadsNum){
        return;
    }

    int blk_Q = (sli_A * local_rows * We * sizeof(float) < 210 * 1024) ? local_rows : 210 * 1024 / (sli_A * We * sizeof(float));

    int rem_Q = blk_Q;
    if(blk_Q != local_rows){
        rem_Q = local_rows % blk_Q;
    }

    float* local_Q = ldm_malloc(sizeof(float) * sli_A * blk_Q * We);
    //printf("myid %d H %d He %d blk_Q %d rem_Q %d local_start %d local_rows %d local_end %d\n", myid, H, He, blk_Q, rem_Q, local_start, local_rows, local_end);
    for(int local_now = local_start; local_now < local_end; local_now += blk_Q){
        int curr_Q = blk_Q;
        if(local_now + blk_Q < H){//right zeros only
            curr_Q = local_end - local_now < blk_Q ? rem_Q : blk_Q;
#ifdef _SWOPS_DEBUG
            printf("local_now %d blk_Q %d curr_Q %d\n", local_now, blk_Q, curr_Q);
#endif
            athread_dma_get_stride(local_Q,
                                    Ap + local_now * We,
                                    sizeof(float) * sli_A * curr_Q * We,
                                    sizeof(float) * curr_Q * We,
                                    sizeof(float) * (He * We - curr_Q * We));

            for(int s = 1; s < sli_A; s++){
                for(int m = 0; m < curr_Q; m++){
                    for(int n = 0; n < W; n++){
                        local_Q[m * We + n] += local_Q[s * curr_Q * We + m * We + n];
                    }
                }
            }

            for(int m = 0; m < curr_Q; m++){
                for(int n = 0; n < W; n++){
                    local_Q[m * W + n] = local_Q[m * We + n];
                }
            }

            athread_dma_put(A + local_now * W,
                            local_Q,
                            sizeof(float) * curr_Q * W);
        }
        else if(local_now + blk_Q >= H && local_now < H){//right and bottom zeros    There are some problems
            curr_Q = local_end - local_now < blk_Q ? rem_Q : blk_Q;
            int res_Q = H - local_now;
#ifdef _SWOPS_DEBUG
            printf("local_now %d blk_Q %d curr_Q %d res_Q %d\n", local_now, blk_Q, curr_Q, res_Q);
#endif
            athread_dma_get_stride(local_Q,
                                    Ap + local_now * We,
                                    sizeof(float) * sli_A * res_Q * We,
                                    sizeof(float) * res_Q * We,
                                    sizeof(float) * (He * We - res_Q * We));

            for(int s = 1; s < sli_A; s++){
                for(int m = 0; m < res_Q; m++){
                    for(int n = 0; n < W; n++){
                        local_Q[m * We + n] += local_Q[s * res_Q * We + m * We + n];
                    }
                }
            }

            for(int m = 0; m < res_Q; m++){
                for(int n = 0; n < W; n++){
                    local_Q[m * W + n] = local_Q[m * We + n];
                }
            }
            athread_dma_put(A + local_now * W,
                            local_Q,
                            sizeof(float) * res_Q * W);
        }
    }
    ldm_free(local_Q, sizeof(float) * sli_A * blk_Q * We);
}

inline void sw_slave_gemm_copy_all_back(const int ThreadsStart, const int ThreadsEnd,
                                        const float* A, const float* Ap,
                                        const int H, const int He,
                                        const int W, const int We){

    const int ThreadsNum = ThreadsEnd - ThreadsStart + 1;
    const int myid = CRTS_cgn * 64 + CRTS_tid - ThreadsStart;

    if(ThreadsNum <= 0){
        printf("sw_slave_gemm_copy_all ThreadsNum Error!\n");
        return;
    }
    /* if(H == He && W == We){
        printf("sw_slave_gemm_copy_all Don't Need Copy All!\n");
        return;
    } */
    if(myid < 0){
        //printf("_MYID %d sw_slave_gemm_copy_all myid %d return!\n", CRTS_cgn * 64 + CRTS_tid, myid);
        return;
    }

    const int counts_Q = H;
    
    const int local_count = (counts_Q + ThreadsNum - 1) / ThreadsNum;
    const int local_start = myid * local_count;
    const int local_end = ((local_start + local_count > counts_Q) ? counts_Q : (local_start + local_count));
    const int local_rows = local_end - local_start;

    if(local_start >= counts_Q || myid >= ThreadsNum){
        return;
    }

    int blk_Q = (local_rows * We * sizeof(float) < 210 * 1024) ? local_rows : 210 * 1024 / (We * sizeof(float));

    int rem_Q = blk_Q;
    if(blk_Q != local_rows){
        rem_Q = local_rows % blk_Q;
    }
    float* local_Q = ldm_malloc(sizeof(float) * blk_Q * We);
    //printf("myid %d H %d He %d blk_Q %d rem_Q %d local_start %d local_rows %d local_end %d\n", myid, H, He, blk_Q, rem_Q, local_start, local_rows, local_end);
    for(int local_now = local_start; local_now < local_end; local_now += blk_Q){
        int curr_Q = blk_Q;
        if(local_now + blk_Q < H){//right zeros only
            curr_Q = local_end - local_now < blk_Q ? rem_Q : blk_Q;
            athread_dma_get(local_Q,
                            Ap + local_now * We,
                            sizeof(float) * curr_Q * We);

            for(int m = 0; m < curr_Q; m++){
                for(int n = 0; n < W; n++){
                    local_Q[m * W + n] = local_Q[m * We + n];
                }
            }

            athread_dma_put(A + local_now * W,
                            local_Q,
                            sizeof(float) * curr_Q * W);
        }
        else if(local_now + blk_Q >= H && local_now < H){//right and bottom zeros    There are some problems
            curr_Q = local_end - local_now < blk_Q ? rem_Q : blk_Q;
            int res_Q = H - local_now;
#ifdef _SWOPS_DEBUG
            printf("local_now %d blk_Q %d curr_Q %d res_Q %d\n", local_now, blk_Q, curr_Q, res_Q);
#endif
            athread_dma_get(local_Q,
                            Ap + local_now * We,
                            sizeof(float) * res_Q * We);
            for(int m = 0; m < res_Q; m++){
                for(int n = 0; n < W; n++){
                    local_Q[m * W + n] = local_Q[m * We + n];
                }
            }
            athread_dma_put(A + local_now * W,
                            local_Q,
                            sizeof(float) * res_Q * W);
        }
    }
    ldm_free(local_Q, sizeof(float) * blk_Q * We);
}

inline void sw_slave_gemm_copy_all(const int ThreadsStart, const int ThreadsEnd,
                                    const float* A, const float* Ap,
                                    const int H, const int He,
                                    const int W, const int We){

    const int ThreadsNum = ThreadsEnd - ThreadsStart + 1;
    const int myid = CRTS_cgn * 64 + CRTS_tid - ThreadsStart;

    if(ThreadsNum <= 0){
        printf("sw_slave_gemm_copy_all ThreadsNum Error!\n");
        return;
    }
    /* if(H == He && W == We){
        //printf("sw_slave_gemm_copy_all Don't Need Copy All!\n");
        return;
    } */
    if(myid < 0){
        //printf("_MYID %d sw_slave_gemm_copy_all myid %d return!\n", CRTS_cgn * 64 + CRTS_tid, myid);
        return;
    }

    const int counts_Q = He;
    const int local_count = (counts_Q + ThreadsNum - 1) / ThreadsNum;
    const int local_start = myid * local_count;
    const int local_end = ((local_start + local_count > counts_Q) ? counts_Q : (local_start + local_count));
    const int local_rows = local_end - local_start;

    if(local_start >= counts_Q || myid >= ThreadsNum){
        return;
    }

    int blk_Q = (local_rows * We * sizeof(float) < 210 * 1024) ? local_rows : 210 * 1024 / (We * sizeof(float));

    int rem_Q = blk_Q;
    if(blk_Q != local_rows){
        rem_Q = local_rows % blk_Q;
    }
    float* local_Q = ldm_malloc(sizeof(float) * blk_Q * We);
    //printf("myid %d H %d He %d blk_Q %d rem_Q %d local_start %d local_rows %d local_end %d\n", myid, H, He, blk_Q, rem_Q, local_start, local_rows, local_end);
    for(int local_now = local_start; local_now < local_end; local_now += blk_Q){
        int curr_Q = blk_Q;
        if(local_now + blk_Q < H){//right zeros only
            curr_Q = local_end - local_now < blk_Q ? rem_Q : blk_Q;
            athread_dma_get(local_Q,
                            A + local_now * W,
                            sizeof(float) * curr_Q * W);
            for(int m = curr_Q - 1; m >= 0; m--){
                for(int n = W - 1; n >= 0; n--){
                    local_Q[m * We + n] = local_Q[m * W + n];
                }
                for(int n = We - 1; n >= W; n--){
                    local_Q[m * We+ n] = 0;
                }
            }
            athread_dma_put(Ap + local_now * We,
                            local_Q,
                            sizeof(float) * curr_Q * We);
        }
        else if(local_now + blk_Q >= H && local_now < H){//right and bottom zeros    There are some problems
            curr_Q = local_end - local_now < blk_Q ? rem_Q : blk_Q;
            int res_Q = H - local_now;
#ifdef _SWOPS_DEBUG
            printf("local_now %d blk_Q %d curr_Q %d res_Q %d\n", local_now, blk_Q, curr_Q, res_Q);
#endif
            athread_dma_get(local_Q,
                            A + local_now * W,
                            sizeof(float) * res_Q * W);
            for(int m = res_Q - 1; m >= 0; m--){//copy right zeros
                for(int n = W - 1; n >= 0; n--){
                    local_Q[m * We + n] = local_Q[m * W + n];
                }
                for(int n = We - 1; n >= W; n--){
                    local_Q[m * We+ n] = 0;
                }
            }
            for(int m = curr_Q - 1; m >= res_Q; m--){//copy bottom zeros
                for(int n = 0; n < We; n++){
                    local_Q[m * We + n] = 0;
                }
            }
            athread_dma_put(Ap + local_now * We,
                            local_Q,
                            sizeof(float) * curr_Q * We);
        }
        else if(local_now >= H){//bottom zeros only
            curr_Q = local_end - local_now < blk_Q ? rem_Q : blk_Q;
            for(int m = curr_Q - 1; m >= 0; m--){//copy bottom zeros
                for(int n = We; n >= 0; n--){
                    local_Q[m * We + n] = 0;
                }
            }
            athread_dma_put(Ap + local_now * We,
                            local_Q,
                            sizeof(float) * curr_Q * We);
        }
    }
    ldm_free(local_Q, sizeof(float) * blk_Q * We);
}

void sw_slave_gemm_copy_all_f32(sw_gemmPara *_){
    sw_gemmPara *para = (sw_gemmPara *)para_cross;
    const float *src_A = para->A;
    const float *src_Ap = para->Ap;
    const float *src_B = para->B;
    const float *src_Bp = para->Bp;
    const float *src_C = para->C;
    const float *src_Cp = para->Cp;
    const float *src_T = para->T;
    const float *src_Tp = para->Tp;
    const size_t slice = para->slice;
    const size_t M = para->M;
    const size_t Ms = para->Ms;
    const size_t Me = para->Me;
    const size_t Mp = para->Mp;
    const size_t N = para->N;
    const size_t Ns = para->Ns;
    const size_t Ne = para->Ne;
    const size_t Np = para->Np;
    const size_t K = para->K;
    const size_t Ks = para->Ks;
    const size_t Ke = para->Ke;
    const size_t Kp = para->Kp;
    const size_t blk_M = para->blk_M;
    const size_t blk_N = para->blk_N;
    const size_t blk_K = para->blk_K;
    sw_slave_gemm_copy_all(0, 383, src_A, src_Ap,
                            M, Me,
                            K, Ke);
    athread_ssync_node();
    sw_slave_gemm_copy_all_back(0, 383, src_B, src_Ap,
                                M, Me,
                                K, Ke);      
    athread_ssync_node();               
}

inline void sw_slave_gemm_copy_all_H_32_to_64(const int CGN_id,
                                                const float* A, const float* Ap,
                                                const int H_pad, const int W_pad,
                                                const int H, const int Hs, const int He, const int blk_H,
                                                const int W, const int Ws, const int We, const int blk_W){
    if(CRTS_cgn != CGN_id){
        return;
    }
    if(H_pad && W_pad){
        printf("sw_slave_gemm_copy_all_32_to_64 error\n");
        return;
    }
    if(We > 30720){
        printf("error in copy border, We: %d larger than 30720", We);
        return;
    }
    if(H_pad){
        if(CRTS_tid < 32){
            const int CPY_tid = CRTS_tid;
            float* start_A = A + CPY_tid * W;
            float* start_Ap = Ap + CPY_tid * We;
            float* local_A = ldm_malloc(sizeof(float) * We);
            athread_dma_get(local_A,
                            start_A,
                            sizeof(float) * W);
            for(int i = W; i < We; i++){
                local_A[i] = 0;
            }
            athread_dma_put(start_Ap,
                            local_A,
                            sizeof(float) * We);
            ldm_free(local_A, sizeof(float) * We);
        }
        else{
            const int CPY_tid = CRTS_tid - 32;
            float* start_A = A + (CPY_tid + 32) * W;
            float* start_Ap = Ap + (CPY_tid + 32) * We;
            float* local_A = ldm_malloc(sizeof(float) * We);
            for(int i = 0; i < We; i++){
                local_A[i] = 0;
            }
            athread_dma_put(start_Ap,
                            local_A,
                            sizeof(float) * We);
            ldm_free(local_A, sizeof(float) * We);
        }
    }
}

void sw_slave_gemm_copy_all_H_f32(sw_gemmPara *_){
    sw_gemmPara *para = (sw_gemmPara *)para_cross;
    const float *src_A = para->A;
    const float *src_Ap = para->Ap;
    const float *src_B = para->B;
    const float *src_Bp = para->Bp;
    const float *src_C = para->C;
    const float *src_Cp = para->Cp;
    const float *src_T = para->T;
    const float *src_Tp = para->Tp;
    const size_t slice = para->slice;
    const size_t M = para->M;
    const size_t Ms = para->Ms;
    const size_t Me = para->Me;
    const size_t N = para->N;
    const size_t Ns = para->Ns;
    const size_t Ne = para->Ne;
    const size_t K = para->K;
    const size_t Ks = para->Ks;
    const size_t Ke = para->Ke;
    const size_t blk_M = para->blk_M;
    const size_t blk_N = para->blk_N;
    const size_t blk_K = para->blk_K;
    if(_MYID == 0){
        printf("processing copy all on H slaves\n");
    }
    sw_slave_gemm_copy_all_H_32_to_64(0, src_A, src_Ap,
                                        1, 0,
                                        M, Ms, Me, blk_M,
                                        K, Ks, Ke, blk_K);
}

inline void sw_slave_gemm_copy_border_back_f32_cgn(const int CGN_id,
                                                    const float* A, const float* Ap,
                                                    const int H, const int Hs, const int He, const int blk_H,
                                                    const int W, const int Ws, const int We, const int blk_W){
    if(CRTS_cgn != CGN_id){
        return;
    }
    if(blk_W > 30720){
        if(_MYID == 0){
            printf("error in sw_slave_gemm_copy_border_f32_cgn\n blk_W: %d > 30720",blk_W);
        }
    }
    
    size_t HW_size = H * W;
    if(CRTS_tid < 32){
        if(Ws == We){
            return;
        }
        const int CPY_tid = CRTS_tid;
        const size_t Cols_P = Hs / 32;
        const float* start_A = A + CPY_tid * Cols_P * W + Ws;
        const float* start_Ap = Ap + CPY_tid * Cols_P * We + Ws;
        const size_t blk_P = Cols_P * blk_W * sizeof(float) < (210 * 1024) ? Cols_P : (210 * 1024) / (blk_W * sizeof(float));//larger than 220 * 1024 will cause memory access error
        const size_t num_P = (Cols_P + blk_P - 1) / blk_P;
        const size_t rem_blk_P = Cols_P - num_P * blk_P == 0 ? blk_P : Cols_P - (num_P - 1) * blk_P;
        const size_t rem_blk_W = W - Ws;
        const size_t local_P_size = blk_P * blk_W;
        size_t curr_blk_P = blk_P;
        size_t next_blk_P = blk_P;
        float* local_P = ldm_malloc(blk_P * blk_W * sizeof(float));
        /* if(local_P == 0){
            printf("local_P ldm_malloc error!!! %d\n", _MYID);
            return;
        } */
        for(int curr_P = 0; curr_P < num_P; curr_P++){
            curr_blk_P = curr_P < num_P - 1 ? blk_P : rem_blk_P;
            for(int i = 0; i < local_P_size; i++){
                local_P[i] = 0;
            }
            athread_dma_get_stride(local_P, 
                                    start_Ap + blk_P * curr_P * We,
                                    sizeof(float) * curr_blk_P * blk_W, 
                                    sizeof(float) * blk_W, 
                                    sizeof(float) * (We - blk_W));
            for(int m = 0; m < curr_blk_P; m++){
                for(int n = 0; n < rem_blk_W; n++){
                    local_P[m * rem_blk_W + n] = local_P[m * blk_W + n];
                }
            }
            athread_dma_put_stride(start_A + curr_P * blk_P * W,//error here
                                   local_P,
                                   sizeof(float) * curr_blk_P * rem_blk_W,
                                   sizeof(float) * rem_blk_W,
                                   sizeof(float) * (W - rem_blk_W));
        }
        ldm_free(local_P,blk_P * blk_W * sizeof(float));
    }
    else{
        if(Hs == He){
            return;
        }
        //printf("Cores %d Copying Border Back\n", CRTS_tid);
        const int CPY_tid = CRTS_tid - 32;// 0 ~ 32
        const size_t Cols_Q = blk_H / 32;
        const size_t num_Q = Cols_Q;
        const size_t rem_blk_H = H - Hs;
        const float* start_A = A + Hs * W + CPY_tid * Cols_Q * W;
        const float* start_Ap = Ap + Hs * We + CPY_tid * Cols_Q * We;
        float* local_Q = ldm_malloc(We * sizeof(float));
        if(We > 30720){
            printf("error in copy border, We: %d larger than 30720", We);
            return;
        }
        for(int curr_Q = 0; curr_Q < num_Q; curr_Q++){
            if(CPY_tid * Cols_Q + curr_Q < rem_blk_H ){//not zero only
                athread_dma_get(local_Q,
                                start_Ap + curr_Q * We,
                                sizeof(float) * We);
                athread_dma_put(start_A + curr_Q * W,
                                local_Q,
                                sizeof(float) * W);
            }
        }
        ldm_free(local_Q, We * sizeof(float));
    }
}
//H the hight dim, W the low dim
//run on CRTS_cgn

inline void sw_slave_gemm_copy_border_f32_cgn(const int CGN_id,
                                              const float* src, const float* dst,
                                              const int H, const int Hs, const int He, const int blk_H,
                                              const int W, const int Ws, const int We, const int blk_W){
    if(CRTS_cgn != CGN_id){
        return;
    }
    if(blk_W > 30720){
        if(_MYID == 0){
            printf("error in sw_slave_gemm_copy_border_f32_cgn\n blk_W: %d > 30720",blk_W);
        }
    }
    size_t HW_size = H*W;
    if(CRTS_tid < 32){
        //do dstp
        if(Ws == We){
            return;
        }
        const int CPY_tid = CRTS_tid;// 0 ~ 31
        const size_t Cols_P = Hs / 32;
        const float* src_P = src + CPY_tid * Cols_P * W + Ws;
        const float* dst_P = dst + CPY_tid * Cols_P * We + Ws;
        const size_t blk_P = Cols_P * blk_W * sizeof(float) < (210 * 1024) ? Cols_P : (210 * 1024) / (blk_W * sizeof(float));//larger than 220 * 1024 will cause memory access error
        const size_t num_P = (Cols_P + blk_P - 1) / blk_P;
        const size_t rem_blk_P = Cols_P - num_P * blk_P == 0 ? blk_P : Cols_P - (num_P - 1) * blk_P;
        const size_t rem_blk_W = W - Ws;
        const size_t local_P_size = blk_P * blk_W;
        size_t curr_blk_P = blk_P;
        size_t next_blk_P = blk_P;
        float* local_P = ldm_malloc(blk_P * blk_W * sizeof(float));
        /* if(local_P == 0){
            printf("local_P ldm_malloc error!!! %d\n", _MYID);
            return;
        } */
        for(int curr_P = 0; curr_P < num_P; curr_P++){
            curr_blk_P = curr_P < num_P - 1 ? blk_P : rem_blk_P;
            for(int i = 0; i < local_P_size; i++){
                local_P[i] = 0;
            }
            athread_dma_get_stride(local_P, 
                                    src_P + blk_P * curr_P * W, //src_P
                                    sizeof(float) * curr_blk_P * rem_blk_W, 
                                    sizeof(float) * rem_blk_W, 
                                    sizeof(float) * (W - rem_blk_W));
            for(int m = curr_blk_P - 1; m >= 0; m--){
                for(int n = rem_blk_W - 1; n >= 0; n--){
                    local_P[m * blk_W + n] = local_P[m * rem_blk_W + n];
                }
                for(int n = blk_W - 1; n >= rem_blk_W; n--){
                    local_P[m * blk_W + n] = 0;
                }
            }
            athread_dma_put_stride(dst_P + curr_P * blk_P * We,//error here
                                   local_P,
                                   sizeof(float) * curr_blk_P * blk_W,
                                   sizeof(float) * blk_W,
                                   sizeof(float) * (We - blk_W));
        }
        ldm_free(local_P,blk_P * blk_W * sizeof(float));
    }
    else{
        if(Hs == He){
            return;
        }
        const int CPY_tid = CRTS_tid - 32;// 0 ~ 32
        const size_t Cols_Q = blk_H / 32;
        const size_t num_Q = Cols_Q;
        const size_t rem_blk_H = H - Hs;
        const float* src_Q = src + Hs * W + CPY_tid * Cols_Q * W;
        const float* dst_Q = dst + Hs * We + CPY_tid * Cols_Q * We;
        float* local_Q = ldm_malloc(We * sizeof(float));
        if(We > 30720){
            printf("error in copy border, We: %d larger than 30720", We);
        }
        for(int curr_Q = 0; curr_Q < num_Q; curr_Q++){
            if(CPY_tid * Cols_Q + curr_Q < rem_blk_H ){//not zero only
                athread_dma_get(local_Q,
                                src_Q + curr_Q * W,
                                sizeof(float) * W);
                for(int i = W; i < We; i++){
                    local_Q[i] = 0;
                }
                athread_dma_put(dst_Q + curr_Q * We,
                                local_Q,
                                sizeof(float) * We);
            }
            else{
                for(int i = 0; i < We; i++){
                    local_Q[i] = 0;
                }
                athread_dma_put(dst_Q + curr_Q * We,
                                local_Q,
                                sizeof(float) * We);
            }
        }
        ldm_free(local_Q, We * sizeof(float));
    }
}

void sw_slave_gemm_crr_cgn(const int CGN_id,
                           const float* A, const float* Ap,
                           const float* B, const float* Bp,
                           const float* C, const float* Cp,
                           const size_t M, const size_t Ms, const size_t Me, const size_t blk_M,
                           const size_t N, const size_t Ns, const size_t Ne, const size_t blk_N,
                           const size_t K, const size_t Ks, const size_t Ke, const size_t blk_K){
    if(CRTS_cgn != CGN_id){
        return;
    }
    const int cid = CRTS_tid % 8;
    const int rid = CRTS_tid / 8;

    const size_t num_M = (M + blk_M - 1) / blk_M;
    const size_t num_N = (N + blk_N - 1) / blk_N;
    const size_t num_K = (K + blk_K - 1) / blk_K;

    const size_t rem_blk_M = num_M * blk_M - M == 0 ? blk_M : M - (num_M-1) * blk_M;
    const size_t rem_blk_N = num_N * blk_N - N == 0 ? blk_N : N - (num_N-1) * blk_N;
    const size_t rem_blk_K = num_K * blk_K - K == 0 ? blk_K : K - (num_K-1) * blk_K;

    const size_t local_A_size = blk_M * blk_K / 64;
    const size_t local_B_size = blk_K * blk_N / 64;
    const size_t local_C_size = blk_M * blk_N / 64;

    float* local_A = (float*)ldm_malloc(sizeof(float) * 2 * blk_M * blk_K / 64);
    float* local_B = (float*)ldm_malloc(sizeof(float) * 2 * blk_K * blk_N / 64);
    double* local_C = (double*)ldm_malloc(sizeof(double) * blk_M * blk_N / 64);

    float* local_A_dma = (float*)ldm_malloc(sizeof(float) * blk_M * blk_K / 64);
    float* local_B_dma = (float*)ldm_malloc(sizeof(float) * blk_K * blk_N / 64);
    float* local_C_dma = (float*)ldm_malloc(sizeof(float) * blk_M * blk_N / 64);

    const float* start_A = A + cid * blk_K/8 * M + rid * blk_M/8;
    const float* start_B = B + rid * blk_K/8 * N + cid * blk_N/8;
    const float* start_C = C + rid * blk_M/8 * N + cid * blk_N/8;

    const float* start_Ap = Ap + cid * blk_K/8 * Me + rid * blk_M/8;
    const float* start_Bp = Bp + rid * blk_K/8 * Ne + cid * blk_N/8;
    const float* start_Cp = Cp + rid * blk_M/8 * Ne + cid * blk_N/8;

    float* next_A = A;
    float* next_B = B;
    float* next_C = C;
    float* curr_C = C;

    size_t A_step = M - blk_M/8;
    size_t B_step = N - blk_N/8;
    size_t C_step = N - blk_N/8;

    size_t curr_M = M;
    size_t curr_N = N;
    size_t curr_K = K;

    size_t curr_blk_M = blk_M;
    size_t curr_blk_N = blk_N;
    size_t curr_blk_K = blk_K;

    size_t next_A_offset = 0;
    size_t next_B_offset = 0;
    size_t next_C_offset = 0;

    size_t next_blk_M = blk_M;
    size_t next_blk_N = blk_N;
    size_t next_blk_K = blk_K;

    volatile athread_rply_t dma_get_A = 0, dma_get_B = 0, dma_put_C = 0;
    volatile athread_rply_t rma_local_A = 0, rma_local_B = 0;
    volatile athread_rply_t rma_A[8] = {0,0,0,0,0,0,0,0}, rma_B[8] = {0,0,0,0,0,0,0,0};

    volatile int double_buffer_A = 0, double_buffer_B = 0; //for rma

    athread_dma_iget_stride(local_A_dma, 
                            start_A, 
                            sizeof(float) * local_A_size, 
                            sizeof(float) * blk_M/8, 
                            sizeof(float) * A_step,
                            &dma_get_A);
    athread_dma_iget_stride(local_B_dma, 
                            start_B, 
                            sizeof(float) * local_B_size, 
                            sizeof(float) * blk_N/8, 
                            sizeof(float) * B_step,
                            &dma_get_B);
    athread_dma_wait_value(&dma_get_A, 1);
    athread_dma_wait_value(&dma_get_B, 1);

    for(int c_M = 0; c_M < num_M; c_M++){
        curr_blk_M = c_M < num_M - 1 ? blk_M : rem_blk_M;
        for(int c_N = 0; c_N < num_N; c_N++){
        curr_blk_N = c_N < num_N - 1 ? blk_N : rem_blk_N;
            for(int i = 0; i < local_C_size; i++){
                local_C[i] = 0;
            }
            for(int i = 0; i < local_C_size; i++){
                local_C_dma[i] = 0;
            }
            for(int c_K = 0; c_K < num_K; c_K++){
                curr_blk_K = c_K < num_K - 1 ? blk_K : rem_blk_K;
                if(c_M * num_N * num_K + c_N * num_K + c_K + 1 < num_M * num_N * num_K){
                    next_blk_M = blk_M;
                    next_blk_N = blk_N;
                    next_blk_K = blk_K;
                    if(c_K == num_K - 2){
                        next_blk_K = rem_blk_K;
                    }
                    if(c_K == num_K - 1){
                        next_blk_K = blk_K;
                        if(c_N == num_N - 2){
                            next_blk_N = rem_blk_N;
                        }
                        if(c_N == num_N - 1){
                            next_blk_N = blk_N;
                            if(c_M == num_M - 2){
                                next_blk_M = rem_blk_M;
                            }
                        }
                    }//here's bugs
                    if(c_M == num_M - 1){
                        next_blk_M = rem_blk_M;
                    }
                    //There must be more conditions
                    if(c_N == num_N - 1 && c_K != num_K - 1){//not the last c_K
                        next_blk_N = rem_blk_N;
                    }
                    //All conditions have been checked
                    if(next_blk_K != blk_K || next_blk_M != blk_M){
                        next_A = start_Ap;
                        curr_M = Me;
                        A_step = Me - blk_M/8;
                    }
                    else{
                        next_A = start_A;
                        curr_M = M;
                        A_step = M - blk_M/8;
                    }
                    if(next_blk_K != blk_K || next_blk_N != blk_N){
                        next_B = start_Bp;
                        curr_N = Ne;
                        B_step = Ne - blk_N/8;
                    }
                    else{
                        next_B = start_B;
                        curr_N = N;
                        B_step = N - blk_N/8;
                    }
                }

                if(c_K == num_K - 1){
                    if(c_N == num_N - 1){
                        next_A_offset = (c_M + 1) * blk_M;
                        next_B_offset = 0;
                    }
                    else{
                        next_A_offset = c_M * blk_M;
                        next_B_offset = (c_N + 1) * blk_N;
                    }
                }
                else{
                    next_A_offset = (c_K + 1) * blk_K * curr_M + c_M * blk_M;
                    next_B_offset = (c_K + 1) * blk_K * curr_N + c_N * blk_N;
                }

                double_buffer_A = 0;
                double_buffer_B = 0;
                athread_ssync_array();
                if(cid == 0){
                    athread_dma_wait_value(&dma_get_A, 1);
                    athread_rma_row_ibcast(local_A + double_buffer_A * local_A_size, 
                                            local_A_dma,
                                            sizeof(float) * local_A_size,
                                            &rma_local_A,
                                            &rma_A[cid]);
                }
                if(rid == 0){
                    athread_dma_wait_value(&dma_get_B, 1);
                    athread_rma_col_ibcast(local_B + double_buffer_B * local_B_size,
                                            local_B_dma,
                                            sizeof(float) * local_B_size,
                                            &rma_local_B,
                                            &rma_B[rid]);
                }

                for(int c_R = 0; c_R < 8; c_R++){

                    if(cid == c_R && c_M * num_N * num_K + c_N * num_K + c_K + 1 < num_M * num_N * num_K){
                        athread_dma_wait_value(&dma_get_A, 1);
                        athread_rma_wait_value(&rma_local_A, 1);
                        dma_get_A = 0;
                        rma_local_A = 0;
                        athread_dma_iget_stride(local_A_dma, 
                                                next_A + next_A_offset, 
                                                sizeof(float) * local_A_size, 
                                                sizeof(float) * blk_M/8, 
                                                sizeof(float) * A_step,
                                                &dma_get_A);
                    }
                    if(rid == c_R && c_M * num_N * num_K + c_N * num_K + c_K + 1 < num_M * num_N * num_K){
                        athread_dma_wait_value(&dma_get_B, 1);
                        athread_rma_wait_value(&rma_local_B, 1);
                        dma_get_B = 0;
                        rma_local_B = 0;
                        athread_dma_iget_stride(local_B_dma, 
                                                next_B + next_B_offset, 
                                                sizeof(float) * local_B_size, 
                                                sizeof(float) * blk_N/8, 
                                                sizeof(float) * B_step,
                                                &dma_get_B);//这个get要护
                    }

                    athread_rma_wait_value(&rma_A[c_R], 1);
                    athread_rma_wait_value(&rma_B[c_R], 1);

                    athread_ssync_array();//here must synchronization

                    double_buffer_A = 1 - double_buffer_A;
                    double_buffer_B = 1 - double_buffer_B;
                    rma_A[c_R] = 0;
                    rma_B[c_R] = 0;

                    if(cid == c_R + 1){
                        athread_rma_row_ibcast(local_A + double_buffer_A * local_A_size, 
                                                local_A_dma,
                                                sizeof(float) * local_A_size,
                                                &rma_local_A,
                                                &rma_A[c_R+1]);
                    }
                    if(rid == c_R + 1){
                        athread_rma_col_ibcast(local_B + double_buffer_B * local_B_size,
                                                local_B_dma,
                                                sizeof(float) * local_B_size,
                                                &rma_local_B,
                                                &rma_B[c_R+1]);
                    }

                    for(int m = 0; m < (blk_M/8); m++){
                        for(int n = 0; n < (blk_N/8); n++){
                            for(int k = 0; k < (blk_K/8); k++){
                                local_C_dma[m * (blk_N/8) + n] += 
                                local_A[k * (blk_M/8) + m + (1 - double_buffer_A) * local_A_size] *
                                local_B[k * (blk_N/8) + n + (1 - double_buffer_B) * local_B_size];
                            }
                        }
                    }
                }
            }
            if(curr_blk_M == blk_M && curr_blk_N == blk_N){
                curr_C = start_C;
                curr_N = N;
                C_step = N - blk_N/8;
            }
            /* if(c_M == num_M - 1 || c_N == num_N - 1){
                curr_C = start_Cp;
                curr_N = Ne;
                C_step = Ne - blk_N/8;
            } */
            else{
                curr_C = start_Cp;
                curr_N = Ne;
                C_step = Ne - blk_N/8;
            }
            athread_dma_put_stride(curr_C + c_M * blk_M * curr_N + c_N * blk_N,
                                    local_C_dma,
                                    sizeof(float) * local_C_size,
                                    sizeof(float) * blk_N/8,
                                    sizeof(float) * C_step);
        }
    }
    athread_ssync_array();
    ldm_free(local_A, sizeof(float) * 2 * blk_M * blk_K / 64);
    ldm_free(local_B, sizeof(float) * 2 * blk_K * blk_N / 64);
    ldm_free(local_C, sizeof(double) * blk_M * blk_N / 64);
    ldm_free(local_A_dma, sizeof(float) * blk_M * blk_K / 64);
    ldm_free(local_B_dma, sizeof(float) * blk_K * blk_N / 64);
    ldm_free(local_C_dma, sizeof(float) * blk_M * blk_N / 64);
}


void sw_slave_gemm_rrr_cgn(const int CGN_id,
                           const float* A, const float* Ap,
                           const float* B, const float* Bp,
                           const float* C, const float* Cp,
                           const size_t M, const size_t Ms, const size_t Me, const size_t blk_M,
                           const size_t N, const size_t Ns, const size_t Ne, const size_t blk_N,
                           const size_t K, const size_t Ks, const size_t Ke, const size_t blk_K){
    if(CRTS_cgn != CGN_id){
        return;
    }

    const int cid = CRTS_tid % 8;
    const int rid = CRTS_tid / 8;

    const size_t num_M = (M + blk_M - 1) / blk_M;
    const size_t num_N = (N + blk_N - 1) / blk_N;
    const size_t num_K = (K + blk_K - 1) / blk_K;

    const size_t rem_blk_M = num_M * blk_M - M == 0 ? blk_M : M - (num_M-1) * blk_M;
    const size_t rem_blk_N = num_N * blk_N - N == 0 ? blk_N : N - (num_N-1) * blk_N;
    const size_t rem_blk_K = num_K * blk_K - K == 0 ? blk_K : K - (num_K-1) * blk_K;

    const size_t local_A_size = blk_M * blk_K / 64;
    const size_t local_B_size = blk_K * blk_N / 64;
    const size_t local_C_size = blk_M * blk_N / 64;

    float* local_A = (float*)ldm_malloc(sizeof(float) * 2 * blk_M * blk_K / 64);
    float* local_B = (float*)ldm_malloc(sizeof(float) * 2 * blk_K * blk_N / 64);
    double* local_C = (double*)ldm_malloc(sizeof(double) * blk_M * blk_N / 64);

    float* local_A_dma = (float*)ldm_malloc(sizeof(float) * blk_M * blk_K / 64);
    float* local_B_dma = (float*)ldm_malloc(sizeof(float) * blk_K * blk_N / 64);
    float* local_C_dma = (float*)ldm_malloc(sizeof(float) * blk_M * blk_N / 64);

    const float* start_A = A + rid * blk_M/8 * K + cid * blk_K/8;
    const float* start_B = B + rid * blk_K/8 * N + cid * blk_N/8;
    const float* start_C = C + rid * blk_M/8 * N + cid * blk_N/8;

    const float* start_Ap = Ap + rid * blk_M/8 * Ke + cid * blk_K/8;
    const float* start_Bp = Bp + rid * blk_K/8 * Ne + cid * blk_N/8;
    const float* start_Cp = Cp + rid * blk_M/8 * Ne + cid * blk_N/8;

    float* next_A = A;
    float* next_B = B;
    float* next_C = C;
    float* curr_C = C;

    size_t A_step = K - blk_K/8;
    size_t B_step = N - blk_N/8;
    size_t C_step = N - blk_N/8;

    size_t curr_M = M;
    size_t curr_N = N;
    size_t curr_K = K;

    size_t curr_blk_M = blk_M;
    size_t curr_blk_N = blk_N;
    size_t curr_blk_K = blk_K;

    size_t next_A_offset = 0;
    size_t next_B_offset = 0;
    size_t next_C_offset = 0;

    size_t next_blk_M = blk_M;
    size_t next_blk_N = blk_N;
    size_t next_blk_K = blk_K;

    volatile athread_rply_t dma_get_A = 0, dma_get_B = 0, dma_put_C = 0;
    volatile athread_rply_t rma_local_A = 0, rma_local_B = 0;
    volatile athread_rply_t rma_A[8] = {0,0,0,0,0,0,0,0}, rma_B[8] = {0,0,0,0,0,0,0,0};

    volatile int double_buffer_A = 0, double_buffer_B = 0; //for rma

    athread_dma_iget_stride(local_A_dma, 
                            start_A, 
                            sizeof(float) * local_A_size, 
                            sizeof(float) * blk_K/8, 
                            sizeof(float) * A_step,
                            &dma_get_A);
    athread_dma_iget_stride(local_B_dma, 
                            start_B, 
                            sizeof(float) * local_B_size, 
                            sizeof(float) * blk_N/8, 
                            sizeof(float) * B_step,
                            &dma_get_B);
    athread_dma_wait_value(&dma_get_A, 1);
    athread_dma_wait_value(&dma_get_B, 1);

    for(int c_M = 0; c_M < num_M; c_M++){
        curr_blk_M = c_M < num_M - 1 ? blk_M : rem_blk_M;
        for(int c_N = 0; c_N < num_N; c_N++){
        curr_blk_N = c_N < num_N - 1 ? blk_N : rem_blk_N;
            for(int i = 0; i < local_C_size; i++){
                local_C[i] = 0;
            }
            for(int i = 0; i < local_C_size; i++){
                local_C_dma[i] = 0;
            }
            for(int c_K = 0; c_K < num_K; c_K++){
                curr_blk_K = c_K < num_K - 1 ? blk_K : rem_blk_K;
                if(c_M * num_N * num_K + c_N * num_K + c_K + 1 < num_M * num_N * num_K){
                    next_blk_M = blk_M;
                    next_blk_N = blk_N;
                    next_blk_K = blk_K;
                    if(c_K == num_K - 2){
                        next_blk_K = rem_blk_K;
                    }
                    if(c_K == num_K - 1){
                        next_blk_K = blk_K;
                        if(c_N == num_N - 2){
                            next_blk_N = rem_blk_N;
                        }
                        if(c_N == num_N - 1){
                            next_blk_N = blk_N;
                            if(c_M == num_M - 2){
                                next_blk_M = rem_blk_M;
                            }
                        }
                    }//here's bugs
                    if(c_M == num_M - 1){
                        next_blk_M = rem_blk_M;
                    }
                    //There must be more conditions
                    if(c_N == num_N - 1 && c_K != num_K - 1){//not the last c_K
                        next_blk_N = rem_blk_N;
                    }
                    //All conditions have been checked
                    if(next_blk_K != blk_K || next_blk_M != blk_M){
                        next_A = start_Ap;
                        curr_K = Ke;
                        A_step = Ke - blk_K/8;
                    }
                    else{
                        next_A = start_A;
                        curr_K = K;
                        A_step = K - blk_K/8;
                    }
                    if(next_blk_K != blk_K || next_blk_N != blk_N){
                        next_B = start_Bp;
                        curr_N = Ne;
                        B_step = Ne - blk_N/8;
                    }
                    else{
                        next_B = start_B;
                        curr_N = N;
                        B_step = N - blk_N/8;
                    }
                }

                if(c_K == num_K - 1){
                    if(c_N == num_N - 1){
                        next_A_offset = (c_M + 1) * blk_M * curr_K;
                        next_B_offset = 0;
                    }
                    else{
                        next_A_offset = c_M * blk_M * curr_K;
                        next_B_offset = (c_N + 1) * blk_N;
                    }
                }
                else{
                    next_A_offset = c_M * blk_M * curr_K + (c_K + 1) * blk_K;
                    next_B_offset = (c_K + 1) * blk_K * curr_N + c_N * blk_N;
                }

                double_buffer_A = 0;
                double_buffer_B = 0;
                athread_ssync_array();
                if(cid == 0){
                    athread_dma_wait_value(&dma_get_A, 1);
                    athread_rma_row_ibcast(local_A + double_buffer_A * local_A_size, 
                                            local_A_dma,
                                            sizeof(float) * local_A_size,
                                            &rma_local_A,
                                            &rma_A[cid]);
                }
                if(rid == 0){
                    athread_dma_wait_value(&dma_get_B, 1);
                    athread_rma_col_ibcast(local_B + double_buffer_B * local_B_size,
                                            local_B_dma,
                                            sizeof(float) * local_B_size,
                                            &rma_local_B,
                                            &rma_B[rid]);
                }

                for(int c_R = 0; c_R < 8; c_R++){

                    if(cid == c_R && c_M * num_N * num_K + c_N * num_K + c_K + 1 < num_M * num_N * num_K){
                        athread_dma_wait_value(&dma_get_A, 1);
                        athread_rma_wait_value(&rma_local_A, 1);
                        dma_get_A = 0;
                        rma_local_A = 0;
                        athread_dma_iget_stride(local_A_dma, 
                                                next_A + next_A_offset, 
                                                sizeof(float) * local_A_size, 
                                                sizeof(float) * blk_K/8, 
                                                sizeof(float) * A_step,
                                                &dma_get_A);
                    }
                    if(rid == c_R && c_M * num_N * num_K + c_N * num_K + c_K + 1 < num_M * num_N * num_K){
                        athread_dma_wait_value(&dma_get_B, 1);
                        athread_rma_wait_value(&rma_local_B, 1);
                        dma_get_B = 0;
                        rma_local_B = 0;
                        athread_dma_iget_stride(local_B_dma, 
                                                next_B + next_B_offset, 
                                                sizeof(float) * local_B_size, 
                                                sizeof(float) * blk_N/8, 
                                                sizeof(float) * B_step,
                                                &dma_get_B);//这个get要护
                    }

                    athread_rma_wait_value(&rma_A[c_R], 1);
                    athread_rma_wait_value(&rma_B[c_R], 1);

                    athread_ssync_array();//here must synchronization

                    double_buffer_A = 1 - double_buffer_A;
                    double_buffer_B = 1 - double_buffer_B;
                    rma_A[c_R] = 0;
                    rma_B[c_R] = 0;

                    if(cid == c_R + 1){
                        athread_rma_row_ibcast(local_A + double_buffer_A * local_A_size, 
                                                local_A_dma,
                                                sizeof(float) * local_A_size,
                                                &rma_local_A,
                                                &rma_A[c_R+1]);
                    }
                    if(rid == c_R + 1){
                        athread_rma_col_ibcast(local_B + double_buffer_B * local_B_size,
                                                local_B_dma,
                                                sizeof(float) * local_B_size,
                                                &rma_local_B,
                                                &rma_B[c_R+1]);
                    }

                    for(int m = 0; m < (blk_M/8); m++){
                        for(int n = 0; n < (blk_N/8); n++){
                            for(int k = 0; k < (blk_K/8); k++){
                                local_C_dma[m * (blk_N/8) + n] += 
                                local_A[m * (blk_K/8) + k + (1 - double_buffer_A) * local_A_size] *
                                local_B[k * (blk_N/8) + n + (1 - double_buffer_B) * local_B_size];
                            }
                        }
                    }
                }
            }
            if(curr_blk_M == blk_M && curr_blk_N == blk_N){
                curr_C = start_C;
                curr_N = N;
                C_step = N - blk_N/8;
            }
            /* if(c_M == num_M - 1 || c_N == num_N - 1){
                curr_C = start_Cp;
                curr_N = Ne;
                C_step = Ne - blk_N/8;
            } */
            else{
                curr_C = start_Cp;
                curr_N = Ne;
                C_step = Ne - blk_N/8;
            }
            athread_dma_put_stride(curr_C + c_M * blk_M * curr_N + c_N * blk_N,
                                    local_C_dma,
                                    sizeof(float) * local_C_size,
                                    sizeof(float) * blk_N/8,
                                    sizeof(float) * C_step);
        }
    }
    athread_ssync_array();
    ldm_free(local_A, sizeof(float) * 2 * blk_M * blk_K / 64);
    ldm_free(local_B, sizeof(float) * 2 * blk_K * blk_N / 64);
    ldm_free(local_C, sizeof(double) * blk_M * blk_N / 64);
    ldm_free(local_A_dma, sizeof(float) * blk_M * blk_K / 64);
    ldm_free(local_B_dma, sizeof(float) * blk_K * blk_N / 64);
    ldm_free(local_C_dma, sizeof(float) * blk_M * blk_N / 64);
}

void sw_slave_gemm_rrr4_cgn(const int CGN_id,
                            const float* A, const float* Ap,
                            const float* B, const float* Bp,
                            const float* C, const float* Cp,
                            const size_t M, const size_t Ms, const size_t Me, const size_t blk_M,
                            const size_t N, const size_t Ns, const size_t Ne, const size_t blk_N,
                            const size_t K, const size_t Ks, const size_t Ke, const size_t blk_K){
    if(CRTS_cgn != CGN_id){
        return;
    }
    const int cid = CRTS_tid % 8;
    const int rid = CRTS_tid / 8;

    const size_t num_M = (M + blk_M - 1) / blk_M;
    const size_t num_N = (N + blk_N - 1) / blk_N;
    const size_t num_K = (K + blk_K - 1) / blk_K;

    const size_t rem_blk_M = num_M * blk_M - M == 0 ? blk_M : M - (num_M-1) * blk_M;
    const size_t rem_blk_N = num_N * blk_N - N == 0 ? blk_N : N - (num_N-1) * blk_N;
    const size_t rem_blk_K = num_K * blk_K - K == 0 ? blk_K : K - (num_K-1) * blk_K;

    const size_t local_A_size = blk_M * blk_K / 64;
    const size_t local_B_size = blk_K * blk_N / 64;
    const size_t local_C_size = blk_M * blk_N / 64;

    float* local_A = (float*)ldm_malloc(sizeof(float) * 4 * blk_M * blk_K / 64);
    float* local_B = (float*)ldm_malloc(sizeof(float) * 4 * blk_K * blk_N / 64);
    double* local_C = (double*)ldm_malloc(sizeof(double) * blk_M * blk_N / 64);

    float* local_A_dma = (float*)ldm_malloc(sizeof(float) * blk_M * blk_K / 64);
    float* local_B_dma = (float*)ldm_malloc(sizeof(float) * blk_K * blk_N / 64);
    float* local_C_dma = (float*)ldm_malloc(sizeof(float) * blk_M * blk_N / 64);

    const float* start_A = A + rid * blk_M/8 * K + cid * blk_K/8;
    const float* start_B = B + rid * blk_K/8 * N + cid * blk_N/8;
    const float* start_C = C + rid * blk_M/8 * N + cid * blk_N/8;

    const float* start_Ap = Ap + rid * blk_M/8 * Ke + cid * blk_K/8;
    const float* start_Bp = Bp + rid * blk_K/8 * Ne + cid * blk_N/8;
    const float* start_Cp = Cp + rid * blk_M/8 * Ne + cid * blk_N/8;

    float* next_A = A;
    float* next_B = B;
    float* next_C = C;
    float* curr_C = C;

    size_t A_step = K - blk_K/8;
    size_t B_step = N - blk_N/8;
    size_t C_step = N - blk_N/8;

    size_t curr_M = M;
    size_t curr_N = N;
    size_t curr_K = K;

    size_t curr_blk_M = blk_M;
    size_t curr_blk_N = blk_N;
    size_t curr_blk_K = blk_K;

    size_t next_A_offset = 0;
    size_t next_B_offset = 0;
    size_t next_C_offset = 0;

    size_t next_blk_M = blk_M;
    size_t next_blk_N = blk_N;
    size_t next_blk_K = blk_K;

    volatile athread_rply_t dma_get_A = 0, dma_get_B = 0, dma_put_C = 0;
    volatile athread_rply_t rma_local_A = 0, rma_local_B = 0;
    volatile athread_rply_t rma_A[8] = {0,0,0,0,0,0,0,0}, rma_B[8] = {0,0,0,0,0,0,0,0};

    volatile int double_buffer_A = 0, double_buffer_B = 0; //for rma

    athread_dma_iget_stride(local_A_dma, 
                            start_A, 
                            sizeof(float) * local_A_size, 
                            sizeof(float) * blk_K/8, 
                            sizeof(float) * A_step,
                            &dma_get_A);
    athread_dma_iget_stride(local_B_dma, 
                            start_B, 
                            sizeof(float) * local_B_size, 
                            sizeof(float) * blk_N/8, 
                            sizeof(float) * B_step,
                            &dma_get_B);
    athread_dma_wait_value(&dma_get_A, 1);
    athread_dma_wait_value(&dma_get_B, 1);

    for(int c_M = 0; c_M < num_M; c_M++){
        curr_blk_M = c_M < num_M - 1 ? blk_M : rem_blk_M;
        for(int c_N = 0; c_N < num_N; c_N++){
        curr_blk_N = c_N < num_N - 1 ? blk_N : rem_blk_N;
            for(int i = 0; i < local_C_size; i++){
                local_C[i] = 0;
            }
            for(int i = 0; i < local_C_size; i++){
                local_C_dma[i] = 0;
            }
            for(int c_K = 0; c_K < num_K; c_K++){
                curr_blk_K = c_K < num_K - 1 ? blk_K : rem_blk_K;
                if(c_M * num_N * num_K + c_N * num_K + c_K + 1 < num_M * num_N * num_K){
                    next_blk_M = blk_M;
                    next_blk_N = blk_N;
                    next_blk_K = blk_K;
                    if(c_K == num_K - 2){
                        next_blk_K = rem_blk_K;
                    }
                    if(c_K == num_K - 1){
                        next_blk_K = blk_K;
                        if(c_N == num_N - 2){
                            next_blk_N = rem_blk_N;
                        }
                        if(c_N == num_N - 1){
                            next_blk_N = blk_N;
                            if(c_M == num_M - 2){
                                next_blk_M = rem_blk_M;
                            }
                        }
                    }//here's bugs
                    if(c_M == num_M - 1){
                        next_blk_M = rem_blk_M;
                    }
                    //There must be more conditions
                    if(c_N == num_N - 1 && c_K != num_K - 1){//not the last c_K
                        next_blk_N = rem_blk_N;
                    }
                    //All conditions have been checked
                    if(next_blk_K != blk_K || next_blk_M != blk_M){
                        next_A = start_Ap;
                        curr_K = Ke;
                        A_step = Ke - blk_K/8;
                    }
                    else{
                        next_A = start_A;
                        curr_K = K;
                        A_step = K - blk_K/8;
                    }
                    if(next_blk_K != blk_K || next_blk_N != blk_N){
                        next_B = start_Bp;
                        curr_N = Ne;
                        B_step = Ne - blk_N/8;
                    }
                    else{
                        next_B = start_B;
                        curr_N = N;
                        B_step = N - blk_N/8;
                    }
                }

                if(c_K == num_K - 1){
                    if(c_N == num_N - 1){
                        next_A_offset = (c_M + 1) * blk_M * curr_K;
                        next_B_offset = 0;
                    }
                    else{
                        next_A_offset = c_M * blk_M * curr_K;
                        next_B_offset = (c_N + 1) * blk_N;
                    }
                }
                else{
                    next_A_offset = c_M * blk_M * curr_K + (c_K + 1) * blk_K;
                    next_B_offset = (c_K + 1) * blk_K * curr_N + c_N * blk_N;
                }
                double_buffer_A = 0;// 2 - A
                double_buffer_B = 0;// 2 - B
                athread_ssync_array();
                if(cid == 0){
                    athread_dma_wait_value(&dma_get_A, 1);
                    athread_rma_row_ibcast(local_A + double_buffer_A * local_A_size, 
                                            local_A_dma,
                                            sizeof(float) * local_A_size,
                                            &rma_local_A,
                                            &rma_A[cid]);
                }
                if(cid == 1){
                    athread_dma_wait_value(&dma_get_A, 1);
                    athread_rma_row_ibcast(local_A + (double_buffer_A + 1) * local_A_size, 
                                            local_A_dma,
                                            sizeof(float) * local_A_size,
                                            &rma_local_A,
                                            &rma_A[cid]);
                }
                if(rid == 0){
                    athread_dma_wait_value(&dma_get_B, 1);
                    athread_rma_col_ibcast(local_B + double_buffer_B * local_B_size,
                                            local_B_dma,
                                            sizeof(float) * local_B_size,
                                            &rma_local_B,
                                            &rma_B[rid]);
                }
                if(rid == 1){
                    athread_dma_wait_value(&dma_get_B, 1);
                    athread_rma_col_ibcast(local_B + (double_buffer_B + 1) * local_B_size,
                                            local_B_dma,
                                            sizeof(float) * local_B_size,
                                            &rma_local_B,
                                            &rma_B[rid]);
                }
                for(int c_R = 0; c_R < 8; c_R += 2){

                    if(cid == c_R && c_M * num_N * num_K + c_N * num_K + c_K + 1 < num_M * num_N * num_K){
                        athread_dma_wait_value(&dma_get_A, 1);
                        athread_rma_wait_value(&rma_local_A, 1);
                        dma_get_A = 0;
                        rma_local_A = 0;
                        athread_dma_iget_stride(local_A_dma, 
                                                next_A + next_A_offset, 
                                                sizeof(float) * local_A_size, 
                                                sizeof(float) * blk_K/8, 
                                                sizeof(float) * A_step,
                                                &dma_get_A);
                    }
                    if(cid == (c_R + 1) && c_M * num_N * num_K + c_N * num_K + c_K + 1 < num_M * num_N * num_K){
                        athread_dma_wait_value(&dma_get_A, 1);
                        athread_rma_wait_value(&rma_local_A, 1);
                        dma_get_A = 0;
                        rma_local_A = 0;
                        athread_dma_iget_stride(local_A_dma, 
                                                next_A + next_A_offset, 
                                                sizeof(float) * local_A_size, 
                                                sizeof(float) * blk_K/8, 
                                                sizeof(float) * A_step,
                                                &dma_get_A);
                    }
                    if(rid == c_R && c_M * num_N * num_K + c_N * num_K + c_K + 1 < num_M * num_N * num_K){
                        athread_dma_wait_value(&dma_get_B, 1);
                        athread_rma_wait_value(&rma_local_B, 1);
                        dma_get_B = 0;
                        rma_local_B = 0;
                        athread_dma_iget_stride(local_B_dma, 
                                                next_B + next_B_offset, 
                                                sizeof(float) * local_B_size, 
                                                sizeof(float) * blk_N/8, 
                                                sizeof(float) * B_step,
                                                &dma_get_B);//这个get要护
                    }
                    if(rid == (c_R + 1) && c_M * num_N * num_K + c_N * num_K + c_K + 1 < num_M * num_N * num_K){
                        athread_dma_wait_value(&dma_get_B, 1);
                        athread_rma_wait_value(&rma_local_B, 1);
                        dma_get_B = 0;
                        rma_local_B = 0;
                        athread_dma_iget_stride(local_B_dma, 
                                                next_B + next_B_offset, 
                                                sizeof(float) * local_B_size, 
                                                sizeof(float) * blk_N/8, 
                                                sizeof(float) * B_step,
                                                &dma_get_B);//这个get要护
                    }

                    athread_rma_wait_value(&rma_A[c_R], 1);
                    athread_rma_wait_value(&rma_B[c_R], 1);
                    athread_rma_wait_value(&rma_A[c_R + 1], 1);
                    athread_rma_wait_value(&rma_B[c_R + 1], 1);
                    
                    athread_ssync_array();//here must synchronization

                    double_buffer_A = 2 - double_buffer_A;
                    double_buffer_B = 2 - double_buffer_B;

                    rma_A[c_R] = 0;
                    rma_A[c_R + 1] = 0;

                    rma_B[c_R] = 0;
                    rma_B[c_R + 1] = 0;

                    if(cid == c_R + 2){
                        athread_rma_row_ibcast(local_A + double_buffer_A * local_A_size, 
                                                local_A_dma,
                                                sizeof(float) * local_A_size,
                                                &rma_local_A,
                                                &rma_A[cid]);
                    }
                    if(cid == c_R + 3){
                        athread_rma_row_ibcast(local_A + (double_buffer_A + 1) * local_A_size, 
                                                local_A_dma,
                                                sizeof(float) * local_A_size,
                                                &rma_local_A,
                                                &rma_A[cid]);
                    }
                    if(rid == c_R + 2){
                        athread_rma_col_ibcast(local_B + double_buffer_B * local_B_size,
                                                local_B_dma,
                                                sizeof(float) * local_B_size,
                                                &rma_local_B,
                                                &rma_B[rid]);
                    }
                    if(rid == c_R + 3){
                        athread_rma_col_ibcast(local_B + (double_buffer_B + 1) * local_B_size,
                                                local_B_dma,
                                                sizeof(float) * local_B_size,
                                                &rma_local_B,
                                                &rma_B[rid]);
                    }
                    for(int m = 0; m < (blk_M/8); m++){
                        for(int n = 0; n < (blk_N/8); n++){
                            for(int k = 0; k < (blk_K/8); k++){
                                local_C_dma[m * (blk_N/8) + n] += 
                                local_A[m * (blk_K/8) + k + (2 - double_buffer_A) * local_A_size] *
                                local_B[k * (blk_N/8) + n + (2 - double_buffer_B) * local_B_size];
                            }
                        }
                    }
                    for(int m = 0; m < (blk_M/8); m++){
                        for(int n = 0; n < (blk_N/8); n++){
                            for(int k = 0; k < (blk_K/8); k++){
                                local_C_dma[m * (blk_N/8) + n] += 
                                local_A[m * (blk_K/8) + k + (2 - double_buffer_A + 1) * local_A_size] *
                                local_B[k * (blk_N/8) + n + (2 - double_buffer_B + 1) * local_B_size];
                            }
                        }
                    }
                    //slower that 8 c_R
                }
            }
            if(curr_blk_M == blk_M && curr_blk_N == blk_N){
                curr_C = start_C;
                curr_N = N;
                C_step = N - blk_N/8;
            }
            /* if(c_M == num_M - 1 || c_N == num_N - 1){
                curr_C = start_Cp;
                curr_N = Ne;
                C_step = Ne - blk_N/8;
            } */
            else{
                curr_C = start_Cp;
                curr_N = Ne;
                C_step = Ne - blk_N/8;
            }
            athread_dma_put_stride(curr_C + c_M * blk_M * curr_N + c_N * blk_N,
                                    local_C_dma,
                                    sizeof(float) * local_C_size,
                                    sizeof(float) * blk_N/8,
                                    sizeof(float) * C_step);
        }
    }
    athread_ssync_array();
    ldm_free(local_A, sizeof(float) * 4 * blk_M * blk_K / 64);
    ldm_free(local_B, sizeof(float) * 4 * blk_K * blk_N / 64);
    ldm_free(local_C, sizeof(double) * blk_M * blk_N / 64);
    ldm_free(local_A_dma, sizeof(float) * blk_M * blk_K / 64);
    ldm_free(local_B_dma, sizeof(float) * blk_K * blk_N / 64);
    ldm_free(local_C_dma, sizeof(float) * blk_M * blk_N / 64);
}

inline void sw_slave_trans_f32_cgn(const int CGN_id,
                                    const float* C, const float* Cp,
                                    const float* T, const float* Tp,
                                    const size_t M, const size_t Ms, const size_t Me, const size_t blk_M,
                                    const size_t N, const size_t Ns, const size_t Ne, const size_t blk_N){
    if(CRTS_cgn != CGN_id){
        return;
    }
    const int cid = CRTS_tid % 8;
    const int rid = CRTS_tid / 8;

    const size_t num_M = (M + blk_M - 1) / blk_M;
    const size_t num_N = (N + blk_N - 1) / blk_N;

    const size_t rem_blk_M = num_M * blk_M - M == 0 ? blk_M : M - (num_M-1) * blk_M;
    const size_t rem_blk_N = num_N * blk_N - N == 0 ? blk_N : N - (num_N-1) * blk_N;

    const size_t local_C_size = blk_M * blk_N / 64;
    const size_t local_T_size = blk_M * blk_N / 64;

    float* local_C = (float*)ldm_malloc(sizeof(float) * blk_M * blk_N / 64);
    float* local_T = (float*)ldm_malloc(sizeof(float) * blk_M * blk_N / 64);
    
    const float* start_C = C + rid * blk_M/8 * N + cid * blk_N/8;
    const float* start_T = T + cid * blk_N/8 * M + rid * blk_M/8;

    const float* start_Cp = Cp + rid * blk_M/8 * Ne + cid * blk_N/8;
    const float* start_Tp = Tp + cid * blk_N/8 * Me + rid * blk_M/8;;

    float* curr_C = start_C;
    float* curr_T = start_T;

    size_t C_step = N - blk_N/8;
    size_t T_step = M - blk_M/8;

    size_t curr_M = M;
    size_t curr_N = N;

    size_t curr_blk_M = blk_M;
    size_t curr_blk_N = blk_N;

    size_t next_blk_M = blk_M;
    size_t next_blk_N = blk_N;

    for(int c_M = 0; c_M < num_M; c_M++){
        curr_blk_M = c_M < num_M - 1 ? blk_M : rem_blk_M;
        for(int c_N = 0; c_N < num_N; c_N++){
        curr_blk_N = c_N < num_N - 1 ? blk_N : rem_blk_N;
            if(curr_blk_M == blk_M && curr_blk_N == blk_N){
                curr_C = start_C;
                curr_T = start_T;
                curr_N = N;
                curr_M = M;
                C_step = N - blk_N/8;
                T_step = M - blk_M/8;
            }
            else{
                curr_C = start_Cp;
                curr_T = start_Tp;
                curr_N = Ne;
                curr_M = Me;
                C_step = Ne - blk_N/8;
                T_step = Me - blk_M/8;
            }
            athread_dma_get_stride(local_C,
                                    curr_C + c_M * blk_M * curr_N + c_N * blk_N,
                                    sizeof(float) * local_C_size,
                                    sizeof(float) * blk_N/8,
                                    sizeof(float) * C_step);

            for(int m = 0; m < blk_M/8; m++){
                for(int n = 0; n < blk_N/8; n++){
                    local_T[n * blk_M/8 + m] = local_C[m * blk_N/8 + n];
                }
            }

            athread_dma_put_stride(curr_T + c_N * blk_N * curr_M + c_M * blk_M,
                                    local_T,
                                    sizeof(float) * local_T_size,
                                    sizeof(float) * blk_M/8,
                                    sizeof(float) * T_step);
        }
    }
    ldm_free(local_C, sizeof(float) * blk_M * blk_N / 64);
    ldm_free(local_T, sizeof(float) * blk_M * blk_N / 64);
}

void sw_slave_gemm_crr_f32(sw_gemmPara *_){
    sw_gemmPara *para = (sw_gemmPara *)para_cross;
    const float *src_A = para->A;
    const float *src_Ap = para->Ap;
    const float *src_B = para->B;
    const float *src_Bp = para->Bp;
    const float *src_C = para->C;
    const float *src_Cp = para->Cp;
    size_t M = para->M;
    size_t Ms = para->Ms;
    size_t Me = para->Me;
    size_t N = para->N;
    size_t Ns = para->Ns;
    size_t Ne = para->Ne;
    size_t K = para->K;
    size_t Ks = para->Ks;
    size_t Ke = para->Ke;
    size_t blk_M = para->blk_M;
    size_t blk_N = para->blk_N;
    size_t blk_K = para->blk_K;
    size_t counts = para->counts;
    sw_slave_gemm_copy_border_f32_cgn(0, src_A, src_Ap, M, Ms, Me, blk_M, K, Ks, Ke ,blk_K);
    sw_slave_gemm_copy_border_f32_cgn(1, src_B, src_Bp, K, Ks, Ke, blk_K, N, Ns, Ne ,blk_N);
    athread_ssync_node();
    sw_slave_gemm_crr_cgn(0,src_A, src_Ap, 
                            src_B, src_Bp, 
                            src_C, src_Cp,
                            M, Ms, Me, blk_M,
                            N, Ns, Ne, blk_N,
                            K, Ks, Ke, blk_K);
    athread_ssync_node();
    sw_slave_gemm_copy_border_back_f32_cgn(0, src_C, src_Cp, M, Ms, Me, blk_M, N, Ns, Ne, blk_N);
}

void sw_slave_gemm_rrr_f32(sw_gemmPara *_){
    sw_gemmPara *para = (sw_gemmPara *)para_cross;
    const float *src_A = para->A;
    const float *src_Ap = para->Ap;
    const float *src_B = para->B;
    const float *src_Bp = para->Bp;
    const float *src_C = para->C;
    const float *src_Cp = para->Cp;
    size_t M = para->M;
    size_t Ms = para->Ms;
    size_t Me = para->Me;
    size_t N = para->N;
    size_t Ns = para->Ns;
    size_t Ne = para->Ne;
    size_t K = para->K;
    size_t Ks = para->Ks;
    size_t Ke = para->Ke;
    size_t blk_M = para->blk_M;
    size_t blk_N = para->blk_N;
    size_t blk_K = para->blk_K;
    size_t counts = para->counts;
    //sw_slave_gemm_copy_border_f32_cgn(0, src_A, src_Ap, M, Ms, Me, blk_M, K, Ks, Ke ,blk_K);
    //sw_slave_gemm_copy_border_f32_cgn(1, src_B, src_Bp, K, Ks, Ke, blk_K, N, Ns, Ne ,blk_N);
    sw_slave_gemm_copy_all(0,383,src_A, src_Ap, M, Me, K, Ke);
    sw_slave_gemm_copy_all(0,383,src_B, src_Bp, K, Ke, N, Ne);
    athread_ssync_node();
    sw_slave_gemm_rrr_cgn(0,src_Ap, src_Ap, 
                            src_Bp, src_Bp, 
                            src_Cp, src_Cp,
                            Me, Me, Me, blk_M,
                            Ne, Ne, Ne, blk_N,
                            Ke, Ke, Ke, blk_K);
    athread_ssync_node();
    sw_slave_gemm_copy_all_back(0,383,src_C, src_Cp, M, Me, N, Ne);
    //sw_slave_gemm_copy_border_back_f32_cgn(0, src_C, src_Cp, M, Ms, Me, blk_M, N, Ns, Ne, blk_N);
}

void sw_slave_gemm_crr_sli_cgn_f32(sw_gemmPara *_){
    sw_gemmPara *para = (sw_gemmPara *)para_cross;
    const float *A = para->A;
    const float *Ap = para->Ap;
    const float *B = para->B;
    const float *Bp = para->Bp;
    const float *C = para->C;
    const float *Cp = para->Cp;
    const float *A_cgn;
    const float *B_cgn;
    const float *C_cgn;
    const size_t M = para->M;
    const size_t Ms = para->Ms;
    const size_t Me = para->Me;
    const size_t N = para->N;
    const size_t Ns = para->Ns;
    const size_t Ne = para->Ne;
    const size_t K = para->K;
    const size_t Ks = para->Ks;
    const size_t Ke = para->Ke;
    const size_t blk_M = para->blk_M;
    const size_t blk_N = para->blk_N;
    const size_t blk_K = para->blk_K;
    const size_t sli_C = para->sli_C;
    if(M == Me && K == Ke){
        A_cgn = para->A_sli[CRTS_cgn];
    }
    else{
        A_cgn = para->Ap_sli[CRTS_cgn];
        sw_slave_gemm_copy_all(0, 383, A, Ap, K, Ke, M, Me);// be care ful!!!!
    }
    if(K == Ke && N == Ne){
        B_cgn = para->B_sli[CRTS_cgn];
    }
    else{
        B_cgn = para->Bp_sli[CRTS_cgn];
        sw_slave_gemm_copy_all(0, 383, B, Bp, K, Ke, N, Ne);
    }
    if(M == Me && N == Ne){
        C_cgn = para->C_sli[CRTS_cgn];
    }
    else{
        C_cgn = para->Cp_sli[CRTS_cgn];
    }
    const size_t K_cgn = para->sli_K[CRTS_cgn];
    athread_ssync_node();
#ifdef _SWOPS_DEBUG
    if(CRTS_tid == 0){
        printf("CRTS_cgn %d K_cgn %d\n",CRTS_cgn,K_cgn);
        printf("sli_C %d\n", sli_C);
    }
#endif
    if(K_cgn > 0){
        sw_slave_gemm_crr_cgn(CRTS_cgn,
                            A_cgn, A_cgn, 
                            B_cgn, B_cgn, 
                            C_cgn, C_cgn,
                            Me, Me, Me, blk_M,
                            Ne, Ne, Ne, blk_N,
                            K_cgn, K_cgn, K_cgn, blk_K);
    }
    athread_ssync_node();
    sw_slave_gemm_copy_all_back_add(0, 383, sli_C, C, Cp, M, Me, N, Ne);
    athread_ssync_node();
}

void sw_slave_gemm_rrr_sli_cgn_f32(sw_gemmPara *_){
    sw_gemmPara *para = (sw_gemmPara *)para_cross;
    const float *A = para->A;
    const float *Ap = para->Ap;
    const float *B = para->B;
    const float *Bp = para->Bp;
    const float *C = para->C;
    const float *Cp = para->Cp;
    const float *A_cgn;
    const float *B_cgn;
    const float *C_cgn;
    const size_t M = para->M;
    const size_t Ms = para->Ms;
    const size_t Me = para->Me;
    const size_t N = para->N;
    const size_t Ns = para->Ns;
    const size_t Ne = para->Ne;
    const size_t K = para->K;
    const size_t Ks = para->Ks;
    const size_t Ke = para->Ke;
    const size_t blk_M = para->blk_M;
    const size_t blk_N = para->blk_N;
    const size_t blk_K = para->blk_K;
    if(M == Me && K == Ke){
        A_cgn = para->A_sli[CRTS_cgn];
    }
    else{
        A_cgn = para->Ap_sli[CRTS_cgn];
        sw_slave_gemm_copy_all(0, 383, A, Ap, M, Me, K, Ke);
    }
    if(K == Ke && N == Ne){
        B_cgn = para->B_sli[CRTS_cgn];
    }
    else{
        B_cgn = para->Bp_sli[CRTS_cgn];
        sw_slave_gemm_copy_all(0, 383, B, Bp, K, Ke, N, Ne);
    }
    if(M == Me && N == Ne){
        C_cgn = para->C_sli[CRTS_cgn];
    }
    else{
        C_cgn = para->Cp_sli[CRTS_cgn];
    }
    const int M_cgn = para->sli_M[CRTS_cgn];
    athread_ssync_node();
#ifdef _SWOPS_DEBUG
    if(CRTS_tid == 0){
        printf("CRTS_cgn %d M_cgn %d\n",CRTS_cgn,M_cgn);
    }
#endif
    if(M_cgn > 0){
        sw_slave_gemm_rrr_cgn(CRTS_cgn,
                            A_cgn, A_cgn, 
                            B_cgn, B_cgn, 
                            C_cgn, C_cgn,
                            M_cgn, M_cgn, M_cgn, blk_M,
                            Ne, Ne, Ne, blk_N,
                            Ke, Ke, Ke, blk_K);
    }
    athread_ssync_node();
    if(M != Me || N != Ne){
        sw_slave_gemm_copy_all_back(0, 383, C, Cp, M, Me, N, Ne);
    }
    athread_ssync_node();
}

void sw_slave_gemm_rrr4_f32(sw_gemmPara *_){
    sw_gemmPara *para = (sw_gemmPara *)para_cross;
    const float *src_A = para->A;
    const float *src_Ap = para->Ap;
    const float *src_B = para->B;
    const float *src_Bp = para->Bp;
    const float *src_C = para->C;
    const float *src_Cp = para->Cp;
    size_t M = para->M;
    size_t Ms = para->Ms;
    size_t Me = para->Me;
    size_t N = para->N;
    size_t Ns = para->Ns;
    size_t Ne = para->Ne;
    size_t K = para->K;
    size_t Ks = para->Ks;
    size_t Ke = para->Ke;
    size_t blk_M = para->blk_M;
    size_t blk_N = para->blk_N;
    size_t blk_K = para->blk_K;
    size_t counts = para->counts;
    if(((blk_M % 32) != 0) || ((blk_N % 32) != 0) || ((blk_K % 32) != 0)){
        if(_MYID == 0){
            printf("gemm rrr block size error!!!, blk_M %d blk_N %d blk_K %d\n", blk_M, blk_N,blk_K);
        }
        return;
    }
    sw_slave_gemm_copy_border_f32_cgn(0, src_A, src_Ap, M, Ms, Me, blk_M, K, Ks, Ke ,blk_K);
    sw_slave_gemm_copy_border_f32_cgn(1, src_B, src_Bp, K, Ks, Ke, blk_K, N, Ns, Ne ,blk_N);
    athread_ssync_node();
    sw_slave_gemm_rrr4_cgn(0,src_A, src_Ap, 
                            src_B, src_Bp, 
                            src_C, src_Cp,
                            M, Ms, Me, blk_M,
                            N, Ns, Ne, blk_N,
                            K, Ks, Ke, blk_K);
    athread_ssync_node();
    sw_slave_gemm_copy_border_back_f32_cgn(0, src_C, src_Cp, M, Ms, Me, blk_M, N, Ns, Ne, blk_N);
}

void sw_slave_gemm_rcr_cgn_f32(sw_gemmPara *_){
    sw_gemmPara *para = (sw_gemmPara *)para_cross;
    const float *src_A = para->A;
    const float *src_Ap = para->Ap;
    const float *src_B = para->B;
    const float *src_Bp = para->Bp;
    const float *src_C = para->C;
    const float *src_Cp = para->Cp;
    const float *src_T = para->T;
    const float *src_Tp = para->Tp;
    const size_t slice = para->slice;
    const size_t M = para->M;
    const size_t Ms = para->Ms;
    const size_t Me = para->Me;
    const size_t N = para->N;
    const size_t Ns = para->Ns;
    const size_t Ne = para->Ne;
    const size_t K = para->K;
    const size_t Ks = para->Ks;
    const size_t Ke = para->Ke;
    const size_t blk_M = para->blk_M;
    const size_t blk_N = para->blk_N;
    const size_t blk_K = para->blk_K;
    sw_slave_gemm_copy_border_f32_cgn(0, src_B, src_Bp, 
                                        N, Ns, Ne ,blk_N,
                                        K, Ks, Ke, blk_K);

    sw_slave_trans_f32_cgn(0, src_B, src_Bp, 
                            src_T, src_Tp,
                            N, Ns, Ne ,blk_N,
                            K, Ks, Ke, blk_K);
    sw_slave_gemm_rrr_cgn(0, 
                            src_A, src_Ap, 
                            src_T, src_Tp,
                            src_C, src_Cp,
                            M, Ms, Me, blk_M,
                            N, Ns, Ne, blk_N,
                            K, Ks, Ke, blk_K);
    sw_slave_gemm_copy_border_back_f32_cgn(0, 
                                            src_C, src_Cp, 
                                            M, Ms, Me, blk_M, 
                                            N, Ns, Ne, blk_N);
}

void sw_slave_gemm_rcr_all_cgn_f32(sw_gemmPara *_){
    sw_gemmPara *para = (sw_gemmPara *)para_cross;
    const float *src_A = para->A;
    const float *src_Ap = para->Ap;
    const float *src_B = para->B;
    const float *src_Bp = para->Bp;
    const float *src_C = para->C;
    const float *src_Cp = para->Cp;
    const float *src_T = para->T;
    const float *src_Tp = para->Tp;
    const size_t slice = para->slice;
    const size_t M = para->M;
    const size_t Ms = para->Ms;
    const size_t Me = para->Me;
    const size_t N = para->N;
    const size_t Ns = para->Ns;
    const size_t Ne = para->Ne;
    const size_t K = para->K;
    const size_t Ks = para->Ks;
    const size_t Ke = para->Ke;
    const size_t blk_M = para->blk_M;
    const size_t blk_N = para->blk_N;
    const size_t blk_K = para->blk_K;
    sw_slave_gemm_copy_border_f32_cgn(0, src_B, src_Bp, 
                                        N, Ns, Ne ,blk_N,
                                        K, Ks, Ke, blk_K);

    athread_ssync_node();
    sw_slave_trans_f32_cgn(0, src_B, src_Bp, 
                            src_T, src_Tp,
                            N, Ns, Ne ,blk_N,
                            K, Ks, Ke, blk_K);
    athread_ssync_node();
    for(int i = 0; i < slice; i++){
        sw_slave_gemm_copy_border_f32_cgn(i % 6, 
                                            src_A + i * (M*K), src_Ap + i * (Me*Ke), 
                                            M, Ms, Me, blk_M, 
                                            K, Ks, Ke ,blk_K);
        sw_slave_gemm_rrr_cgn(i % 6, 
                                src_A + i * (M*K), src_Ap + i * (Me*Ke), 
                                src_T, src_Tp,
                                src_C + i * (M*N), src_Cp + i * (Me*Ne),
                                M, Ms, Me, blk_M,
                                N, Ns, Ne, blk_N,
                                K, Ks, Ke, blk_K);
        sw_slave_gemm_copy_border_back_f32_cgn(i % 6, 
                                                src_C + i * (M*N), src_Cp + i * (Me*Ne), 
                                                M, Ms, Me, blk_M, 
                                                N, Ns, Ne, blk_N);
    }
}



void sw_slave_bmm_rrr(sw_bmmPara *_){
    const int myid = CRTS_cgn * 64 + CRTS_tid;
    sw_bmmPara *para = (sw_bmmPara *)para_cross;
    const float *src_A = para->A;
    const float *src_B = para->B;
    const float *src_C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    size_t blk_M = para->blk_M;
    size_t blk_N = para->blk_N;
    size_t blk_K = para->blk_K;
    size_t counts = para->counts;
    const int local_count = (counts + 383) / 384;
    const int local_start = myid * local_count;
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
    reply_get_A = 0;
    reply_get_B = 0;
    for(int local_now = local_start; local_now < local_end; ++local_now){
        start_A = src_A + MK_size * local_now;
        start_B = src_B + KN_size * local_now;
        start_C = src_C + MN_size * local_now;
        for(int c_M = 0; c_M < num_M; c_M++){//K N M order
            curr_blk_M = c_M < num_M - 1 ? blk_M : rem_blk_M;
            for(int c_N = 0; c_N < num_N; c_N++){
                curr_blk_N = c_N < num_N - 1 ? blk_N : rem_blk_N;
                memset(local_C + (1 - double_buffer_flag_C) * local_C_size, 0, local_C_size * sizeof(float));
                for(int c_K = 0; c_K < num_K; c_K++){
                    curr_blk_K = c_K < num_K - 1 ? blk_K : rem_blk_K;
                    if(c_N * num_M * num_K + c_M * num_K + c_K +1 < num_M * num_N * num_K){//still in local_now
                        next_blk_M = curr_blk_M;
                        next_blk_N = curr_blk_N;
                        next_blk_K = curr_blk_K;
                        if(c_K == num_K - 2){
                            next_blk_K = rem_blk_K;
                        }
                        if(c_K == num_K - 1){
                            next_blk_K = blk_K;
                            if(c_N == num_N - 2){
                                next_blk_N = rem_blk_N;
                            }
                            if(c_N == num_N - 1){
                                next_blk_N = blk_N;
                                if(c_M == num_M - 2){
                                    next_blk_M = rem_blk_M;
                                }
                                if(c_M == num_M - 1){//to local_now + 1
                                    next_blk_M = blk_M;
                                }
                            }
                        }
                        if(c_K == num_K - 1){
                            if(c_N == num_N - 1){
                                athread_dma_iget_stride(local_A + (double_buffer_flag_AB * local_A_size),
                                                        start_A + (c_M + 1) * blk_M * K + 0 * blk_K,
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
                            }
                            else{
                                athread_dma_iget_stride(local_A + (double_buffer_flag_AB * local_A_size),
                                                        start_A + (c_M * blk_M * K) + 0 * blk_K,
                                                        sizeof(float) * next_blk_M * next_blk_K,
                                                        sizeof(float) * next_blk_K,
                                                        sizeof(float) * (K - next_blk_K),
                                                        &reply_get_A);
                                athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_B_size),
                                                        start_B + 0 * blk_K * N + (c_N + 1) * blk_N,
                                                        sizeof(float) * next_blk_K * next_blk_N,
                                                        sizeof(float) * next_blk_N,
                                                        sizeof(float) * (N - next_blk_N),
                                                        &reply_get_B);
                            }
                        }
                        else{
                            athread_dma_iget_stride(local_A + (double_buffer_flag_AB * local_A_size),
                                                    start_A + (c_M * blk_M * K) + (c_K +1) * blk_K,
                                                    sizeof(float) * next_blk_M * next_blk_K,
                                                    sizeof(float) * next_blk_K,
                                                    sizeof(float) * (K - next_blk_K),
                                                    &reply_get_A);
                            athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_B_size),
                                                    start_B + (c_K + 1) * blk_K * N + (c_N * blk_N),
                                                    sizeof(float) * next_blk_K * next_blk_N,
                                                    sizeof(float) * next_blk_N,
                                                    sizeof(float) * (N - next_blk_N),
                                                    &reply_get_B);
                        }
                    }
                    else if(local_now < local_end - 1){
                        athread_dma_iget_stride(local_A + (double_buffer_flag_AB * local_A_size), 
                                                start_A + MK_size, 
                                                sizeof(float) * blk_M * blk_K, 
                                                sizeof(float) * blk_K, 
                                                sizeof(float) * (K - blk_K),
                                                &reply_get_A);
                        athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_B_size), 
                                                start_B + KN_size, 
                                                sizeof(float) * blk_K * blk_N, 
                                                sizeof(float) * blk_N, 
                                                sizeof(float) * (N - blk_N),
                                                &reply_get_B);
                    }
                    for(int m = 0; m < curr_blk_M; m++)
                        for(int n = 0; n < curr_blk_N; n++)
                            for(int k = 0; k < curr_blk_K; k++){
                                local_C[(1 - double_buffer_flag_C) * local_C_size + m * curr_blk_N + n]
                             += local_A[(1 - double_buffer_flag_AB) * local_A_size + m * curr_blk_K + k]
                              * local_B[(1 - double_buffer_flag_AB) * local_B_size + k * curr_blk_N + n];
                            }
                    //gemm
                    if(c_N * num_M * num_K + c_M * num_K + c_K +1 < num_M * num_N * num_K || local_now < local_end - 1){
                        athread_dma_wait_value(&reply_get_A, 1);
                        athread_dma_wait_value(&reply_get_B, 1);
                        reply_get_A = 0;
                        reply_get_B = 0;
                        double_buffer_flag_AB = 1 - double_buffer_flag_AB;
                    }
                }
                /* athread_dma_wait_value(&reply_put_C, 1);
                reply_put_C = 0;
                athread_dma_iput_stride(start_C + c_M * blk_M * N + c_N * blk_N,
                                        local_C + (1 - double_buffer_flag_C) * local_C_size,
                                        sizeof(float) * curr_blk_M * curr_blk_N,
                                        sizeof(float) * curr_blk_N,
                                        sizeof(float) * (N - curr_blk_N),
                                        &reply_put_C);
                double_buffer_flag_C = 1 - double_buffer_flag_C; */

                //Without Double buffer
                athread_dma_put_stride(start_C + c_M * blk_M * N + c_N * blk_N,
                                       local_C + (1 - double_buffer_flag_C) * local_C_size,
                                       sizeof(float) * curr_blk_M * curr_blk_N,
                                       sizeof(float) * curr_blk_N,
                                       sizeof(float) * (N - curr_blk_N));
            }
        }
    }
    ldm_free(local_A,sizeof(float) * blk_M * blk_K * 2);
    ldm_free(local_B,sizeof(float) * blk_K * blk_N * 2);
    ldm_free(local_C,sizeof(float) * blk_M * blk_N * 2);
}

void sw_slave_bmm_rcr(sw_bmmPara *_){
    const int myid = CRTS_cgn * 64 + CRTS_tid;
    sw_bmmPara *para = (sw_bmmPara *)para_cross;
    const float *src_A = para->A;
    const float *src_B = para->B;
    const float *src_C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    size_t blk_M = para->blk_M;
    size_t blk_N = para->blk_N;
    size_t blk_K = para->blk_K;
    size_t counts = para->counts;
    const int local_count = (counts + 383) / 384;
    const int local_start = myid * local_count;
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
                            sizeof(float) * blk_K, 
                            sizeof(float) * (K - blk_K),
                            &reply_get_B);
    athread_dma_wait_value(&reply_get_A, 1);
    athread_dma_wait_value(&reply_get_B, 1);
    reply_get_A = 0;
    reply_get_B = 0;
    for(int local_now = local_start; local_now < local_end; ++local_now){
        start_A = src_A + MK_size * local_now;
        start_B = src_B + KN_size * local_now;
        start_C = src_C + MN_size * local_now;
        for(int c_M = 0; c_M < num_M; c_M++){//K N M order
            curr_blk_M = c_M < num_M - 1 ? blk_M : rem_blk_M;
            for(int c_N = 0; c_N < num_N; c_N++){
                curr_blk_N = c_N < num_N - 1 ? blk_N : rem_blk_N;
                memset(local_C + (1 - double_buffer_flag_C) * local_C_size, 0, local_C_size * sizeof(float));
                for(int c_K = 0; c_K < num_K; c_K++){
                    curr_blk_K = c_K < num_K - 1 ? blk_K : rem_blk_K;
                    if(c_N * num_M * num_K + c_M * num_K + c_K +1 < num_M * num_N * num_K){//still in local_now
                        next_blk_M = curr_blk_M;
                        next_blk_N = curr_blk_N;
                        next_blk_K = curr_blk_K;
                        if(c_K == num_K - 2){
                            next_blk_K = rem_blk_K;
                        }
                        if(c_K == num_K - 1){
                            next_blk_K = blk_K;
                            if(c_N == num_N - 2){
                                next_blk_N = rem_blk_N;
                            }
                            if(c_N == num_N - 1){
                                next_blk_N = blk_N;
                                if(c_M == num_M - 2){
                                    next_blk_M = rem_blk_M;
                                }
                                if(c_M == num_M - 1){//to local_now + 1
                                    next_blk_M = blk_M;
                                }
                            }
                        }
                        if(c_K == num_K - 1){
                            if(c_N == num_N - 1){
                                athread_dma_iget_stride(local_A + (double_buffer_flag_AB * local_A_size),
                                                        start_A + (c_M + 1) * blk_M * K,
                                                        sizeof(float) * next_blk_M * next_blk_K,
                                                        sizeof(float) * next_blk_K,
                                                        sizeof(float) * (K - next_blk_K),
                                                        &reply_get_A);
                                athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_B_size),
                                                        start_B,
                                                        sizeof(float) * next_blk_K * next_blk_N,
                                                        sizeof(float) * next_blk_K,
                                                        sizeof(float) * (K - next_blk_K),
                                                        &reply_get_B);
                            }
                            else{
                                athread_dma_iget_stride(local_A + (double_buffer_flag_AB * local_A_size),
                                                        start_A + c_M * blk_M * K,
                                                        sizeof(float) * next_blk_M * next_blk_K,
                                                        sizeof(float) * next_blk_K,
                                                        sizeof(float) * (K - next_blk_K),
                                                        &reply_get_A);
                                athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_B_size),
                                                        start_B + (c_N + 1) * blk_N * K,
                                                        sizeof(float) * next_blk_K * next_blk_N,
                                                        sizeof(float) * next_blk_K,
                                                        sizeof(float) * (K - next_blk_K),
                                                        &reply_get_B);
                            }
                        }
                        else{
                            athread_dma_iget_stride(local_A + (double_buffer_flag_AB * local_A_size),
                                                    start_A + c_M * blk_M * K + (c_K + 1) * blk_K,
                                                    sizeof(float) * next_blk_M * next_blk_K,
                                                    sizeof(float) * next_blk_K,
                                                    sizeof(float) * (K - next_blk_K),
                                                    &reply_get_A);
                            athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_B_size),
                                                    start_B + c_N * blk_N * K + (c_K + 1) * blk_K,
                                                    sizeof(float) * next_blk_K * next_blk_N,
                                                    sizeof(float) * next_blk_K,
                                                    sizeof(float) * (K - next_blk_K),
                                                    &reply_get_B);
                        }
                    }
                    else if(local_now < local_end - 1){
                        athread_dma_iget_stride(local_A + (double_buffer_flag_AB * local_A_size), 
                                                start_A + MK_size, 
                                                sizeof(float) * blk_M * blk_K, 
                                                sizeof(float) * blk_K, 
                                                sizeof(float) * (K - blk_K),
                                                &reply_get_A);
                        athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_B_size), 
                                                start_B + KN_size, 
                                                sizeof(float) * blk_K * blk_N, 
                                                sizeof(float) * blk_K,
                                                sizeof(float) * (K - blk_K),
                                                &reply_get_B);
                    }
                    for(int m = 0; m < curr_blk_M; m++)
                        for(int n = 0; n < curr_blk_N; n++)
                            for(int k = 0; k < curr_blk_K; k++){
                                local_C[(1 - double_buffer_flag_C) * local_C_size + m * curr_blk_N + n]
                             += local_A[(1 - double_buffer_flag_AB) * local_A_size + m * curr_blk_K + k]
                              * local_B[(1 - double_buffer_flag_AB) * local_B_size + n * curr_blk_K + k];
                            }
                    //gemm
                    if(c_N * num_M * num_K + c_M * num_K + c_K +1 < num_M * num_N * num_K || local_now < local_end - 1){
                        athread_dma_wait_value(&reply_get_A, 1);
                        athread_dma_wait_value(&reply_get_B, 1);
                        reply_get_A = 0;
                        reply_get_B = 0;
                        double_buffer_flag_AB = 1 - double_buffer_flag_AB;
                    }
                }
                /* athread_dma_wait_value(&reply_put_C, 1);
                reply_put_C = 0;
                athread_dma_iput_stride(start_C + c_M * blk_M * N + c_N * blk_N,
                                        local_C + (1 - double_buffer_flag_C) * local_C_size,
                                        sizeof(float) * curr_blk_M * curr_blk_N,
                                        sizeof(float) * curr_blk_N,
                                        sizeof(float) * (N - curr_blk_N),
                                        &reply_put_C);
                double_buffer_flag_C = 1 - double_buffer_flag_C; */

                //Without Double buffer
                athread_dma_put_stride(start_C + c_M * blk_M * N + c_N * blk_N,
                                       local_C + (1 - double_buffer_flag_C) * local_C_size,
                                       sizeof(float) * curr_blk_M * curr_blk_N,
                                       sizeof(float) * curr_blk_N,
                                       sizeof(float) * (N - curr_blk_N));
            }
        }
    }
    ldm_free(local_A,sizeof(float) * blk_M * blk_K * 2);
    ldm_free(local_B,sizeof(float) * blk_K * blk_N * 2);
    ldm_free(local_C,sizeof(float) * blk_M * blk_N * 2);
}

void sw_slave_bmm_crr(sw_bmmPara *_){
    const int myid = CRTS_cgn * 64 + CRTS_tid;
    sw_bmmPara *para = (sw_bmmPara *)para_cross;
    const float *src_A = para->A;
    const float *src_B = para->B;
    const float *src_C = para->C;
    size_t M = para->M;
    size_t N = para->N;
    size_t K = para->K;
    size_t blk_M = para->blk_M;
    size_t blk_N = para->blk_N;
    size_t blk_K = para->blk_K;
    size_t counts = para->counts;
    const int local_count = (counts + 383) / 384;
    const int local_start = myid * local_count;
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
                            sizeof(float) * blk_M, 
                            sizeof(float) * (M - blk_M),
                            &reply_get_A);
    athread_dma_iget_stride(local_B + (1 - double_buffer_flag_AB) * local_B_size, 
                            start_B, 
                            sizeof(float) * blk_K * blk_N, 
                            sizeof(float) * blk_N, 
                            sizeof(float) * (N - blk_N),
                            &reply_get_B);
    athread_dma_wait_value(&reply_get_A, 1);
    athread_dma_wait_value(&reply_get_B, 1);
    reply_get_A = 0;
    reply_get_B = 0;
    for(int local_now = local_start; local_now < local_end; ++local_now){
        start_A = src_A + MK_size * local_now;
        start_B = src_B + KN_size * local_now;
        start_C = src_C + MN_size * local_now;
        for(int c_M = 0; c_M < num_M; c_M++){//K N M order
            curr_blk_M = c_M < num_M - 1 ? blk_M : rem_blk_M;
            for(int c_N = 0; c_N < num_N; c_N++){
                curr_blk_N = c_N < num_N - 1 ? blk_N : rem_blk_N;
                memset(local_C + (1 - double_buffer_flag_C) * local_C_size, 0, local_C_size * sizeof(float));
                for(int c_K = 0; c_K < num_K; c_K++){
                    curr_blk_K = c_K < num_K - 1 ? blk_K : rem_blk_K;
                    if(c_N * num_M * num_K + c_M * num_K + c_K +1 < num_M * num_N * num_K){//still in local_now
                        next_blk_M = curr_blk_M;
                        next_blk_N = curr_blk_N;
                        next_blk_K = curr_blk_K;
                        if(c_K == num_K - 2){
                            next_blk_K = rem_blk_K;
                        }
                        if(c_K == num_K - 1){
                            next_blk_K = blk_K;
                            if(c_N == num_N - 2){
                                next_blk_N = rem_blk_N;
                            }
                            if(c_N == num_N - 1){
                                next_blk_N = blk_N;
                                if(c_M == num_M - 2){
                                    next_blk_M = rem_blk_M;
                                }
                                if(c_M == num_M - 1){//to local_now + 1
                                    next_blk_M = blk_M;
                                }
                            }
                        }
                        if(c_K == num_K - 1){
                            if(c_N == num_N - 1){
                                athread_dma_iget_stride(local_A + (double_buffer_flag_AB * local_A_size),
                                                        start_A + (c_M + 1) * blk_M,
                                                        sizeof(float) * next_blk_M * next_blk_K,
                                                        sizeof(float) * next_blk_M,
                                                        sizeof(float) * (M - next_blk_M),
                                                        &reply_get_A);
                                athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_B_size),
                                                        start_B,
                                                        sizeof(float) * next_blk_K * next_blk_N,
                                                        sizeof(float) * next_blk_N,
                                                        sizeof(float) * (N - next_blk_N),
                                                        &reply_get_B);
                            }
                            else{
                                athread_dma_iget_stride(local_A + (double_buffer_flag_AB * local_A_size),
                                                        start_A + c_M * blk_M,
                                                        sizeof(float) * next_blk_M * next_blk_K,
                                                        sizeof(float) * next_blk_M,
                                                        sizeof(float) * (M - next_blk_M),
                                                        &reply_get_A);
                                athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_B_size),
                                                        start_B + 0 * blk_K * N + (c_N + 1) * blk_N,
                                                        sizeof(float) * next_blk_K * next_blk_N,
                                                        sizeof(float) * next_blk_N,
                                                        sizeof(float) * (N - next_blk_N),
                                                        &reply_get_B);
                            }
                        }
                        else{
                            athread_dma_iget_stride(local_A + (double_buffer_flag_AB * local_A_size),
                                                    start_A + (c_K + 1) * blk_K * M + c_M * blk_M,
                                                    sizeof(float) * next_blk_M * next_blk_K,
                                                    sizeof(float) * next_blk_M,
                                                    sizeof(float) * (M - next_blk_M),
                                                    &reply_get_A);
                            athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_B_size),
                                                    start_B + (c_K + 1) * blk_K * N + c_N * blk_N,
                                                    sizeof(float) * next_blk_K * next_blk_N,
                                                    sizeof(float) * next_blk_N,
                                                    sizeof(float) * (N - next_blk_N),
                                                    &reply_get_B);
                        }
                    }
                    else if(local_now < local_end - 1){
                        athread_dma_iget_stride(local_A + (double_buffer_flag_AB * local_A_size), 
                                                start_A + MK_size, 
                                                sizeof(float) * blk_M * blk_K, 
                                                sizeof(float) * blk_M, 
                                                sizeof(float) * (M - blk_M),
                                                &reply_get_A);
                        athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_B_size), 
                                                start_B + KN_size, 
                                                sizeof(float) * blk_K * blk_N, 
                                                sizeof(float) * blk_N, 
                                                sizeof(float) * (N - blk_N),
                                                &reply_get_B);
                    }
                    for(int m = 0; m < curr_blk_M; m++)
                        for(int n = 0; n < curr_blk_N; n++)
                            for(int k = 0; k < curr_blk_K; k++){
                                local_C[(1 - double_buffer_flag_C) * local_C_size + m * curr_blk_N + n]
                             += local_A[(1 - double_buffer_flag_AB) * local_A_size + k * curr_blk_M + m]
                              * local_B[(1 - double_buffer_flag_AB) * local_B_size + k * curr_blk_N + n];
                            }
                    //gemm
                    if(c_N * num_M * num_K + c_M * num_K + c_K +1 < num_M * num_N * num_K || local_now < local_end - 1){
                        athread_dma_wait_value(&reply_get_A, 1);
                        athread_dma_wait_value(&reply_get_B, 1);
                        reply_get_A = 0;
                        reply_get_B = 0;
                        double_buffer_flag_AB = 1 - double_buffer_flag_AB;
                    }
                }
                /* athread_dma_wait_value(&reply_put_C, 1);
                reply_put_C = 0;
                athread_dma_iput_stride(start_C + c_M * blk_M * N + c_N * blk_N,
                                        local_C + (1 - double_buffer_flag_C) * local_C_size,
                                        sizeof(float) * curr_blk_M * curr_blk_N,
                                        sizeof(float) * curr_blk_N,
                                        sizeof(float) * (N - curr_blk_N),
                                        &reply_put_C);
                double_buffer_flag_C = 1 - double_buffer_flag_C; */

                //Without Double buffer
                athread_dma_put_stride(start_C + c_M * blk_M * N + c_N * blk_N,
                                       local_C + (1 - double_buffer_flag_C) * local_C_size,
                                       sizeof(float) * curr_blk_M * curr_blk_N,
                                       sizeof(float) * curr_blk_N,
                                       sizeof(float) * (N - curr_blk_N));
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
