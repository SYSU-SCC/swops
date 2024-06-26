#include <crts.h>
#include <simd.h>
#include <slave.h>
#include <stdio.h>
#include <stdlib.h>

#include "args.h"

__cross void *para_cross = NULL;

__cross void *para_cross_1 = NULL;

__cross void *para_cross_2 = NULL;

#define thread_num (64 * 6)


#define sgemm_8_8_8(A, B, C, K, N)\
({\
	asm volatile(\
	"vlds $32, 0(%0)\n"/*A0*/\
		"s4addl %3, %0, $1\n"/*&A1 at $1*/\
	"vlds $40, 0(%1)\n"/*B0*/\
		"s8addl %4, %2, $5\n"/*&C1 at $5*/\
	"vldd $42, 0(%2)\n"/*C0*/\
		"s4addl %3, $1, $2\n"/*&A2 at $2*/\
	"vlds $33, 0($1)\n"/*A1*/\
		"s8addl %4, $5, $6\n"/*&C2 at $6*/\
	"vldd $43, 0($5)\n"/*C1*/\
		"s4addl %3, $2, $3\n"/*&A3 at $3*/\
	"vlds $34, 0($2)\n"/*A2*/\
		"s8addl %4, $6, $7\n"/*&C3 at $7*/\
	"vldd $44, 0($6)\n"/*C2*/\
		"s4addl %3, $3, $4\n"/*&A4 at $4*/\
	"vlds $35, 0($3)\n"/*A3*/\
		"s8addl %4, $7, $8\n"/*&C4 at $8*/\
	"ldi $9,7($31)\n"  /*get interval*/\
		"s4addl %3, $4, $1\n"/*&A5 at $1*/\
	"wcsr $9, 0x92\n"  /*set interval*/\
		"s8addl %4, $8, $5\n"/*&C5 at $5*/\
	"vldd $45, 0($7)\n"/*C3*/\
		"s4addl %3, $1, $2\n"/*&A6 at $2*/\
	"vlds $36, 0($4)\n"/*A4*/\
		"s8addl %4, $5, $6\n"/*&C6 at $6*/\
	"vldd $46, 0($8)\n"/*C4*/\
		"s4addl %3, $2, $3\n"/*&A7 at $3*/\
	"vlds $37, 0($1)\n"/*A5*/\
		"s8addl %4, $6, $7\n"/*&C7 at $7*/\
	"vldd $47, 0($5)\n"/*C5*/\
		"vlenma $32, $40, $42, $42\n"\
	"vlds $38, 0($2)\n"/*A6*/\
		"vlenma $33, $40, $43, $43\n"\
	"vldd $48, 0($6)\n"/*C6*/\
		"vlenma $34, $40, $44, $44\n"\
	"vlds $39, 0($3)\n"/*A7*/\
		"vlenma $35, $40, $45, $45\n"\
	"vldd $49, 0($7)\n"/*C7*/\
		"vlenma $36, $40, $46, $46\n"\
	"s4addl %4, %1, $9\n"/*&B1 at $9*/\
		"vlenma $37, $40, $47, $47\n"\
	"vlds $41, 0($9)\n"/*B1*/\
		"vlenma $38, $40, $48, $48\n"\
	"s4addl %4, $9, $10\n"/*&B2 at $10*/\
		"vlenma $39, $40, $49, $49\n"\
	"ldi $1,0(%2)\n"   /*get CSR_VLENMAS_STADDR*/\
		"vlenma $32, $41, $42, $42\n"\
	"vlds $40, 0($10)\n"/*B2*/\
		"vlenma $33, $41, $43, $43\n"\
	"wcsr $1, 0x90\n"  /*set CSR_VLENMAS_STADDR*/\
		"vlenma $34, $41, $44, $44\n"\
	"s8addl %4, $31, $2\n"\
		"vlenma $35, $41, $45, $45\n"\
	"wcsr $2, 0x91\n"\
		"vlenma $36, $41, $46, $46\n"\
	"s4addl %4, $10, $9\n"/*&B3 at $9*/\
		"vlenma $37, $41, $47, $47\n"\
		"vlenma $38, $41, $48, $48\n"\
	"s4addl %4, $9, $10\n"/*&B4 at $10*/\
		"vlenma $39, $41, $49, $49\n"\
		"vlenma $32, $40, $42, $42\n"\
	"vlds $41, 0($9)\n"/*B3*/\
		"vlenma $33, $40, $43, $43\n"\
	"s4addl %4, $10, $9\n"/*&B5 at $9*/\
		"vlenma $34, $40, $44, $44\n"\
		"vlenma $35, $40, $45, $45\n"\
		"vlenma $36, $40, $46, $46\n"\
		"vlenma $37, $40, $47, $47\n"\
		"vlenma $38, $40, $48, $48\n"\
		"vlenma $39, $40, $49, $49\n"\
		"vlenma $32, $41, $42, $42\n"\
	"vlds $40, 0($10)\n"/*B4*/\
		"vlenma $33, $41, $43, $43\n"\
	"s4addl %4, $9, $10\n"/*B6 at $10*/\
		"vlenma $34, $41, $44, $44\n"\
		"vlenma $35, $41, $45, $45\n"\
		"vlenma $36, $41, $46, $46\n"\
		"vlenma $37, $41, $47, $47\n"\
		"vlenma $38, $41, $48, $48\n"\
		"vlenma $39, $41, $49, $49\n"\
		"vlenma $32, $40, $42, $42\n"\
	"vlds $41, 0($9)\n"/*B5*/\
		"vlenma $33, $40, $43, $43\n"\
	"s4addl %4, $10, $9\n"/*B7 at $9*/\
		"vlenma $34, $40, $44, $44\n"\
		"vlenma $35, $40, $45, $45\n"\
		"vlenma $36, $40, $46, $46\n"\
		"vlenma $37, $40, $47, $47\n"\
		"vlenma $38, $40, $48, $48\n"\
		"vlenma $39, $40, $49, $49\n"\
		"vlenma $32, $41, $42, $42\n"\
	"vlds $40, 0($10)\n"/*B6*/\
		"vlenma $33, $41, $43, $43\n"\
		"vlenma $34, $41, $44, $44\n"\
		"vlenma $35, $41, $45, $45\n"\
		"vlenma $36, $41, $46, $46\n"\
		"vlenma $37, $41, $47, $47\n"\
		"vlenma $38, $41, $48, $48\n"\
		"vlenma $39, $41, $49, $49\n"\
		"vlenma $32, $40, $42, $42\n"\
	"vlds $41, 0($9)\n"/*B7*/\
		"vlenma $33, $40, $43, $43\n"\
		"vlenma $34, $40, $44, $44\n"\
		"vlenma $35, $40, $45, $45\n"\
		"vlenma $36, $40, $46, $46\n"\
		"vlenma $37, $40, $47, $47\n"\
		"vlenma $38, $40, $48, $48\n"\
		"vlenma $39, $40, $49, $49\n"\
		"vlenmas $32, $41, $42, $42\n"\
		"vlenmas $33, $41, $43, $43\n"\
		"vlenmas $34, $41, $44, $44\n"\
		"vlenmas $35, $41, $45, $45\n"\
		"vlenmas $36, $41, $46, $46\n"\
		"vlenmas $37, $41, $47, $47\n"\
		"vlenmas $38, $41, $48, $48\n"\
		"vlenmas $39, $41, $49, $49\n"\
	:\
	:"r"(A),"r"(B),"r"(C),"r"(K),"r"(N)\
	:"$32","$33","$34","$35","$36","$37","$38","$39",/*A*/\
	 "$40","41",/*B*/\
	 "$42","$43","$44","$45","$46","$47","$48","$49",/*C*/\
	 "$1","$2","$3","$4","$5","$6","$7","$8","$9","$10",\
	 "memory");\
	0;\
})

#define SWAP_8_8(a1, a2, a3, a4, a5, a6, a7, a8)    \
{                                                   \
    floatv8 b1 = simd_vshfs(a1, a2, shuffle1);      \
    floatv8 b2 = simd_vshfs(a1, a2, shuffle2);      \
    floatv8 b3 = simd_vshfs(a3, a4, shuffle1);      \
    floatv8 b4 = simd_vshfs(a3, a4, shuffle2);      \
    floatv8 b5 = simd_vshfs(a5, a6, shuffle1);      \
    floatv8 b6 = simd_vshfs(a5, a6, shuffle2);      \
    floatv8 b7 = simd_vshfs(a7, a8, shuffle1);      \
    floatv8 b8 = simd_vshfs(a7, a8, shuffle2);      \
    floatv8 c1 = simd_vshfs(b1, b3, shuffle3);      \
    floatv8 c2 = simd_vshfs(b1, b3, shuffle4);      \
    floatv8 c3 = simd_vshfs(b2, b4, shuffle3);      \
    floatv8 c4 = simd_vshfs(b2, b4, shuffle4);      \
    floatv8 c5 = simd_vshfs(b5, b7, shuffle3);      \
    floatv8 c6 = simd_vshfs(b5, b7, shuffle4);      \
    floatv8 c7 = simd_vshfs(b6, b8, shuffle3);      \
    floatv8 c8 = simd_vshfs(b6, b8, shuffle4);      \
    a1 = simd_vshfs(c1, c5, shuffle5);              \
    a2 = simd_vshfs(c1, c5, shuffle6);              \
    a3 = simd_vshfs(c2, c6, shuffle5);              \
    a4 = simd_vshfs(c2, c6, shuffle6);              \
    a5 = simd_vshfs(c3, c7, shuffle5);              \
    a6 = simd_vshfs(c3, c7, shuffle6);              \
    a7 = simd_vshfs(c4, c8, shuffle5);              \
    a8 = simd_vshfs(c4, c8, shuffle6);              \
}

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

    int blk_Q = (sli_A * local_rows * We * sizeof(float) < 200 * 1024) ? local_rows : 200 * 1024 / (sli_A * We * sizeof(float));

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

    int blk_Q = (local_rows * We * sizeof(float) < 200 * 1024) ? local_rows : 200 * 1024 / (We * sizeof(float));

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

    int blk_Q = (local_rows * We * sizeof(float) < 200 * 1024) ? local_rows : 200 * 1024 / (We * sizeof(float));

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

inline void sw_slave_gemm_copy_all_back_cgn(const int CGN_id,
                                            const float* A, const float* Ap,
                                            const int H, const int He,
                                            const int W, const int We){

    if(CRTS_cgn != CGN_id){
        return;
    }
    const int myid = CRTS_tid;
    /* if(H == He && W == We){
        printf("sw_slave_gemm_copy_all Don't Need Copy All!\n");
        return;
    } */
    if(myid < 0){
        //printf("_MYID %d sw_slave_gemm_copy_all myid %d return!\n", CRTS_cgn * 64 + CRTS_tid, myid);
        return;
    }

    const int counts_Q = H;
    
    const int local_count = (counts_Q + 64 - 1) / 64;
    const int local_start = myid * local_count;
    const int local_end = ((local_start + local_count > counts_Q) ? counts_Q : (local_start + local_count));
    const int local_rows = local_end - local_start;

    if(local_start >= counts_Q){
        return;
    }

    int blk_Q = (local_rows * We * sizeof(float) < 200 * 1024) ? local_rows : 200 * 1024 / (We * sizeof(float));

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

inline void sw_slave_gemm_copy_all_cgn(const int CGN_id,
                                        const float* A, const float* Ap,
                                        const int H, const int He,
                                        const int W, const int We){

    if(CRTS_cgn != CGN_id){
        return;
    }
    const int myid = CRTS_tid;
    /* if(H == He && W == We){
        printf("sw_slave_gemm_copy_all Don't Need Copy All!\n");
        return;
    } */
    if(myid < 0){
        //printf("_MYID %d sw_slave_gemm_copy_all myid %d return!\n", CRTS_cgn * 64 + CRTS_tid, myid);
        return;
    }

    const int counts_Q = He;
    const int local_count = (counts_Q + 64 - 1) / 64;
    const int local_start = myid * local_count;
    const int local_end = ((local_start + local_count > counts_Q) ? counts_Q : (local_start + local_count));
    const int local_rows = local_end - local_start;

    if(local_start >= counts_Q){
        return;
    }

    int blk_Q = (local_rows * We * sizeof(float) < 200 * 1024) ? local_rows : 200 * 1024 / (We * sizeof(float));

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
    const int M = para->M;
    const int Ms = para->Ms;
    const int Me = para->Me;
    const int N = para->N;
    const int Ns = para->Ns;
    const int Ne = para->Ne;
    const int K = para->K;
    const int Ks = para->Ks;
    const int Ke = para->Ke;
    const int blk_M = para->blk_M;
    const int blk_N = para->blk_N;
    const int blk_K = para->blk_K;
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
    
    int HW_size = H * W;
    if(CRTS_tid < 32){
        if(Ws == We){
            return;
        }
        const int CPY_tid = CRTS_tid;
        const int Cols_P = Hs / 32;
        const float* start_A = A + CPY_tid * Cols_P * W + Ws;
        const float* start_Ap = Ap + CPY_tid * Cols_P * We + Ws;
        const int blk_P = Cols_P * blk_W * sizeof(float) < (200 * 1024) ? Cols_P : (200 * 1024) / (blk_W * sizeof(float));//larger than 220 * 1024 will cause memory access error
        const int num_P = (Cols_P + blk_P - 1) / blk_P;
        const int rem_blk_P = Cols_P - num_P * blk_P == 0 ? blk_P : Cols_P - (num_P - 1) * blk_P;
        const int rem_blk_W = W - Ws;
        const int local_P_size = blk_P * blk_W;
        int curr_blk_P = blk_P;
        int next_blk_P = blk_P;
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
        const int Cols_Q = blk_H / 32;
        const int num_Q = Cols_Q;
        const int rem_blk_H = H - Hs;
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
    int HW_size = H*W;
    if(CRTS_tid < 32){
        //do dstp
        if(Ws == We){
            return;
        }
        const int CPY_tid = CRTS_tid;// 0 ~ 31
        const int Cols_P = Hs / 32;
        const float* src_P = src + CPY_tid * Cols_P * W + Ws;
        const float* dst_P = dst + CPY_tid * Cols_P * We + Ws;
        const int blk_P = Cols_P * blk_W * sizeof(float) < (200 * 1024) ? Cols_P : (200 * 1024) / (blk_W * sizeof(float));//larger than 220 * 1024 will cause memory access error
        const int num_P = (Cols_P + blk_P - 1) / blk_P;
        const int rem_blk_P = Cols_P - num_P * blk_P == 0 ? blk_P : Cols_P - (num_P - 1) * blk_P;
        const int rem_blk_W = W - Ws;
        const int local_P_size = blk_P * blk_W;
        int curr_blk_P = blk_P;
        int next_blk_P = blk_P;
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
        const int Cols_Q = blk_H / 32;
        const int num_Q = Cols_Q;
        const int rem_blk_H = H - Hs;
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

void sw_slave_gemm_crr_asm_cgn(const int CGN_id,
                                const float* A, const float* Ap,
                                const float* B, const float* Bp,
                                const float* C, const float* Cp,
                                const int M, const int Ms, const int Me, const int blk_M,
                                const int N, const int Ns, const int Ne, const int blk_N,
                                const int K, const int Ks, const int Ke, const int blk_K){
    if(CRTS_cgn != CGN_id){
        return;
    }
    const int cid = CRTS_tid % 8;
    const int rid = CRTS_tid / 8;

    const int Md = blk_M/8;
    const int Nd = blk_N/8;
    const int Kd = blk_K/8;

    const int num_M = (M + blk_M - 1) / blk_M;
    const int num_N = (N + blk_N - 1) / blk_N;
    const int num_K = (K + blk_K - 1) / blk_K;

    const int rem_blk_M = num_M * blk_M - M == 0 ? blk_M : M - (num_M-1) * blk_M;
    const int rem_blk_N = num_N * blk_N - N == 0 ? blk_N : N - (num_N-1) * blk_N;
    const int rem_blk_K = num_K * blk_K - K == 0 ? blk_K : K - (num_K-1) * blk_K;

    const int local_A_size = blk_M * blk_K / 64;
    const int local_B_size = blk_K * blk_N / 64;
    const int local_C_size = blk_M * blk_N / 64;

    float* local_A = (float*)ldm_malloc(sizeof(float) * 2 * blk_M * blk_K / 64);
    float* local_B = (float*)ldm_malloc(sizeof(float) * 2 * blk_K * blk_N / 64);
    double* local_C = (double*)ldm_malloc(sizeof(double) * blk_M * blk_N / 64);

    float* trans_A = (float*)ldm_malloc(sizeof(float) * blk_M * blk_K / 64);

    float* local_A_dma = (float*)ldm_malloc(sizeof(float) * 2 * blk_M * blk_K / 64);
    float* local_B_dma = (float*)ldm_malloc(sizeof(float) * 2 * blk_K * blk_N / 64);
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

    int A_step = M - blk_M/8;
    int B_step = N - blk_N/8;
    int C_step = N - blk_N/8;

    int curr_M = M;
    int curr_N = N;
    int curr_K = K;

    int curr_blk_M = blk_M;
    int curr_blk_N = blk_N;
    int curr_blk_K = blk_K;

    int next_A_offset = 0;
    int next_B_offset = 0;
    int next_C_offset = 0;

    int next_blk_M = blk_M;
    int next_blk_N = blk_N;
    int next_blk_K = blk_K;

    volatile athread_rply_t dma_get_A = 0, dma_get_B = 0, dma_put_C = 0;
    volatile athread_rply_t rma_local_A = 0, rma_local_B = 0;
    volatile athread_rply_t rma_A[8] = {0,0,0,0,0,0,0,0}, rma_B[8] = {0,0,0,0,0,0,0,0};

    volatile int double_buffer_A = 0, double_buffer_B = 0; //for rma
    volatile int double_buffer_DMA = 0;

    /* if(CRTS_tid ==0){
        printf("CGN %d\nM %d Ms %d Me %d\nN %d Ns %d Ne %d\nK %d Ks %d Ke %d\n", 
                CRTS_cgn, M, Ms, Me, N, Ns, Ne, K, Ks, Ke);
    } */

    athread_dma_iget_stride(local_A_dma + (1 - double_buffer_DMA) * local_A_size, 
                            start_A, 
                            sizeof(float) * local_A_size, 
                            sizeof(float) * blk_M/8, 
                            sizeof(float) * A_step,
                            &dma_get_A);
    athread_dma_iget_stride(local_B_dma + (1 - double_buffer_DMA) * local_B_size, 
                            start_B, 
                            sizeof(float) * local_B_size, 
                            sizeof(float) * blk_N/8, 
                            sizeof(float) * B_step,
                            &dma_get_B);

    athread_dma_wait_value(&dma_get_A, 1);
    athread_dma_wait_value(&dma_get_B, 1);
    dma_get_A = 0;
    dma_get_B = 0;

    intv16 shuffle1 = {0x8628C020, 0x6AD0A49C, 0xBD8E, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle2 = {0x96ACE128, 0xEEF1ACDE, 0xFF9E, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle3 = {0xA3018820, 0x4398A49C, 0xBDAB, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle4 = {0xB385A928, 0xC7B9ACDE, 0xFFBB, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle5 = {0x8A418820, 0x49CA3039, 0xBDAB, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle6 = {0x9AC5A928, 0xCDEB387B, 0xFFBB, 0,0,0,0,0,0,0,0,0,0,0,0,0};

    floatv8 a1,a2,a3,a4,a5,a6,a7,a8;

    doublev8 vec_double;
    floatv8 vec_float;

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
                    athread_dma_iget_stride(local_A_dma + double_buffer_DMA * local_A_size, 
                                            next_A + next_A_offset, 
                                            sizeof(float) * local_A_size, 
                                            sizeof(float) * blk_M/8, 
                                            sizeof(float) * A_step,
                                            &dma_get_A);
                    athread_dma_iget_stride(local_B_dma + double_buffer_DMA * local_B_size, 
                                            next_B + next_B_offset, 
                                            sizeof(float) * local_B_size, 
                                            sizeof(float) * blk_N/8, 
                                            sizeof(float) * B_step,
                                            &dma_get_B);//这个get要护
                }
                //trans A
                /* for(int k = 0; k < Kd; k++){
                    for(int m = 0; m < Md; m++){
                        trans_A[m * Kd + k] = local_A_dma[k * Md + m + (1 - double_buffer_DMA) * local_A_size];
                    }
                } */

                float* origin_A = local_A_dma + (1 - double_buffer_DMA) * local_A_size;
                for(int k = 0; k < Kd; k += 8){
                    for(int m = 0; m < Md; m += 8){
                        int index = k * Md + m;
                        simd_load(a1, origin_A + index);
                        index += Md;
                        simd_load(a2, origin_A + index);
                        index += Md;
                        simd_load(a3, origin_A + index);
                        index += Md;
                        simd_load(a4, origin_A + index);
                        index += Md;
                        simd_load(a5, origin_A + index);
                        index += Md;
                        simd_load(a6, origin_A + index);
                        index += Md;
                        simd_load(a7, origin_A + index);
                        index += Md;
                        simd_load(a8, origin_A + index);
                            
                        SWAP_8_8(a1,a2,a3,a4,a5,a6,a7,a8);

                        index = m * Kd + k;
                        simd_store(a1,trans_A + index);
                        index += Kd;
                        simd_store(a2,trans_A + index);
                        index += Kd;
                        simd_store(a3,trans_A + index);
                        index += Kd;
                        simd_store(a4,trans_A + index);
                        index += Kd;
                        simd_store(a5,trans_A + index);
                        index += Kd;
                        simd_store(a6,trans_A + index);
                        index += Kd;
                        simd_store(a7,trans_A + index);
                        index += Kd;
                        simd_store(a8,trans_A + index);
                    }
                }

                double_buffer_A = 0;
                double_buffer_B = 0;
                athread_ssync_array();
                if(cid == 0){
                    athread_rma_row_ibcast(local_A + double_buffer_A * local_A_size,
                                            trans_A,//local_A_dma + (1 - double_buffer_DMA) * local_A_size,
                                            sizeof(float) * local_A_size,
                                            &rma_local_A,
                                            &rma_A[cid]);
                }
                if(rid == 0){
                    athread_rma_col_ibcast(local_B + double_buffer_B * local_B_size,
                                            local_B_dma + (1 - double_buffer_DMA) * local_B_size,
                                            sizeof(float) * local_B_size,
                                            &rma_local_B,
                                            &rma_B[rid]);
                }

                for(int c_R = 0; c_R < 8; c_R++){

                    athread_rma_wait_value(&rma_A[c_R], 1);
                    athread_rma_wait_value(&rma_B[c_R], 1);

                    athread_ssync_array();//here must synchronization

                    double_buffer_A = 1 - double_buffer_A;
                    double_buffer_B = 1 - double_buffer_B;
                    rma_A[c_R] = 0;
                    rma_B[c_R] = 0;

                    if(cid == c_R + 1){
                        athread_rma_row_ibcast(local_A + double_buffer_A * local_A_size, 
                                                trans_A,//local_A_dma + (1 - double_buffer_DMA) * local_A_size,
                                                sizeof(float) * local_A_size,
                                                &rma_local_A,
                                                &rma_A[c_R+1]);
                    }
                    if(rid == c_R + 1){
                        athread_rma_col_ibcast(local_B + double_buffer_B * local_B_size,
                                                local_B_dma + (1 - double_buffer_DMA) * local_B_size,
                                                sizeof(float) * local_B_size,
                                                &rma_local_B,
                                                &rma_B[c_R+1]);
                    }

                    float* comp_A = local_A + (1 - double_buffer_A) * local_A_size;
                    float* comp_B = local_B + (1 - double_buffer_B) * local_B_size;
                    double* comp_C = local_C;

                    for(int m = 0; m < Md; m += 8){
                        for(int n = 0; n < Nd; n += 8){
                            for(int k = 0; k < Kd; k += 8){
                                sgemm_8_8_8(comp_A + m * Kd + k, 
                                            comp_B + k * Nd + n, 
                                            comp_C + m * Nd + n, 
                                            Kd, Nd);
                            }
                        }
                    }

                    /* for(int m = 0; m < Md; m++){
                        for(int n = 0; n < Nd; n++){
                            for(int k = 0; k < Kd; k++){
                                local_C_dma[m * Nd + n] += 
                                local_A[m * Kd + k + (1 - double_buffer_A) * local_A_size] *
                                local_B[k * Nd + n + (1 - double_buffer_B) * local_B_size];
                            }
                        }
                    } */
                }
                if(c_M * num_N * num_K + c_N * num_K + c_K + 1 < num_M * num_N * num_K){
                    athread_dma_wait_value(&dma_get_A, 1);
                    dma_get_A = 0;
                    rma_local_A = 0;
                    athread_dma_wait_value(&dma_get_B, 1);
                    dma_get_B = 0;
                    rma_local_B = 0;
                    double_buffer_DMA = 1 - double_buffer_DMA;
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
            /* for(int m = 0; m < (blk_M/8); m++){
                for(int n = 0; n < (blk_N/8); n++){
                    local_C_dma[m * (blk_N/8) + n] -= local_C[m * (blk_N/8) + n];
                }
            } */

            for(int i = 0; i < Md * Nd; i += 8){
                simd_load(vec_double, local_C + i);
                vec_float = - (floatv8)vec_double;
                simd_store(vec_float, local_C_dma + i);
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

    ldm_free(trans_A,sizeof(float) * blk_M * blk_K / 64);

    ldm_free(local_A_dma, sizeof(float) * 2 * blk_M * blk_K / 64);
    ldm_free(local_B_dma, sizeof(float) * 2 * blk_K * blk_N / 64);
    ldm_free(local_C_dma, sizeof(float) * blk_M * blk_N / 64);
}

void sw_slave_gemm_rrr_asm_cgn(const int CGN_id,
                                const float* A, const float* Ap,
                                const float* B, const float* Bp,
                                const float* C, const float* Cp,
                                const int M, const int Ms, const int Me, const int blk_M,
                                const int N, const int Ns, const int Ne, const int blk_N,
                                const int K, const int Ks, const int Ke, const int blk_K){
    if(CRTS_cgn != CGN_id){
        return;
    }

    const int cid = CRTS_tid % 8;
    const int rid = CRTS_tid / 8;

    const int Md = blk_M/8;
    const int Nd = blk_N/8;
    const int Kd = blk_K/8;

    const int num_M = (M + blk_M - 1) / blk_M;
    const int num_N = (N + blk_N - 1) / blk_N;
    const int num_K = (K + blk_K - 1) / blk_K;

    const int rem_blk_M = num_M * blk_M - M == 0 ? blk_M : M - (num_M-1) * blk_M;
    const int rem_blk_N = num_N * blk_N - N == 0 ? blk_N : N - (num_N-1) * blk_N;
    const int rem_blk_K = num_K * blk_K - K == 0 ? blk_K : K - (num_K-1) * blk_K;

    const int local_A_size = blk_M * blk_K / 64;
    const int local_B_size = blk_K * blk_N / 64;
    const int local_C_size = blk_M * blk_N / 64;

    float* local_A = (float*)ldm_malloc(sizeof(float) * 2 * blk_M * blk_K / 64);
    float* local_B = (float*)ldm_malloc(sizeof(float) * 2 * blk_K * blk_N / 64);
    double* local_C = (double*)ldm_malloc(sizeof(double) * blk_M * blk_N / 64);

    float* local_A_dma = (float*)ldm_malloc(sizeof(float) * 2 * blk_M * blk_K / 64);
    float* local_B_dma = (float*)ldm_malloc(sizeof(float) * 2 * blk_K * blk_N / 64);
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

    int A_step = K - blk_K/8;
    int B_step = N - blk_N/8;
    int C_step = N - blk_N/8;

    int curr_M = M;
    int curr_N = N;
    int curr_K = K;

    int curr_blk_M = blk_M;
    int curr_blk_N = blk_N;
    int curr_blk_K = blk_K;

    int next_A_offset = 0;
    int next_B_offset = 0;
    int next_C_offset = 0;

    int next_blk_M = blk_M;
    int next_blk_N = blk_N;
    int next_blk_K = blk_K;

    volatile athread_rply_t dma_get_A = 0, dma_get_B = 0, dma_put_C = 0;
    volatile athread_rply_t rma_local_A = 0, rma_local_B = 0;
    volatile athread_rply_t rma_A[8] = {0,0,0,0,0,0,0,0}, rma_B[8] = {0,0,0,0,0,0,0,0};

    int double_buffer_A = 0, double_buffer_B = 0; //for rma
    int double_buffer_DMA = 0;

    athread_dma_iget_stride(local_A_dma + (1 - double_buffer_DMA) * local_A_size, 
                            start_A, 
                            sizeof(float) * local_A_size, 
                            sizeof(float) * blk_K/8, 
                            sizeof(float) * A_step,
                            &dma_get_A);
    athread_dma_iget_stride(local_B_dma + (1 - double_buffer_DMA) * local_B_size, 
                            start_B, 
                            sizeof(float) * local_B_size, 
                            sizeof(float) * blk_N/8, 
                            sizeof(float) * B_step,
                            &dma_get_B);

    athread_dma_wait_value(&dma_get_A, 1);
    athread_dma_wait_value(&dma_get_B, 1);

    dma_get_A = 0;
    dma_get_B = 0;

    doublev8 vec_double;
    floatv8 vec_float;

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
                    athread_dma_iget_stride(local_A_dma + double_buffer_DMA * local_A_size, 
                                        next_A + next_A_offset, 
                                        sizeof(float) * local_A_size, 
                                        sizeof(float) * blk_K/8, 
                                        sizeof(float) * A_step,
                                        &dma_get_A);
                    athread_dma_iget_stride(local_B_dma + double_buffer_DMA * local_B_size, 
                                        next_B + next_B_offset, 
                                        sizeof(float) * local_B_size, 
                                        sizeof(float) * blk_N/8, 
                                        sizeof(float) * B_step,
                                        &dma_get_B);//这个get要护
                }
                double_buffer_A = 0;
                double_buffer_B = 0;
                athread_ssync_array();
                if(cid == 0){
                    athread_rma_row_ibcast(local_A + double_buffer_A * local_A_size, 
                                            local_A_dma + (1 - double_buffer_DMA) * local_A_size,
                                            sizeof(float) * local_A_size,
                                            &rma_local_A,
                                            &rma_A[cid]);
                }
                if(rid == 0){
                    athread_rma_col_ibcast(local_B + double_buffer_B * local_B_size,
                                            local_B_dma + (1 - double_buffer_DMA) * local_B_size,
                                            sizeof(float) * local_B_size,
                                            &rma_local_B,
                                            &rma_B[rid]);
                }

                for(int c_R = 0; c_R < 8; c_R++){

                    athread_rma_wait_value(&rma_A[c_R], 1);
                    athread_rma_wait_value(&rma_B[c_R], 1);
                    athread_ssync_array();//here must synchronization
                    double_buffer_A = 1 - double_buffer_A;
                    double_buffer_B = 1 - double_buffer_B;
                    rma_A[c_R] = 0;
                    rma_B[c_R] = 0;

                    if(cid == c_R + 1){
                        athread_rma_row_ibcast(local_A + double_buffer_A * local_A_size, 
                                                local_A_dma + (1 - double_buffer_DMA) * local_A_size,
                                                sizeof(float) * local_A_size,
                                                &rma_local_A,
                                                &rma_A[c_R+1]);
                    }
                    if(rid == c_R + 1){
                        athread_rma_col_ibcast(local_B + double_buffer_B * local_B_size,
                                                local_B_dma + (1 - double_buffer_DMA) * local_B_size,
                                                sizeof(float) * local_B_size,
                                                &rma_local_B,
                                                &rma_B[c_R+1]);
                    }

                    /* for(int m = 0; m < (blk_M/8); m++){
                        for(int n = 0; n < (blk_N/8); n++){
                            for(int k = 0; k < (blk_K/8); k++){
                                local_C_dma[m * (blk_N/8) + n] += 
                                local_A[m * (blk_K/8) + k + (1 - double_buffer_A) * local_A_size] *
                                local_B[k * (blk_N/8) + n + (1 - double_buffer_B) * local_B_size];
                            }
                        }
                    } */


                    //sgemm_8_8_8(A, B, C, K, N);
                    float* comp_A = local_A + (1 - double_buffer_A) * local_A_size;
                    float* comp_B = local_B + (1 - double_buffer_B) * local_B_size;
                    double* comp_C = local_C;

                    for(int m = 0; m < Md; m += 8){
                        for(int n = 0; n < Nd; n += 8){
                            for(int k = 0; k < Kd; k += 8){
                                sgemm_8_8_8(comp_A + m * Kd + k, 
                                            comp_B + k * Nd + n, 
                                            comp_C + m * Nd + n, 
                                            Kd, Nd);
                            }
                        }
                    }
                }
                if(c_M * num_N * num_K + c_N * num_K + c_K + 1 < num_M * num_N * num_K){
                    athread_dma_wait_value(&dma_get_A, 1);
                    dma_get_A = 0;
                    rma_local_A = 0;
                    athread_dma_wait_value(&dma_get_B, 1);
                    dma_get_B = 0;
                    rma_local_B = 0;
                    double_buffer_DMA = 1 - double_buffer_DMA;
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

            /* for(int m = 0; m < Md; m++){
                for(int n = 0; n < Nd; n++){
                    local_C_dma[m * Nd + n] -= local_C[m * Nd + n];
                }
            } */

            for(int i = 0; i < Md * Nd; i += 8){
                simd_load(vec_double, local_C + i);
                vec_float = - (floatv8)vec_double;
                simd_store(vec_float, local_C_dma + i);
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
    ldm_free(local_A_dma, sizeof(float) * 2 * blk_M * blk_K / 64);
    ldm_free(local_B_dma, sizeof(float) * 2 * blk_K * blk_N / 64);
    ldm_free(local_C_dma, sizeof(float) * blk_M * blk_N / 64);
}

void sw_slave_gemm_rcr_asm_cgn(const int CGN_id,
                                const float* A, const float* Ap,
                                const float* B, const float* Bp,
                                const float* C, const float* Cp,
                                const int M, const int Ms, const int Me, const int blk_M,
                                const int N, const int Ns, const int Ne, const int blk_N,
                                const int K, const int Ks, const int Ke, const int blk_K){
    if(CRTS_cgn != CGN_id){
        return;
    }

    const int cid = CRTS_tid % 8;
    const int rid = CRTS_tid / 8;

    const int Md = blk_M/8;
    const int Nd = blk_N/8;
    const int Kd = blk_K/8;

    const int num_M = (M + blk_M - 1) / blk_M;
    const int num_N = (N + blk_N - 1) / blk_N;
    const int num_K = (K + blk_K - 1) / blk_K;

    const int rem_blk_M = num_M * blk_M - M == 0 ? blk_M : M - (num_M-1) * blk_M;
    const int rem_blk_N = num_N * blk_N - N == 0 ? blk_N : N - (num_N-1) * blk_N;
    const int rem_blk_K = num_K * blk_K - K == 0 ? blk_K : K - (num_K-1) * blk_K;

    const int local_A_size = blk_M * blk_K / 64;
    const int local_B_size = blk_K * blk_N / 64;
    const int local_C_size = blk_M * blk_N / 64;

    float* local_A = (float*)ldm_malloc(sizeof(float) * 2 * blk_M * blk_K / 64);
    float* local_B = (float*)ldm_malloc(sizeof(float) * 2 * blk_K * blk_N / 64);
    double* local_C = (double*)ldm_malloc(sizeof(double) * blk_M * blk_N / 64);

    float* trans_B = (float*)ldm_malloc(sizeof(float) * blk_K * blk_N / 64);

    float* local_A_dma = (float*)ldm_malloc(sizeof(float) * 2 * blk_M * blk_K / 64);
    float* local_B_dma = (float*)ldm_malloc(sizeof(float) * 2 * blk_K * blk_N / 64);
    float* local_C_dma = (float*)ldm_malloc(sizeof(float) * blk_M * blk_N / 64);

    const float* start_A = A + rid * blk_M/8 * K + cid * blk_K/8;
    const float* start_B = B + cid * blk_N/8 * K + rid * blk_K/8;
    const float* start_C = C + rid * blk_M/8 * N + cid * blk_N/8;

    const float* start_Ap = Ap + rid * blk_M/8 * Ke + cid * blk_K/8;
    const float* start_Bp = Bp + cid * blk_N/8 * Ke + rid * blk_K/8;
    const float* start_Cp = Cp + rid * blk_M/8 * Ne + cid * blk_N/8;

    float* next_A = A;
    float* next_B = B;
    float* next_C = C;
    float* curr_C = C;

    int A_step = K - blk_K/8;
    int B_step = K - blk_K/8;
    int C_step = N - blk_N/8;

    int curr_M = M;
    int curr_N = N;
    int curA_K = K;
    int curB_K = K;

    int curr_blk_M = blk_M;
    int curr_blk_N = blk_N;
    int curr_blk_K = blk_K;

    int next_A_offset = 0;
    int next_B_offset = 0;
    int next_C_offset = 0;

    int next_blk_M = blk_M;
    int next_blk_N = blk_N;
    int next_blk_K = blk_K;

    volatile athread_rply_t dma_get_A = 0, dma_get_B = 0, dma_put_C = 0;
    volatile athread_rply_t rma_local_A = 0, rma_local_B = 0;
    volatile athread_rply_t rma_A[8] = {0,0,0,0,0,0,0,0}, rma_B[8] = {0,0,0,0,0,0,0,0};

    int double_buffer_A = 0, double_buffer_B = 0; //for rma
    int double_buffer_DMA = 0;

    athread_dma_iget_stride(local_A_dma + (1 - double_buffer_DMA) * local_A_size, 
                            start_A, 
                            sizeof(float) * local_A_size, 
                            sizeof(float) * blk_K/8, 
                            sizeof(float) * A_step,
                            &dma_get_A);
    athread_dma_iget_stride(local_B_dma + (1 - double_buffer_DMA) * local_B_size, 
                            start_B, 
                            sizeof(float) * local_B_size, 
                            sizeof(float) * blk_K/8, 
                            sizeof(float) * B_step,
                            &dma_get_B);

    athread_dma_wait_value(&dma_get_A, 1);
    athread_dma_wait_value(&dma_get_B, 1);

    dma_get_A = 0;
    dma_get_B = 0;

    intv16 shuffle1 = {0x8628C020, 0x6AD0A49C, 0xBD8E, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle2 = {0x96ACE128, 0xEEF1ACDE, 0xFF9E, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle3 = {0xA3018820, 0x4398A49C, 0xBDAB, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle4 = {0xB385A928, 0xC7B9ACDE, 0xFFBB, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle5 = {0x8A418820, 0x49CA3039, 0xBDAB, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle6 = {0x9AC5A928, 0xCDEB387B, 0xFFBB, 0,0,0,0,0,0,0,0,0,0,0,0,0};

    floatv8 a1,a2,a3,a4,a5,a6,a7,a8;

    doublev8 vec_double;
    floatv8 vec_float;

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
                        curA_K = Ke;
                        A_step = Ke - blk_K/8;
                    }
                    else{
                        next_A = start_A;
                        curA_K = K;
                        A_step = K - blk_K/8;
                    }
                    if(next_blk_K != blk_K || next_blk_N != blk_N){
                        next_B = start_Bp;
                        curB_K = Ke;
                        B_step = Ke - blk_K/8;
                    }
                    else{
                        next_B = start_B;
                        curB_K = K;
                        B_step = K - blk_K/8;
                    }
                    if(c_K == num_K - 1){
                        if(c_N == num_N - 1){
                            next_A_offset = (c_M + 1) * blk_M * curA_K;
                            next_B_offset = 0;
                        }
                        else{
                            next_A_offset = c_M * blk_M * curA_K;
                            next_B_offset = (c_N + 1) * blk_N * curB_K;
                        }
                    }
                    else{
                        next_A_offset = c_M * blk_M * curA_K + (c_K + 1) * blk_K;
                        next_B_offset = c_N * blk_N * curB_K + (c_K + 1) * blk_K;
                    }
                    athread_dma_iget_stride(local_A_dma + double_buffer_DMA * local_A_size, 
                                        next_A + next_A_offset, 
                                        sizeof(float) * local_A_size, 
                                        sizeof(float) * blk_K/8, 
                                        sizeof(float) * A_step,
                                        &dma_get_A);
                    athread_dma_iget_stride(local_B_dma + double_buffer_DMA * local_B_size, 
                                        next_B + next_B_offset, 
                                        sizeof(float) * local_B_size, 
                                        sizeof(float) * blk_K/8, 
                                        sizeof(float) * B_step,
                                        &dma_get_B);//这个get要护
                }
                //trans_B

                /* for(int n = 0; n < Nd; n++){
                    for(int k = 0; k < Kd; k++){
                        trans_B[k * Nd + n] = local_B_dma[n * Kd + k + (1 - double_buffer_DMA) * local_B_size];
                    }
                } */

                float* origin_B = local_B_dma + (1 - double_buffer_DMA) * local_B_size;
                for(int n = 0; n < Nd; n += 8){
                    for(int k = 0; k < Kd; k += 8){
                        int index = n * Kd + k;
                        simd_load(a1, origin_B + index);
                        index += Kd;
                        simd_load(a2, origin_B + index);
                        index += Kd;
                        simd_load(a3, origin_B + index);
                        index += Kd;
                        simd_load(a4, origin_B + index);
                        index += Kd;
                        simd_load(a5, origin_B + index);
                        index += Kd;
                        simd_load(a6, origin_B + index);
                        index += Kd;
                        simd_load(a7, origin_B + index);
                        index += Kd;
                        simd_load(a8, origin_B + index);

                        SWAP_8_8(a1,a2,a3,a4,a5,a6,a7,a8);

                        index = k * Nd + n;
                        simd_store(a1, trans_B + index);
                        index += Nd;
                        simd_store(a2, trans_B + index);
                        index += Nd;
                        simd_store(a3, trans_B + index);
                        index += Nd;
                        simd_store(a4, trans_B + index);
                        index += Nd;
                        simd_store(a5, trans_B + index);
                        index += Nd;
                        simd_store(a6, trans_B + index);
                        index += Nd;
                        simd_store(a7, trans_B + index);
                        index += Nd;
                        simd_store(a8, trans_B + index);
                    }
                }

                double_buffer_A = 0;
                double_buffer_B = 0;
                athread_ssync_array();
                if(cid == 0){
                    athread_rma_row_ibcast(local_A + double_buffer_A * local_A_size, 
                                            local_A_dma + (1 - double_buffer_DMA) * local_A_size,
                                            sizeof(float) * local_A_size,
                                            &rma_local_A,
                                            &rma_A[cid]);
                }
                if(rid == 0){
                    athread_rma_col_ibcast(local_B + double_buffer_B * local_B_size,
                                            trans_B,//local_B_dma + (1 - double_buffer_DMA) * local_B_size,
                                            sizeof(float) * local_B_size,
                                            &rma_local_B,
                                            &rma_B[rid]);
                }

                for(int c_R = 0; c_R < 8; c_R++){

                    athread_rma_wait_value(&rma_A[c_R], 1);
                    athread_rma_wait_value(&rma_B[c_R], 1);

                    athread_ssync_array();//here must synchronization

                    double_buffer_A = 1 - double_buffer_A;
                    double_buffer_B = 1 - double_buffer_B;
                    rma_A[c_R] = 0;
                    rma_B[c_R] = 0;

                    if(cid == c_R + 1){
                        athread_rma_row_ibcast(local_A + double_buffer_A * local_A_size, 
                                                local_A_dma + (1 - double_buffer_DMA) * local_A_size,
                                                sizeof(float) * local_A_size,
                                                &rma_local_A,
                                                &rma_A[c_R+1]);
                    }
                    if(rid == c_R + 1){
                        athread_rma_col_ibcast(local_B + double_buffer_B * local_B_size,
                                                trans_B,//local_B_dma + (1 - double_buffer_DMA) * local_B_size,
                                                sizeof(float) * local_B_size,
                                                &rma_local_B,
                                                &rma_B[c_R+1]);
                    }

                    /* for(int m = 0; m < Md; m++){
                        for(int n = 0; n < Nd; n++){
                            for(int k = 0; k < Kd; k++){
                                local_C_dma[m * Nd + n] +=
                                local_A[m * Kd + k + (1 - double_buffer_A) * local_A_size] *
                                local_B[k * Nd + n + (1 - double_buffer_B) * local_B_size];
                            }
                        }
                    } */
                    
                    /* for(int m = 0; m < Md; m++){
                        for(int n = 0; n < Nd; n++){
                            for(int k = 0; k < Kd; k++){
                                local_C_dma[m * Nd + n] +=
                                local_A[m * Kd + k + (1 - double_buffer_A) * local_A_size] *
                                local_B[n * Kd + k + (1 - double_buffer_B) * local_B_size];
                            }
                        }
                    } */


                    //sgemm_8_8_8(A, B, C, K, N);

                    float* comp_A = local_A + (1 - double_buffer_A) * local_A_size;
                    float* comp_B = local_B + (1 - double_buffer_B) * local_B_size;
                    double* comp_C = local_C;

                    for(int m = 0; m < Md; m += 8){
                        for(int n = 0; n < Nd; n += 8){
                            for(int k = 0; k < Kd; k += 8){
                                sgemm_8_8_8(comp_A + m * Kd + k, 
                                            comp_B + k * Nd + n, 
                                            comp_C + m * Nd + n, 
                                            Kd, Nd);
                            }
                        }
                    }
                }
                if(c_M * num_N * num_K + c_N * num_K + c_K + 1 < num_M * num_N * num_K){
                    athread_dma_wait_value(&dma_get_A, 1);
                    dma_get_A = 0;
                    rma_local_A = 0;
                    athread_dma_wait_value(&dma_get_B, 1);
                    dma_get_B = 0;
                    rma_local_B = 0;
                    double_buffer_DMA = 1 - double_buffer_DMA;
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

            /* for(int m = 0; m < Md; m++){
                for(int n = 0; n < Nd; n++){
                    local_C_dma[m * Nd + n] -= local_C[m * Nd + n];
                }
            } */

            for(int i = 0; i < Md * Nd; i += 8){
                simd_load(vec_double, local_C + i);
                vec_float = - (floatv8)vec_double;
                simd_store(vec_float, local_C_dma + i);
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

    ldm_free(trans_B, sizeof(float) * blk_K * blk_N / 64);

    ldm_free(local_A_dma, sizeof(float) * 2 * blk_M * blk_K / 64);
    ldm_free(local_B_dma, sizeof(float) * 2 * blk_K * blk_N / 64);
    ldm_free(local_C_dma, sizeof(float) * blk_M * blk_N / 64);
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
    const int M = para->M;
    const int Ms = para->Ms;
    const int Me = para->Me;
    const int N = para->N;
    const int Ns = para->Ns;
    const int Ne = para->Ne;
    const int K = para->K;
    const int Ks = para->Ks;
    const int Ke = para->Ke;
    const int blk_M = para->blk_M;
    const int blk_N = para->blk_N;
    const int blk_K = para->blk_K;
    const int sli_C = para->sli_C;
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
    const int K_cgn = para->sli_K[CRTS_cgn];
    athread_ssync_node();
#ifdef _SWOPS_DEBUG
    if(CRTS_tid == 0){
        printf("CRTS_cgn %d K_cgn %d\n",CRTS_cgn,K_cgn);
        printf("sli_C %d\n", sli_C);
    }
#endif
    if(K_cgn > 0){
        sw_slave_gemm_crr_asm_cgn(CRTS_cgn,
                                    A_cgn, A_cgn, 
                                    B_cgn, B_cgn, 
                                    C_cgn, C_cgn,
                                    Me, Me, Me, blk_M,
                                    Ne, Ne, Ne, blk_N,
                                    K_cgn, K_cgn, K_cgn, blk_K);
    }
    athread_ssync_node();
    sw_slave_gemm_copy_all_back_add(0, 383, sli_C, C, Cp, M, Me, N, Ne);
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
    const int M = para->M;
    const int Ms = para->Ms;
    const int Me = para->Me;
    const int N = para->N;
    const int Ns = para->Ns;
    const int Ne = para->Ne;
    const int K = para->K;
    const int Ks = para->Ks;
    const int Ke = para->Ke;
    const int blk_M = para->blk_M;
    const int blk_N = para->blk_N;
    const int blk_K = para->blk_K;
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
        sw_slave_gemm_rrr_asm_cgn(CRTS_cgn,
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
}

void sw_slave_gemm_rcr_sli_cgn_f32(sw_gemmPara *_){
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
    const int M = para->M;
    const int Ms = para->Ms;
    const int Me = para->Me;
    const int N = para->N;
    const int Ns = para->Ns;
    const int Ne = para->Ne;
    const int K = para->K;
    const int Ks = para->Ks;
    const int Ke = para->Ke;
    const int blk_M = para->blk_M;
    const int blk_N = para->blk_N;
    const int blk_K = para->blk_K;
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
        sw_slave_gemm_copy_all(0, 383, B, Bp, N, Ne, K, Ke);//highDim lowDim
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
        sw_slave_gemm_rcr_asm_cgn(CRTS_cgn,
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
}
























void sw_slave_addm_thread(const int ThreadsStart, const int ThreadsEnd,
                            const float* A, const float* B, float* C,
                            const int MN_size){

    const int ThreadsNum = ThreadsEnd - ThreadsStart + 1;
    const int myid = CRTS_cgn * 64 + CRTS_tid - ThreadsStart;

    if(ThreadsNum <= 0){
        printf("sw_slave_gemm_copy_all ThreadsNum Error!\n");
        return;
    }
    if(myid < 0){
        //printf("_MYID %d sw_slave_gemm_copy_all myid %d return!\n", CRTS_cgn * 64 + CRTS_tid, myid);
        return;
    }

    const int counts_Q = MN_size;
    
    const int local_count = (counts_Q + ThreadsNum - 1) / ThreadsNum;
    const int local_start = myid * local_count;
    const int local_end = ((local_start + local_count > counts_Q) ? counts_Q : (local_start + local_count));
    
    const int local_adds = local_end - local_start;//how much add operation will be processed in this thread

    //printf("myid %d MN_size %d local_count %d local_start %d local_end %d\n", myid, MN_size, local_count, local_start, local_end);

    if(local_start >= counts_Q || myid >= ThreadsNum){
        return;
    }
    
    const int blk_Q = local_adds * sizeof(float) * 2 < 215 * 1024 ? local_adds : 215 * 1024 / (2 * sizeof(float));
    const int rem_Q = local_adds % blk_Q == 0 ? blk_Q : local_adds % blk_Q;

    float* local_A = ldm_malloc(sizeof(float) * ((blk_Q + 8 - 1) / 8) * 8);//padding to 8x for simd;
    float* local_B = ldm_malloc(sizeof(float) * ((blk_Q + 8 - 1) / 8) * 8);
    float* local_C = local_A;

    floatv8 vec_A, vec_B, vec_C;

    for(int local_now = local_start; local_now < local_end; local_now += blk_Q){
        int curr_Q = local_now + blk_Q < local_end ? blk_Q: rem_Q;
        athread_dma_get(local_A,
                        A + local_now,
                        sizeof(float) * curr_Q);
        athread_dma_get(local_B,
                        B + local_now,
                        sizeof(float) * curr_Q);
        /* for(int i = 0; i < curr_Q; i++){
            local_C[i] = local_A[i] + local_B[i];
        } */
        for(int i = 0; i < curr_Q; i += 8){
            simd_load(vec_A, local_A + i);
            simd_load(vec_B, local_B + i);
            vec_C = vec_A + vec_B;
            simd_store(vec_C, local_C + i);
        }
        athread_dma_put(C + local_now,
                        local_C,
                        sizeof(float) * curr_Q);
        //C[local_now] = A[local_now] + B[local_now];
    }
    ldm_free(local_A, sizeof(float) * ((blk_Q + 8 - 1) / 8) * 8);
    ldm_free(local_B, sizeof(float) * ((blk_Q + 8 - 1) / 8) * 8);
}
void sw_slave_addm_all(sw_addmPara *_){
    sw_addmPara *para = (sw_addmPara *)para_cross;
    const float* A = para->A;
    const float* B = para->B;
    const int MN_size = para->MN_size;
    float* C = para->C;
    sw_slave_addm_thread(0, 199, A, B, C, MN_size);
}



















void sw_slave_gemm_rrr4_cgn(const int CGN_id,
                            const float* A, const float* Ap,
                            const float* B, const float* Bp,
                            const float* C, const float* Cp,
                            const int M, const int Ms, const int Me, const int blk_M,
                            const int N, const int Ns, const int Ne, const int blk_N,
                            const int K, const int Ks, const int Ke, const int blk_K){
    if(CRTS_cgn != CGN_id){
        return;
    }
    const int cid = CRTS_tid % 8;
    const int rid = CRTS_tid / 8;

    const int num_M = (M + blk_M - 1) / blk_M;
    const int num_N = (N + blk_N - 1) / blk_N;
    const int num_K = (K + blk_K - 1) / blk_K;

    const int rem_blk_M = num_M * blk_M - M == 0 ? blk_M : M - (num_M-1) * blk_M;
    const int rem_blk_N = num_N * blk_N - N == 0 ? blk_N : N - (num_N-1) * blk_N;
    const int rem_blk_K = num_K * blk_K - K == 0 ? blk_K : K - (num_K-1) * blk_K;

    const int local_A_size = blk_M * blk_K / 64;
    const int local_B_size = blk_K * blk_N / 64;
    const int local_C_size = blk_M * blk_N / 64;

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

    int A_step = K - blk_K/8;
    int B_step = N - blk_N/8;
    int C_step = N - blk_N/8;

    int curr_M = M;
    int curr_N = N;
    int curr_K = K;

    int curr_blk_M = blk_M;
    int curr_blk_N = blk_N;
    int curr_blk_K = blk_K;

    int next_A_offset = 0;
    int next_B_offset = 0;
    int next_C_offset = 0;

    int next_blk_M = blk_M;
    int next_blk_N = blk_N;
    int next_blk_K = blk_K;

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
                                    const int M, const int Ms, const int Me, const int blk_M,
                                    const int N, const int Ns, const int Ne, const int blk_N){
    if(CRTS_cgn != CGN_id){
        return;
    }
    const int cid = CRTS_tid % 8;
    const int rid = CRTS_tid / 8;

    const int num_M = (M + blk_M - 1) / blk_M;
    const int num_N = (N + blk_N - 1) / blk_N;

    const int rem_blk_M = num_M * blk_M - M == 0 ? blk_M : M - (num_M-1) * blk_M;
    const int rem_blk_N = num_N * blk_N - N == 0 ? blk_N : N - (num_N-1) * blk_N;

    const int local_C_size = blk_M * blk_N / 64;
    const int local_T_size = blk_M * blk_N / 64;

    float* local_C = (float*)ldm_malloc(sizeof(float) * blk_M * blk_N / 64);
    float* local_T = (float*)ldm_malloc(sizeof(float) * blk_M * blk_N / 64);
    
    const float* start_C = C + rid * blk_M/8 * N + cid * blk_N/8;
    const float* start_T = T + cid * blk_N/8 * M + rid * blk_M/8;

    const float* start_Cp = Cp + rid * blk_M/8 * Ne + cid * blk_N/8;
    const float* start_Tp = Tp + cid * blk_N/8 * Me + rid * blk_M/8;;

    float* curr_C = start_C;
    float* curr_T = start_T;

    int C_step = N - blk_N/8;
    int T_step = M - blk_M/8;

    int curr_M = M;
    int curr_N = N;

    int curr_blk_M = blk_M;
    int curr_blk_N = blk_N;

    int next_blk_M = blk_M;
    int next_blk_N = blk_N;

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
    int M = para->M;
    int Ms = para->Ms;
    int Me = para->Me;
    int N = para->N;
    int Ns = para->Ns;
    int Ne = para->Ne;
    int K = para->K;
    int Ks = para->Ks;
    int Ke = para->Ke;
    int blk_M = para->blk_M;
    int blk_N = para->blk_N;
    int blk_K = para->blk_K;
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
    int M = para->M;
    int Ms = para->Ms;
    int Me = para->Me;
    int N = para->N;
    int Ns = para->Ns;
    int Ne = para->Ne;
    int K = para->K;
    int Ks = para->Ks;
    int Ke = para->Ke;
    int blk_M = para->blk_M;
    int blk_N = para->blk_N;
    int blk_K = para->blk_K;
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

void sw_slave_gemm_rrr4_f32(sw_gemmPara *_){
    sw_gemmPara *para = (sw_gemmPara *)para_cross;
    const float *src_A = para->A;
    const float *src_Ap = para->Ap;
    const float *src_B = para->B;
    const float *src_Bp = para->Bp;
    const float *src_C = para->C;
    const float *src_Cp = para->Cp;
    int M = para->M;
    int Ms = para->Ms;
    int Me = para->Me;
    int N = para->N;
    int Ns = para->Ns;
    int Ne = para->Ne;
    int K = para->K;
    int Ks = para->Ks;
    int Ke = para->Ke;
    int blk_M = para->blk_M;
    int blk_N = para->blk_N;
    int blk_K = para->blk_K;
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

void sw_slave_bmm_cgn(sw_bmmPara *_){
    const int myid = CRTS_cgn * 64 + CRTS_tid;
    sw_bmmPara *para = ldm_malloc(sizeof(sw_bmmPara));
    athread_dma_get(para, para_cross, sizeof(sw_bmmPara));
    if(myid == 0){
        printf("sw_slave_bmm_cgn !!!\n");
    }
    const float *A = para->A;
    const float *B = para->B;
    const float *C = para->C;

    int M = para->M;
    int N = para->N;
    int K = para->K;

    int Ms = para->Ms;
    int Ns = para->Ns;
    int Ks = para->Ks;

    int Me = para->Me;
    int Ne = para->Ne;
    int Ke = para->Ke;

    int transposeA = para->transposeA;
    int transposeB = para->transposeB;

    int blk_M = para->blk_M;
    int blk_N = para->blk_N;
    int blk_K = para->blk_K;

    int counts = para->counts;

    const int MK_size = M * K;
    const int KN_size = K * N;
    const int MN_size = M * N;

    const int local_count = (counts + 5) / 6;
    const int local_start = CRTS_cgn * local_count;
    const int local_end = ((local_start + local_count > counts) ? counts : (local_start + local_count));
    
    /* printf("myid %d CGN %d counts %d local_count %d local_start %d local_end %d\n",
            myid, CRTS_cgn, counts, local_count, local_start, local_end); */

    if (local_start >= counts){
        return;
    }

    //printf("myid %d thread_start_cgn %d thread_end_cgn %d\n",myid,thread_start_cgn,thread_end_cgn);

    int local_now = local_start;

    float *A_cgn = A + MK_size * local_start;
    float *B_cgn = B + KN_size * local_start;
    float *C_cgn = C + MN_size * local_start;

    float *Ap_cgn = para->Ap + CRTS_cgn * Me * Ke;
    float *Bp_cgn = para->Bp + CRTS_cgn * Ke * Ne;
    float *Cp_cgn = para->Cp + CRTS_cgn * Me * Ne;

    for(int local_now = local_start; local_now < local_end; ++local_now){
        A_cgn = A + MK_size * local_now;
        B_cgn = B + KN_size * local_now;
        C_cgn = C + MN_size * local_now;
        //C_cgn = C +  MK_size * local_now;
        sw_slave_gemm_copy_all_cgn(CRTS_cgn,
                                    A_cgn, Ap_cgn,
                                    M, Me,
                                    K, Ke);
        sw_slave_gemm_copy_all_cgn(CRTS_cgn,
                                    B_cgn, Bp_cgn,
                                    K, Ke,
                                    N, Ne);                        
        athread_ssync_array();
        sw_slave_gemm_rrr_asm_cgn(CRTS_cgn,
                                Ap_cgn, Ap_cgn, 
                                Bp_cgn, Bp_cgn, 
                                Cp_cgn, Cp_cgn,
                                Me, Me, Me, blk_M,
                                Ne, Ne, Ne, blk_N,
                                Ke, Ke, Ke, blk_K);
        athread_ssync_array();
        sw_slave_gemm_copy_all_back_cgn(CRTS_cgn,
                                        C_cgn, Cp_cgn,
                                        M, Me,
                                        N, Ne);
    }
    ldm_free(para, sizeof(sw_bmmPara));
}



void sw_slave_bmm_rrr(sw_bmmPara *_){
    const int myid = CRTS_cgn * 64 + CRTS_tid;
    sw_bmmPara *para = (sw_bmmPara *)para_cross;
    const float *src_A = para->A;
    const float *src_B = para->B;
    const float *src_C = para->C;
    int M = para->M;
    int N = para->N;
    int K = para->K;
    int blk_M = para->blk_M;
    int blk_N = para->blk_N;
    int blk_K = para->blk_K;
    int counts = para->counts;
    const int local_count = (counts + 383) / 384;
    const int local_start = myid * local_count;
    const int local_end = ((local_start + local_count > counts) ? counts : (local_start + local_count));
    if (local_start >= counts){
        return;
    }
    const int local_A_size = blk_M * blk_K;
    const int local_B_size = blk_K * blk_N;
    const int local_C_size = blk_M * blk_N;
    const int MK_size = M * K;
    const int KN_size = K * N;
    const int MN_size = M * N;

    //num_M here might has problems
    const int num_M = (M + blk_M - 1) / blk_M;
    const int num_N = (N + blk_N - 1) / blk_N; //这一定能够被整除
    const int num_K = (K + blk_K - 1) / blk_K;

    const int rem_blk_M = num_M * blk_M - M == 0 ? blk_M : M - (num_M-1) * blk_M;
    const int rem_blk_N = num_N * blk_N - N == 0 ? blk_N : N - (num_N-1) * blk_N;
    const int rem_blk_K = num_K * blk_K - K == 0 ? blk_K : K - (num_K-1) * blk_K;

    //change curr_blk_MNK
    const int first_blk_M = M < blk_M ? M : blk_M;
    const int first_blk_N = N < blk_N ? N : blk_N;
    const int first_blk_K = K < blk_K ? K : blk_K;

    const int rem_blk_M_not_aligned = rem_blk_M != blk_M;
    const int rem_blk_N_not_aligned = rem_blk_N != blk_N;
    const int rem_blk_K_not_aligned = rem_blk_K != blk_K;

    const int first_blk_M_not_aligned = first_blk_M != blk_M;
    const int first_blk_N_not_aligned = first_blk_N != blk_N;
    const int first_blk_K_not_aligned = first_blk_K != blk_K;

#ifdef _SWOPS_DEBUG
    if(myid == 0){
        printf("rem_blk_M %d rem_blk_N %d rem_blk_K %d\n", rem_blk_M, rem_blk_N, rem_blk_K);
        printf("rem_blk_M_not_aligned %d rem_blk_N_not_aligned %d rem_blk_K_not_aligned %d\n", rem_blk_M_not_aligned, rem_blk_N_not_aligned, rem_blk_K_not_aligned);
    }
#endif

#ifdef _SWOPS_DEBUG
    if(myid == 0){
        printf("first_blk_M %d first_blk_N %d first_blk_K %d\n", first_blk_M, first_blk_N, first_blk_K);
        printf("first_blk_M_not_aligned %d first_blk_N_not_aligned %d first_blk_K_not_aligned %d\n", first_blk_M_not_aligned, first_blk_N_not_aligned, first_blk_K_not_aligned);
    }
#endif
    int curr_blk_M = first_blk_M;
    int curr_blk_N = first_blk_N;
    int curr_blk_K = first_blk_K;

    int next_blk_M = blk_M;
    int next_blk_N = blk_N;
    int next_blk_K = blk_K;
    
    float *start_A = src_A + MK_size * local_start;
    float *start_B = src_B + KN_size * local_start;
    float *start_C = src_C + MN_size * local_start;

    float *local_A = (float *)ldm_malloc(sizeof(float) * blk_M * blk_K * 2);
    float *local_B = (float *)ldm_malloc(sizeof(float) * blk_K * blk_N * 2);
    float *local_C = (float *)ldm_malloc(sizeof(float) * blk_M * blk_N * 2);
    double *local_C_temp = (double *)ldm_malloc(sizeof(double) * blk_M * blk_N);

    volatile int double_buffer_flag_AB = 0;
    volatile int double_buffer_flag_C = 0;

    volatile athread_rply_t reply_get_A = 0, reply_get_B = 0, reply_put_C = 1;//start 1

    int local_now = local_start;

    athread_dma_iget_stride(local_A + (1 - double_buffer_flag_AB) * local_A_size, 
                            start_A, 
                            sizeof(float) * first_blk_M * first_blk_K, 
                            sizeof(float) * first_blk_K, 
                            sizeof(float) * (K - first_blk_K),
                            &reply_get_A);
    athread_dma_iget_stride(local_B + (1 - double_buffer_flag_AB) * local_B_size, 
                            start_B, 
                            sizeof(float) * first_blk_K * first_blk_N, 
                            sizeof(float) * first_blk_N, 
                            sizeof(float) * (N - first_blk_N),
                            &reply_get_B);

    athread_dma_wait_value(&reply_get_A, 1);
    athread_dma_wait_value(&reply_get_B, 1);

    reply_get_A = 0;
    reply_get_B = 0;

    doublev8 vec_double;
    floatv8 vec_float;

    for(int local_now = local_start; local_now < local_end; ++local_now){
        start_A = src_A + MK_size * local_now;
        start_B = src_B + KN_size * local_now;
        start_C = src_C + MN_size * local_now;
        for(int c_M = 0; c_M < num_M; c_M++){//K N M order
            curr_blk_M = c_M < num_M - 1 ? blk_M : rem_blk_M;
            curr_blk_M = num_M == 1 ? first_blk_M : curr_blk_M;
            for(int c_N = 0; c_N < num_N; c_N++){
                curr_blk_N = c_N < num_N - 1 ? blk_N : rem_blk_N;
                curr_blk_N = num_N == 1 ? first_blk_N : curr_blk_N;
                for(int i = 0; i < local_C_size; i++){
                    local_C[i + (1 - double_buffer_flag_C) * local_C_size] = 0;
                }
                for(int i = 0; i < local_C_size; i++){
                    local_C_temp[i] = 0;
                }
                for(int c_K = 0; c_K < num_K; c_K++){
                    curr_blk_K = c_K < num_K - 1 ? blk_K : rem_blk_K;
                    curr_blk_K = num_K == 1 ? first_blk_K : curr_blk_K;
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
                                                sizeof(float) * first_blk_M * first_blk_K, 
                                                sizeof(float) * first_blk_K, 
                                                sizeof(float) * (K - first_blk_K),
                                                &reply_get_A);
                        athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_B_size), 
                                                start_B + KN_size, 
                                                sizeof(float) * first_blk_K * first_blk_N, 
                                                sizeof(float) * first_blk_N, 
                                                sizeof(float) * (N - first_blk_N),
                                                &reply_get_B);
                    }

                    if((curr_blk_M == first_blk_M && first_blk_M_not_aligned) 
                    || (curr_blk_K == first_blk_K && first_blk_K_not_aligned)
                    || (curr_blk_M == rem_blk_M && rem_blk_M_not_aligned)
                    || (curr_blk_K == rem_blk_K && rem_blk_K_not_aligned)){

                        /* if(myid == 0){
                            printf("padding start\n");
                            for(int m = 0; m < blk_M; m++){
                                for(int k = 0; k < blk_K; k++){
                                    printf("%f ", local_A[m * blk_K + k + (1 - double_buffer_flag_AB) * local_A_size]);
                                }
                                printf("\n");
                            }
                        } */
                        
                        for(int m = curr_blk_M - 1; m >= 0; m--){
                            for(int k = curr_blk_K - 1; k >= 0; k--){
                                local_A[m * blk_K + k + (1 - double_buffer_flag_AB) * local_A_size] 
                                = local_A[m * curr_blk_K + k + (1 - double_buffer_flag_AB) * local_A_size];
                            }
                            for(int k = blk_K - 1; k >= curr_blk_K; k--){
                                local_A[m * blk_K + k + (1 - double_buffer_flag_AB) * local_A_size] = 0;
                            }
                        }
                        for(int m = blk_M - 1; m >= curr_blk_M; m--){
                            for(int k = blk_K - 1; k >= 0; k--){
                                local_A[m * blk_K + k + (1 - double_buffer_flag_AB) * local_A_size] = 0;
                            }
                        }

                        /* if(myid == 0){
                            printf("padding end\n");
                            for(int m = 0; m < blk_M; m++){
                                for(int k = 0; k < blk_K; k++){
                                    printf("%f ", local_A[m * blk_K + k + (1 - double_buffer_flag_AB) * local_A_size]);
                                }
                                printf("\n");
                            }
                        } */
                    }

                    if((curr_blk_K == first_blk_K && first_blk_K_not_aligned)
                    || (curr_blk_N == first_blk_N && first_blk_N_not_aligned)
                    || (curr_blk_K == rem_blk_K && rem_blk_K_not_aligned)
                    || (curr_blk_N == rem_blk_N && rem_blk_N_not_aligned)){
                        /* if(myid == 0){
                            printf("padding start\n");
                            for(int k = 0; k < blk_K; k++){
                                for(int n = 0; n < blk_N; n++){
                                    printf("%f ", local_B[k * blk_N + n + (1 - double_buffer_flag_AB) * local_B_size]);
                                }
                                printf("\n");
                            }
                        } */
                        for(int k = curr_blk_K - 1; k >= 0; k--){
                            for(int n = curr_blk_N - 1; n >= 0; n--){
                                local_B[k * blk_N + n + (1 - double_buffer_flag_AB) * local_B_size]
                                = local_B[k * curr_blk_N + n + (1 - double_buffer_flag_AB) * local_B_size];
                            }
                            for(int n = blk_N - 1; n >= curr_blk_N; n--){
                                local_B[k * blk_N + n + (1 - double_buffer_flag_AB) * local_B_size] = 0;
                            }
                        }
                        for(int k = blk_K - 1; k >= curr_blk_K; k--){
                            for(int n = blk_N - 1; n >= 0; n--){
                                local_B[k * blk_N + n + (1 - double_buffer_flag_AB) * local_B_size] = 0;
                            }
                        }
                        /* if(myid == 0){
                            printf("padding end\n");
                            for(int k = 0; k < blk_K; k++){
                                for(int n = 0; n < blk_N; n++){
                                    printf("%f ", local_B[k * blk_N + n + (1 - double_buffer_flag_AB) * local_B_size]);
                                }
                                printf("\n");
                            }
                        } */
                    }

                    /* for(int m = 0; m < blk_M; m++){
                        for(int n = 0; n < blk_N; n++){
                            for(int k = 0; k < blk_K; k++){
                                local_C[m * blk_N + n + (1 - double_buffer_flag_C) * local_C_size]
                             += local_A[m * blk_K + k + (1 - double_buffer_flag_AB) * local_A_size]
                              * local_B[k * blk_N + n + (1 - double_buffer_flag_AB) * local_B_size];
                            }
                        }
                    } */

                    float* comp_A = local_A + (1 - double_buffer_flag_AB) * local_A_size;
                    float* comp_B = local_B + (1 - double_buffer_flag_AB) * local_B_size;
                    double* comp_C = local_C_temp;

                    for(int m = 0; m < blk_M; m += 8){
                        for(int n = 0; n <blk_N; n += 8){
                            for(int k = 0; k < blk_K; k += 8){
                                sgemm_8_8_8(comp_A + m * blk_K + k, 
                                            comp_B + k * blk_N + n, 
                                            comp_C + m * blk_N + n, 
                                            blk_K, blk_N);
                            }
                        }
                    }

                    /* for(int m = 0; m < curr_blk_M; m++)
                        for(int n = 0; n < curr_blk_N; n++)
                            for(int k = 0; k < curr_blk_K; k++){
                                local_C[(1 - double_buffer_flag_C) * local_C_size + m * curr_blk_N + n]
                             += local_A[(1 - double_buffer_flag_AB) * local_A_size + m * curr_blk_K + k]
                              * local_B[(1 - double_buffer_flag_AB) * local_B_size + k * curr_blk_N + n];
                            } */
                    //gemm
                    if(c_N * num_M * num_K + c_M * num_K + c_K +1 < num_M * num_N * num_K || local_now < local_end - 1){
                        athread_dma_wait_value(&reply_get_A, 1);
                        athread_dma_wait_value(&reply_get_B, 1);
                        reply_get_A = 0;
                        reply_get_B = 0;
                        double_buffer_flag_AB = 1 - double_buffer_flag_AB;
                    }
                }

                //double to float
                /* for(int m = 0; m < blk_M; m++){
                    for(int n = 0; n < blk_N; n++){
                        local_C[m * blk_N + n + (1 - double_buffer_flag_C) * local_C_size] 
                        -= local_C_temp[m * blk_N + n];
                    }
                } */

                for(int i = 0; i < blk_M * blk_N; i += 8){
                    simd_load(vec_double, local_C_temp + i);
                    vec_float = - (floatv8)vec_double;
                    simd_store(vec_float, local_C + i + (1 - double_buffer_flag_C) * local_C_size);
                }

                if((curr_blk_M == first_blk_M && first_blk_M_not_aligned)
                || (curr_blk_N == first_blk_N && first_blk_N_not_aligned)
                || (curr_blk_M == rem_blk_M && rem_blk_M_not_aligned)
                || (curr_blk_N == rem_blk_N && rem_blk_N_not_aligned)){
                    /* if(myid == 0){
                            printf("unpadding start\n");
                            for(int m = 0; m < blk_M; m++){
                                for(int n = 0; n < blk_N; n++){
                                    printf("%f ", local_C[m * blk_N + n + (1 - double_buffer_flag_C) * local_C_size]);
                                }
                                printf("\n");
                            }
                        } */
                    for(int m = 0; m < curr_blk_M; m++){
                        for(int n = 0; n < curr_blk_N; n++){
                            local_C[m * curr_blk_N + n + (1 - double_buffer_flag_C) * local_C_size]
                            = local_C[m * blk_N + n + (1 - double_buffer_flag_C) * local_C_size];
                        }
                    }
                    /* if(myid == 0){
                            printf("unpadding end\n");
                            for(int m = 0; m < blk_M; m++){
                                for(int n = 0; n < blk_N; n++){
                                    printf("%f ", local_C[m * blk_N + n + (1 - double_buffer_flag_C) * local_C_size]);
                                }
                                printf("\n");
                            }
                        } */
                }
                athread_dma_put_stride(start_C + c_M * blk_M * N + c_N * blk_N,
                                       local_C + (1 - double_buffer_flag_C) * local_C_size,
                                       sizeof(float) * curr_blk_M * curr_blk_N,
                                       sizeof(float) * curr_blk_N,
                                       sizeof(float) * (N - curr_blk_N));
                /* athread_dma_wait_value(&reply_put_C, 1);
                reply_put_C = 0;
                athread_dma_iput_stride(start_C + c_M * blk_M * N + c_N * blk_N,
                                        local_C + (1 - double_buffer_flag_C) * local_C_size,
                                        sizeof(float) * curr_blk_M * curr_blk_N,
                                        sizeof(float) * curr_blk_N,
                                        sizeof(float) * (N - curr_blk_N),
                                        &reply_put_C);
                double_buffer_flag_C = 1 - double_buffer_flag_C; */
            }
        }
    }
    ldm_free(local_A,sizeof(float) * blk_M * blk_K * 2);
    ldm_free(local_B,sizeof(float) * blk_K * blk_N * 2);
    ldm_free(local_C,sizeof(float) * blk_M * blk_N * 2);
    ldm_free(local_C_temp,sizeof(double) * blk_M * blk_N);
}

void sw_slave_bmm_rcr(sw_bmmPara *_){
    const int myid = CRTS_cgn * 64 + CRTS_tid;
    sw_bmmPara *para = (sw_bmmPara *)para_cross;
    const float *src_A = para->A;
    const float *src_B = para->B;
    const float *src_C = para->C;
    int M = para->M;
    int N = para->N;
    int K = para->K;
    int blk_M = para->blk_M;
    int blk_N = para->blk_N;
    int blk_K = para->blk_K;
    int counts = para->counts;
    const int local_count = (counts + 383) / 384;
    const int local_start = myid * local_count;
    const int local_end = ((local_start + local_count > counts) ? counts : (local_start + local_count));
    if (local_start >= counts){
        return;
    }
    const int local_A_size = blk_M * blk_K;
    const int local_B_size = blk_K * blk_N;
    const int local_C_size = blk_M * blk_N;

    const int MK_size = M * K;
    const int KN_size = K * N;
    const int MN_size = M * N;

    const int num_M = (M + blk_M - 1) / blk_M;
    const int num_N = (N + blk_N - 1) / blk_N; //这一定能够被整除
    const int num_K = (K + blk_K - 1) / blk_K;

    const int rem_blk_M = num_M * blk_M - M == 0 ? blk_M : M - (num_M-1) * blk_M;
    const int rem_blk_N = num_N * blk_N - N == 0 ? blk_N : N - (num_N-1) * blk_N;
    const int rem_blk_K = num_K * blk_K - K == 0 ? blk_K : K - (num_K-1) * blk_K;

    //change curr_blk_MNK
    const int first_blk_M = M < blk_M ? M : blk_M;
    const int first_blk_N = N < blk_N ? N : blk_N;
    const int first_blk_K = K < blk_K ? K : blk_K;

    const int rem_blk_M_not_aligned = rem_blk_M != blk_M;
    const int rem_blk_N_not_aligned = rem_blk_N != blk_N;
    const int rem_blk_K_not_aligned = rem_blk_K != blk_K;

    const int first_blk_M_not_aligned = first_blk_M != blk_M;
    const int first_blk_N_not_aligned = first_blk_N != blk_N;
    const int first_blk_K_not_aligned = first_blk_K != blk_K;

#ifdef _SWOPS_DEBUG
    if(myid == 0){
        printf("rem_blk_M %d rem_blk_N %d rem_blk_K %d\n", rem_blk_M, rem_blk_N, rem_blk_K);
        printf("rem_blk_M_not_aligned %d rem_blk_N_not_aligned %d rem_blk_K_not_aligned %d\n", rem_blk_M_not_aligned, rem_blk_N_not_aligned, rem_blk_K_not_aligned);
    }
#endif

#ifdef _SWOPS_DEBUG
    if(myid == 0){
        printf("first_blk_M %d first_blk_N %d first_blk_K %d\n", first_blk_M, first_blk_N, first_blk_K);
        printf("first_blk_M_not_aligned %d first_blk_N_not_aligned %d first_blk_K_not_aligned %d\n", first_blk_M_not_aligned, first_blk_N_not_aligned, first_blk_K_not_aligned);
    }
#endif
    int curr_blk_M = first_blk_M;
    int curr_blk_N = first_blk_N;
    int curr_blk_K = first_blk_K;

    int next_blk_M = blk_M;
    int next_blk_N = blk_N;
    int next_blk_K = blk_K;

    float *start_A = src_A + MK_size * local_start;
    float *start_B = src_B + KN_size * local_start;
    float *start_C = src_C + MN_size * local_start;

    float *local_A = (float *)ldm_malloc(sizeof(float) * blk_M * blk_K * 2);
    float *local_B = (float *)ldm_malloc(sizeof(float) * blk_K * blk_N * 2);
    float *local_C = (float *)ldm_malloc(sizeof(float) * blk_M * blk_N * 2);
    float *trans_B = (float *)ldm_malloc(sizeof(float) * blk_K * blk_N);
    double *local_C_temp = (double *)ldm_malloc(sizeof(double) * blk_M * blk_N);

    volatile int double_buffer_flag_AB = 0;
    volatile int double_buffer_flag_C = 0;
    
    volatile athread_rply_t reply_get_A = 0, reply_get_B = 0, reply_put_C = 1;//start 1

    int local_now = local_start;

    athread_dma_iget_stride(local_A + (1 - double_buffer_flag_AB) * local_A_size, 
                            start_A, 
                            sizeof(float) * first_blk_M * first_blk_K, 
                            sizeof(float) * first_blk_K, 
                            sizeof(float) * (K - first_blk_K),
                            &reply_get_A);
    athread_dma_iget_stride(local_B + (1 - double_buffer_flag_AB) * local_B_size, 
                            start_B, 
                            sizeof(float) * first_blk_N * first_blk_K, 
                            sizeof(float) * first_blk_K, 
                            sizeof(float) * (K - first_blk_K),
                            &reply_get_B);
    athread_dma_wait_value(&reply_get_A, 1);
    athread_dma_wait_value(&reply_get_B, 1);

    reply_get_A = 0;
    reply_get_B = 0;

    intv16 shuffle1 = {0x8628C020, 0x6AD0A49C, 0xBD8E, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle2 = {0x96ACE128, 0xEEF1ACDE, 0xFF9E, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle3 = {0xA3018820, 0x4398A49C, 0xBDAB, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle4 = {0xB385A928, 0xC7B9ACDE, 0xFFBB, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle5 = {0x8A418820, 0x49CA3039, 0xBDAB, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle6 = {0x9AC5A928, 0xCDEB387B, 0xFFBB, 0,0,0,0,0,0,0,0,0,0,0,0,0};

    floatv8 a1,a2,a3,a4,a5,a6,a7,a8;

    doublev8 vec_double;
    floatv8 vec_float;

    for(int local_now = local_start; local_now < local_end; ++local_now){

        start_A = src_A + MK_size * local_now;
        start_B = src_B + KN_size * local_now;
        start_C = src_C + MN_size * local_now;

        for(int c_M = 0; c_M < num_M; c_M++){//K N M order

            curr_blk_M = c_M < num_M - 1 ? blk_M : rem_blk_M;
            curr_blk_M = num_M == 1 ? first_blk_M : curr_blk_M;

            for(int c_N = 0; c_N < num_N; c_N++){

                curr_blk_N = c_N < num_N - 1 ? blk_N : rem_blk_N;
                curr_blk_N = num_N == 1 ? first_blk_N : curr_blk_N;

                for(int i = 0; i < local_C_size; i++){
                    local_C[i + (1 - double_buffer_flag_C) * local_C_size] = 0;
                }
                for(int i = 0; i < local_C_size; i++){
                    local_C_temp[i] = 0;
                }

                for(int c_K = 0; c_K < num_K; c_K++){

                    curr_blk_K = c_K < num_K - 1 ? blk_K : rem_blk_K;
                    curr_blk_K = num_K == 1 ? first_blk_K : curr_blk_K;

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
                                                sizeof(float) * first_blk_M * first_blk_K, 
                                                sizeof(float) * first_blk_K, 
                                                sizeof(float) * (K - first_blk_K),
                                                &reply_get_A);
                        athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_B_size), 
                                                start_B + KN_size, 
                                                sizeof(float) * first_blk_N * first_blk_K,
                                                sizeof(float) * first_blk_K,
                                                sizeof(float) * (K - first_blk_K),
                                                &reply_get_B);
                    }
                    //align A
                    if((curr_blk_M == first_blk_M && first_blk_M_not_aligned) 
                    || (curr_blk_K == first_blk_K && first_blk_K_not_aligned)
                    || (curr_blk_M == rem_blk_M && rem_blk_M_not_aligned)
                    || (curr_blk_K == rem_blk_K && rem_blk_K_not_aligned)){

                        /* if(myid == 0){
                            printf("padding start\n");
                            for(int m = 0; m < blk_M; m++){
                                for(int k = 0; k < blk_K; k++){
                                    printf("%f ", local_A[m * blk_K + k + (1 - double_buffer_flag_AB) * local_A_size]);
                                }
                                printf("\n");
                            }
                        } */
                        
                        for(int m = curr_blk_M - 1; m >= 0; m--){
                            for(int k = curr_blk_K - 1; k >= 0; k--){
                                local_A[m * blk_K + k + (1 - double_buffer_flag_AB) * local_A_size] 
                                = local_A[m * curr_blk_K + k + (1 - double_buffer_flag_AB) * local_A_size];
                            }
                            for(int k = blk_K - 1; k >= curr_blk_K; k--){
                                local_A[m * blk_K + k + (1 - double_buffer_flag_AB) * local_A_size] = 0;
                            }
                        }
                        for(int m = blk_M - 1; m >= curr_blk_M; m--){
                            for(int k = blk_K - 1; k >= 0; k--){
                                local_A[m * blk_K + k + (1 - double_buffer_flag_AB) * local_A_size] = 0;
                            }
                        }

                        /* if(myid == 0){
                            printf("padding end\n");
                            for(int m = 0; m < blk_M; m++){
                                for(int k = 0; k < blk_K; k++){
                                    printf("%f ", local_A[m * blk_K + k + (1 - double_buffer_flag_AB) * local_A_size]);
                                }
                                printf("\n");
                            }
                        } */
                    }
                    //align B
                    if((curr_blk_K == first_blk_K && first_blk_K_not_aligned)
                    || (curr_blk_N == first_blk_N && first_blk_N_not_aligned)
                    || (curr_blk_K == rem_blk_K && rem_blk_K_not_aligned)
                    || (curr_blk_N == rem_blk_N && rem_blk_N_not_aligned)){
                        
                        /* if(myid == 0){
                            printf("padding start\n");
                            for(int n = 0; n < blk_N; n++){
                                for(int k = 0; k < blk_K; k++){
                                    printf("%f ", local_B[n * blk_K + k + (1 - double_buffer_flag_AB) * local_B_size]);
                                }
                                printf("\n");
                            }
                        } */

                        for(int n = curr_blk_N - 1; n >= 0; n--){
                            for(int k = curr_blk_K - 1; k >= 0; k--){
                                local_B[n * blk_K + k + (1 - double_buffer_flag_AB) * local_B_size]
                                = local_B[n * curr_blk_K + k + (1 - double_buffer_flag_AB) * local_B_size];
                            }
                            for(int k = blk_K - 1; k >= curr_blk_K; k--){
                                local_B[n * blk_K + k + (1 - double_buffer_flag_AB) * local_B_size] = 0;
                            }
                        }
                        for(int n = blk_N - 1; n >= curr_blk_N; n--){
                            for(int k = blk_K - 1; k >= 0; k--){
                                local_B[n * blk_K + k + (1 - double_buffer_flag_AB) * local_B_size] = 0;
                            }
                        }

                        /* if(myid == 0){
                            printf("padding end\n");
                            for(int n = 0; n < blk_N; n++){
                                for(int k = 0; k < blk_K; k++){
                                    printf("%f ", local_B[n * blk_K + k + (1 - double_buffer_flag_AB) * local_B_size]);
                                }
                                printf("\n");
                            }
                        } */
                    }

                    //trans_B

                    /* for(int n = 0; n < blk_N; n++){
                        for(int k = 0; k < blk_K; k++){
                            trans_B[k * blk_N + n] = local_B[n * blk_K + k + (1 - double_buffer_flag_AB) * local_B_size];
                        }
                    } */


                    float* origin_B = local_B + (1 - double_buffer_flag_AB) * local_B_size;
                    for(int n = 0; n < blk_N; n += 8){
                        for(int k = 0; k < blk_K; k += 8){
                            int index = n * blk_K + k;
                            simd_load(a1, origin_B + index);
                            index += blk_K;
                            simd_load(a2, origin_B + index);
                            index += blk_K;
                            simd_load(a3, origin_B + index);
                            index += blk_K;
                            simd_load(a4, origin_B + index);
                            index += blk_K;
                            simd_load(a5, origin_B + index);
                            index += blk_K;
                            simd_load(a6, origin_B + index);
                            index += blk_K;
                            simd_load(a7, origin_B + index);
                            index += blk_K;
                            simd_load(a8, origin_B + index);
                            
                            SWAP_8_8(a1,a2,a3,a4,a5,a6,a7,a8);

                            index = k * blk_N + n;
                            simd_store(a1, trans_B + index);
                            index += blk_N;
                            simd_store(a2, trans_B + index);
                            index += blk_N;
                            simd_store(a3, trans_B + index);
                            index += blk_N;
                            simd_store(a4, trans_B + index);
                            index += blk_N;
                            simd_store(a5, trans_B + index);
                            index += blk_N;
                            simd_store(a6, trans_B + index);
                            index += blk_N;
                            simd_store(a7, trans_B + index);
                            index += blk_N;
                            simd_store(a8, trans_B + index);
                        }
                    }

                    /* if(myid == 0){
                        printf("trans end\n");
                        for(int k = 0; k < blk_K; k++){
                            for(int n = 0; n < blk_N; n++){
                                printf("%f ", trans_B[k * blk_N + n]);
                            }
                            printf("\n");
                        }
                    } */

                    /* for(int m = 0; m < blk_M; m++){
                        for(int n = 0; n < blk_N; n++){
                            for(int k = 0; k < blk_K; k++){
                                local_C[m * blk_N + n + (1 - double_buffer_flag_C) * local_C_size]
                            +=  local_A[m * blk_K + k + (1 - double_buffer_flag_AB) * local_A_size]
                             *  trans_B[k * blk_N + n];
                            }
                        }
                    } */

                    float* comp_A = local_A + (1 - double_buffer_flag_AB) * local_A_size;
                    float* comp_B = trans_B;
                    double* comp_C = local_C_temp;

                    for(int m = 0; m < blk_M; m += 8){
                        for(int n = 0; n <blk_N; n += 8){
                            for(int k = 0; k < blk_K; k += 8){
                                sgemm_8_8_8(comp_A + m * blk_K + k, 
                                            comp_B + k * blk_N + n, 
                                            comp_C + m * blk_N + n, 
                                            blk_K, blk_N);
                            }
                        }
                    }

                    /* for(int m = 0; m < curr_blk_M; m++)
                        for(int n = 0; n < curr_blk_N; n++)
                            for(int k = 0; k < curr_blk_K; k++){
                                local_C[(1 - double_buffer_flag_C) * local_C_size + m * curr_blk_N + n]
                             += local_A[(1 - double_buffer_flag_AB) * local_A_size + m * curr_blk_K + k]
                              * local_B[(1 - double_buffer_flag_AB) * local_B_size + n * curr_blk_K + k];
                            } */

                    //gemm
                    if(c_N * num_M * num_K + c_M * num_K + c_K +1 < num_M * num_N * num_K || local_now < local_end - 1){
                        athread_dma_wait_value(&reply_get_A, 1);
                        athread_dma_wait_value(&reply_get_B, 1);
                        reply_get_A = 0;
                        reply_get_B = 0;
                        double_buffer_flag_AB = 1 - double_buffer_flag_AB;
                    }
                }

                //double to float
                /* for(int m = 0; m < blk_M; m++){
                    for(int n = 0; n < blk_N; n++){
                        local_C[m * blk_N + n + (1 - double_buffer_flag_C) * local_C_size] 
                        -= local_C_temp[m * blk_N + n];
                    }
                } */

                for(int i = 0; i < blk_M * blk_N; i += 8){
                    simd_load(vec_double, local_C_temp + i);
                    vec_float = - (floatv8)vec_double;
                    simd_store(vec_float, local_C + i + (1 - double_buffer_flag_C) * local_C_size);
                }

                if((curr_blk_M == first_blk_M && first_blk_M_not_aligned)
                || (curr_blk_N == first_blk_N && first_blk_N_not_aligned)
                || (curr_blk_M == rem_blk_M && rem_blk_M_not_aligned)
                || (curr_blk_N == rem_blk_N && rem_blk_N_not_aligned)){
                    /* if(myid == 0){
                            printf("unpadding start\n");
                            for(int m = 0; m < blk_M; m++){
                                for(int n = 0; n < blk_N; n++){
                                    printf("%f ", local_C[m * blk_N + n + (1 - double_buffer_flag_C) * local_C_size]);
                                }
                                printf("\n");
                            }
                        } */
                    for(int m = 0; m < curr_blk_M; m++){
                        for(int n = 0; n < curr_blk_N; n++){
                            local_C[m * curr_blk_N + n + (1 - double_buffer_flag_C) * local_C_size]
                            = local_C[m * blk_N + n + (1 - double_buffer_flag_C) * local_C_size];
                        }
                    }
                    /* if(myid == 0){
                            printf("unpadding end\n");
                            for(int m = 0; m < blk_M; m++){
                                for(int n = 0; n < blk_N; n++){
                                    printf("%f ", local_C[m * blk_N + n + (1 - double_buffer_flag_C) * local_C_size]);
                                }
                                printf("\n");
                            }
                        } */
                }
                athread_dma_put_stride(start_C + c_M * blk_M * N + c_N * blk_N,
                                       local_C + (1 - double_buffer_flag_C) * local_C_size,
                                       sizeof(float) * curr_blk_M * curr_blk_N,
                                       sizeof(float) * curr_blk_N,
                                       sizeof(float) * (N - curr_blk_N));
                /* athread_dma_wait_value(&reply_put_C, 1);
                reply_put_C = 0;
                athread_dma_iput_stride(start_C + c_M * blk_M * N + c_N * blk_N,
                                        local_C + (1 - double_buffer_flag_C) * local_C_size,
                                        sizeof(float) * curr_blk_M * curr_blk_N,
                                        sizeof(float) * curr_blk_N,
                                        sizeof(float) * (N - curr_blk_N),
                                        &reply_put_C);
                double_buffer_flag_C = 1 - double_buffer_flag_C; */
            }
        }
    }
    ldm_free(local_A,sizeof(float) * blk_M * blk_K * 2);
    ldm_free(local_B,sizeof(float) * blk_K * blk_N * 2);
    ldm_free(local_C,sizeof(float) * blk_M * blk_N * 2);
    ldm_free(trans_B,sizeof(float) * blk_K * blk_N);
    ldm_free(local_C_temp,sizeof(double) * blk_M * blk_N);
}

void sw_slave_bmm_crr(sw_bmmPara *_){
    const int myid = CRTS_cgn * 64 + CRTS_tid;
    sw_bmmPara *para = (sw_bmmPara *)para_cross;
    const float *src_A = para->A;
    const float *src_B = para->B;
    const float *src_C = para->C;
    int M = para->M;
    int N = para->N;
    int K = para->K;
    int blk_M = para->blk_M;
    int blk_N = para->blk_N;
    int blk_K = para->blk_K;
    int counts = para->counts;
    const int local_count = (counts + 383) / 384;
    const int local_start = myid * local_count;
    const int local_end = ((local_start + local_count > counts) ? counts : (local_start + local_count));
    if (local_start >= counts){
        return;
    }
    const int local_A_size = blk_M * blk_K;
    const int local_B_size = blk_K * blk_N;
    const int local_C_size = blk_M * blk_N;

    const int MK_size = M * K;
    const int KN_size = K * N;
    const int MN_size = M * N;

    const int num_M = (M + blk_M - 1) / blk_M;
    const int num_N = (N + blk_N - 1) / blk_N; //这一定能够被整除
    const int num_K = (K + blk_K - 1) / blk_K;

    const int rem_blk_M = num_M * blk_M - M == 0 ? blk_M : M - (num_M-1) * blk_M;
    const int rem_blk_N = num_N * blk_N - N == 0 ? blk_N : N - (num_N-1) * blk_N;
    const int rem_blk_K = num_K * blk_K - K == 0 ? blk_K : K - (num_K-1) * blk_K;
    //change curr_blk_MNK
    const int first_blk_M = M < blk_M ? M : blk_M;
    const int first_blk_N = N < blk_N ? N : blk_N;
    const int first_blk_K = K < blk_K ? K : blk_K;

    const int rem_blk_M_not_aligned = rem_blk_M != blk_M;
    const int rem_blk_N_not_aligned = rem_blk_N != blk_N;
    const int rem_blk_K_not_aligned = rem_blk_K != blk_K;

    const int first_blk_M_not_aligned = first_blk_M != blk_M;
    const int first_blk_N_not_aligned = first_blk_N != blk_N;
    const int first_blk_K_not_aligned = first_blk_K != blk_K;

#ifdef _SWOPS_DEBUG
    if(myid == 0){
        printf("rem_blk_M %d rem_blk_N %d rem_blk_K %d\n", rem_blk_M, rem_blk_N, rem_blk_K);
        printf("rem_blk_M_not_aligned %d rem_blk_N_not_aligned %d rem_blk_K_not_aligned %d\n", rem_blk_M_not_aligned, rem_blk_N_not_aligned, rem_blk_K_not_aligned);
    }
#endif


#ifdef _SWOPS_DEBUG
    if(myid == 0){
        printf("first_blk_M %d first_blk_N %d first_blk_K %d\n", first_blk_M, first_blk_N, first_blk_K);
        printf("first_blk_M_not_aligned %d first_blk_N_not_aligned %d first_blk_K_not_aligned %d\n", first_blk_M_not_aligned, first_blk_N_not_aligned, first_blk_K_not_aligned);
    }
#endif
    int curr_blk_M = first_blk_M;
    int curr_blk_N = first_blk_N;
    int curr_blk_K = first_blk_K;

    int next_blk_M = blk_M;
    int next_blk_N = blk_N;
    int next_blk_K = blk_K;

    float *start_A = src_A + MK_size * local_start;
    float *start_B = src_B + KN_size * local_start;
    float *start_C = src_C + MN_size * local_start;

    float *local_A = (float *)ldm_malloc(sizeof(float) * blk_M * blk_K * 2);
    float *local_B = (float *)ldm_malloc(sizeof(float) * blk_K * blk_N * 2);
    float *local_C = (float *)ldm_malloc(sizeof(float) * blk_M * blk_N * 2);
    float *trans_A = (float *)ldm_malloc(sizeof(float) * blk_M * blk_K);
    double *local_C_temp = (double *)ldm_malloc(sizeof(double) * blk_M * blk_N);

    volatile int double_buffer_flag_AB = 0;
    volatile int double_buffer_flag_C = 0;

    volatile athread_rply_t reply_get_A = 0, reply_get_B = 0, reply_put_C = 1;//start 1

    int local_now = local_start;

    athread_dma_iget_stride(local_A + (1 - double_buffer_flag_AB) * local_A_size, 
                            start_A, 
                            sizeof(float) * first_blk_K * first_blk_M, 
                            sizeof(float) * first_blk_M,
                            sizeof(float) * (M - first_blk_M),
                            &reply_get_A);
    athread_dma_iget_stride(local_B + (1 - double_buffer_flag_AB) * local_B_size, 
                            start_B, 
                            sizeof(float) * first_blk_K * first_blk_N, 
                            sizeof(float) * first_blk_N, 
                            sizeof(float) * (N - first_blk_N),
                            &reply_get_B);

    athread_dma_wait_value(&reply_get_A, 1);
    athread_dma_wait_value(&reply_get_B, 1);
    reply_get_A = 0;
    reply_get_B = 0;

    intv16 shuffle1 = {0x8628C020, 0x6AD0A49C, 0xBD8E, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle2 = {0x96ACE128, 0xEEF1ACDE, 0xFF9E, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle3 = {0xA3018820, 0x4398A49C, 0xBDAB, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle4 = {0xB385A928, 0xC7B9ACDE, 0xFFBB, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle5 = {0x8A418820, 0x49CA3039, 0xBDAB, 0,0,0,0,0,0,0,0,0,0,0,0,0};
    intv16 shuffle6 = {0x9AC5A928, 0xCDEB387B, 0xFFBB, 0,0,0,0,0,0,0,0,0,0,0,0,0};

    floatv8 a1,a2,a3,a4,a5,a6,a7,a8;

    doublev8 vec_double;
    floatv8 vec_float;

    for(int local_now = local_start; local_now < local_end; ++local_now){

        start_A = src_A + MK_size * local_now;
        start_B = src_B + KN_size * local_now;
        start_C = src_C + MN_size * local_now;

        for(int c_M = 0; c_M < num_M; c_M++){//K N M order

            curr_blk_M = c_M < num_M - 1 ? blk_M : rem_blk_M;
            curr_blk_M = num_M == 1 ? first_blk_M : curr_blk_M;

            for(int c_N = 0; c_N < num_N; c_N++){

                curr_blk_N = c_N < num_N - 1 ? blk_N : rem_blk_N;
                curr_blk_N = num_N == 1 ? first_blk_N : curr_blk_N;

                for(int i = 0; i < local_C_size; i++){
                    local_C[i + (1 - double_buffer_flag_C) * local_C_size] = 0;
                }
                for(int i = 0; i < local_C_size; i++){
                    local_C_temp[i] = 0;
                }

                for(int c_K = 0; c_K < num_K; c_K++){

                    curr_blk_K = c_K < num_K - 1 ? blk_K : rem_blk_K;
                    curr_blk_K = num_K == 1 ? first_blk_K : curr_blk_K;

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
                                                sizeof(float) * first_blk_K * first_blk_M,
                                                sizeof(float) * first_blk_M, 
                                                sizeof(float) * (M - first_blk_M),
                                                &reply_get_A);
                        athread_dma_iget_stride(local_B + (double_buffer_flag_AB * local_B_size), 
                                                start_B + KN_size, 
                                                sizeof(float) * first_blk_K * first_blk_N, 
                                                sizeof(float) * first_blk_N, 
                                                sizeof(float) * (N - first_blk_N),
                                                &reply_get_B);
                    }
                    //align A
                    if((curr_blk_M == first_blk_M && first_blk_M_not_aligned) 
                    || (curr_blk_K == first_blk_K && first_blk_K_not_aligned)
                    || (curr_blk_M == rem_blk_M && rem_blk_M_not_aligned)
                    || (curr_blk_K == rem_blk_K && rem_blk_K_not_aligned)){

                        /* if(myid ==0){
                            printf("padding start\n");
                            for(int k = 0; k < blk_K; k++){
                                for(int m = 0; m < blk_M; m++){
                                    printf("%f ", local_A[k * blk_M + m + (1 - double_buffer_flag_AB) * local_A_size]);
                                }
                                printf("\n");
                            }
                        } */

                        for(int k = curr_blk_K - 1; k >= 0; k--){
                            for(int m = curr_blk_M - 1; m >= 0; m--){
                                local_A[k * blk_M + m + (1 - double_buffer_flag_AB) * local_A_size] 
                                = local_A[k * curr_blk_M + m + + (1 - double_buffer_flag_AB) * local_A_size];
                            }
                            for(int m = blk_M - 1; m >= curr_blk_M; m--){
                                local_A[k * blk_M + m + (1 - double_buffer_flag_AB) * local_A_size] = 0;
                            }
                        }
                        for(int k = blk_K - 1; k >= curr_blk_K; k--){
                            for(int m = blk_M - 1; m >= 0; m--){
                                local_A[k * blk_M + m + (1 - double_buffer_flag_AB) * local_A_size] = 0;
                            }
                        }

                        /* if(myid ==0){
                            printf("padding end\n");
                            for(int k = 0; k < blk_K; k++){
                                for(int m = 0; m < blk_M; m++){
                                    printf("%f ", local_A[k * blk_M + m + (1 - double_buffer_flag_AB) * local_A_size]);
                                }
                                printf("\n");
                            }
                        } */
                    }
                    //align B
                    if((curr_blk_K == first_blk_K && first_blk_K_not_aligned)
                    || (curr_blk_N == first_blk_N && first_blk_N_not_aligned)
                    || (curr_blk_K == rem_blk_K && rem_blk_K_not_aligned)
                    || (curr_blk_N == rem_blk_N && rem_blk_N_not_aligned)){
                        /* if(myid == 0){
                            printf("padding start\n");
                            for(int k = 0; k < blk_K; k++){
                                for(int n = 0; n < blk_N; n++){
                                    printf("%f ", local_B[k * blk_N + n + (1 - double_buffer_flag_AB) * local_B_size]);
                                }
                                printf("\n");
                            }
                        } */
                        for(int k = curr_blk_K - 1; k >= 0; k--){
                            for(int n = curr_blk_N - 1; n >= 0; n--){
                                local_B[k * blk_N + n + (1 - double_buffer_flag_AB) * local_B_size]
                                = local_B[k * curr_blk_N + n + (1 - double_buffer_flag_AB) * local_B_size];
                            }
                            for(int n = blk_N - 1; n >= curr_blk_N; n--){
                                local_B[k * blk_N + n + (1 - double_buffer_flag_AB) * local_B_size] = 0;
                            }
                        }
                        for(int k = blk_K - 1; k >= curr_blk_K; k--){
                            for(int n = blk_N - 1; n >= 0; n--){
                                local_B[k * blk_N + n + (1 - double_buffer_flag_AB) * local_B_size] = 0;
                            }
                        }
                        /* if(myid == 0){
                            printf("padding end\n");
                            for(int k = 0; k < blk_K; k++){
                                for(int n = 0; n < blk_N; n++){
                                    printf("%f ", local_B[k * blk_N + n + (1 - double_buffer_flag_AB) * local_B_size]);
                                }
                                printf("\n");
                            }
                        } */
                    }

                    /* for(int k = 0; k < blk_K; k++){
                        for(int m = 0; m < blk_M; m++){
                            trans_A[m * blk_K + k] = local_A[k * blk_M + m + (1 - double_buffer_flag_AB) * local_A_size];
                        }
                    } */

                    float* origin_A = local_A + (1 - double_buffer_flag_AB) * local_A_size;
                    for(int k = 0; k < blk_K; k += 8){
                        for(int m = 0; m < blk_M; m += 8){
                            int index = k * blk_M + m;
                            simd_load(a1, origin_A + index);
                            index += blk_M;
                            simd_load(a2, origin_A + index);
                            index += blk_M;
                            simd_load(a3, origin_A + index);
                            index += blk_M;
                            simd_load(a4, origin_A + index);
                            index += blk_M;
                            simd_load(a5, origin_A + index);
                            index += blk_M;
                            simd_load(a6, origin_A + index);
                            index += blk_M;
                            simd_load(a7, origin_A + index);
                            index += blk_M;
                            simd_load(a8, origin_A + index);
                            
                            SWAP_8_8(a1,a2,a3,a4,a5,a6,a7,a8);

                            index = m * blk_K + k;
                            simd_store(a1,trans_A + index);
                            index += blk_K;
                            simd_store(a2,trans_A + index);
                            index += blk_K;
                            simd_store(a3,trans_A + index);
                            index += blk_K;
                            simd_store(a4,trans_A + index);
                            index += blk_K;
                            simd_store(a5,trans_A + index);
                            index += blk_K;
                            simd_store(a6,trans_A + index);
                            index += blk_K;
                            simd_store(a7,trans_A + index);
                            index += blk_K;
                            simd_store(a8,trans_A + index);
                        }
                    }

                    float* comp_A = trans_A;
                    float* comp_B = local_B + (1 - double_buffer_flag_AB) * local_B_size;
                    double* comp_C = local_C_temp;

                    for(int m = 0; m < blk_M; m += 8){
                        for(int n = 0; n <blk_N; n += 8){
                            for(int k = 0; k < blk_K; k += 8){
                                sgemm_8_8_8(comp_A + m * blk_K + k, 
                                            comp_B + k * blk_N + n, 
                                            comp_C + m * blk_N + n, 
                                            blk_K, blk_N);
                            }
                        }
                    }


                    /* for(int m = 0; m < blk_M; m++){
                        for(int n = 0; n < blk_N; n++){
                            for(int k = 0; k < blk_K; k++){
                                local_C[m * blk_N + n + (1 - double_buffer_flag_C) * local_C_size]
                             += trans_A[m * blk_K + k]
                              * local_B[k * blk_N + n + (1 - double_buffer_flag_AB) * local_B_size];
                            }
                        }
                    } */


                    /* for(int m = 0; m < curr_blk_M; m++)
                        for(int n = 0; n < curr_blk_N; n++)
                            for(int k = 0; k < curr_blk_K; k++){
                                local_C[(1 - double_buffer_flag_C) * local_C_size + m * curr_blk_N + n]
                             += local_A[(1 - double_buffer_flag_AB) * local_A_size + k * curr_blk_M + m]
                              * local_B[(1 - double_buffer_flag_AB) * local_B_size + k * curr_blk_N + n];
                            } */
                    //gemm
                    if(c_N * num_M * num_K + c_M * num_K + c_K +1 < num_M * num_N * num_K || local_now < local_end - 1){
                        athread_dma_wait_value(&reply_get_A, 1);
                        athread_dma_wait_value(&reply_get_B, 1);
                        reply_get_A = 0;
                        reply_get_B = 0;
                        double_buffer_flag_AB = 1 - double_buffer_flag_AB;
                    }
                }

                //double to float
                /* for(int m = 0; m < blk_M; m++){
                    for(int n = 0; n < blk_N; n++){
                        local_C[m * blk_N + n + (1 - double_buffer_flag_C) * local_C_size] 
                        -= local_C_temp[m * blk_N + n];
                    }
                } */

                for(int i = 0; i < blk_M * blk_N; i += 8){
                    simd_load(vec_double, local_C_temp + i);
                    vec_float = - (floatv8)vec_double;
                    simd_store(vec_float, local_C + i + (1 - double_buffer_flag_C) * local_C_size);
                }

                if((curr_blk_M == first_blk_M && first_blk_M_not_aligned)
                || (curr_blk_N == first_blk_N && first_blk_N_not_aligned)
                || (curr_blk_M == rem_blk_M && rem_blk_M_not_aligned)
                || (curr_blk_N == rem_blk_N && rem_blk_N_not_aligned)){
                    /* if(myid == 0){
                            printf("unpadding start\n");
                            for(int m = 0; m < blk_M; m++){
                                for(int n = 0; n < blk_N; n++){
                                    printf("%f ", local_C[m * blk_N + n + (1 - double_buffer_flag_C) * local_C_size]);
                                }
                                printf("\n");
                            }
                        } */
                    for(int m = 0; m < curr_blk_M; m++){
                        for(int n = 0; n < curr_blk_N; n++){
                            local_C[m * curr_blk_N + n + (1 - double_buffer_flag_C) * local_C_size]
                            = local_C[m * blk_N + n + (1 - double_buffer_flag_C) * local_C_size];
                        }
                    }
                    /* if(myid == 0){
                            printf("unpadding end\n");
                            for(int m = 0; m < blk_M; m++){
                                for(int n = 0; n < blk_N; n++){
                                    printf("%f ", local_C[m * blk_N + n + (1 - double_buffer_flag_C) * local_C_size]);
                                }
                                printf("\n");
                            }
                        } */
                }

                athread_dma_put_stride(start_C + c_M * blk_M * N + c_N * blk_N,
                                       local_C + (1 - double_buffer_flag_C) * local_C_size,
                                       sizeof(float) * curr_blk_M * curr_blk_N,
                                       sizeof(float) * curr_blk_N,
                                       sizeof(float) * (N - curr_blk_N));
                /* athread_dma_wait_value(&reply_put_C, 1);
                reply_put_C = 0;
                athread_dma_iput_stride(start_C + c_M * blk_M * N + c_N * blk_N,
                                        local_C + (1 - double_buffer_flag_C) * local_C_size,
                                        sizeof(float) * curr_blk_M * curr_blk_N,
                                        sizeof(float) * curr_blk_N,
                                        sizeof(float) * (N - curr_blk_N),
                                        &reply_put_C);
                double_buffer_flag_C = 1 - double_buffer_flag_C; */
            }
        }
    }
    ldm_free(local_A,sizeof(float) * blk_M * blk_K * 2);
    ldm_free(local_B,sizeof(float) * blk_K * blk_N * 2);
    ldm_free(local_C,sizeof(float) * blk_M * blk_N * 2);
    ldm_free(trans_A,sizeof(float) * blk_M * blk_K);
    ldm_free(local_C_temp,sizeof(double) * blk_M * blk_N);
}

void sw_slave_mm_ABT(swptex_mmPara *_){
    int myid = CRTS_cgn * 64 + CRTS_tid;
    swptex_mmPara *para = (swptex_mmPara *)para_cross;
    const float *A = para->A;
    const float *B = para->B;
    float *C = para->C;
    int M = para->M;
    int N = para->N;
    int K = para->K;
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
    int M = para->M;
    int N = para->N;
    int K = para->K;
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
    int M = para->M;
    int N = para->N;
    int K = para->K;
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
