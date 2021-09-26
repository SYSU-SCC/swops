__cross void *para_cross_1 = NULL;

__cross void *para_cross_2 = NULL;

void sw_slave_bmm_rrr_flx(const int ThreadsStart, const int ThreadsEnd, 
                            sw_bmmPara *para){

    const int ThreadsNum = ThreadsEnd - ThreadsStart + 1;
    const int myid = CRTS_cgn * 64 + CRTS_tid - ThreadsStart;

    if(ThreadsNum <= 0){
        printf("sw_slave_bmm_rrr_flx ThreadsNum Error!\n");
        return;
    }
    if(myid < 0){
        //printf("_MYID %d sw_slave_gemm_copy_all myid %d return!\n", CRTS_cgn * 64 + CRTS_tid, myid);
        return;
    }

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

    const int local_count = (counts + ThreadsNum - 1) / ThreadsNum;
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

#ifdef _SWOPS_DEBUG
    if(myid == 0){
        printf("rem_blk_M %d rem_blk_N %d rem_blk_K %d\n", rem_blk_M, rem_blk_N, rem_blk_K);
        printf("rem_blk_M_not_aligned %d rem_blk_N_not_aligned %d rem_blk_K_not_aligned %d\n", rem_blk_M_not_aligned, rem_blk_N_not_aligned, rem_blk_K_not_aligned);
    }
#endif
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

void sw_slave_bmm_rcr_flx(const int ThreadsStart, const int ThreadsEnd, 
                            sw_bmmPara *para){

    const int ThreadsNum = ThreadsEnd - ThreadsStart + 1;
    const int myid = CRTS_cgn * 64 + CRTS_tid - ThreadsStart;

    if(ThreadsNum <= 0){
        printf("sw_slave_bmm_rrr_flx ThreadsNum Error!\n");
        return;
    }
    if(myid < 0){
        //printf("_MYID %d sw_slave_gemm_copy_all myid %d return!\n", CRTS_cgn * 64 + CRTS_tid, myid);
        return;
    }

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

    const int local_count = (counts + ThreadsNum - 1) / ThreadsNum;
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
#ifdef _SWOPS_DEBUG
    if(myid == 0){
        printf("rem_blk_M %d rem_blk_N %d rem_blk_K %d\n", rem_blk_M, rem_blk_N, rem_blk_K);
        printf("rem_blk_M_not_aligned %d rem_blk_N_not_aligned %d rem_blk_K_not_aligned %d\n", rem_blk_M_not_aligned, rem_blk_N_not_aligned, rem_blk_K_not_aligned);
    }
#endif
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

void sw_slave_bmm_crr_flx(const int ThreadsStart, const int ThreadsEnd, 
                            sw_bmmPara *para){

    const int ThreadsNum = ThreadsEnd - ThreadsStart + 1;
    const int myid = CRTS_cgn * 64 + CRTS_tid - ThreadsStart;

    if(ThreadsNum <= 0){
        printf("sw_slave_bmm_rrr_flx ThreadsNum Error!\n");
        return;
    }
    if(myid < 0){
        //printf("_MYID %d sw_slave_gemm_copy_all myid %d return!\n", CRTS_cgn * 64 + CRTS_tid, myid);
        return;
    }

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

    const int local_count = (counts + ThreadsNum - 1) / ThreadsNum;
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
#ifdef _SWOPS_DEBUG
    if(myid == 0){
        printf("rem_blk_M %d rem_blk_N %d rem_blk_K %d\n", rem_blk_M, rem_blk_N, rem_blk_K);
        printf("rem_blk_M_not_aligned %d rem_blk_N_not_aligned %d rem_blk_K_not_aligned %d\n", rem_blk_M_not_aligned, rem_blk_N_not_aligned, rem_blk_K_not_aligned);
    }
#endif
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

void sw_slave_bmm_flx(sw_bmmPara *_){
    sw_bmmPara *para;
    int Threadstart = 0;
    int Threadend = 383;
    if(CRTS_cgn < 3){
        para = (sw_bmmPara *)para_cross_1;
        Threadstart = 0;
        Threadend = 191;
    }
    else{
        para = (sw_bmmPara *)para_cross_2;
        Threadstart = 192;
        Threadend = 383;
    }
    if(para->transposeA && (!para->transposeB)){
        sw_slave_bmm_crr_flx(Threadstart, Threadend, para);
    }
    else if((!para->transposeA) && para->transposeB){
        sw_slave_bmm_rcr_flx(Threadstart, Threadend, para);
    }
    else{
        sw_slave_bmm_rrr_flx(Threadstart, Threadend, para);
    }
}
