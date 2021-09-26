

void sw_bmm_double(const void *A_1, const void *B_1, void *C_1,
                    const int M_1, const int N_1, const int K_1,
                    const int transposeA_1, const int transposeB_1,
                    const void *A_2, const void *B_2, void *C_2,
                    const int M_2, const int N_2, const int K_2,
                    const int transposeA_2, const int transposeB_2,
                    const int batch){
    sw_bmmPara para;

    sw_bmmPara para_1;
    int blk_M_1 = 8;
    int blk_N_1 = 8;
    int blk_K_1 = 8;

    sw_bmm_get_best_blk(M_1, N_1, K_1, transposeA_1, transposeB_1, &blk_M_1, &blk_N_1, &blk_K_1);

    para_1.A = A_1;
    para_1.B = B_1;
    para_1.C = C_1;
    para_1.M = M_1;
    para_1.N = N_1;
    para_1.K = K_1;
    para_1.transposeA = transposeA_1;
    para_1.transposeB = transposeB_1;
    para_1.blk_M = blk_M_1;
    para_1.blk_N = blk_N_1;
    para_1.blk_K = blk_K_1;
    para_1.counts = batch;

    para_cross_1 = &para_1;

    sw_bmmPara para_2;
    int blk_M_2 = 8;
    int blk_N_2 = 8;
    int blk_K_2 = 8;

    sw_bmm_get_best_blk(M_2, N_2, K_2, transposeA_2, transposeB_2, &blk_M_2, &blk_N_2, &blk_K_2);

    para_2.A = A_2;
    para_2.B = B_2;
    para_2.C = C_2;
    para_2.M = M_2;
    para_2.N = N_2;
    para_2.K = K_2;
    para_2.transposeA = transposeA_2;
    para_2.transposeB = transposeB_2;
    para_2.blk_M = blk_M_2;
    para_2.blk_N = blk_N_2;
    para_2.blk_K = blk_K_2;
    para_2.counts = batch;

    para_cross_2 = &para_2;

    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_slave_bmm_flx, &para);
    athread_join_cgs();
}



void test_sw_bmm_double_all(){

    int bn = 192;//384

    int M = 256;
    int N = 512;
    int K = 256;

    float *A_1 = _sw_xmalloc(sizeof(float) * bn * M * K);
    float *B_1 = _sw_xmalloc(sizeof(float) * bn * K * N);
    float *C_1 = _sw_xmalloc(sizeof(float) * bn * M * N);
    float *check_C_1 = _sw_xmalloc(sizeof(float) * bn * M * N);

    float *A_2 = _sw_xmalloc(sizeof(float) * bn * M * K);
    float *B_2 = _sw_xmalloc(sizeof(float) * bn * K * N);
    float *C_2 = _sw_xmalloc(sizeof(float) * bn * M * N);
    float *check_C_2 = _sw_xmalloc(sizeof(float) * bn * M * N);

#ifdef _SWOPS_DEBUG
    printf("A_1 ptr %p\n", A_1);
    printf("B_1 ptr %p\n", B_1);
    printf("C_1 ptr %p\n", C_1);
    printf("check_C_1 ptr %p\n", check_C_1);

    printf("A_2 ptr %p\n", A_2);
    printf("B_2 ptr %p\n", B_2);
    printf("C_2 ptr %p\n", C_2);
    printf("check_C_2 ptr %p\n", check_C_2);
#endif
    for (int i = 0; i < bn * M * K; i++){
        A_1[i] = rand()*1.0/RAND_MAX;
        A_2[i] = rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < bn * K * N; i++){
        B_1[i] = rand()*1.0/RAND_MAX;
        B_2[i] = rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < bn * M * N; i++){
        C_1[i] = 0;
        C_2[i] = 0;
    }
    for (int i = 0; i < bn * M * N; i++){
        check_C_1[i] = 0;
        check_C_2[i] = 0;
    }



    struct timeval tv1, tv2;

    gettimeofday(&tv1, NULL);
    sw_bmm_double(A_1, B_1, C_1, M, N, K, 1, 0,
                  A_2, B_2, C_2, M, N, K, 0, 0, bn);
    gettimeofday(&tv2, NULL);

    double optimized_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;

    gettimeofday(&tv1, NULL);

    //swptex_bmm(A_1,B_1,check_C_1,bn,M,N,K,1,0);
    //swptex_bmm(A_2,B_2,check_C_2,bn,M,N,K,0,0);

    sw_bmm(A_1,B_1,check_C_1,bn,M,N,K,1,0);
    sw_bmm(A_2,B_2,check_C_2,bn,M,N,K,0,0);

    gettimeofday(&tv2, NULL);

    double origin_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;

    for(int i = 0; i < bn * M * N; i++){
        if(NEQUAL_F(check_C_1[i], C_1[i])){
            printf("error at %d check_C: %f C %f\n", i, check_C_1[i], C_1[i]);
            break;
        }
    }
    for(int i = 0; i < bn * M * N; i++){
        if(NEQUAL_F(check_C_2[i], C_2[i])){
            printf("error at %d check_C: %f C %f\n", i, check_C_2[i], C_2[i]);
            break;
        }
    }
    printf("bmm original: %lf\n", origin_seconds);
    printf("bmm double launch: %lf\n", optimized_seconds);


    _sw_xfree(A_1);
    _sw_xfree(B_1);
    _sw_xfree(C_1);
    _sw_xfree(check_C_1);

    _sw_xfree(A_2);
    _sw_xfree(B_2);
    _sw_xfree(C_2);
    _sw_xfree(check_C_2);
}