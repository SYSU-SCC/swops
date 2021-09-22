#include <crts.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "args.h"

#define _F_PRECISION 1e-5
#define EQUAL_F(a, b) (fabs((a)-(b)) < (fabs(a)+fabs(b))*_F_PRECISION)
#define NEQUAL_F(a, b) (fabs((a)-(b)) >= (fabs(a)+fabs(b))*_F_PRECISION)

extern SLAVE_FUN(sw_slave_mm_AB)(swptex_mmPara_t);
extern SLAVE_FUN(sw_slave_mm_ATB)(swptex_mmPara_t);
extern SLAVE_FUN(sw_slave_mm_ABT)(swptex_mmPara_t);

extern SLAVE_FUN(sw_slave_gemm_crr_f32)(sw_gemmPara_t);
extern SLAVE_FUN(sw_slave_gemm_rrr4_f32)(sw_gemmPara_t);
extern SLAVE_FUN(sw_slave_gemm_rrr_f32)(sw_gemmPara_t);
extern SLAVE_FUN(sw_slave_gemm_rcr_cgn_f32)(sw_gemmPara_t);

extern SLAVE_FUN(sw_slave_gemm_copy_all_H_f32)(sw_gemmPara_t);
extern SLAVE_FUN(sw_slave_gemm_copy_all_f32)(sw_gemmPara_t);

extern SLAVE_FUN(sw_slave_gemm_rcr_all_cgn_no_trans_f32)(sw_gemmPara_t);
extern SLAVE_FUN(sw_slave_gemm_rrr_all_cgn_f32)(sw_gemmPara_t);
extern SLAVE_FUN(sw_slave_gemm_rcr_all_cgn_f32)(sw_gemmPara_t);
extern SLAVE_FUN(sw_slave_gemm_trans_f32)(sw_gemmPara_t);
extern SLAVE_FUN(sw_slave_padding_only_f32)(sw_gemmPara_t);
extern SLAVE_FUN(sw_slave_bmm_rrr)(sw_bmmPara_t);
extern SLAVE_FUN(sw_slave_bmm_rcr)(sw_bmmPara_t);
extern SLAVE_FUN(sw_slave_bmm_crr)(sw_bmmPara_t);











































//gemm final
extern SLAVE_FUN(sw_slave_gemm_rrr_sli_cgn_f32)(sw_gemmPara_t);
extern SLAVE_FUN(sw_slave_gemm_crr_sli_cgn_f32)(sw_gemmPara_t);

extern void *para_cross; // param on cross seg



float estimite_compute_time(int blkM, int blkN, int blkK, int M, int N, int K)
{
	//performance model achieves an estimited performance
	int bsizeN = blkN / 8 * sizeof(float);
	int bsizeM = blkM / 8 * sizeof(float);

	double a = 9.55371467e-09;
	double b = 4.80294349e-10;
	double c = 3.85210279e-11;//change here
	double d = 1.36105221e-05;
	double T_compute = (a * blkN + b * blkM * blkN + c * blkM * blkK * blkN + d) / 10 * M / blkM * K / blkK * N / blkN;
	return T_compute;
}

void get_best_blk_rrr(int M, int N, int K, int *best_blk_M, int *best_blk_N, int *best_blk_K){
    
    #define MAX(x,y) (x>y?x:y)
    double MBW_map[] = {3362.3000000000002, 6342.6000000000004, 9091.3999999999996, 11966.799999999999, 14464.4, 10109.4, 10826.799999999999, 13355.9, 14225.6, 16268.0, 17285.200000000001, 19322.400000000001, 20039.099999999999, 8748.6000000000004, 16397.0, 17568.099999999999, 18846.599999999999, 19078.799999999999, 17884.799999999999, 21040.299999999999, 21277.799999999999, 18181.299999999999, 18960.400000000001, 19724.799999999999, 20330.599999999999, 21263.700000000001, 21535.799999999999, 11486.1, 22908.099999999999, 19666.900000000001, 20302.700000000001, 21102.5, 21682.700000000001, 21875.700000000001, 22555.200000000001, 23501.799999999999, 21774.299999999999, 20105.700000000001, 21358.700000000001, 21932.099999999999, 21482.5, 19173.299999999999, 22579.200000000001, 23836.799999999999, 23775.400000000001, 21602.5, 21919.700000000001, 22429.5, 22826.400000000001, 23273.799999999999, 23630.599999999999, 24175.700000000001, 24429.099999999999, 22369.0, 21537.200000000001, 20850.400000000001, 21515.099999999999, 23762.900000000001, 23600.599999999999, 24484.400000000001, 24604.900000000001, 22800.0, 22921.599999999999, 23484.599999999999, 3390.3000000000002, 6033.3000000000002, 9180.6000000000004, 11876.4, 13766.799999999999, 10086.6, 10522.6, 13365.1, 14133.6, 16451.5, 17313.799999999999, 19445.799999999999, 19996.900000000001, 10185.9, 16442.299999999999, 17509.299999999999, 18274.200000000001, 19345.099999999999, 19300.099999999999, 20965.700000000001, 20429.400000000001, 18137.299999999999, 18710.0, 19884.799999999999, 20117.299999999999, 21051.299999999999, 21009.099999999999, 12401.799999999999, 22579.200000000001, 19770.599999999999, 20368.599999999999, 21030.200000000001, 21637.900000000001, 22160.900000000001, 22898.099999999999, 23052.599999999999, 22213.900000000001, 18106.900000000001, 21088.0, 21992.099999999999, 22231.400000000001, 18906.799999999999, 22901.299999999999, 23413.0, 23620.400000000001, 21629.400000000001, 21916.799999999999, 22201.5, 22933.200000000001, 23258.599999999999, 23628.900000000001, 24179.299999999999, 24568.700000000001, 22159.0, 22024.0, 21703.700000000001, 22000.900000000001, 23542.5, 23898.5, 24653.200000000001, 24752.200000000001, 22704.700000000001, 22908.099999999999, 23676.799999999999, 23607.799999999999, 24061.099999999999, 24265.099999999999, 24463.0, 24885.200000000001, 22663.700000000001, 23200.099999999999, 23771.5, 23822.200000000001, 23878.200000000001, 24192.599999999999, 24364.599999999999, 24847.700000000001, 23300.099999999999, 23664.5, 23730.5, 24296.799999999999, 24190.299999999999, 24188.599999999999, 24158.900000000001, 24806.599999999999, 23510.599999999999, 23840.200000000001, 24218.299999999999, 24235.200000000001, 24448.099999999999, 24828.099999999999, 25147.5, 25309.599999999999, 23574.0, 23646.700000000001, 23983.700000000001, 24372.200000000001, 24844.0, 24924.700000000001, 25109.700000000001, 25378.700000000001, 23925.400000000001, 24214.799999999999, 24551.299999999999, 24587.900000000001, 24818.299999999999, 25051.5, 25338.299999999999, 25317.400000000001, 24099.799999999999, 24158.900000000001, 24436.799999999999, 23581.200000000001, 24539.900000000001, 24685.900000000001, 24942.099999999999, 25381.200000000001, 24162.900000000001, 24270.900000000001, 24740.599999999999, 24684.099999999999, 24816.400000000001, 25079.599999999999, 25461.5, 25446.700000000001, 23707.400000000001, 23754.0, 24641.700000000001, 24930.400000000001, 25199.900000000001, 25365.099999999999, 25548.299999999999, 25656.099999999999, 24629.700000000001, 24703.0, 24873.5, 24806.599999999999, 25210.400000000001, 24867.400000000001, 24863.200000000001, 25700.5, 24606.799999999999, 24904.5, 24893.200000000001, 25083.400000000001, 25196.0, 25277.099999999999, 25601.799999999999, 24222.299999999999, 24871.599999999999, 24890.799999999999, 24978.0, 24897.0, 24061.099999999999, 25350.400000000001, 25539.799999999999, 25816.200000000001, 24777.299999999999, 25125.5, 24796.799999999999, 25261.099999999999, 25498.400000000001, 25510.700000000001, 25738.099999999999, 25643.599999999999, 24911.099999999999, 24659.200000000001, 25003.599999999999, 25063.900000000001, 25409.5, 25566.599999999999, 26173.099999999999, 25681.5, 24767.5, 25033.900000000001, 25204.200000000001, 25141.799999999999, 25419.799999999999, 25579.5, 25675.5, 25702.5, 25407.5, 24873.5, 25331.900000000001, 25430.599999999999, 24968.099999999999, 24645.400000000001, 25573.0, 25950.200000000001, 25004.099999999999, 25081.5, 25333.900000000001, 25364.599999999999, 25558.599999999999, 25914.599999999999, 25814.200000000001, 25803.599999999999, 25248.599999999999, 25129.799999999999, 25356.799999999999, 25370.900000000001, 25616.700000000001, 25525.0, 25742.700000000001, 25208.0, 25129.299999999999, 25232.599999999999, 25451.200000000001, 25446.700000000001, 25710.5, 25742.700000000001, 25948.700000000001, 25412.0, 25397.299999999999, 25393.400000000001, 25510.700000000001, 25635.599999999999, 25446.700000000001, 25925.299999999999, 25914.099999999999, 25230.700000000001, 25107.299999999999, 23748.400000000001, 25196.0, 24577.900000000001, 25139.400000000001, 25759.200000000001, 25725.599999999999, 25258.700000000001, 25432.0, 25490.0, 25510.700000000001, 25539.400000000001, 25746.700000000001, 25806.099999999999, 25706.5, 25155.700000000001, 25206.099999999999, 25381.200000000001, 25307.200000000001, 25506.299999999999, 25769.799999999999, 25683.5, 25912.0, 24691.5, 25299.400000000001, 25432.5, 25417.799999999999, 25702.5, 25772.299999999999, 24928.5, 26051.700000000001, 25049.599999999999, 25459.5, 25444.799999999999};
    float est_best_time = 1000000;

    int temp_M = M > 64 ? M : 64;
    int temp_N = N > 64 ? N : 64;
    int temp_K = K > 64 ? K : 64;
    int temp_blk_M = 64;
    int temp_blk_N = 64;
    int temp_blk_K = 64;
    int blk_M = 64;
    int blk_N = 64;
    int blk_K = 64;

    int ldm_use = sizeof(float) * (4 * temp_N * temp_K + 4 * temp_K * 64 + 3 * temp_N * 64) / 64;// try blk_M = 64

    if(ldm_use < 220 * 1024 && temp_N % 64 == 0 && temp_K % 64 == 0){
        blk_N = temp_N;
        blk_K = temp_K;
        for(blk_M = 64; blk_M <= temp_M && blk_M <= 8192 && blk_M * 6 <= temp_M + blk_M/2; blk_M += 64){
            ldm_use = sizeof(float) * (4 * blk_N * blk_K + 4 * blk_K * blk_M + 3 * blk_N * blk_M ) / 64;
            if(ldm_use < 220 * 1024){
                /* int bsizeN = blk_N / 8 * sizeof(float);
			    int bsizeM = blk_M / 8 * sizeof(float);

                double T_dma = temp_N / blk_N * temp_M / blk_M * temp_K / blk_K * (1.0 * blk_N * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1] + 1.0 * blk_M * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1]) +
								1.0 * temp_N / blk_N * temp_M / blk_M * blk_M * blk_N * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1];

			    double T_init_dma = (1.0 * blk_N * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1] +
									1.0 * blk_M * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeM / 16 - 1]);
			    float T_compute = estimite_compute_time(blk_M, blk_N, blk_K, temp_M, temp_N, temp_K);
                double est_time = MAX(T_compute, T_dma) + T_init_dma;
                if(est_time < est_best_time){
				    est_best_time = est_time;
                    temp_blk_M = blk_M;
				    temp_blk_N = blk_N;						
                    temp_blk_K = blk_K;
			    } */
                temp_blk_M = blk_M;
				temp_blk_N = blk_N;						
                temp_blk_K = blk_K;
            }
        }
    }
    else{
        for(blk_M = 64; blk_M <= temp_M && blk_M <= 8192; blk_M += 64){
            for(blk_N = 64; blk_N <= temp_N && blk_N <= 8192; blk_N += 64){
                for(blk_K = 64; blk_K <= temp_K && blk_K <= 8192; blk_K += 64){
                    ldm_use = sizeof(float) * (4 * blk_N * blk_K + 4 * blk_K * blk_M + 3 * blk_N * blk_M ) / 64;
                    if(ldm_use < 220 * 1024 && temp_N % blk_N == 0 && temp_K % blk_K == 0 && blk_M * 6 <= temp_M + blk_M){
                        int bsizeN = blk_N / 8 * sizeof(float);
			            int bsizeM = blk_M / 8 * sizeof(float);

                        double T_dma = temp_N / blk_N * temp_M / blk_M * temp_K / blk_K * (1.0 * blk_N * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1] + 1.0 * blk_M * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1]) +
								1.0 * temp_N / blk_N * temp_M / blk_M * blk_M * blk_N * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1];

			            double T_init_dma = (1.0 * blk_N * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1] +
										    1.0 * blk_M * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeM / 16 - 1]);
			            
                        float T_compute = estimite_compute_time(blk_M, blk_N, blk_K, temp_M, temp_N, temp_K);
                        double est_time = MAX(T_compute, T_dma) + T_init_dma;
                        if(est_time < est_best_time){
				            est_best_time = est_time;
                            temp_blk_M = blk_M;
				            temp_blk_N = blk_N;						
                            temp_blk_K = blk_K;
			            }
                    }
                }
            }
        }
    }
    best_blk_M[0] = temp_blk_M;
    best_blk_N[0] = temp_blk_N;
    best_blk_K[0] = temp_blk_K;
#ifdef _SWOPS_DEBUG
    printf("temp_blk_M %d temp_blk_N %d temp_blk_K %d\n", temp_blk_M, temp_blk_N, temp_blk_K);
#endif
}

void get_best_blk_crr(int M, int N, int K, int *best_blk_M, int *best_blk_N, int *best_blk_K){
    
    #define MAX(x,y) (x>y?x:y)
    double MBW_map[] = {3362.3000000000002, 6342.6000000000004, 9091.3999999999996, 11966.799999999999, 14464.4, 10109.4, 10826.799999999999, 13355.9, 14225.6, 16268.0, 17285.200000000001, 19322.400000000001, 20039.099999999999, 8748.6000000000004, 16397.0, 17568.099999999999, 18846.599999999999, 19078.799999999999, 17884.799999999999, 21040.299999999999, 21277.799999999999, 18181.299999999999, 18960.400000000001, 19724.799999999999, 20330.599999999999, 21263.700000000001, 21535.799999999999, 11486.1, 22908.099999999999, 19666.900000000001, 20302.700000000001, 21102.5, 21682.700000000001, 21875.700000000001, 22555.200000000001, 23501.799999999999, 21774.299999999999, 20105.700000000001, 21358.700000000001, 21932.099999999999, 21482.5, 19173.299999999999, 22579.200000000001, 23836.799999999999, 23775.400000000001, 21602.5, 21919.700000000001, 22429.5, 22826.400000000001, 23273.799999999999, 23630.599999999999, 24175.700000000001, 24429.099999999999, 22369.0, 21537.200000000001, 20850.400000000001, 21515.099999999999, 23762.900000000001, 23600.599999999999, 24484.400000000001, 24604.900000000001, 22800.0, 22921.599999999999, 23484.599999999999, 3390.3000000000002, 6033.3000000000002, 9180.6000000000004, 11876.4, 13766.799999999999, 10086.6, 10522.6, 13365.1, 14133.6, 16451.5, 17313.799999999999, 19445.799999999999, 19996.900000000001, 10185.9, 16442.299999999999, 17509.299999999999, 18274.200000000001, 19345.099999999999, 19300.099999999999, 20965.700000000001, 20429.400000000001, 18137.299999999999, 18710.0, 19884.799999999999, 20117.299999999999, 21051.299999999999, 21009.099999999999, 12401.799999999999, 22579.200000000001, 19770.599999999999, 20368.599999999999, 21030.200000000001, 21637.900000000001, 22160.900000000001, 22898.099999999999, 23052.599999999999, 22213.900000000001, 18106.900000000001, 21088.0, 21992.099999999999, 22231.400000000001, 18906.799999999999, 22901.299999999999, 23413.0, 23620.400000000001, 21629.400000000001, 21916.799999999999, 22201.5, 22933.200000000001, 23258.599999999999, 23628.900000000001, 24179.299999999999, 24568.700000000001, 22159.0, 22024.0, 21703.700000000001, 22000.900000000001, 23542.5, 23898.5, 24653.200000000001, 24752.200000000001, 22704.700000000001, 22908.099999999999, 23676.799999999999, 23607.799999999999, 24061.099999999999, 24265.099999999999, 24463.0, 24885.200000000001, 22663.700000000001, 23200.099999999999, 23771.5, 23822.200000000001, 23878.200000000001, 24192.599999999999, 24364.599999999999, 24847.700000000001, 23300.099999999999, 23664.5, 23730.5, 24296.799999999999, 24190.299999999999, 24188.599999999999, 24158.900000000001, 24806.599999999999, 23510.599999999999, 23840.200000000001, 24218.299999999999, 24235.200000000001, 24448.099999999999, 24828.099999999999, 25147.5, 25309.599999999999, 23574.0, 23646.700000000001, 23983.700000000001, 24372.200000000001, 24844.0, 24924.700000000001, 25109.700000000001, 25378.700000000001, 23925.400000000001, 24214.799999999999, 24551.299999999999, 24587.900000000001, 24818.299999999999, 25051.5, 25338.299999999999, 25317.400000000001, 24099.799999999999, 24158.900000000001, 24436.799999999999, 23581.200000000001, 24539.900000000001, 24685.900000000001, 24942.099999999999, 25381.200000000001, 24162.900000000001, 24270.900000000001, 24740.599999999999, 24684.099999999999, 24816.400000000001, 25079.599999999999, 25461.5, 25446.700000000001, 23707.400000000001, 23754.0, 24641.700000000001, 24930.400000000001, 25199.900000000001, 25365.099999999999, 25548.299999999999, 25656.099999999999, 24629.700000000001, 24703.0, 24873.5, 24806.599999999999, 25210.400000000001, 24867.400000000001, 24863.200000000001, 25700.5, 24606.799999999999, 24904.5, 24893.200000000001, 25083.400000000001, 25196.0, 25277.099999999999, 25601.799999999999, 24222.299999999999, 24871.599999999999, 24890.799999999999, 24978.0, 24897.0, 24061.099999999999, 25350.400000000001, 25539.799999999999, 25816.200000000001, 24777.299999999999, 25125.5, 24796.799999999999, 25261.099999999999, 25498.400000000001, 25510.700000000001, 25738.099999999999, 25643.599999999999, 24911.099999999999, 24659.200000000001, 25003.599999999999, 25063.900000000001, 25409.5, 25566.599999999999, 26173.099999999999, 25681.5, 24767.5, 25033.900000000001, 25204.200000000001, 25141.799999999999, 25419.799999999999, 25579.5, 25675.5, 25702.5, 25407.5, 24873.5, 25331.900000000001, 25430.599999999999, 24968.099999999999, 24645.400000000001, 25573.0, 25950.200000000001, 25004.099999999999, 25081.5, 25333.900000000001, 25364.599999999999, 25558.599999999999, 25914.599999999999, 25814.200000000001, 25803.599999999999, 25248.599999999999, 25129.799999999999, 25356.799999999999, 25370.900000000001, 25616.700000000001, 25525.0, 25742.700000000001, 25208.0, 25129.299999999999, 25232.599999999999, 25451.200000000001, 25446.700000000001, 25710.5, 25742.700000000001, 25948.700000000001, 25412.0, 25397.299999999999, 25393.400000000001, 25510.700000000001, 25635.599999999999, 25446.700000000001, 25925.299999999999, 25914.099999999999, 25230.700000000001, 25107.299999999999, 23748.400000000001, 25196.0, 24577.900000000001, 25139.400000000001, 25759.200000000001, 25725.599999999999, 25258.700000000001, 25432.0, 25490.0, 25510.700000000001, 25539.400000000001, 25746.700000000001, 25806.099999999999, 25706.5, 25155.700000000001, 25206.099999999999, 25381.200000000001, 25307.200000000001, 25506.299999999999, 25769.799999999999, 25683.5, 25912.0, 24691.5, 25299.400000000001, 25432.5, 25417.799999999999, 25702.5, 25772.299999999999, 24928.5, 26051.700000000001, 25049.599999999999, 25459.5, 25444.799999999999};
    float est_best_time = 1000000;

    int temp_M = M > 64 ? M : 64;
    int temp_N = N > 64 ? N : 64;
    int temp_K = K > 64 ? K : 64;
    int temp_blk_M = 64;
    int temp_blk_N = 64;
    int temp_blk_K = 64;
    int blk_M = 64;
    int blk_N = 64;
    int blk_K = 64;

    int ldm_use = sizeof(float) * (4 * temp_N * 64 + 4 * 64 * temp_M + 3 * temp_N * temp_M) / 64;// try blk_M = 64

    if(ldm_use < 220 * 1024 && temp_M % 64 == 0 && temp_N % 64 == 0){
        blk_M = temp_M;
        blk_N = temp_N;
        for(blk_K = 64; blk_K <= temp_K && blk_K <= 8192 && blk_K * 6 <= temp_K + blk_K/2 ; blk_K += 64){
            ldm_use = sizeof(float) * (4 * blk_N * blk_K + 4 * blk_K * blk_M + 3 * blk_N * blk_M) / 64;
            if(ldm_use < 220 * 1024){

                /* int bsizeN = blk_N / 8 * sizeof(float);
			    int bsizeM = blk_M / 8 * sizeof(float);

                double T_dma = temp_N / blk_N * temp_M / blk_M * temp_K / blk_K * (1.0 * blk_N * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1] + 1.0 * blk_M * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1]) +
								1.0 * temp_N / blk_N * temp_M / blk_M * blk_M * blk_N * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1];

			    double T_init_dma = (1.0 * blk_N * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1] +
									1.0 * blk_M * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeM / 16 - 1]);
			    
                float T_compute = estimite_compute_time(blk_M, blk_N, blk_K, temp_M, temp_N, temp_K);
                
                double est_time = MAX(T_compute, T_dma) + T_init_dma;
                
                if(est_time < est_best_time){
				    est_best_time = est_time;
                    temp_blk_M = blk_M;
				    temp_blk_N = blk_N;						
                    temp_blk_K = blk_K;
			    } */
                temp_blk_M = blk_M;
				temp_blk_N = blk_N;						
                temp_blk_K = blk_K;
            }
        }
    }
    else{
        for(blk_M = 64; blk_M <= temp_M && blk_M <= 8192; blk_M += 64){
            for(blk_N = 64; blk_N <= temp_N && blk_N <= 8192; blk_N += 64){
                for(blk_K = 64; blk_K <= temp_K && blk_K <= 8192; blk_K += 64){
                    
                    ldm_use = sizeof(float) * (4 * blk_N * blk_K + 4 * blk_K * blk_M + 3 * blk_N * blk_M ) / 64;
                    
                    if(ldm_use < 220 * 1024 && temp_M % blk_M == 0 && temp_N % blk_N == 0 && blk_K * 6 <= temp_K + blk_K){
                        
                        int bsizeN = blk_N / 8 * sizeof(float);
			            int bsizeM = blk_M / 8 * sizeof(float);

                        double T_dma = temp_N / blk_N * temp_M / blk_M * K / blk_K * (1.0 * blk_N * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1] + 1.0 * blk_M * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1]) +
								1.0 * temp_N / blk_N * temp_M / blk_M * blk_M * blk_N * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1];

			            double T_init_dma = (1.0 * blk_N * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1] +
										    1.0 * blk_M * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeM / 16 - 1]);
			            
                        float T_compute = estimite_compute_time(blk_M, blk_N, blk_K, temp_M, temp_N, temp_K);
                        
                        double est_time = MAX(T_compute, T_dma) + T_init_dma;
                        
                        if(est_time < est_best_time){
				            est_best_time = est_time;
                            temp_blk_M = blk_M;
				            temp_blk_N = blk_N;						
                            temp_blk_K = blk_K;
			            }
                    }
                }
            }
        }
    }
    best_blk_M[0] = temp_blk_M;
    best_blk_N[0] = temp_blk_N;
    best_blk_K[0] = temp_blk_K;
#ifdef _SWOPS_DEBUG
    printf("temp_blk_M %d temp_blk_N %d temp_blk_K %d\n", temp_blk_M, temp_blk_N, temp_blk_K);
#endif
}

void gemm_crr_all(float* A, float* B, float* C, int M, int N, int K){
#ifdef _SWOPS_DEBUG
    printf("gemm crr all API\n");
#endif
    /* if(K < N|| K < M){
        printf("GEMM CRR can't perform well\n");
        return;
    } */
    int blk_M = 64;
    int blk_N = 64;
    int blk_K = 64;
    int sli_M[6] = {0,0,0,0,0,0};
    int sli_N[6] = {0,0,0,0,0,0};
    int sli_K[6] = {0,0,0,0,0,0};

    get_best_blk_crr(M,N,K,&blk_M,&blk_N,&blk_K);

    int temp_M = M > 64 ? M : 64;
    int temp_N = N > 64 ? N : 64;
    int temp_K = K > 64 ? K : 64;

    int Ms = (temp_M / blk_M) * blk_M;
    int Ns = (temp_N / blk_N) * blk_N;
    int Ks = (temp_K / blk_K) * blk_K;
    int Me = temp_M % blk_M != 0 ? Ms + blk_M : Ms;
    int Ne = temp_N % blk_N != 0 ? Ns + blk_N : Ns;
    int Ke = temp_K % blk_K != 0 ? Ks + blk_K : Ks;

    int ldm_use = (4 * blk_M * blk_K + 4 * blk_K * blk_N + 3 * blk_M * blk_N) * sizeof(float)/64;
    printf("CRR M %d N %d K %d\nMs %d Ns %d Ks %d\nMe %d Ne %d Ke %d\nblk_M %d blk_N %d blk_K %d\n", M, N, K, Ms, Ns, Ks, Me, Ne, Ke, blk_M, blk_N, blk_K);
    printf("CRR ldm_use %d total_ldm %d\n", ldm_use, 220 * 1024);

    float* Ap = _sw_xmalloc(sizeof(float) * Me * Ke);
    float* Bp = _sw_xmalloc(sizeof(float) * Ke * Ne);
    float* Cp = _sw_xmalloc(sizeof(float) * 6 * Me * Ne);// all six
#ifdef _SWOPS_DEBUG
    printf("Ap ptr %p\n",Ap);
    printf("Bp ptr %p\n",Bp);
    printf("Cp ptr %p\n",Cp);
#endif

    int num_sli = (temp_K + blk_K - 1)/ blk_K;//the totol num of blk_M
#ifdef _SWOPS_DEBUG
    printf("num_sli %d\n", num_sli);
#endif
    const int counts_Q = num_sli;//count jobs

    int local_count = (num_sli + 6 - 1)/6;

    int local_start[6];
    int local_end[6];
    int local_sli[6];
    for(int i = 0; i < 6; i++){
        local_start[i] = i * local_count;
        local_end[i] = ((local_start[i] + local_count > counts_Q) ? counts_Q : (local_start[i] + local_count));
        local_sli[i] = local_end[i] - local_start[i];
#ifdef _SWOPS_DEBUG
        printf("cgn %d local_start %d local_end %d local_sli %d\n",i,local_start[i],local_end[i],local_sli[i]);
#endif    
    }

    for(int i = 0; i < 6; i++){
        sli_K[i] = local_sli[i] * blk_K;
    }

    sw_gemmPara para;

    para.A_sli[0] = A;
    para.Ap_sli[0] = Ap;
    para.B_sli[0] = B;
    para.Bp_sli[0] = Bp;
    //All result will be stored in Cp
    para.C_sli[0] = Cp;
    para.Cp_sli[0] = Cp;
    para.sli_C = 0;

    for(int i = 0; i < 6; i++){
        para.sli_K[i] = sli_K[i];
        if(sli_K[i] > 0){
            para.sli_C++;
        }
    }

    for(int i = 1; i < 6; i++){
        para.A_sli[i] = A + local_start[i] * blk_K * M;
        para.Ap_sli[i] = Ap + local_start[i] * blk_K * Me;
        para.B_sli[i] = B + local_start[i] * blk_K * N;
        para.Bp_sli[i] = Bp + local_start[i] * blk_K * Ne;
        para.C_sli[i] = Cp + i * Me * Ne;
        para.Cp_sli[i] = Cp + i * Me * Ne;
    }//All result will be stored in Cp

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
    ret = athread_spawn_cgs(sw_slave_gemm_crr_sli_cgn_f32, &para);
    athread_join_cgs();
#ifdef _SWOPS_DEBUG
    printf("Cp[0] %f Cp[1] %f Cp[2] %f\n", Cp[0], Cp[1], Cp[2]);
    printf("C[0] %f C[1] %f C[2] %f\n", C[0], C[1], C[2]);
#endif
    /* for(int s = 0; s < 6; s++){
        for(int m = 0; m < Me; m++){
            for(int n = 0; n < Ne; n++){
                if(Cp[s * Me * Ne + m * Ne + n] != 2688 && Cp[s * Me * Ne + m * Ne + n] != 0){
                    printf("Cp %d %d %d error, value %f\n", s, m, n, Cp[s * Me * Ne + m * Ne + n]);
                }
            }
        }
    } */

    /* for(int s = 0; s < para.sli_C; s++){
        for(int m = 0; m < M; m++){
            for(int n = 0; n < N; n++){
                C[m * N + n] += Cp[m * Ne + n + s * Me * Ne];
            }
        }
    } */
    

    _sw_xfree(Ap);
    _sw_xfree(Bp);
    _sw_xfree(Cp);
}

void gemm_rrr_all(float *A, float *B, float* C, int M, int N, int K){
#ifdef _SWOPS_DEBUG
    printf("gemm rrr all API\n");
#endif

    /* if(M < N || M < K){
        printf("GEMM RRR can't perform well\n");
        return;
    } */

    int blk_M = 64;
    int blk_N = 64;
    int blk_K = 64;

    int sli_M[6] = {0,0,0,0,0,0};
    int sli_N[6] = {0,0,0,0,0,0};
    int sli_K[6] = {0,0,0,0,0,0};

    get_best_blk_rrr(M, N, K, &blk_M, &blk_N, &blk_K);

    int temp_M = M > 64 ? M : 64;
    int temp_N = N > 64 ? N : 64;
    int temp_K = K > 64 ? K : 64;

    int ldm_use = (4 * blk_M * blk_K + 4 * blk_K * blk_N + 3 * blk_M * blk_N) * sizeof(float)/64;

    printf("M %d N %d K %d blk_M %d blk_N %d blk_K %d RRR ldm_use %d total_ldm %d\n", M, N, K, blk_M, blk_N, blk_K, ldm_use, 220 * 1024);


    int Ms = (temp_M / blk_M) * blk_M;
    int Ns = (temp_N / blk_N) * blk_N;
    int Ks = (temp_K / blk_K) * blk_K;
    int Me = temp_M % blk_M != 0 ? Ms + blk_M : Ms;
    int Ne = temp_N % blk_N != 0 ? Ns + blk_N : Ns;
    int Ke = temp_K % blk_K != 0 ? Ks + blk_K : Ks;

#ifdef _SWOPS_DEBUG
    printf("M %d N %d K %d blk_M %d blk_N %d blk_K %d\n", M, N, K, blk_M, blk_N, blk_K);
    printf("Ms %d Ns %d Ks %d Me %d Ne %d Ke %d\n", Ms, Ns, Ks, Me, Ne, Ke);
#endif

    float* Ap = _sw_xmalloc(sizeof(float) * Me * Ke);
    float* Bp = _sw_xmalloc(sizeof(float) * Ke * Ne);
    float* Cp = _sw_xmalloc(sizeof(float) * Me * Ne);
#ifdef _SWOPS_DEBUG
    printf("Ap ptr %p\n",Ap);
    printf("Bp ptr %p\n",Bp);
    printf("Cp ptr %p\n",Cp);
#endif


    int num_sli = (temp_M + blk_M - 1)/ blk_M;//the totol num of blk_M
#ifdef _SWOPS_DEBUG
    printf("num_sli %d\n", num_sli);
#endif

    const int counts_Q = num_sli;//count jobs

    int local_count = (num_sli + 6 - 1)/6;
    int local_start[6];
    int local_end[6];
    int local_sli[6];
    for(int i = 0; i < 6; i++){
        local_start[i] = i * local_count;
        local_end[i] = ((local_start[i] + local_count > counts_Q) ? counts_Q : (local_start[i] + local_count));
        local_sli[i] = local_end[i] - local_start[i];
#ifdef _SWOPS_DEBUG
        printf("cgn %d local_start %d local_end %d local_sli %d\n",i,local_start[i],local_end[i],local_sli[i]);
#endif    
    }

    for(int i = 0; i < 6; i++){
        sli_M[i] = local_sli[i] * blk_M;
    }

    sw_gemmPara para;


    para.A_sli[0] = A;
    para.Ap_sli[0] = Ap;
    para.B_sli[0] = B;
    para.Bp_sli[0] = Bp;
    para.C_sli[0] = C;
    para.Cp_sli[0] = Cp;

    for(int i = 0; i < 6; i++){
        para.sli_M[i] = sli_M[i];
    }

    for(int i = 1; i < 6; i++){
        para.A_sli[i] = A + local_start[i] * blk_M * K;
        para.Ap_sli[i] = Ap + local_start[i] * blk_M * Ke;
        para.B_sli[i] = B;
        para.Bp_sli[i] = Bp;
        para.C_sli[i] = C + local_start[i] * blk_M * N;
        para.Cp_sli[i] = Cp + local_start[i] * blk_M * Ne;
    }

    /* for(int i = 0; i < 6; i++){
        printf("para.A_sli[%d] %d ", i, para.A_sli[i]);
        printf("para.Ap_sli[%d] %d ", i, para.Ap_sli[i]);
        printf("para.B_sli[%d] %d ", i, para.B_sli[i]);
        printf("para.Bp_sli[%d] %d ", i, para.Bp_sli[i]);
        printf("para.C_sli[%d] %d ", i, para.C_sli[i]);
        printf("para.Cp_sli[%d] %d ", i, para.Cp_sli[i]);
        printf("\n");
    } */

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
    ret = athread_spawn_cgs(sw_slave_gemm_rrr_sli_cgn_f32, &para);
    athread_join_cgs();

    _sw_xfree(Ap);
    _sw_xfree(Bp);
    _sw_xfree(Cp);
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
            if(NEQUAL_F(C[m * N + n], check_C[m * N + n])){
                printf("checking C all error m %d n %d C %f check_C %f\n", m, n, C[m * N + n], check_C[m * N + n]);
                return ;
            }
        }
    }
    printf("checking C all passed\n");
}






void test_gemm_crr_all(){
    struct timeval tv1, tv2;
    int M = 4096;
    int N = 2048;
    int K = 6144;
    float *A = _sw_xmalloc(sizeof(float) * M * K);
    float *B = _sw_xmalloc(sizeof(float) * K * N);
    float *C = _sw_xmalloc(sizeof(float) * M * N);
    float *check_C = _sw_xmalloc(sizeof(float) * M * N);
#ifdef _SWOPS_DEBUG
    printf("A ptr %p\n",A);
    printf("B ptr %p\n",B);
    printf("C ptr %p\n",C);
    printf("check_C ptr %p\n",check_C);
#endif
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


    gettimeofday(&tv1, NULL);

    gemm_crr_all(A,B,C,M,N,K);

    gettimeofday(&tv2, NULL);

    double optimized_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
#ifdef _SWOPS_DEBUG
    printf("Result of gemm crr f32 cgn all, triple buffer with asm: %lf\n", optimized_seconds);
#endif
    

    gettimeofday(&tv1, NULL);

    //swptex_mm(A, B, check_C, M, N, K, 1, 0);

    gettimeofday(&tv2, NULL);

    double origin_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
#ifdef _SWOPS_DEBUG
    printf("Result of gemm crr f32 hardware cache: %lf\n", origin_seconds);
#endif

    check_C_all_f32(C, check_C, M, N);

    _sw_xfree(A);
    _sw_xfree(B);
    _sw_xfree(C);
    _sw_xfree(check_C);
    
}

void test_gemm_rrr_all(){
    struct timeval tv1, tv2;
    int M = 12288;
    int N = 2048;
    int K = 12288;
    float *A = _sw_xmalloc(sizeof(float) * M * K);
    float *B = _sw_xmalloc(sizeof(float) * K * N);
    float *C = _sw_xmalloc(sizeof(float) * M * N);
    float *check_C = _sw_xmalloc(sizeof(float) * M * N);
#ifdef _SWOPS_DEBUG
    printf("A ptr %p\n",A);
    printf("B ptr %p\n",B);
    printf("C ptr %p\n",C);
    printf("check_C ptr %p\n",check_C);
#endif

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

    double flops = (double)2 * K / 1024 * N / 1024 * M / 1024;

    gettimeofday(&tv1, NULL);

    gemm_rrr_all(A, B, C, M, N, K);

    gettimeofday(&tv2, NULL);

    double optimized_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("Result of gemm rrr f32 cgn all, quadruple buffer with asm time: %lf flops %lfG\n", optimized_seconds, flops/optimized_seconds);

    

    gettimeofday(&tv1, NULL);

    //swptex_mm(A, B, check_C, M, N, K, 0, 0);

    gettimeofday(&tv2, NULL);

    double origin_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("Result of gemm rrr f32 hardware cache: %lf\n", origin_seconds);

    check_C_all_f32(C, check_C, M, N);

    _sw_xfree(A);
    _sw_xfree(B);
    _sw_xfree(C);
    _sw_xfree(check_C);
}

void test_gemm_rrr(){
    struct timeval tv1, tv2;
    int M = 12800;
    int N = 128;
    int K = 1024;
    int blk_M = 640;
    int blk_N = 128;
    int blk_K = 1024;
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

    gettimeofday(&tv1, NULL);

    float* Ap = malloc(sizeof(float) * bn * Me * Ke);
    float* Bp = malloc(sizeof(float) * bn * Ke * Ne);
    float* Cp = malloc(sizeof(float) * bn * Me * Ne);

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
    printf("Result of gemm rrr f32 one cgn, triple buffer with asm no: %lf\n", optimized_seconds);

    //check_copy_border_f32(A, Ap, M, Ms, Me, blk_M, K, Ks, Ke, blk_K);
    //check_copy_border_f32(B, Bp, K, Ks, Ke, blk_K, N, Ns, Ne, blk_N);
    //check_A_B_f32(A, Ap, B, Bp, M, Ms, Me, K, Ks, Ke);

    //check_copy_border_f32(check_C, Cp, M, Ms, Me, blk_M, N, Ns, Ne, blk_N);

    gettimeofday(&tv1, NULL);
    swptex_mm(A,B,check_C,M,N,K,0,0);
    gettimeofday(&tv2, NULL);
    double origin_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("Result of gemm rrr f32 hardware cache: %lf\n", origin_seconds);

    check_C_all_f32(C, check_C, M, N);

    free(A);
    free(Ap);
    free(B);
    free(Bp);
    free(C);
    free(Cp);
    free(check_C);
}





















































void test_gemm_crr(){
    struct timeval tv1, tv2;
    int M = 64;
    int N = 768;
    int K = 12800;
    int blk_M = 64;
    int blk_N = 768;
    int blk_K = 1280;
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
    printf("Testing GEMM CRR F32 triple buffer asm no\n");
    gettimeofday(&tv1, NULL);

    float* Ap = malloc(sizeof(float) * bn * Me * Ke);
    float* Bp = malloc(sizeof(float) * bn * Ke * Ne);
    float* Cp = malloc(sizeof(float) * bn * Me * Ne);

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
    ret = athread_spawn_cgs(sw_slave_gemm_crr_f32, &para);
    athread_join_cgs();

    gettimeofday(&tv2, NULL);

    double optimized_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("Result of GEMM CRR F32 triple buffer asm no: %lf\n", optimized_seconds);

    //check_copy_border_f32(A, Ap, M, Ms, Me, blk_M, K, Ks, Ke, blk_K);
    //check_copy_border_f32(B, Bp, K, Ks, Ke, blk_K, N, Ns, Ne, blk_N);
    //check_A_B_f32(A, Ap, B, Bp, M, Ms, Me, K, Ks, Ke);
    printf("Testing GEMM CRR F32 hardware cache \n");
    gettimeofday(&tv1, NULL);
    swptex_mm(A,B,check_C,M,N,K,1,0);
    gettimeofday(&tv2, NULL);
    double origin_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("Result of GEMM CRR F32 hardware cache: %lf\n", origin_seconds);

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

void test_gemm_rrr4(){
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
    float *check_C = malloc(sizeof(float) * bn * M * N);//rand()*1.0/RAND_MAX;
    for(int m = 0; m < M; m++){
        for(int k = 0; k < K; k++){
            A[m * K + k] = k;
        }
    }
    for(int n = 0; n < N; n++){
        for(int k = 0; k < K; k++){
            B[k * N + n] = k;
        }
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
    ret = athread_spawn_cgs(sw_slave_gemm_rrr4_f32, &para);
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
    int N = 64;
    int K = 768;
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
    //M = M / 10;
    const int Ms = (M / blk_M) * blk_M;
    const int Ns = (N / blk_N) * blk_N;
    const int Ks = (K / blk_K) * blk_K;
    const int Me = M % blk_M != 0 ? Ms + blk_M : Ms;
    const int Ne = N % blk_N != 0 ? Ns + blk_N : Ns;
    const int Ke = K % blk_K != 0 ? Ks + blk_K : Ks;
    const int slice = 6;
    //const int slice = 10;

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

void test_copy_all_H(){

    int M = 32;
    int N = 1777;
    int K = 1777;
    int blk_M = 64;
    int blk_N = 512;
    int blk_K = 512;
    float *A = malloc(sizeof(float) * M * K);
    float *B = malloc(sizeof(float) * K * N);
    float *C = malloc(sizeof(float) * M * N);
    float *check_C = malloc(sizeof(float) * M * N);
    for (int i = 0; i < M * K; i++){
        A[i] = 1;
    }
    for (int i = 0; i < K * N; i++){
        B[i] = 1;
    }
    for (int i = 0; i < M * N; i++){
        C[i] = 0;
    }
    for (int i = 0; i < M * N; i++){
        check_C[i] = 0;
    }

    int Ms = (M / blk_M) * blk_M;
    int Ns = (N / blk_N) * blk_N;
    int Ks = (K / blk_K) * blk_K;
    int Me = M % blk_M != 0 ? Ms + blk_M : Ms;
    int Ne = N % blk_N != 0 ? Ns + blk_N : Ns;
    int Ke = K % blk_K != 0 ? Ks + blk_K : Ks;
    int copy_all_A = 0;
    int copy_all_B = 0;
    int copy_all_C = 0;

    if(M == 32){
        M = 32;
        blk_M = 64;
        Ms = 64;
        Me = 64;
        copy_all_A = 1;
        copy_all_B = 0;
        copy_all_C = 1;
    }

    float* Ap = malloc(sizeof(float) * Me * Ke);
    float *check_A = malloc(sizeof(float) * Me * Ke);
    float* Bp = malloc(sizeof(float) * Ke * Ne);
    float* Cp = malloc(sizeof(float) * Me * Ne);

    for(int i = 0; i < Me * Ke; i++){
        check_A[i] = 0;
    }

    for(int m = 0; m < M; m++){
        for(int k = 0; k < K; k++){
            check_A[m * Ke + k] = A[m * K + k];
        }
    }

    printf("K %d Ks %d Ke %d\n", K, Ks, Ke);


    sw_gemmPara para;

    para.blk_M = blk_M;
    para.blk_N = blk_N;
    para.blk_K = blk_K;
    para.A = A;
    para.Ap = Ap;
    para.copy_all_A = 1;
    para.B = B;
    para.Bp = Bp;
    para.copy_all_B = 0;
    para.C = C;
    para.Cp = Cp;
    para.copy_all_C = 1;
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
    ret = athread_spawn_cgs(sw_slave_gemm_copy_all_H_f32, &para);
    athread_join_cgs();

    check_C_all_f32(Ap, check_A, Me, Ke);

    printf("test_copy_all_H \n");

    printf("A ptr %d\n",A);
    printf("B ptr %d\n",B);
    printf("C ptr %d\n",C);
    printf("Ap ptr %d\n",Ap);
    printf("Bp ptr %d\n",Bp);
    printf("Cp ptr %d\n",Cp);
    printf("check_C ptr %d\n",check_C);
    printf("check_A ptr %d\n",check_A);

    free(A);
    free(B);
    free(C);
    free(Ap);
    free(Bp);
    free(Cp);
    free(check_C);
    free(check_A);

}

void test_copy_all(){

    int M = 15360;
    int N = 1024;
    int K = 64;
    int blk_M = 960;
    int blk_N = 1024;
    int blk_K = 64;
    float *A = malloc(sizeof(float) * M * K);
    float *A_back = malloc(sizeof(float) * M * K);
    float *B = malloc(sizeof(float) * K * N);
    float *C = malloc(sizeof(float) * M * N);
    float *check_C = malloc(sizeof(float) * M * N);
    /* for (int i = 0; i < M * K; i++){
        A_back[i] = 0;
    } */
    for (int i = 0; i < M * K; i++){
        A[i] = 1;//rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < K * N; i++){
        B[i] = 1;//rand()*1.0/RAND_MAX;
    }
    for (int i = 0; i < M * N; i++){
        C[i] = 0;
    }
    for (int i = 0; i < M * N; i++){
        check_C[i] = 0;
    }

    int Ms = (M / blk_M) * blk_M;
    int Ns = (N / blk_N) * blk_N;
    int Ks = (K / blk_K) * blk_K;
    int Me = M % blk_M != 0 ? Ms + blk_M : Ms;
    int Ne = N % blk_N != 0 ? Ns + blk_N : Ns;
    int Ke = K % blk_K != 0 ? Ks + blk_K : Ks;
    int copy_all_A = 0;
    int copy_all_B = 0;
    int copy_all_C = 0;
    int Mp = M;
    int Np = N;
    int Kp = K;

    printf("testing copy all W\n");

    float* Ap = malloc(sizeof(float) * Me * Ke);
    float *check_A = malloc(sizeof(float) * Me * Ke);
    float* Bp = malloc(sizeof(float) * Ke * Ne);
    float* Cp = malloc(sizeof(float) * Me * Ne);

    for(int i = 0; i < Me * Ke; i++){
        check_A[i] = 0;
    }

    for(int m = 0; m < M; m++){
        for(int k = 0; k < K; k++){
            check_A[m * Ke + k] = A[m * K + k];
        }
    }
    printf("M %d Ms %d Me %d\n", M, Ms, Me);
    printf("N %d Ns %d Ne %d\n", N, Ns, Ne);
    printf("K %d Ks %d Ke %d\n", K, Ks, Ke);
    sw_gemmPara para;
    para.blk_M = blk_M;
    para.blk_N = blk_N;
    para.blk_K = blk_K;
    para.A = A;
    para.Ap = Ap;
    para.copy_all_A = copy_all_A;
    para.B = A_back;
    para.Bp = Bp;
    para.copy_all_B = copy_all_B;
    para.C = C;
    para.Cp = Cp;
    para.copy_all_C = copy_all_C;
    para.M = M;
    para.Ms = Ms;
    para.Me = Me;
    para.Mp = Mp;
    para.N = N;
    para.Ns = Ns;
    para.Ne = Ne;
    para.Np = Np;
    para.K = K;
    para.Ks = Ks;
    para.Ke = Ke;
    para.Kp = Kp;
    para_cross = &para;

    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_slave_gemm_copy_all_f32, &para);
    athread_join_cgs();

    check_C_all_f32(Ap, check_A, Me, Ke);
    check_C_all_f32(A_back, A, M, K);

    printf("test_copy_all_W \n");

    printf("A ptr %d\n",A);
    printf("B ptr %d\n",B);
    printf("C ptr %d\n",C);
    printf("Ap ptr %d\n",Ap);
    printf("Bp ptr %d\n",Bp);
    printf("Cp ptr %d\n",Cp);
    printf("check_C ptr %d\n",check_C);
    printf("check_A ptr %d\n",check_A);

    free(A);
    free(B);
    free(C);
    free(Ap);
    free(Bp);
    free(Cp);
    free(A_back);
    free(check_C);
    free(check_A);

}

void gemm_rcr_all_cgn(float* A, float* B, float* C,int M,int N,int K){

    int blk_M = 8192;
    int blk_N = 64;
    int blk_K = 64;

    int local_A_LDM_size = blk_M * blk_K / 64;
    int local_B_LDM_size = blk_K * blk_N / 64;
    int local_C_LDM_size = blk_M * blk_N / 64;

    int Total_LDM_size = 3 * sizeof(float) * (local_A_LDM_size + local_B_LDM_size + local_C_LDM_size);
    if(Total_LDM_size > 200 * 1024){
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
    struct timeval tv1, tv2;
    int M = 8192;
    int N = 64;
    int K = 64;
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
    printf("GEMM size: M %d N %d K %d\n", M, N, K);
    printf("Testing GEMM RCR F32 triple buffer asm no\n");
    gettimeofday(&tv1, NULL);
    gemm_rcr_all_cgn(A,B,C,M,N,K);
    gettimeofday(&tv2, NULL);
    double optimized_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("Result of GEMM RCR F32 triple buffer asm no: %lf\n", optimized_seconds);

    printf("Testing GEMM RCR F32 hardware cache \n");
    gettimeofday(&tv1, NULL);
    swptex_mm(A ,B ,check_C ,M ,N ,K , 0, 1);
    gettimeofday(&tv2, NULL);

    double origin_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("Result of GEMM RCR F32 hardware cache: %lf\n", origin_seconds);


    check_C_all_f32(C, check_C, M, N);
    free(A);
    free(B);
    free(C);
    free(check_C);
}

void test_gemm_rcr_all_cgn(){
    struct timeval tv1, tv2;
    int M = 12800;
    int N = 256;
    int K = 64;
    int blk_M = 512;
    int blk_N = 256;
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

void test_gemm_rcr_cgn(){

    struct timeval tv1, tv2;
    int M = 12800;
    int N = 32;
    int K = 64;
    int blk_M = 512;
    int blk_N = 32;
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
    const int Ms = (M / blk_M) * blk_M;
    const int Ns = (N / blk_N) * blk_N;
    const int Ks = (K / blk_K) * blk_K;
    const int Me = M % blk_M != 0 ? Ms + blk_M : Ms;
    const int Ne = N % blk_N != 0 ? Ns + blk_N : Ns;
    const int Ke = K % blk_K != 0 ? Ks + blk_K : Ks;
    const int slice = 6;

    /* printf("M %d Ms %d Me %d blk_M %d\nN %d Ns %d Ne %d blk_N %d\nK %d Ks %d Ke %d blk_K %d\n",
            M,Ms,Me,blk_M,N,Ns,Ne,blk_N,K,Ks,Ke,blk_K); */
    printf("GEMM size: M %d N %d K %d\n", M, N, K);
    printf("Testing GEMM RCR F32 triple buffer asm no\n");

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
    gettimeofday(&tv1, NULL);
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
    para.T = BT;
    para.Tp = BTp;
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
    ret = athread_spawn_cgs(sw_slave_gemm_rcr_cgn_f32, &para);
    athread_join_cgs();

    gettimeofday(&tv2, NULL);

    double optimized_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("Result of GEMM RCR F32 triple buffer asm no: %lf\n", optimized_seconds);

    //check_copy_border_f32(A, Ap, M, Ms, Me, blk_M, K, Ks, Ke, blk_K);
    //check_copy_border_f32(B, Bp, K, Ks, Ke, blk_K, N, Ns, Ne, blk_N);
    //check_A_B_f32(A, Ap, B, Bp, M, Ms, Me, K, Ks, Ke);
    printf("Testing GEMM RCR F32 hardware cache \n");
    gettimeofday(&tv1, NULL);
    swptex_mm(A ,B ,check_C ,M ,N ,K , 0, 1);
    gettimeofday(&tv2, NULL);
    double origin_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;
    printf("Result of GEMM RCR F32 hardware cache: %lf\n", origin_seconds);

    //check_copy_border_f32(check_C, Cp, M, Ms, Me, blk_M, N, Ns, Ne, blk_N);
    check_C_all_f32(C, check_C, M, N);

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

void test_gemm_rcr_all_cgn_no_trans(){
    struct timeval tv1, tv2;
    int M = 12800;
    int N = 32;
    int K = 64;
    int blk_M = 12800;
    int blk_N = 32;
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
    printf("Testing GEMM RCR F32 triple buffer asm no\n");

    float* BT = malloc(sizeof(float) * bn * K * N);
    /* for(int n = 0; n < N; n++){
        for(int k = 0; k < K; k++){
            BT[k * N + n] = B[n * K + k];
        }
    } */

    float* Ap = malloc(sizeof(float) * slice * Me * Ke);
    float* Bp = malloc(sizeof(float) * Ke * Ne);
    float* BTp = malloc(sizeof(float) * Ke * Ne);
    float* Cp = malloc(sizeof(float) * slice * Me * Ne);

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
    gettimeofday(&tv1, NULL);
    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_slave_gemm_rcr_all_cgn_no_trans_f32, &para);
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

void sw_bmm_rrr(const void *A, const void *B, void *C, int batch, 
                int M, int N, int K, int blk_M, int blk_N, int blk_K){
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
    para.counts = batch;
    para_cross = &para;
    int LDM_size_Ax2 = 2 * para.blk_M * para.blk_K *sizeof(float);
    int LDM_size_Bx2 = 2 * para.blk_K * para.blk_N *sizeof(float);
    int LDM_size_Cx2 = 2 * para.blk_M * para.blk_N *sizeof(float);
    if(LDM_size_Ax2 + LDM_size_Bx2 + LDM_size_Cx2 >= 256 * 1024){
        printf("ldm_malloc error, size of blk: %d, max size: 262144\n",LDM_size_Ax2 + LDM_size_Bx2 + LDM_size_Cx2);
        return;
    }
    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_slave_bmm_rrr, &para);
    athread_join_cgs();
}

void sw_bmm_rcr(const void *A, const void *B, void *C, int batch, 
                int M, int N, int K, int blk_M, int blk_N, int blk_K){
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
    para.counts = batch;
    para_cross = &para;
    int LDM_size_Ax2 = 2 * para.blk_M * para.blk_K *sizeof(float);
    int LDM_size_Bx2 = 2 * para.blk_K * para.blk_N *sizeof(float);
    int LDM_size_Cx2 = 2 * para.blk_M * para.blk_N *sizeof(float);
    if(LDM_size_Ax2 + LDM_size_Bx2 + LDM_size_Cx2 >= 256 * 1024){
        printf("ldm_malloc error, size of blk: %d, max size: 262144\n",LDM_size_Ax2 + LDM_size_Bx2 + LDM_size_Cx2);
        return;
    }
    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_slave_bmm_rcr, &para);
    athread_join_cgs();
}

void sw_bmm_crr(const void *A, const void *B, void *C, int batch, 
                int M, int N, int K, int blk_M, int blk_N, int blk_K){
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
    para.counts = batch;
    para_cross = &para;
    int LDM_size_Ax2 = 2 * para.blk_M * para.blk_K *sizeof(float);
    int LDM_size_Bx2 = 2 * para.blk_K * para.blk_N *sizeof(float);
    int LDM_size_Cx2 = 2 * para.blk_M * para.blk_N *sizeof(float);
    if(LDM_size_Ax2 + LDM_size_Bx2 + LDM_size_Cx2 >= 256 * 1024){
        printf("ldm_malloc error, size of blk: %d, max size: 262144\n",LDM_size_Ax2 + LDM_size_Bx2 + LDM_size_Cx2);
        return;
    }
    int ret = athread_init_cgs();
    ret = athread_spawn_cgs(sw_slave_bmm_crr, &para);
    athread_join_cgs();
}

void sw_bmm(const void *A, const void *B, void *C, int batch, int M,
               int N, int K, int transposeA, int transposeB){
    const int temp_blk_M = (M/8 + 1) * 8;
    const int temp_blk_N = (N/8 + 1) * 8;
    const int temp_blk_K = (K/8 + 1) * 8;
    int best_blk_M = 64;
    int best_blk_N = 64;
    int best_blk_K = 64;//temp_blk_M * temp_blk_K * transposeA + 
    if((temp_blk_M * temp_blk_K * 2 + 
        temp_blk_K * temp_blk_N * 2 + 
        temp_blk_M * temp_blk_N * 4 + 
        temp_blk_M * temp_blk_K * transposeA + 
        temp_blk_K * temp_blk_N * transposeB )
        * sizeof(float) < 200 * 1024){
        best_blk_M = temp_blk_M;
        best_blk_N = temp_blk_N;
        best_blk_K = temp_blk_K;
    }
    else{
        float est_best_time = 1000000;
        double MBW_map[] = {3362.3000000000002, 6342.6000000000004, 9091.3999999999996, 11966.799999999999, 14464.4, 10109.4, 10826.799999999999, 13355.9, 14225.6, 16268.0, 17285.200000000001, 19322.400000000001, 20039.099999999999, 8748.6000000000004, 16397.0, 17568.099999999999, 18846.599999999999, 19078.799999999999, 17884.799999999999, 21040.299999999999, 21277.799999999999, 18181.299999999999, 18960.400000000001, 19724.799999999999, 20330.599999999999, 21263.700000000001, 21535.799999999999, 11486.1, 22908.099999999999, 19666.900000000001, 20302.700000000001, 21102.5, 21682.700000000001, 21875.700000000001, 22555.200000000001, 23501.799999999999, 21774.299999999999, 20105.700000000001, 21358.700000000001, 21932.099999999999, 21482.5, 19173.299999999999, 22579.200000000001, 23836.799999999999, 23775.400000000001, 21602.5, 21919.700000000001, 22429.5, 22826.400000000001, 23273.799999999999, 23630.599999999999, 24175.700000000001, 24429.099999999999, 22369.0, 21537.200000000001, 20850.400000000001, 21515.099999999999, 23762.900000000001, 23600.599999999999, 24484.400000000001, 24604.900000000001, 22800.0, 22921.599999999999, 23484.599999999999, 3390.3000000000002, 6033.3000000000002, 9180.6000000000004, 11876.4, 13766.799999999999, 10086.6, 10522.6, 13365.1, 14133.6, 16451.5, 17313.799999999999, 19445.799999999999, 19996.900000000001, 10185.9, 16442.299999999999, 17509.299999999999, 18274.200000000001, 19345.099999999999, 19300.099999999999, 20965.700000000001, 20429.400000000001, 18137.299999999999, 18710.0, 19884.799999999999, 20117.299999999999, 21051.299999999999, 21009.099999999999, 12401.799999999999, 22579.200000000001, 19770.599999999999, 20368.599999999999, 21030.200000000001, 21637.900000000001, 22160.900000000001, 22898.099999999999, 23052.599999999999, 22213.900000000001, 18106.900000000001, 21088.0, 21992.099999999999, 22231.400000000001, 18906.799999999999, 22901.299999999999, 23413.0, 23620.400000000001, 21629.400000000001, 21916.799999999999, 22201.5, 22933.200000000001, 23258.599999999999, 23628.900000000001, 24179.299999999999, 24568.700000000001, 22159.0, 22024.0, 21703.700000000001, 22000.900000000001, 23542.5, 23898.5, 24653.200000000001, 24752.200000000001, 22704.700000000001, 22908.099999999999, 23676.799999999999, 23607.799999999999, 24061.099999999999, 24265.099999999999, 24463.0, 24885.200000000001, 22663.700000000001, 23200.099999999999, 23771.5, 23822.200000000001, 23878.200000000001, 24192.599999999999, 24364.599999999999, 24847.700000000001, 23300.099999999999, 23664.5, 23730.5, 24296.799999999999, 24190.299999999999, 24188.599999999999, 24158.900000000001, 24806.599999999999, 23510.599999999999, 23840.200000000001, 24218.299999999999, 24235.200000000001, 24448.099999999999, 24828.099999999999, 25147.5, 25309.599999999999, 23574.0, 23646.700000000001, 23983.700000000001, 24372.200000000001, 24844.0, 24924.700000000001, 25109.700000000001, 25378.700000000001, 23925.400000000001, 24214.799999999999, 24551.299999999999, 24587.900000000001, 24818.299999999999, 25051.5, 25338.299999999999, 25317.400000000001, 24099.799999999999, 24158.900000000001, 24436.799999999999, 23581.200000000001, 24539.900000000001, 24685.900000000001, 24942.099999999999, 25381.200000000001, 24162.900000000001, 24270.900000000001, 24740.599999999999, 24684.099999999999, 24816.400000000001, 25079.599999999999, 25461.5, 25446.700000000001, 23707.400000000001, 23754.0, 24641.700000000001, 24930.400000000001, 25199.900000000001, 25365.099999999999, 25548.299999999999, 25656.099999999999, 24629.700000000001, 24703.0, 24873.5, 24806.599999999999, 25210.400000000001, 24867.400000000001, 24863.200000000001, 25700.5, 24606.799999999999, 24904.5, 24893.200000000001, 25083.400000000001, 25196.0, 25277.099999999999, 25601.799999999999, 24222.299999999999, 24871.599999999999, 24890.799999999999, 24978.0, 24897.0, 24061.099999999999, 25350.400000000001, 25539.799999999999, 25816.200000000001, 24777.299999999999, 25125.5, 24796.799999999999, 25261.099999999999, 25498.400000000001, 25510.700000000001, 25738.099999999999, 25643.599999999999, 24911.099999999999, 24659.200000000001, 25003.599999999999, 25063.900000000001, 25409.5, 25566.599999999999, 26173.099999999999, 25681.5, 24767.5, 25033.900000000001, 25204.200000000001, 25141.799999999999, 25419.799999999999, 25579.5, 25675.5, 25702.5, 25407.5, 24873.5, 25331.900000000001, 25430.599999999999, 24968.099999999999, 24645.400000000001, 25573.0, 25950.200000000001, 25004.099999999999, 25081.5, 25333.900000000001, 25364.599999999999, 25558.599999999999, 25914.599999999999, 25814.200000000001, 25803.599999999999, 25248.599999999999, 25129.799999999999, 25356.799999999999, 25370.900000000001, 25616.700000000001, 25525.0, 25742.700000000001, 25208.0, 25129.299999999999, 25232.599999999999, 25451.200000000001, 25446.700000000001, 25710.5, 25742.700000000001, 25948.700000000001, 25412.0, 25397.299999999999, 25393.400000000001, 25510.700000000001, 25635.599999999999, 25446.700000000001, 25925.299999999999, 25914.099999999999, 25230.700000000001, 25107.299999999999, 23748.400000000001, 25196.0, 24577.900000000001, 25139.400000000001, 25759.200000000001, 25725.599999999999, 25258.700000000001, 25432.0, 25490.0, 25510.700000000001, 25539.400000000001, 25746.700000000001, 25806.099999999999, 25706.5, 25155.700000000001, 25206.099999999999, 25381.200000000001, 25307.200000000001, 25506.299999999999, 25769.799999999999, 25683.5, 25912.0, 24691.5, 25299.400000000001, 25432.5, 25417.799999999999, 25702.5, 25772.299999999999, 24928.5, 26051.700000000001, 25049.599999999999, 25459.5, 25444.799999999999};
        for(int blk_M = 8; blk_M <= M && blk_M <= 8192; blk_M += 8){
            for(int blk_N = 8; blk_N <= N && blk_N <= 8192; blk_N += 8){
                for(int blk_K = 8; blk_K <= K && blk_K <= 8192; blk_K += 8){
                    int ldm_use = ( blk_M * blk_K * 2 + 
                                    blk_K * blk_N * 2 + 
                                    blk_M * blk_N * 4 + 
                                    blk_M * blk_K * transposeA + 
                                    blk_K * blk_N * transposeB )
                                    * sizeof(float);
                    if(ldm_use < 200 * 1024){
                        int bsizeN = blk_N * sizeof(float);
			            int bsizeM = blk_M * sizeof(float);
                        double T_dma = N / blk_N * M / blk_M * K / blk_K * (1.0 * blk_N * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1] + 1.0 * blk_M * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1]) +
									    1.0 * N / blk_N * M / blk_M * blk_M * blk_N * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1];

			            double T_init_dma = (1.0 * blk_N * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeN / 16 - 1] +
										    1.0 * blk_M * blk_K * sizeof(float) / 1e6 / MBW_map[bsizeM / 16 - 1]);
			            float T_compute = estimite_compute_time(blk_M, blk_N, blk_K, M, N, K);
                        double est_time = MAX(T_compute, T_dma) + T_init_dma;
                        if(est_time < est_best_time){
				            est_best_time = est_time;
                            best_blk_M = blk_M;
				            best_blk_N = blk_N;						
                            best_blk_K = blk_K;
			            }
                    }
                }
            }
        }
    }
    printf("blk_M %d blk_N %d blk_K %d\n", best_blk_M, best_blk_N, best_blk_K);
    if(transposeA && (!transposeB)){//crr
        sw_bmm_crr(A,B,C,batch,M,N,K,best_blk_M,best_blk_N,best_blk_K);
    }
    else if((!transposeA) && transposeB){
        sw_bmm_rcr(A,B,C,batch,M,N,K,best_blk_M,best_blk_N,best_blk_K);
    }
    else{
        sw_bmm_rrr(A,B,C,batch,M,N,K,best_blk_M,best_blk_N,best_blk_K);
    }
}

void test_sw_bmm_all(){
    int M = 77;
    int N = 131;
    int K = 719;
    int bn = 1536;//384
    float *A = _sw_xmalloc(sizeof(float) * bn * M * K);
    float *B = _sw_xmalloc(sizeof(float) * bn * K * N);
    float *C = _sw_xmalloc(sizeof(float) * bn * M * N);
    float *check_C = _sw_xmalloc(sizeof(float) * bn * M * N);
    printf("A ptr %p\n", A);
    printf("B ptr %p\n", B);
    printf("C ptr %p\n", C);
    printf("check_C ptr %p\n", check_C);
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
    sw_bmm(A,B,C,bn,M,N,K,0,0);
    gettimeofday(&tv2, NULL);

    double optimized_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;

    gettimeofday(&tv1, NULL);
    swptex_bmm(A,B,check_C,bn,M,N,K,0,0);
    gettimeofday(&tv2, NULL);

    double origin_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;

    for(int i = 0; i < bn * M * N; i++){
        if(NEQUAL_F(check_C[i], C[i])){
            printf("error at %d check_C: %f C %f\n", i, check_C[i], C[i]);
            break;
        }
    }
    printf("bmm rrr original: %lf\n", origin_seconds);
    printf("bmm rrr optimized: %lf\n", optimized_seconds);

    gettimeofday(&tv1, NULL);
    sw_bmm(A,B,C,bn,M,N,K,1,0);
    gettimeofday(&tv2, NULL);

    optimized_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;

    gettimeofday(&tv1, NULL);
    swptex_bmm(A,B,check_C,bn,M,N,K,1,0);
    gettimeofday(&tv2, NULL);

    origin_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;

    for(int i = 0; i < bn * M * N; i++){
        if(NEQUAL_F(check_C[i], C[i])){
            printf("error at %d check_C: %f C %f\n", i, check_C[i], C[i]);
            break;
        }
    }
    printf("bmm crr original: %lf\n", origin_seconds);
    printf("bmm crr optimized: %lf\n", optimized_seconds);

    gettimeofday(&tv1, NULL);
    sw_bmm(A,B,C,bn,M,N,K,0,1);
    gettimeofday(&tv2, NULL);

    optimized_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;

    gettimeofday(&tv1, NULL);
    swptex_bmm(A,B,check_C,bn,M,N,K,0,1);
    gettimeofday(&tv2, NULL);

    origin_seconds = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1.0e-6;

    for(int i = 0; i < bn * M * N; i++){
        if(NEQUAL_F(check_C[i], C[i])){
            printf("error at %d check_C: %f C %f\n", i, check_C[i], C[i]);
            break;
        }
    }
    printf("bmm rcr original: %lf\n", origin_seconds);
    printf("bmm rcr optimized: %lf\n", optimized_seconds);


    _sw_xfree(A);
    _sw_xfree(B);
    _sw_xfree(C);
    _sw_xfree(check_C);
}

int swptex_mm(const void *A, const void *B, void *C, int M, int N,
              int K, int transposeA, int transposeB)
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

int swptex_bmm(const void *A, const void *B, void *C, int batch, int M,
               int N, int K, int transposeA, int transposeB)
{
    int bn;
    for (bn = 0; bn < batch; ++bn)
    {
        swptex_mm((float *)A + bn * M * K, (float *)B + bn * N * K,
                  (float *)C + bn * M * N, M, N, K, transposeA, transposeB);
    }
}

int swptex_softmax(void *x_, int M, int N)
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

int swptex_dsoftmax(void *dy_, const void *y_, int M, int N)
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
                               int B, int N, int S, int D)
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
                               int B, int N, int S, int D)
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

int swptex_split(const void *QKV_, void *QKVT_, int B, int N, int S,
                 int D)
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

int swptex_merge(const void *QKV_, void *QKVT_, int B, int N, int S,
                 int D)
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

int swptex_scale(void *x_, int len, float scaling)
{
    float *x = (float *)x_;
    int i;
    for (i = 0; i < len; ++i)
    {
        x[i] *= scaling;
    }
}
