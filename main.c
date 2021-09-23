#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char *argv[]){
    for(int i = 0; i < 999; i++){
        test_sw_bmm_all();
        //test_gemm_rrr4();

        //test_copy_all();

        //test_gemm_rrr();

        //test_gemm_crr();
        test_gemm_rcr_all();
        test_gemm_rrr_all();
        test_gemm_crr_all();

    }
    return 0;
}