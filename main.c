#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char *argv[]){
    for(int i = 0; i < 999; i++){
        test_bmm_cgn();
        test_addm_all();

        //test_sw_bmm_double_all();


        //test_gemm_rcr_all();
        test_gemm_rrr_all();
        test_gemm_crr_all();
        //test_sw_bmm_all();
    }
    return 0;
}