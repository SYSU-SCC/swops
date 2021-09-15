#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char *argv[]){
    test_bmm();
    //test_gemm_rrr();
    test_gemm_rrr_all_cgn();
    test_gemm_rcr_all_cgn();
    return 0;
}