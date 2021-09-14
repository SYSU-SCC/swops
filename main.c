#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern void test_bmm(void);
extern void test_gemm_rrr(void);
int main(int argc, char *argv[]){
    test_bmm();
    test_gemm_rrr();
    return 0;
}