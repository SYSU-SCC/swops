# SWOPS Unit Test of cpc-21-final
Only change the makefile and add a main.c

# ToDo
GEMM RRR RCR CRR  
BMM RRR RCR CRR

# GEMM RRR Result at Sep 14th 2021
```
testing gemm rrr f32 triple buffer asm no
result of gemm triple buffer asm no: 1.700471
checking copy border passed
checking copy border passed
testin gemm hardware cache 
result of gemm hardware cache: 32.298641
checking C no border
checking C no border passed
Job 2958271 has been finished.
[cpc10@sw_hpc_102 swops]$ 
```

# BMM RRR Result at Sep 8th 2021
Seems that the writing double buffer is meaningless
```
bmm rrr original: 0.247718
bmm rrr optimized: 0.016450
bmm crr original: 0.256681
bmm crr optimized: 0.008509
bmm rcr original: 0.248262
bmm rcr optimized: 0.008528
Job 2970526 has been finished.
[cpc10@sw_hpc_102 swops]$
```
