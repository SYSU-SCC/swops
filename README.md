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
test batch mm without writing double buffer
ldm_malloc size of blk: 221184, max size: 262144
M 192 N 192 K 192 blk_M 96 blk_N 96 blk_K 96 batch: 768
bmm original: 30.302411
bmm optimized: 0.249667

test batch mm with writing double buffer
ldm_malloc size of blk: 221184, max size: 262144
M 192 N 192 K 192 blk_M 96 blk_N 96 blk_K 96 batch: 768
bmm original: 30.302411
bmm optimized: 0.249033
```
