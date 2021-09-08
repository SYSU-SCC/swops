# SWOPS Unit Test of cpc-21-final
Only change the makefile and add a main.c

# ToDo
GEMM RRR RCR CRR  
BMM RRR RCR CRR

# BMM RRR Result at Sep 8th 2021
```
test batch mm
ldm_malloc size of blk: 221184, max size: 262144
M 192 N 192 K 192 blk_M 96 blk_N 96 blk_K 96 batch: 768
bmm original: 30.302411
bmm optimized: 0.249667
```
