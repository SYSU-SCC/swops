CC = swgcc
HCFLAGS = -std=gnu99 -O3 -msimd -faddress_align=128 -mhost
SCFLAGS = -std=gnu99 -O3 -msimd -faddress_align=128 -mslave
LDFLAGS = -lm
HEADER_FILES = 

BUILD = ./obj

main : main.o master.o slave.o
	$(CC) -std=gnu99 -O3 -msimd -mhybrid -o $@ $^ $(LDFLAGS) -lm_slave

%.o : %.c $(HEADER_FILES)
	$(CC) $(HCFLAGS) -o $@ -c $< 

slave.o : slave.c args.h
	$(CC) $(SCFLAGS) -o $@ -c $<

run : 
	bsub -I -b -q q_cpc -N 1 -cgsp 64 -host_stack 1024 -ro_size 256 -share_size 1600 -cross_size 60000 -mpecg 6 ./main

asm:
	swgcc -std=gnu99 -O3 -mslave -msimd -fverbose-asm -S slave.c
	swgcc -std=gnu99 -O3 -mhost -msimd -fverbose-asm -S master.c

clean :
	rm -f *.o ./main
