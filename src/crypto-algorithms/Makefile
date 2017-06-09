OUTPUT=rot-13

CC=gcc
NVCC=nvcc
CC_OPT=

.PHONY: all
all: rot-13 rot-13_cuda md2 md2_cuda base64 base64_cuda

rot-13: rot-13_test.c rot-13.c rot-13.h
	$(CC) -o rot-13 $(CC_OPT) rot-13_test.c rot-13.c rot-13.h

rot-13_cuda: rot-13_test.cu rot-13.cu rot-13.h
	$(NVCC) -o rot-13_cuda $(CC_OPT) rot-13_test.cu rot-13.cu 

md2: md2_test.c md2.c md2.h
	$(CC) -o md2 $(CC_OPT) md2_test.c md2.c md2.h

md2_cuda: md2_test.cu md2.cu md2cu.h
	$(NVCC) -o md2_cuda $(CC_OPT) md2_test.cu md2.cu 

base64: base64_test.c base64.c base64.h
	$(CC) -o base64 $(CC_OPT) base64_test.c base64.c base64.h

base64_cuda: base64_test.cu base64.cu base64cu.h
	$(NVCC) -o base64_cuda $(CC_OPT) base64_test.cu base64.cu 

.PHONY: clean
clean:
	rm rot-13 rot-13_cuda md2 md2_cuda base64 base64_cuda
