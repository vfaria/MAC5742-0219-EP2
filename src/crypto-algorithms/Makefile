OUTPUT=rot-13

CC=gcc
NVCC=nvcc
CC_OPT=

.PHONY: all
all: $(OUTPUT) $(OUTPUT)_cuda

$(OUTPUT): $(OUTPUT)_test.c $(OUTPUT).c $(OUTPUT).h
	$(CC) -o $(OUTPUT) $(CC_OPT) $(OUTPUT)_test.c $(OUTPUT).c $(OUTPUT).h

$(OUTPUT)_cuda: $(OUTPUT)_test.cu $(OUTPUT).cu $(OUTPUT).h
	$(NVCC) -o $(OUTPUT)_cuda $(CC_OPT) $(OUTPUT)_test.cu $(OUTPUT).cu 

.PHONY: clean
clean:
	rm $(OUTPUT) $(OUTPUT)_cuda
