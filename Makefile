EXECS:conv_1d.o

all: ${EXECS}

conv_1d.o: conv_1d.c
	gcc -std=c99 conv_1d.c -o conv_1d.o
	
test_cpu: conv_1d.o
	./conv_1d.o 10 10 3 7 5
	
conv_1d_gpu.o: conv_1d_gpu.cu
	git pull
	nvcc -o conv_1d_gpu.o conv_1d_gpu.cu
	
test_gpu: conv_1d_gpu.o
	./ conv_1d_gpu.o 10 10 3 7 5
	
sync: Makefile conv_1d_gpu.cu
	git add Makefile conv_1d_gpu.cu
	git commit -m "Syncing with master"
	git push
	
clean:
	rm ${EXECS}