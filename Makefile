DEVICE=0
EXECS:conv_1d.o

all: ${EXECS}

conv_1d.o: conv_1d.c
	gcc -std=c99 conv_1d.c -o conv_1d.o
	
test_cpu: conv_1d.o
	./conv_1d.o 10 10 32 5 3
	
test_gpu:
	git pull
	nvcc -arch=sm_21 -o conv_1d_gpu.o conv_1d_gpu.cu
	./conv_1d_gpu.o 50 50 32 7 6 ${DEVICE}
	
sync: Makefile conv_1d_gpu.cu
	git add Makefile conv_1d_gpu.cu
	git commit -m "Syncing with master"
	git push
	
clean:
	rm ${EXECS}