EXECS:conv_1d.o

all: ${EXECS}

conv_1d.o: conv_1d.c
	gcc -std=c99 conv_1d.c -o conv_1d.o
	
test_cpu: conv_1d.o
	./conv_1d.o 10 10 3 7 5
	
sync: Makefile conv_1d_gpu.c
	git add Makefile conv_1d_gpu.c
	git commit -m "Syncing with master"
	git push
	
clean:
	rm ${EXECS}