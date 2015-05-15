EXECS:conv_1d.o

all: ${EXECS}

conv_1d.o: conv_1d.c
	gcc -std=c11 conv_1d.c -o conv_1d.o
	
test_cpu: conv_1d.o
	./conv_1d.o 500 10 10
	
clean:
	rm ${EXECS}