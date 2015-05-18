DEVICE=0
N_BATCHES=5000
BATCH_SIZE=1000
DIM=100
KERN_W=5
N_KERNS=28

EXECS:conv_1d.o conv_1d_gpu.o

all: ${EXECS}

test_cpu:
	git pull
	gcc -std=c99 conv_1d.c -o conv_1d.o
	./conv_1d.o ${N_BATCHES} ${BATCH_SIZE} ${DIM} ${KERN_W} ${N_KERNS}

test_gpu:
	git pull
	nvcc -arch=sm_21 -o conv_1d_gpu.o conv_1d_gpu.cu
	./conv_1d_gpu.o ${N_BATCHES} ${BATCH_SIZE} ${DIM} ${KERN_W} ${N_KERNS} ${DEVICE}

test_theano_cpu:
	git pull
	THEANO_FLAGS=floatX=float32 python theano_benchmark.py --n_batches ${N_BATCHES} --batch_size ${BATCH_SIZE} --dim ${DIM} --kern_w ${KERN_W} --n_kerns ${N_KERNS}

test_theano_gpu:
	git pull
	THEANO_FLAGS=mode=FAST_RUN,device=${DEVICE},floatX=float32 python theano_benchmark.py --n_batches ${N_BATCHES} --batch_size ${BATCH_SIZE} --dim ${DIM} --kern_w ${KERN_W} --n_kerns ${N_KERNS}

sync: Makefile conv_1d_gpu.cu
	git add Makefile conv_1d_gpu.cu conv_1d.c theano_benchmark.py
	git commit -m "Syncing with master"
	git push

clean:
	rm ${EXECS}