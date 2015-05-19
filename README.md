# Efficient Theano 1D Convolutions for Variable Length Sequences

Depending on the mini-batch size this implementation achieves speedup of 2x (for min-batch size of 200) 
compared to using Theano's 2D convolution with zero-padding. 

# Instructions for testing the speed-up

We run the speed-up comparisons on sentences extracted from Trip Advisor Movie Review Dataset. You can try varying 
input size, width of kernel and number of kernels and compare the run rime with Theano's implementation.

We assume that you have a working version of NVIDIA drivers for CUDA on your system. We have tested our code on NVIDIA GRID K520 GPU.

To run our implemnetation:
```
make test_gpu
```
Optionally try changing parameters for convolution
```
make test_gpu N_BATCHES=10 BATCH_SIZE=200 DIM=100 KERN_W=7 N_KERNS=32
```

For running Theano code, you need to have Theano (http://deeplearning.net/software/theano/) and CUDA driver working properly.

To run the Theano's implementation with zero-padding:
```
make test_theano_gpu 
```
Similarly, you can try changing parameters for convolution
```
make test_theano_gpu N_BATCHES=10 BATCH_SIZE=200 DIM=100 KERN_W=7 N_KERNS=32
```

