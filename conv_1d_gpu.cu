/*  Serial version of conv_1d for minibatch with variable length instances.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <assert.h>
#include <cuda.h>

#define MIN(a,b) ((a<b)?a:b)
#define MAX(a,b) ((a>b)?a:b)

typedef struct WORDVECS{
    float* w;
    int dim;
    long* lens;
    int b_size;
}WORDVECS;

typedef struct KERNS{
    float* k;
    int num;
    int width;
    int height;
}KERNS;

typedef struct OUTPUTS{
    float* out;
    int dim;
    long* lens;
    int b_size;
}OUTPUTS;


void read_sentence_lens(char* file_path, WORDVECS* wordvecs, int n_batches){
    /*
     Reads sentence lengths from Trip Advisor Dataset. Assumes that sentence 
     lengths are non-zero.
     */
    int len=0;
    FILE *fp = fopen(file_path, "r");
    if (fp == NULL) {
        fprintf(stderr, "Can't open input file %s!\n", file_path);
        exit(1);
    }
    for (int batch=0; batch < n_batches; batch++) {
        int fret = fscanf(fp, "%d\n", &len);
         wordvecs[batch].lens[0] = len;
        for (int i=1; (i<wordvecs[batch].b_size && fret != EOF); i++) {
            fret = fscanf(fp, "%d\n", &len);
            wordvecs[batch].lens[i] = wordvecs[batch].lens[i-1] + len;
        }
    }
    
    fclose(fp);
}

void init_wordvecs(float* wordvecs, int dim, int total_words){
    /*
     Initilizes word vectors. i.e input for convolution
     */
    for (int i=0; i < total_words; i++) {
        for (int d=0; d < dim; d++) {
            wordvecs[i*dim+d] = 1.;
        }
    }
}

void init_kerns(float* kerns, int n_kerns, int kern_w, int kern_h){
    /*
     Initilizes kernels.
     */
    for (int i=0; i < (n_kerns*kern_w*kern_h); i++) {
        kerns[i] = rand()/(float)INT_MAX;
    }
}

void init_out_lens(long** out_lens, long* lens, int b_size, int kern_w){
    (*out_lens)[0] = (lens[0] + kern_w - 1);
    for (int i=1; i < b_size; i++) {
        (*out_lens)[i] = (*out_lens)[i-1] + (lens[i] - lens[i-1]) + kern_w - 1;
    }
}


void print_mat(float* mat, int width,int height){
    /*
     Printing the matrix for verification.
     */
    printf("np.array([");
    for (int i=0; i<width; i++) {
        printf("[");
        for (int j=0; j<height; j++) {
            if (j == height-1) {
                printf(" %.4f", mat[i*height+j]);
            } else {
                printf(" %.4f,", mat[i*height+j]);
            }
        }
        if (i == width-1) {
            printf("]");
        } else {
            printf("],\n");
        }
    }
    printf("])\n");
}

void conv1d(WORDVECS wordvec, KERNS kerns, OUTPUTS output){
    /*
     Performs 1d convolution on CPU for each mini-batch at a time.
     */
    long len, out_len;
    float* wv;
    float* out;
    int dim = wordvec.dim, out_dim=kerns.num;
    for (int inst=0; inst < wordvec.b_size; inst++) {
        if (inst == 0) {
            len = wordvec.lens[inst];
            out_len = output.lens[inst];
            wv = &wordvec.w[dim*0];
            out = &output.out[out_dim*0];
        } else {
            len = wordvec.lens[inst] - wordvec.lens[inst-1];
            out_len = output.lens[inst] - output.lens[inst-1];
            wv = &wordvec.w[dim*wordvec.lens[inst-1]];
            out = &output.out[out_dim*output.lens[inst-1]];
        }
        for (int i=0; i < out_len; i++) {
            for (int k=0; k < kerns.num; k++) {
                float s = 0.;
                for (int j = MAX(0, i-kerns.width+1); j <= MIN(i, len-1); j++) {
                    int k_sub=(kerns.width-1-i+j);
                    for (int d=0; d<dim; d++) {
                        s += (wv[j*dim+d] * kerns.k[k*kerns.width*kerns.height + k_sub*kerns.height + d]);
                    }
                }
                out[i*kerns.num+k] += s;
            }
        }
    }
}

__global__
void conv1d_kernel(WORDVECS wordvec, KERNS kerns, OUTPUTS output){
    /*
     Performs 1d convolution on CPU for each mini-batch at a time.
     */
    int tIdx = threadIdx.x;
    int bIdx = blockIdx.x;
    
    __syncthreads();
    printf("Reporting tIdx=%d, bIdx=%d\n", tIdx, bIdx);
    __syncthreads();
    
    long len, out_len;
    float* wv;
    float* out;
    int dim = wordvec.dim, out_dim=kerns.num;
    
    assert(blockDim.x == dim);
    
    extern __shared__ float s[];
    
    if (bIdx == 0) {
        len = wordvec.lens[bIdx];
        out_len = output.lens[bIdx];
        wv = &wordvec.w[dim*0];
        out = &output.out[out_dim*0];
    } else {
        len = wordvec.lens[bIdx] - wordvec.lens[bIdx-1];
        out_len = output.lens[bIdx] - output.lens[bIdx-1];
        wv = &wordvec.w[dim*wordvec.lens[bIdx-1]];
        out = &output.out[out_dim*output.lens[bIdx-1]];
    }
    __syncthreads();
    
    for (int i=0; i < out_len; i++) {
        for (int k=0; k < kerns.num; k++) {
            s[tIdx] = 0.;
            for (int j = MAX(0, i-kerns.width+1); j <= MIN(i, len-1); j++) {
                int k_sub=(kerns.width-1-i+j);
                s[tIdx] += (wv[j*dim+tIdx] * kerns.k[k*kerns.width*kerns.height + k_sub*kerns.height + tIdx]);
            }
            atomicAdd(&out[i*kerns.num+k], s[tIdx]);
            __syncthreads();
        }
    }
    __syncthreads();
}


int main(int argc, char* argv[]){
    if (argc != 7) {
        printf("USAGE: ./conv_1d.o <n_batches> <batch_size> <dim> <kern_w> <n_kerns> <device_id>");
        exit(1);
    }
    
    //Initilizing random numbers
    srand(20);
    
    // Decalre structs
    KERNS kerns;
    
    //Parsing commandline args and initialize structs
    int n_batches = atoi(argv[1]);
    int batch_size = atoi(argv[2]);
    int dim = atoi(argv[3]);
    kerns.height = dim;
    kerns.width = atoi(argv[4]);
    kerns.num = atoi(argv[5]);
    int device_id = atoi(argv[6]);
    printf("n_batches=%d, batch_size=%d, dim=%d, kern_w=%d, kern_h=%d, n_kerns=%d\n", n_batches, batch_size, dim, kerns.width, kerns.height, kerns.num);
    
    WORDVECS* wordvecs = (WORDVECS*) calloc(n_batches, sizeof(WORDVECS));
    
    //Allocate sentence lengths and read
    for (int batch=0; batch < n_batches; batch++) {
        wordvecs[batch].b_size = batch_size;
        wordvecs[batch].dim = dim;
        wordvecs[batch].lens = (long*) calloc(batch_size, sizeof(long));
    }
    
    // Read mini-batch sentence lengths
    read_sentence_lens("sentence_lens.txt", wordvecs, n_batches);
    
    // Test sentence lens for a given mini-batch
    int test_batch = 9, test_idx = 9;
    printf("i=%d, len=%d \n", 0, wordvecs[test_batch].lens[0]);
    for (int i=1; i < wordvecs[test_batch].b_size; i++) {
        printf("i=%d, len=%d \n", i, wordvecs[test_batch].lens[i] - wordvecs[test_batch].lens[i-1]);
    }
    
    // Allocate word vectors and initialize
    for (int batch=0; batch < n_batches; batch++) {
        wordvecs[batch].w = (float*) calloc(wordvecs[batch].dim*wordvecs[batch].lens[batch_size-1], sizeof(float));
        init_wordvecs(wordvecs[batch].w, wordvecs[batch].dim, wordvecs[batch].lens[batch_size-1]);
    }
    
    //Testing initialization
    printf("Input: \n");
    if (test_idx == 0) {
        print_mat(&(wordvecs[test_batch].w[0*dim]), wordvecs[test_batch].lens[test_idx], wordvecs[test_batch].dim);
    } else {
        print_mat(&(wordvecs[test_batch].w[wordvecs[test_batch].lens[test_idx-1]*dim]), wordvecs[test_batch].lens[test_idx]-wordvecs[test_batch].lens[test_idx-1], wordvecs[test_batch].dim);
    }
    
    
    //Allocate kernels and initilize
    kerns.k = (float *) calloc(kerns.height*kerns.width*kerns.num, sizeof(float));
    init_kerns(kerns.k, kerns.num, kerns.width, kerns.height);
    
    // Test kernel initialization
    printf("CPU kernel values: \n");
    for (int i=0; i<kerns.num; i++) {
        printf("Kernel: %d\n", i);
        print_mat(&kerns.k[i*kerns.height*kerns.width], kerns.width, kerns.height);
        printf("\n\n");
    }
    
    //Allocate and initialize outputs
    OUTPUTS* outputs = (OUTPUTS*) calloc(batch_size, sizeof(OUTPUTS));
    for (int batch=0; batch < n_batches; batch++) {
        outputs[batch].b_size = batch_size;
        outputs[batch].dim = kerns.num;
        outputs[batch].lens = (long*) calloc(batch_size, sizeof(long));
        init_out_lens(&(outputs[batch].lens), wordvecs[batch].lens, batch_size, kerns.width);
    }
    
    // Test output lens for a given mini-batch
    printf("i=%d, len=%d, out_len=%d \n", 0, wordvecs[test_batch].lens[0], outputs[test_batch].lens[0]);
    for (int i=1; i < wordvecs[test_batch].b_size; i++) {
        printf("i=%d, len=%d, out_len=%d \n", i, wordvecs[test_batch].lens[i] - wordvecs[test_batch].lens[i-1],  outputs[test_batch].lens[i]-outputs[test_batch].lens[i-1]);
    }
    
    //Allocate outputs
    for (int batch=0; batch < n_batches; batch++) {
        outputs[batch].out = (float*) calloc(kerns.num*outputs[batch].lens[batch_size-1], sizeof(float));
    }
    
    
    // CPU loop
    for (int batch=0; batch < n_batches; batch++) {
            conv1d(wordvecs[batch], kerns, outputs[batch]);
    }
    
    //Testing computation
    printf("CPU Output: \n");
    if (test_idx == 0) {
        print_mat(&(outputs[test_batch].out[0*kerns.num]), outputs[test_batch].lens[test_idx], kerns.num);
    } else {
        print_mat(&(outputs[test_batch].out[outputs[test_batch].lens[test_idx-1]*kerns.num]), outputs[test_batch].lens[test_idx]-outputs[test_batch].lens[test_idx-1], kerns.num);
    }
    
    //Set back output results to zero.
    for (int batch=0; batch < n_batches; batch++) {
        memset(outputs[batch].out, 0, kerns.num*outputs[batch].lens[batch_size-1]*sizeof(float));
    }
    
    //Select the device you want to run the code.
    cudaSetDevice(device_id);
    
    // Allocate GPU WORDVEC, KERNS and OUTPUT. Planning to pass these structs by value.
    WORDVECS d_wordvec;
    KERNS d_kerns;
    OUTPUTS d_output;
    
    // Allocate and Initialize kerns.k on device
    float* d_k;
    cudaMalloc((void **) &(d_k), sizeof(float)*kerns.num*kerns.width*kerns.height);
    printf("Done allocating d_k \n");
    
    cudaMemcpy(d_k, kerns.k, sizeof(float)*kerns.num*kerns.width*kerns.height, cudaMemcpyHostToDevice);
    printf("Done transfering kerns.k -> d_k \n");
    d_kerns.k = d_k;
    d_kerns.num = kerns.num;
    d_kerns.width = kerns.width;
    d_kerns.height = kerns.height;
    
    // Readback and check if the results are right
    cudaMemcpy(kerns.k, d_k, sizeof(float)*kerns.num*kerns.width*kerns.height, cudaMemcpyDeviceToHost);
    
    printf("GPU kernel values. \n");
    for (int i=0; i<kerns.num; i++) {
        printf("Kernel: %d\n", i);
        print_mat(&kerns.k[i*kerns.height*kerns.width], kerns.width, kerns.height);
        printf("\n\n");
    }
    
    // Allocate and initialize wordvecs.w and wordvecs.lens on GPU
    d_wordvec.dim = dim;
    d_wordvec.b_size = batch_size;
    long* d_wlens;
    cudaMalloc((void **) &(d_wlens), sizeof(long)*batch_size);
    printf("Done allocating d_wlens \n");
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("***ERROR***: %s\n", cudaGetErrorString(err));
    
    cudaMemcpy(d_wlens, wordvecs[test_batch].lens, sizeof(long)*batch_size, cudaMemcpyHostToDevice);
    printf("Done transfering wordvecs[test_batch].lens -> d_wlens \n");
    d_wordvec.lens = d_wlens;
    
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("***ERROR***: %s\n", cudaGetErrorString(err));
    
    float* d_w;
    cudaMalloc((void **) &(d_w), sizeof(float)*dim*wordvecs[test_batch].lens[batch_size-1]);
    printf("Done allocating d_w \n");
    
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("***ERROR***: %s\n", cudaGetErrorString(err));
    
    cudaMemcpy(d_w, wordvecs[test_batch].w, sizeof(float)*dim*wordvecs[test_batch].lens[batch_size-1], cudaMemcpyHostToDevice);
    printf("Done transfering wordvecs[test_batch].w -> d_w \n");
    d_wordvec.w = d_w;
    
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("***ERROR***: %s\n", cudaGetErrorString(err));
    
    // Allocate and initialize outputs.out and outputs.lens on GPU
    d_output.dim = kerns.num;
    d_output.b_size = batch_size;
    long* d_olens;
    cudaMalloc((void **) &(d_olens), sizeof(long)*batch_size);
    printf("Done allocating d_olens \n");
    
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("***ERROR***: %s\n", cudaGetErrorString(err));
    
    cudaMemcpy(d_olens, outputs[test_batch].lens, sizeof(long)*batch_size, cudaMemcpyHostToDevice);
    printf("Done transfering wordvecs[test_batch].lens -> d_olens \n");
    d_output.lens = d_olens;
    
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("***ERROR***: %s\n", cudaGetErrorString(err));
    
    float* d_out;
    cudaMalloc((void **) &(d_out), sizeof(float)*kerns.num*outputs[test_batch].lens[batch_size-1]);
    printf("Done allocating d_out \n");
    
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("***ERROR***: %s\n", cudaGetErrorString(err));
    
    cudaMemcpy(d_out, outputs[test_batch].out, sizeof(float)*kerns.num*outputs[test_batch].lens[batch_size-1], cudaMemcpyHostToDevice);
    printf("Done transfering outputs[test_batch].out -> d_out \n");
    d_output.out = d_out;
    
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("***ERROR***: %s\n", cudaGetErrorString(err));
    
    // Launch the kernel
    
    conv1d_kernel<<<batch_size, dim, sizeof(float)*dim>>>(d_wordvec, d_kerns, d_output);
    
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("***ERROR***: %s\n", cudaGetErrorString(err));
    
    printf("Done launching the kernel. \n");
    
    
    // Get output results back
    cudaMemcpy(outputs[test_batch].out, d_out, sizeof(float)*kerns.num*outputs[test_batch].lens[batch_size-1], cudaMemcpyDeviceToHost);
    printf("Done transfering d_out -> outputs[test_batch].out \n");
    
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("***ERROR***: %s\n", cudaGetErrorString(err));
    
    // Verify GPU Results
    printf("GPU Output: \n");
    if (test_idx == 0) {
        print_mat(&(outputs[test_batch].out[0*kerns.num]), outputs[test_batch].lens[test_idx], kerns.num);
    } else {
        print_mat(&(outputs[test_batch].out[outputs[test_batch].lens[test_idx-1]*kerns.num]), outputs[test_batch].lens[test_idx]-outputs[test_batch].lens[test_idx-1], kerns.num);
    }
    
    
    // Free GPU allocations for mini-batch
    cudaFree(d_wlens);
    cudaFree(d_w);
    cudaFree(d_olens);
    cudaFree(d_out);
    
    //Free all GPU allocated resources.
    cudaFree(d_k);
    
    //Free all host allocated resources
    for (int batch=0; batch < n_batches; batch++) {
        free(wordvecs[batch].w);
        free(wordvecs[batch].lens);
        free(outputs[batch].out);
        free(outputs[batch].lens);
    }
    free(wordvecs);
    free(outputs);
}