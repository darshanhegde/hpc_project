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

#define DEBUG 1

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


void read_sentence_lens(const char* file_path, int* sent_lens, int n_sents){
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
    int fret = 1;
    for (int i=0; (i<n_sents && fret != EOF); i++) {
        fret = fscanf(fp, "%d\n", &len);
        if (len > 125) {
            len = 125;
        }
        sent_lens[i] = len;
    }
    fclose(fp);
}

void init_lens(long* lens, int* sent_lens, int batch_size, int batch){
    lens[0] = sent_lens[batch*batch_size];
    for (int i=1; i < batch_size; i++) {
        lens[i] = lens[i-1] + sent_lens[batch*batch_size + i];
    }
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
}


int main(int argc, char* argv[]){

    if (argc != 7) {
        printf("USAGE: ./conv_1d.o <n_batches> <batch_size> <dim> <kern_w> <n_kerns> <device_id>");
        exit(1);
    }
    
    //Initilizing random numbers
    srand(20);
    
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
    
    // Read mini-batch sentence lengths
    int* sent_lens = (int*) calloc(n_batches*batch_size, sizeof(int));
    read_sentence_lens("sentence_lens.txt", sent_lens, n_batches*batch_size);
    
    //Allocate kernels and initilize
    kerns.k = (float *)calloc(kerns.height*kerns.width*kerns.num, sizeof(float));
    init_kerns(kerns.k, kerns.num, kerns.width, kerns.height);
    
    // Test kernel initialization
    if (DEBUG) {
        for (int i=0; i<kerns.num; i++) {
            printf("Kernel: %d\n", i);
            print_mat(&kerns.k[i*kerns.height*kerns.width], kerns.width, kerns.height);
            printf("\n\n");
        }
    }
    
    // Define test idxs
    int test_batch = 9, test_idx = 9;
    
    WORDVECS wordvec;
    OUTPUTS output;

    
    //Select the device you want to run the code.
    cudaSetDevice(device_id);
    printf("Using device: %d \n", device_id);
    
    // Allocate GPU WORDVEC, KERNS and OUTPUT. Planning to pass these structs by value.
    WORDVECS d_wordvec;
    KERNS d_kerns;
    OUTPUTS d_output;
    
    // Allocate and Initialize kerns.k on device
    float* d_k;
    cudaMalloc((void **) &(d_k), sizeof(float)*kerns.num*kerns.width*kerns.height);
    if (DEBUG) {
        printf("Done allocating d_k \n");
    }
    
    cudaMemcpy(d_k, kerns.k, sizeof(float)*kerns.num*kerns.width*kerns.height, cudaMemcpyHostToDevice);
    if (DEBUG) {
        printf("Done transfering kerns.k -> d_k \n");
    }
    d_kerns.k = d_k;
    d_kerns.num = kerns.num;
    d_kerns.width = kerns.width;
    d_kerns.height = kerns.height;
    
    // Readback and check if the results are right
    cudaMemcpy(kerns.k, d_k, sizeof(float)*kerns.num*kerns.width*kerns.height, cudaMemcpyDeviceToHost);
    
    if (DEBUG) {
        printf("GPU kernel values. \n");
        for (int i=0; i<kerns.num; i++) {
            printf("Kernel: %d\n", i);
            print_mat(&kerns.k[i*kerns.height*kerns.width], kerns.width, kerns.height);
            printf("\n\n");
        }
    }
    
    for (int batch=0; batch < n_batches; batch++) {
        wordvec.b_size = batch_size;
        wordvec.dim = dim;
        wordvec.lens = (long*) calloc(batch_size, sizeof(long));
        
        init_lens(wordvec.lens, sent_lens, batch_size, batch);
        
        // Test sentence lens for a given mini-batch
        if (DEBUG && (test_batch == batch)) {
            printf("i=%d, len=%ld \n", 0, wordvec.lens[0]);
            for (int i=1; i < wordvec.b_size; i++) {
                printf("i=%d, len=%ld \n", i, wordvec.lens[i] - wordvec.lens[i-1]);
            }
        }
        
        // Allocate word vectors and initialize
        wordvec.w = (float*) calloc(wordvec.dim*wordvec.lens[batch_size-1], sizeof(float));
        init_wordvecs(wordvec.w, wordvec.dim, wordvec.lens[batch_size-1]);
        
        
        //Testing initialization
        if (DEBUG && (test_batch == batch)) {
            printf("Input: \n");
            if (test_idx == 0) {
                print_mat(&(wordvec.w[0*dim]), wordvec.lens[test_idx], wordvec.dim);
            } else {
                print_mat(&(wordvec.w[wordvec.lens[test_idx-1]*dim]), wordvec.lens[test_idx]-wordvec.lens[test_idx-1], wordvec.dim);
            }
        }
        
        //Allocate and initialize outputs
        output.b_size = batch_size;
        output.dim = kerns.num;
        output.lens = (long*) calloc(batch_size, sizeof(long));
        init_out_lens(&(output.lens), wordvec.lens, batch_size, kerns.width);
        
        // Test output lens for a given mini-batch
        if (DEBUG && test_batch == batch) {
            printf("i=%d, len=%ld, out_len=%ld \n", 0, wordvec.lens[0], output.lens[0]);
            for (int i=1; i < wordvec.b_size; i++) {
                printf("i=%d, len=%ld, out_len=%ld \n", i, wordvec.lens[i] - wordvec.lens[i-1],  output.lens[i]-output.lens[i-1]);
            }
        }
        
        //Allocate outputs
        output.out = (float*) calloc(kerns.num*output.lens[batch_size-1], sizeof(float));
        
        //Run on CPU and test output
        if (DEBUG && test_batch == batch) {
            conv1d(wordvec, kerns, output);
            
            printf("CPU Output: \n");
            if (test_idx == 0) {
                print_mat(&(output.out[0*kerns.num]), output.lens[test_idx], kerns.num);
            } else {
                print_mat(&(output.out[output.lens[test_idx-1]*kerns.num]), output.lens[test_idx]-output
                          .lens[test_idx-1], kerns.num);
            }
            
            memset(output, 0, sizeof(float)*kerns.num*output.lens[batch_size-1]);
        }

        
        // Allocate and initialize wordvecs.w and wordvecs.lens on GPU
        d_wordvec.dim = dim;
        d_wordvec.b_size = batch_size;
        long* d_wlens;
        cudaMalloc((void **) &(d_wlens), sizeof(long)*batch_size);
        if (DEBUG) {
            printf("Done allocating d_wlens \n");
        }
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("***ERROR***: %s\n", cudaGetErrorString(err));
        
        cudaMemcpy(d_wlens, wordvec.lens, sizeof(long)*batch_size, cudaMemcpyHostToDevice);
        if (DEBUG) {
            printf("Done transfering wordvecs[batch].lens -> d_wlens \n");
        }
        d_wordvec.lens = d_wlens;
        
        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("***ERROR***: %s\n", cudaGetErrorString(err));
        
        float* d_w;
        cudaMalloc((void **) &(d_w), sizeof(float)*dim*wordvec.lens[batch_size-1]);
        if (DEBUG) {
            printf("Done allocating d_w \n");
        }
        
        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("***ERROR***: %s\n", cudaGetErrorString(err));
        
        cudaMemcpy(d_w, wordvec.w, sizeof(float)*dim*wordvec.lens[batch_size-1], cudaMemcpyHostToDevice);
        if (DEBUG) {
            printf("Done transfering wordvecs[batch].w -> d_w \n");
        }
        d_wordvec.w = d_w;
        
        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("***ERROR***: %s\n", cudaGetErrorString(err));
        
        // Allocate and initialize outputs.out and outputs.lens on GPU
        d_output.dim = kerns.num;
        d_output.b_size = batch_size;
        long* d_olens;
        cudaMalloc((void **) &(d_olens), sizeof(long)*batch_size);
        if (DEBUG) {
            printf("Done allocating d_olens \n");
        }
        
        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("***ERROR***: %s\n", cudaGetErrorString(err));
        
        cudaMemcpy(d_olens, output.lens, sizeof(long)*batch_size, cudaMemcpyHostToDevice);
        if (DEBUG) {
            printf("Done transfering wordvecs[batch].lens -> d_olens \n");
        }
        d_output.lens = d_olens;
        
        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("***ERROR***: %s\n", cudaGetErrorString(err));
        
        float* d_out;
        cudaMalloc((void **) &(d_out), sizeof(float)*kerns.num*output.lens[batch_size-1]);
        if (DEBUG) {
            printf("Done allocating d_out \n");
        }
        
        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("***ERROR***: %s\n", cudaGetErrorString(err));
        
        cudaMemcpy(d_out, output.out, sizeof(float)*kerns.num*output.lens[batch_size-1], cudaMemcpyHostToDevice);
        if (DEBUG) {
            printf("Done transfering outputs[batch].out -> d_out \n");
        }
        d_output.out = d_out;
        
        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("***ERROR***: %s\n", cudaGetErrorString(err));
        
        // Launch the kernel
        
        conv1d_kernel<<<batch_size, dim, sizeof(float)*dim>>>(d_wordvec, d_kerns, d_output);
        
        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("***ERROR***: %s\n", cudaGetErrorString(err));
        if (DEBUG) {
            printf("Done launching the kernel. \n");
        }
        
        // Get output results back
        cudaMemcpy(output.out, d_out, sizeof(float)*kerns.num*output.lens[batch_size-1], cudaMemcpyDeviceToHost);
        if (DEBUG) {
            printf("Done transfering d_out -> outputs[batch].out \n");
        }
        
        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("***ERROR***: %s\n", cudaGetErrorString(err));
        
        // Verify GPU Results
        if (DEBUG) {
            if (batch == test_batch) {
                printf("GPU Output: \n");
                if (test_idx == 0) {
                    print_mat(&(output.out[0*kerns.num]), output.lens[test_idx], kerns.num);
                } else {
                    print_mat(&(output.out[output.lens[test_idx-1]*kerns.num]), output.lens[test_idx]-output.lens[test_idx-1], kerns.num);
                }
            }
        }
        
        // Free GPU allocations for mini-batch
        cudaFree(d_wlens);
        cudaFree(d_w);
        cudaFree(d_olens);
        cudaFree(d_out);
        
        // Free all allocated resources.
        free(wordvec.w);
        free(wordvec.lens);
        free(output.out);
        free(output.lens);
    }
    
    //Free all GPU allocated resources.
    cudaFree(d_k);
    free(kerns.k);
    free(sent_lens);
}