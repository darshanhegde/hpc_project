/*  Serial version of conv_1d for minibatch with variable length instances.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

#define MIN(a,b) ((a<b)?a:b)
#define MAX(a,b) ((a>b)?a:b)

typedef struct WORDVECS{
    float* w;
    int dim;
    int* lens;
    int n_sents;
    int n_batches;
    int batch_size;
}WORDVECS;

typedef struct KERNS{
    float* k;
    int num;
    int width;
    int height;
}KERNS;


void read_sentence_lens(char* file_path, int* sent_lens, int n_sents){
    /*
     Reads sentence lengths from Trip Advisor Dataset.
     */
    int len=0;
    FILE *fp = fopen(file_path, "r");
    if (fp == NULL) {
        fprintf(stderr, "Can't open input file %s!\n", file_path);
        exit(1);
    }
    int fret = fscanf(fp, "%d\n", &len);
    sent_lens[0] = len;
    for(int i=1; (i < n_sents) && (fret != EOF); i++) {
        fret = fscanf(fp, "%d\n", &len);
        sent_lens[i] = sent_lens[i-1] + len;
    }
    fclose(fp);
}

void init_wordvecs(float* wordvecs, int dim, int n_sents){
    /*
     Initilizes word vectors. i.e input for convolution
     */
    for (int i=0; i < (dim*n_sents); i++) {
        wordvecs[i] = 1.;
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

void init_output_idxs(int* output_idxs, int* sent_lens, int n_sents, int kern_w){
    output_idxs[0] = (sent_lens[0] + kern_w - 1);
    for (int i=1; i< n_sents; i++) {
        output_idxs[i] += (output_idxs[i-1] + (sent_lens[i] - sent_lens[i-1] + kern_w - 1));
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

void conv1d_kernel(WORDVECS wordvecs, KERNS kerns, float* outputs, int* output_idxs, int mini_batch){
    /*
     Performs 1d convolution on CPU for each mini-batch at a time.
     */
    int* out_idxs = &output_idxs[mini_batch*wordvecs.batch_size];
    int* lens = &wordvecs.lens[mini_batch*wordvecs.batch_size];
    float* wv;
    float* outs;
    
    int len, out_len, kern_idx, kern_base, real_inst;
    for (int inst=0; inst < wordvecs.batch_size; inst++) {
        real_inst = mini_batch*wordvecs.batch_size + inst;
        if (real_inst == 0) {
            len = lens[inst];
            out_len = out_idxs[inst];
        } else {
            len = lens[inst] - lens[inst-1];
            out_len = out_idxs[inst] - out_idxs[inst-1];
        }
        wv = &wordvecs.w[wordvecs.dim*wordvecs.lens[real_inst]];
        outs = &outputs[kerns.num*output_idxs[real_inst]];
        float* sum = calloc(out_len*kerns.num, sizeof(float));
        for (int i=0; i < out_len; i++) {
            for (int j = MAX(0, i-kerns.width+1); j <= MIN(i, len-1); j++) {
                for (int k=0; k < kerns.num; k++) {
                    int k_sub=(kerns.width-1-i+j);
                    if (real_inst == 7) {
                        printf("out_idx=%d, out_len: %d, len:%d, i=%d, j=%d, k_sub=%d, k=%d\n", kerns.num*out_idxs[inst], out_len, len, i, j, k_sub, k);
                    }
                    float d_sum = 0.;
                    for (int d=0; d<wordvecs.dim; d++) {
                        d_sum += (wv[i*wordvecs.dim+d] * kerns.k[k*kerns.width*kerns.height + k_sub*kerns.height + d]);
                    }
                    sum[i*kerns.num+k] += d_sum;
                    outs[i*kerns.num+k] += d_sum;
                }
            }
        }
        if (real_inst == 7) {
            print_mat(sum, out_len, kerns.num);
            print_mat(outs, out_len, kerns.num);
        }
    }
}

int main(int argc, char* argv[]){
    if (argc != 6) {
        printf("USAGE: ./conv_1d.o <n_batches> <batch_size> <dim> <kern_w> <n_kerns>");
        exit(1);
    }
    
    //Initilizing random numbers
    srand(20);
    
    // Decalre structs
    WORDVECS wordvecs;
    KERNS kerns;
    
    //Parsing commandline args and initialize structs
    wordvecs.n_batches = atoi(argv[1]);
    wordvecs.batch_size = atoi(argv[2]);
    wordvecs.dim = atoi(argv[3]);
    kerns.height = wordvecs.dim;
    kerns.width = atoi(argv[4]);
    kerns.num = atoi(argv[5]);
    printf("n_batches=%d, batch_size=%d, dim=%d, kern_w=%d, n_kerns=%d\n", wordvecs.n_batches, wordvecs.batch_size, wordvecs.dim, kerns.width, kerns.num);
    
    wordvecs.n_sents = wordvecs.n_batches*wordvecs.batch_size;
    
    //Allocate sentence lengths and read
    wordvecs.lens = (int *) calloc(wordvecs.n_sents, sizeof(int));
    read_sentence_lens("sentence_lens.txt", wordvecs.lens, wordvecs.n_sents);
    
    int total_words = wordvecs.lens[wordvecs.n_sents-1];
    //Allocate wordvec and initilize
    wordvecs.w = calloc(total_words*wordvecs.dim, sizeof(float));
    init_wordvecs(wordvecs.w, wordvecs.dim, wordvecs.n_sents);
    
    //Allocate kernels and initilize
    kerns.k = calloc(kerns.height*kerns.width*kerns.num, sizeof(float));
    init_kerns(kerns.k, kerns.num, kerns.width, kerns.height);
    
    for (int i=0; i<kerns.num; i++) {
        printf("Kernel: %d\n", i);
        print_mat(&kerns.k[i*kerns.height*kerns.width], kerns.width, kerns.height);
        printf("\n\n");
    }
    
    //Allocate outputs and corresponding idx and initilize them
    float *outputs = calloc((total_words+(kerns.width-1)*wordvecs.n_sents)*kerns.num, sizeof(float));
    int* output_idxs = (int *) calloc(wordvecs.n_sents, sizeof(int));
    init_output_idxs(output_idxs, wordvecs.lens, wordvecs.n_sents, kerns.width);
    
    // Perform convolution on each mini-batch
    for (int i=0; i<wordvecs.n_batches; i++) {
        conv1d_kernel(wordvecs, kerns, outputs, output_idxs, i);
    }
    
    
    int inst = 7;
    printf("Output of %dth index: \n", inst);
    printf("out_idx=%d, length=%d, output_length=%d \n", kerns.num*output_idxs[inst], wordvecs.lens[inst]-wordvecs.lens[inst-1], output_idxs[inst]-output_idxs[inst-1]);
    print_mat(&outputs[kerns.num*output_idxs[inst]], output_idxs[inst]-output_idxs[inst-1], kerns.num);
}