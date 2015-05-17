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
    long* lens;
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


void read_sentence_lens(char* file_path, long* sent_lens, int n_sents){
    /*
     Reads sentence lengths from Trip Advisor Dataset.
     */
    long len=0;
    FILE *fp = fopen(file_path, "r");
    if (fp == NULL) {
        fprintf(stderr, "Can't open input file %s!\n", file_path);
        exit(1);
    }
    int fret = fscanf(fp, "%d\n", &len);
    sent_lens[0] = len;
    for(int i=1; (i < n_sents) && (fret != EOF); i++) {
        fret = fscanf(fp, "%d\n", &len);
        if (len == 0) {
            printf("Zero length sentence encountered. \n");
        }
        sent_lens[i] = sent_lens[i-1] + len;
    }
    fclose(fp);
}

void init_wordvecs(float* wordvecs, int dim, int total_words){
    /*
     Initilizes word vectors. i.e input for convolution
     */
    for (int i=0; i < (dim*total_words); i++) {
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

void init_out_idxs(long* out_idxs, long* lens, int n_sents, int kern_w){
    out_idxs[0] = (lens[0] + kern_w - 1);
    for (int i=1; i < n_sents; i++) {
        out_idxs[i] = out_idxs[i-1] + (lens[i] - lens[i-1]) + kern_w - 1;
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

void conv1d_kernel(WORDVECS wordvecs, KERNS kerns, float* outputs, long* out_idxs, int mini_batch, int test_idx){
    /*
     Performs 1d convolution on CPU for each mini-batch at a time.
     */
    float* wv;
    float* outs;
    long len, out_len;
    int full_idx;
    // Loop over instances in a mini-batch
    for (int idx=0; idx < wordvecs.batch_size; idx++) {
        full_idx = mini_batch*wordvecs.batch_size + idx;
        wv = &wordvecs.w[wordvecs.dim * wordvecs.lens[full_idx]];
        outs = &outputs[kerns.num * out_idxs[full_idx]];
        if (mini_batch == 0 && idx == 0) {
            len = wordvecs.lens[0];
            out_len = out_idxs[0];
        } else {
            len = wordvecs.lens[full_idx] - wordvecs.lens[full_idx-1];
            out_len = out_idxs[full_idx] - out_idxs[full_idx-1];
        }
        // Debugging code
        if (test_idx == full_idx) {
            printf("testing for instance %d, len: %d, out_len: %d \n", test_idx, len, out_len);
        }
        
        int k_sub;
        float dot_sum;
        // Loop over output positions
        for (int i=0; i < out_len; i++) {
            // Loop over kernels
            for (int k=0; k < kerns.num; k++) {
                // Loop over valid kernel widths
                dot_sum = 0.;
                for (int j=MAX(0, i-kerns.width+1); j <= MIN(i, len-1); j++) {
                    k_sub = (kerns.width-1-i+j);
                    if (test_idx == full_idx) {
                        printf("i=%d, k=%d, j=%d k_sub=%d\n", i, k, j, k_sub);
                    }
                    for (int d=0; d < wordvecs.dim; d++) {
                        dot_sum += (wv[j*wordvecs.dim+d] * kerns.k[k*kerns.width*kerns.height + k_sub*wordvecs.dim + d]);
                    }
                }
                outs[i*kerns.num+k] = dot_sum;
                if (test_idx == full_idx) {
                    printf("outs[i=%d, k=%d]=%f\n", i, k, dot_sum);
                }
            }
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
    wordvecs.lens = (long *) calloc(wordvecs.n_sents, sizeof(long));
    read_sentence_lens("sentence_lens.txt", wordvecs.lens, wordvecs.n_sents);
    
    int total_words = wordvecs.lens[wordvecs.n_sents-1];
    //Allocate wordvec and initilize
    wordvecs.w = calloc(total_words*wordvecs.dim, sizeof(float));
    init_wordvecs(wordvecs.w, wordvecs.dim, total_words);
    
    //Allocate kernels and initilize
    kerns.k = calloc(kerns.height*kerns.width*kerns.num, sizeof(float));
    init_kerns(kerns.k, kerns.num, kerns.width, kerns.height);
    
    for (int i=0; i<kerns.num; i++) {
        printf("Kernel: %d\n", i);
        print_mat(&kerns.k[i*kerns.height*kerns.width], kerns.width, kerns.height);
        printf("\n\n");
    }
    
    printf("Total words:%d \n", total_words);
    
    long* out_idxs = (long*) calloc(wordvecs.n_sents, sizeof(long));
    init_out_idxs(out_idxs, wordvecs.lens, wordvecs.n_sents, kerns.width);
    
//    printf("i=%d, lens=%d, out_idxs=%d \n", 0, wordvecs.lens[0], out_idxs[0]);
//    for (int i=1; i < 50; i++) {
//        printf("i=%d, lens=%d, out_idxs=%d \n", i, wordvecs.lens[i]-wordvecs.lens[i-1], out_idxs[i] - out_idxs[i-1]);
//    }
//    printf("i=%d, lens=%d, out_idxs=%d \n", wordvecs.n_sents-1, wordvecs.lens[wordvecs.n_sents-1]-wordvecs.lens[wordvecs.n_sents-2], out_idxs[wordvecs.n_sents-1] - out_idxs[wordvecs.n_sents-2]);
    
    float* outputs = (float *) calloc(out_idxs[wordvecs.n_sents-1]*kerns.num, sizeof(float));
    
    int test_idx = 4993;
    for (int i=0; i < wordvecs.n_batches; i++) {
        conv1d_kernel(wordvecs, kerns, outputs, out_idxs, i, test_idx);
    }
    
    printf("Test idx: %d \n", test_idx);
    printf("Output: len: %d \n", out_idxs[test_idx]-out_idxs[test_idx-1]);
    print_mat(&outputs[out_idxs[test_idx]*kerns.num], out_idxs[test_idx]-out_idxs[test_idx-1], kerns.num);
    printf("Input: len: %d \n", wordvecs.lens[test_idx]-wordvecs.lens[test_idx-1]);
    print_mat(&wordvecs.w[wordvecs.lens[test_idx]*wordvecs.dim], wordvecs.lens[test_idx]-wordvecs.lens[test_idx-1], wordvecs.dim);
    
    free(wordvecs.lens);
    free(wordvecs.w);
    free(kerns.k);
    free(out_idxs);
    free(outputs);
}