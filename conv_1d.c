/*  Serial version of conv_1d for minibatch with variable length instances.
 */

#include <stdio.h>
#include <stdlib.h>


void read_sentence_lens(char* file_path, int* sent_lens, int num_batches, int batch_size){
    /*
     Reads sentence lengths from Trip Advisor Dataset.
     */
    int len=1, fret=1;
    FILE *fp = fopen(file_path, "r");
    if (fp == NULL) {
        fprintf(stderr, "Can't open input file %s!\n", file_path);
        exit(1);
    }
    for(int i=0; (i < num_batches*batch_size) && (fret != EOF); i++) {
        fret = fscanf(fp, "%d\n", &len);
        sent_lens[i] = len;
    }
    fclose(fp);
}

int main(int argc, char* argv[]){
    if (argc != 4) {
        printf("USAGE: ./conv_1d.o <num_batches> <batch_size> <num_kerns>");
    }
    
    int num_batches = atoi(argv[1]);
    int batch_size = atoi(argv[2]);
    int n_kerns = atoi(argv[3]);
    
    printf("num_batches: %d, batch_size: %d, n_kerns: %d\n", num_batches, batch_size, n_kerns);
    
    int* sent_lens = (int *) calloc(num_batches*batch_size, sizeof(int));
    read_sentence_lens("sentence_lens.txt", sent_lens, num_batches, batch_size);
    
    
    
}