import time
import argparse
import numpy as np

import theano
from theano import tensor as T
from theano.tensor.nnet import conv


def read_sentence_lens(sentence_lens_path):
    sent_lens = []
    sentence_lens_file = open(sentence_lens_path, "rU")
    for str_len in sentence_lens_file:
        length = int(str_len.strip())
        if length > 150:
            length = 150
        sent_lens.append(length)
    return sent_lens


def get_theano_benchmark(sentence_lens, n_batches, batch_size, dim, kern_w, n_kerns, debug=False,
                         test_batch=0, test_idx=0):

    kern_shape = (n_kerns, dim, 1, kern_w)
    h_kerns = np.random.random(size=kern_shape).astype(theano.config.floatX)
    if debug:
        debug_kerns = np.swapaxes(h_kerns, 1, 3)
        debug_kerns = debug_kerns.reshape((n_kerns, kern_w, dim))
        for kern_idx in range(n_kerns):
            print "Kernel %d" % kern_idx
            print debug_kerns[kern_idx, :, :]

    d_kerns = theano.shared(h_kerns, borrow=True)

    wordvec = T.tensor4(name='wordvec', dtype=theano.config.floatX)

    batch_input = T.tensor4(name='batch_input', dtype=theano.config.floatX)

    output = conv.conv2d(input=wordvec, filters=d_kerns, border_mode='full')

    minibatch_conv = theano.function(
        [batch_input],
        output,
        givens={
            wordvec: batch_input
        }
    )

    tic = time.time()
    for batch in range(n_batches):
        batch_lens = sentence_lens[batch*batch_size: (batch+1)*batch_size]
        max_batch_len = max(batch_lens)

        batch_input = np.ones(shape=(batch_size, dim, 1, max_batch_len), dtype=theano.config.floatX)
        out = minibatch_conv(batch_input)

        if debug and batch == test_batch:
            out = np.swapaxes(out, 1, 3)
            out = out.reshape((batch_size, max_batch_len+kern_w-1, n_kerns))
            print out[test_idx, :, :]

    toc = time.time()
    print "n_batches=%d, batch_size=%d, dim=%d, kern_w=%d, kern_h=%d, n_kerns=%d took %f secs" % (n_batches, batch_size,
                                                    dim, kern_w, dim, n_kerns, toc-tic)


def main():
    np.random.seed(10)
    sentence_lens = read_sentence_lens("sentence_lens.txt")

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_batches', required=True, type=int, help='Number of mini-batches to use.')
    parser.add_argument('--batch_size', required=True, type=int, help='Batch size. ')
    parser.add_argument('--dim', required=True, type=int, help='Dimension of word vectors. ')
    parser.add_argument('--kern_w', required=True, type=int, help='Width of 1d kernel. ')
    parser.add_argument('--n_kerns', required=True, type=int, help='Number of 1d kernels. ')

    args = parser.parse_args()


    debug_flag = False
    test_batch = 0
    test_idx = 9
    get_theano_benchmark(sentence_lens, args.n_batches, args.batch_size, args.dim, args.kern_w, args.n_kerns,
                         debug=debug_flag, test_batch=test_batch, test_idx=test_idx)



if __name__ == '__main__':
    main()
