"""
Just gets sentence lengths of Trip Advisor Dataset and plots the histogram of sentence lengths.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv


def get_sentence_lengths(trip_advisor_dataset_path):
    """
    Extracts sentence lengths from Trip Advisor Dataset.
    :param trip_advisor_dataset_path:
    :return:
    """
    sentence_lengths = []
    trip_advisor_dataset = open(trip_advisor_dataset_path, "rU")
    trip_advisor_iter = csv.reader(trip_advisor_dataset, delimiter='\t')
    headers = trip_advisor_iter.next()
    review_idx = headers.index("review")
    print "review idx: ", review_idx
    for data_line in trip_advisor_iter:
        review = data_line[review_idx]
        for sentence in review.split("<*>"):
            sent_len = len(sentence.split())
            if sent_len > 0:
                sentence_lengths.append(len(sentence.split()))

    trip_advisor_dataset.close()
    return sentence_lengths


def plot_sent_len_hist(dataset, hist_file_path, datalen_path):
    """
    Write the sentence lengths and plot the distribution of sentence lengths.
    :param dataset:
    :param hist_file_path:
    :param datalen_path:
    :return:
    """
    sent_lens = get_sentence_lengths(dataset)
    # Writing sentence lengths to a file.
    datalen_file = open(datalen_path, "w")
    datalen_file.write("\n".join([str(sent_len) for sent_len in sent_lens]))
    datalen_file.close()

    #Making histogram
    max_len = max(sent_lens)
    min_len = min(sent_lens)
    print "Max Sentence Length: ", max_len, " Min Sentence Length: ", min_len
    plt.hist(sent_lens, bins=max_len, normed=1, facecolor='green', alpha=0.75)

    plt.xlabel('Sentence Lengths')
    plt.ylabel('Normalized Counts')
    plt.title('Distribution of sentence lengths for Trip Advisor Dataset')
    plt.axis([0, 80, 0, 0.05])
    plt.savefig(hist_file_path)

    plt.show()


def main():
    plot_sent_len_hist("trip_advisor_full.tsv", "sentence_len_dist.pdf", "sentence_lens.txt")


if __name__ == '__main__':
    main()
