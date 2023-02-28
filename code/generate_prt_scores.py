# python3
# coding: utf-8
from docopt import docopt
import numpy as np
import pickle
import logging
from sklearn.decomposition import PCA
from sklearn import preprocessing


# Cosine similarity (COS) algorithm

def main():
    """
    Compute (diachronic) distance between sets of contextualised representations.
    """

    # Get the arguments
    args = docopt("""Compute (diachronic) distance between sets of contextualised representations.

    Usage:
        distance.py [--metric=<d> --frequency] <testSet> <valueFile1> <valueFile2> <outPath>

    Arguments:
        <testSet> = path to file with one target per line
        <valueFile1> = path to file containing usage matrices and snippets
        <valueFile2> = path to file containing usage matrices and snippets
        <outPath> = output path for result file

    Options:
        --metric=<d> how to aggregate embeddings if there are multiple layers speficied. 
                     [default: mean]
    """)

    testset = args['<testSet>']
    value_file1 = args['<valueFile1>']
    value_file2 = args['<valueFile2>']
    outpath = args['<outPath>']
    mode = args['--metric']

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    # start_time = time.time()

    # Load targets
    targets = []
    with open(testset, 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            target = line.strip()
            targets.append(target)

    # Get usages collected from corpus 1
    if value_file1.endswith('.dict'):
        with open(value_file1, 'rb') as f_in:
            usages1 = pickle.load(f_in)
    elif value_file1.endswith('.npz'):
        usages1 = np.load(value_file1)
    else:
        raise ValueError('valueFile 1: wrong format.')

    # Get usages collected from corpus 2
    if value_file2.endswith('.dict'):
        with open(value_file2, 'rb') as f_in:
            usages2 = pickle.load(f_in)
    elif value_file2.endswith('.npz'):
        usages2 = np.load(value_file2)
    else:
        raise ValueError('valueFile 2: wrong format.')

    try:
        f_out = open(outpath, 'w', encoding='utf-8')
    except TypeError:
        f_out = None

    for word in targets:
        try:
            frequency = np.median([usages1[word].shape[0], usages2[word].shape[0]])
        except KeyError:
            print(word)
            continue
        if usages1[word].shape[0] < 3 or usages2[word].shape[0] < 3:
            logger.info('{} omitted because of low frequency'.format(word))

        vectors0 = usages1[word]
        vectors1 = usages2[word]
        vectors = []
        if mode == 'pca':
            for m in [vectors0, vectors1]:
                scaled = (m - np.mean(m, 0)) / np.std(m, 0)
                pca = PCA(n_components=3)
                analysis = pca.fit(scaled)
                vector = analysis.components_[0]
                vectors.append(vector)
        elif mode == 'mean':
            for m in [vectors0, vectors1]:
                vector = np.average(m, axis=0)
                print("here", vector.shape)
                vectors.append(vector)
        elif mode == 'sum':
            for m in [vectors0, vectors1]:
                vector = np.sum(m, axis=0)
                vectors.append(vector)
        vectors = [preprocessing.normalize(v.reshape(1, -1), norm='l2') for v in vectors]
        shift = 1 / np.dot(vectors[0].reshape(-1), vectors[1].reshape(-1))

        print('\t'.join([word, str(shift)]), file=f_out)

    if f_out:
        f_out.close()


if __name__ == '__main__':
    main()
