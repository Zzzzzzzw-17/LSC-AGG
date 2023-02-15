from docopt import docopt
import os
import numpy as np
from sklearn import metrics


def read(path):
    with open(path, 'r') as f:
        dict_ = {}
        for line in f:
            token, score = line.split('\t')
            dict_[token] = float(score[:-1])

    return dict_


def calculate_auc(trueAnsPath, modelAnsDir):
    results = {}
    true = read(trueAnsPath)
    me = ['PRT', 'APD']
    for metric in me:
        for filename in os.listdir(f"{modelAnsDir}/{metric}"):
            if filename != 'discovery_noun' and filename != 'discovery_verb' and filename != 'gold.txt':
                path = os.path.join(f"{modelAnsDir}/{metric}", filename)
                pred = read(path, filename, metric)
                y_true = np.array(list(true.values()))
                y_pred = np.array([pred[k] for k, v in true.items()])

                fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
                results[filename] = metrics.auc(fpr, tpr)
        print(results)
    return results


all_results = calculate_auc()


def main():
    """
    Evaluate lexical semantic change detection results.
    """

    # Get the arguments
    args = docopt("""Evaluate lexical semantic change detection results.

    Usage:
        eval.py <modelAnsDir> <trueAnsPath>

    Arguments:
        #<modelAnsDir> = directory to tab-separated answer file for Task 1 (lemma + "\t" + continuous score)
        #<trueAnsPath> = path to tab-separated gold answer file for Task 1 (lemma + "\t" + binary score)
    """)

    modelAnsDir = args['<modelAnsPath>']
    trueAnsPath = args['<trueAnsPath>']

    results = calculate_auc(trueAnsPath, modelAnsDir)


if __name__ == '__main__':
    main()
