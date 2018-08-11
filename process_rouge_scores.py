import sys
import csv
import numpy as np
from scipy import stats
import math

def confidence_interval(data):
    n, min_max, mean, var, skew, kurt = stats.describe(data)
    std = math.sqrt(var)
    return stats.norm.interval(0.05, loc=mean, scale=std)

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage: python process_rouge_scores.py <csv_filename>")

    filename = sys.argv[1]

    l_scores = []
    one_scores = []
    two_scores = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['ROUGE-Type'] == 'ROUGE-L+StopWordRemoval':
                l_scores.append(float(row['Avg_F-Score']))
            elif row['ROUGE-Type'] == 'ROUGE-1+StopWordRemoval':
               one_scores.append(float(row['Avg_F-Score']))
            elif row['ROUGE-Type'] == 'ROUGE-2+StopWordRemoval':
                two_scores.append(float(row['Avg_F-Score']))

    # print("L: {}".format(np.percentile(l_scores, 95)))
    # print("1: {}".format(np.percentile(one_scores, 95)))
    # print("2: {}".format(np.percentile(two_scores, 95)))
    print("L scores: {}".format(confidence_interval(l_scores)))
    print("1 gram scores: {}".format(confidence_interval(one_scores)))
    print("2 gram scores: {}".format(confidence_interval(two_scores)))
