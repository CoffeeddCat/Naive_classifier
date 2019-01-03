from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np
import argparse
from config import *
import loader

parser = argparse.ArgumentParser()
parser.add_argument('-dgr', '--degree', type=int, default=3)
parser.add_argument('-i', '--task_index', type=int, default=0)
parser.add_argument('-dm', '--dimension', type=int, default=22283)
parser.add_argument('-C', '--C', type=float, default=1)
parser.add_argument('-p', '--penalty', type=str, default='l2')
parser.add_argument('-svl', '--solver', type=str, default='sag')

kernel_list=[ 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

args = parser.parse_args()

param = {
    'C': args.C,
    'penalty': args.penalty,
    'solver': args.solver,
    'max_iter': 100,
    'multi_class': 'multinomial',
    'n_jobs': -1,
    'random_state': 1
}

classifier = LogisticRegression(**param)

loader = loader.Loader(LABEL_FILE_PATH, DATA_FILE_PATH, 10, TRAINING_SET_PERCENT, FIRST_TIME_TO_READ_FILE)
classifier.fit(loader.x_train, np.squeeze(loader.y_train))
y_predict = classifier.predict(loader.x_test)

print(f1_score(np.squeeze(loader.y_test), y_predict, average='macro'))