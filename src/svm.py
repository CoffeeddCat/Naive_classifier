from config import *
from sklearn.svm import SVC
import loader
import numpy as np
import argparse
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--kernel', type=int, default=0)
parser.add_argument('-dgr', '--degree', type=int, default=3)
parser.add_argument('-i', '--task_index', type=int, default=0)
parser.add_argument('-dm', '--dimension', type=int, default=22283)
parser.add_argument('-gm', '--gamma', type=float, default=0.001)
parser.add_argument('-c', '--C', type=float, default=1)
parser.add_argument('-l', '--label', type=int, default=0)
args = parser.parse_args()

kernel_list=[ 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

param = {
    'kernel':kernel_list[args.kernel],
    'degree':7,
    'gamma':args.gamma,
    'C':args.C
}

loader = loader.Loader(LABEL_FILE_PATH, DATA_FILE_PATH, args.label, TRAINING_SET_PERCENT, FIRST_TIME_TO_READ_FILE)

classifier = SVC(**param)
classifier.fit(loader.x_train, np.squeeze(loader.y_train))
y_predict = classifier.predict(loader.x_test)

print(f1_score(np.squeeze(loader.y_test), y_predict, average='macro'))
