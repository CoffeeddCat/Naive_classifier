from config import *
from sklearn.svm import SVC
import loader
import numpy as np
import argparse
from sklearn.metrics import f1_score

# parser = argparse.ArgumentParser()
# parser.add_argument('-k', '--kernel', type=int, default=0)
# parser.add_argument('-dgr', '--degree', type=int, default=3)
# parser.add_argument('-i', '--task_index', type=int, default=0)
# parser.add_argument('-dm', '--dimension', type=int, default=22283)
# parser.add_argument('-gm', '--gamma', type=float, default=0.001)
# parser.add_argument('-c', '--C', type=float, default=1)
# parser.add_argument('-l', '--label', type=int, default=0)
# args = parser.parse_args()

label = [2, 4, 5, 7]
kernel_list=[ 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
gamma_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
C_list = [0.1, 1, 5, 10]
for l in label:
    for k in kernel_list:
        for gm in gamma_list:
            for C_ in C_list:
                param = {
                    'kernel':kernel_list[k],
                    'degree': 3,
                    'gamma':gm,
                    'C':C_
                }

                loader = loader.Loader(LABEL_FILE_PATH, DATA_FILE_PATH, l, TRAINING_SET_PERCENT, False)

                classifier = SVC(**param)
                classifier.fit(loader.x_train, np.squeeze(loader.y_train))
                y_predict = classifier.predict(loader.x_test)

                print('label:', l)
                print(param)
                print(f1_score(np.squeeze(loader.y_test), y_predict, average='macro'))
