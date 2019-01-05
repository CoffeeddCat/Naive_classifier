import tensorflow as tf
import numpy as np
from loader import Loader
from sklearn.metrics import f1_score
from nn_model import Network
from config import *

if __name__ == '__main__':
    network = Network([256, 128, 64, 9], tf.nn.relu, 22283, 1e-4)
    loader = Loader(LABEL_FILE_PATH, DATA_FILE_PATH, 2, TRAINING_SET_PERCENT, FIRST_TIME_TO_READ_FILE)

    # max= 0
    #
    # for i in range(loader.y_test.shape[0]):
    #     if loader.y_test[i][0] > max:
    #         max = loader.y_train[i][0]
    # print(max)

    train_step = 10000

    for i in range(train_step):
        network.train(loader.x_train, np.squeeze(loader.y_train))

    y_predict = np.squeeze(network.output(loader.x_test))

    print(f1_score(np.squeeze(loader.y_test), y_predict, average='macro'))