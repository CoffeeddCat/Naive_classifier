import tensorflow as tf
import numpy as np
import loader
from sklearn.metrics import f1_score
from nn_model import Network
from config import *

if __name__ == '__main__':
    network = Network([], tf.nn.relu, 22283, 1e-4)
    loader = loader.Loader(LABEL_FILE_PATH, DATA_FILE_PATH, 10, TRAINING_SET_PERCENT, FIRST_TIME_TO_READ_FILE)

    train_step = 10000

    for i in range(train_step):
        network.train(loader.x_train, np.squeeze(loader.y_train))

    y_predict = np.squeeze(network.output(loader.x_test))

    print(f1_score(np.squeeze(loader.y_test), y_predict, average='macro'))