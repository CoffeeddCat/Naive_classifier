from loader import *
from sklearn.decomposition import PCA
import numpy as np

class PCA_reduce:
    def __init__(self, variance=0.95, read_file = False):
        if not read_file:
            pca = PCA(n_components=variance)
            print("Start PCA...")
            loader = Loader(LABEL_FILE_PATH, DATA_FILE_PATH, 10, TRAINING_SET_PERCENT, FIRST_TIME_TO_READ_FILE)
            # train and reduce x_train
            self.x_train_reduced = pca.fit_transform(loader.x_train)
            np.save("../data/x_train_reduced.npy", self.x_train_reduced)
            self.feature_num = pca.n_components_
            print(self.feature_num)
            # reduce x_test
            self.x_test_reduced = pca.transform(loader.x_test)
            np.save("../data/x_test_reduced.npy", self.x_test_reduced)
            print("PCA done!")
        else:
            self.x_train_reduced = np.load("../data/x_train_reduced.npy")
            print("Load x_train_reduced over!")
            self.x_test_reduced = np.load("../data/x_test_reduced.npy")
            print("Load x_test_reduced over!")

if __name__ == "__main__":
    pca_reduce = PCA_reduce()
    print(pca_reduce.x_train_reduced)
