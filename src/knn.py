from loader import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np

'''
samples_X = np.array(
[[1,1,1,1,1,1,1],
 [2,2,2,2,2,2,2],
 [3,3,3,3,3,3,3],
 [1,2,1,2,1,2,1],
 [2,2,2,2,2,1,2],
 [1,2,2,2,3,2,2],
 [3,3,1,3,3,3,2]])
samples_y = np.array([1,2,3,1,2,2,3])

tests_X = np.array(
[[1,2,1,1,1,1,1],
 [2,2,1,1,1,1,1],
 [3,3,3,2,2,3,3],
 [1,2,3,3,3,3,3],
 [2,2,2,2,3,1,1],
 [1,1,3,1,1,1,3]])
tests_y = np.array([1,1,3,3,2,1])

def knn(k, samples_X, samples_y, tests_X):
    if k == 0: raise Exception("k cannot be 0!")
    test_y = []
    for test in tests_X: # predict each test
        distances_and_labels = []
        for i, sample in enumerate(samples_X): # get [the distance to each sample, sample's label]
            sample_array = np.array(sample)
            test_array = np.array(test)
            distance_array = sample_array - test_array
            distances_and_labels.append([np.inner(distance_array, distance_array), samples_y[i]])
        distances_and_labels.sort(key=lambda x:x[0]) # sort the samples by distances
        for i in range(len(distances_and_labels)): distances_and_labels[i] = distances_and_labels[i][1]
        nearest_labels = distances_and_labels[:k] # get the k nearest neighbours
        print(nearest_labels)
        label_num = 9999
        label_freq = []
        for i in range(label_num): label_freq.append(0)
        for i in nearest_labels: label_freq[i]+=1
        test_y.append(label_freq.index(max(label_freq)))
    return test_y

def get_accuracy(tests_y, predict_y): # Accuracy is (Correct predictions / All predictions)
    tests_num = len(tests_y)
    correct_num = 0
    for i, value in enumerate(tests_y):
        if value == predict_y[i]: correct_num += 1
    return correct_num / tests_num
'''

if __name__ == "__main__":
    """
    predict_y = knn(3, samples_X, samples_y, tests_X)
    print(predict_y)
    print(get_accuracy(tests_y, predict_y))
    """
    loader = Loader(LABEL_FILE_PATH, DATA_FILE_PATH, 10, TRAINING_SET_PERCENT, FIRST_TIME_TO_READ_FILE)
    try:
        1/0
        y_predict = np.load("../data/knn.npy")
        print("KNN y_predict loaded.")
    except:
        print("Start KNN")
        neigh = KNeighborsClassifier(5)
        neigh.fit(loader.x_train, loader.y_train)
        y_predict = neigh.predict(loader.x_test)
        np.save("../data/knn.npy", y_predict)
        # print("KNN y_predict saved.")

    # print(y_predict.tolist())
    f1 = metrics.f1_score(loader.y_test, y_predict, average='weighted')
    print("KNN f1=",f1)