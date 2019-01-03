from loader import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

'''
X_samples = np.array(
[[1,1,1,1,1,1,1],
 [2,2,2,2,2,2,2],
 [3,3,3,3,3,3,3],
 [1,2,1,2,1,2,1],
 [2,2,2,2,2,1,2],
 [1,2,2,2,3,2,2],
 [3,3,1,3,3,3,2]])
y_samples = np.array([1,2,3,1,2,2,3])

X_tests = np.array(
[[1,2,1,1,1,1,1],
 [2,2,1,1,1,1,1],
 [3,3,3,2,2,3,3],
 [1,2,3,3,3,3,3],
 [2,2,2,2,3,1,1],
 [1,1,3,1,1,1,3]])
y_tests = np.array([1,1,3,3,2,1])
'''

if __name__ == "__main__":
    loader = Loader(LABEL_FILE_PATH, DATA_FILE_PATH, 10, TRAINING_SET_PERCENT, FIRST_TIME_TO_READ_FILE)
    rf = RandomForestClassifier().fit(loader.x_train, loader.y_train)
    y_predict = rf.predict(loader.x_test)
    print(y_predict)

    f1 = metrics.f1_score(loader.y_test, y_predict, average='weighted')
    print("RandomForestClassifier F1=", f1)