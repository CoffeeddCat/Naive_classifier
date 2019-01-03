from config import *
from sklearn.svm import SVC
import loader
import numpy as np

loader = loader.Loader(LABEL_FILE_PATH, DATA_FILE_PATH, 10, TRAINING_SET_PERCENT)

classifier = SVC(kernel='rbf', class_weight='balanced')
