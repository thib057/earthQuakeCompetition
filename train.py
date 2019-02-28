from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle


class Model():

    def __init__(self, X, y , path_to_save = "", save = True):
        self.save = save
        self.path_to_save = path_to_save
        self.model = SVR(kernel="linear", degree=3)
        self.y = y
        self.X = X


    def evaluate_model(self, cv = 5):
        sc = cross_val_score(self.model, self.X, self.y, cv=cv, scoring="neg_mean_absolute_error")
        print("Cross validation returns on training NMSE of %0.2f"%np.mean(sc))

    def train(self):

        self.model = self.model.fit(self.X, self.y)
        if self.save:
            s = pickle.dumps(self.model)
        else:
            return
