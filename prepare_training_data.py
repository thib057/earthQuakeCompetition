import pandas as pd
import numpy as np


class DataProcessing():

    def __init__(self, training_csv_path = "../data/train.csv", chunk_size = 150000):
        print("Reading data")
        self.train_set = pd.read_csv(training_csv_path)
        print("Data read")
        self.chunk_size = chunk_size
        x_train = np.zeros([len(self.train_set) // 150000, chunk_size])
        y_train = []
        # Cut data into chuncks
        for i in range(len(self.train_set) // 150000):
            x_train[i] = self.train_set.loc[i * 150000:(i + 1) * 150000 - 1, "acoustic_data"].values
            y_train.append(self.train_set.loc[(i + 1) * 150000 - 1, "time_to_failure"])

        self.x_train = x_train
        self.y_train = np.array(y_train)

    def get_feature_data(self):
        """
        Just print some figures and returns the data needed for the model to learn
        :return:
        x_train : array of shape (samples, signal)
        y_train : array of label, shape nb of samples
        """
        print("segment size : %i"%self.chunk_size)
        print("number of samples for training : %i"%len(self.x_train))
        return self.x_train, self.y_train


