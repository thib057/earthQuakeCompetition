import os
import numpy as np
import pandas as pd

class Predictor:
    """
    This class predicts, for each segment, the remaining time before EarthQuake
    It needs the folder path refering to the test data

    Then, it saves the output dataframe for submission on Kaggle

    Dependencies : Feature extractor and model, as we need to extract features the same way we did before
    """
    def __init__(self,
                 x_test,
                 y_test,
                 path_to_test_dir = '/Users/drapp thibaut/Documents/10_Personal_projects/kaggle_earthquake/data/test/',
                 model = None,
                 save = True):

        self.model = model
        self.seg_ids = np.array(os.listdir(path_to_test_dir))
        self.path_to_test_dir = path_to_test_dir
        self.save = save

        # Initiate result dataframe
        t_data_init = np.array([[" " * 10, 0.] for i in range(len(self.seg_ids))]) # 10 is length of seg id
        t_data_init[:, 0] = self.seg_ids

        self.df_result = pd.DataFrame(columns=["seg_id", "time_to_failure"], data = t_data_init)

        # empty init
        self.X_features = None


    def predict(self):
        self.df_result.loc[:, "time_to_failure"] = self.model.predict(self.X_features)
        if self.save:
            self.df_result.to_csv("test_result.csv", index=False)

    def load_model(self):
        # update self.model
        pass



