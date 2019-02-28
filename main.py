from feature_extractor import FeatureExtractor
from predict_and_process import Predictor
from prepare_training_data import DataProcessing
from train import Model
from feature_generator import FeatureGenerator
from models import NeuralNetv1

NB_PCA = 10


def main():
    print("Start data preprocessing and feature extraction..")

    training_fg = FeatureGenerator(dtype='train', n_jobs=4, chunk_size=150000)
    training_data = training_fg.generate()

    test_fg = FeatureGenerator(dtype='test', n_jobs=4, chunk_size=None)
    test_data = test_fg.generate()

    training_data.to_csv("../input/train_features.csv", index=False)
    test_data.to_csv("../input/test_features.csv", index=False)

    X = training_data.iloc[:,:-2].values
    y = training_data.loc[:, "target"].values
    print("Train model")
    model = NeuralNetv1(training_data, y, save=False)
    model.evaluate_model()
    model.train()
    print("Model trained")

    print("Build result and predict for Kaggle")
    predictor = Predictor(x_test, y_test, model = model.model)
    predictor.predict()

    print("END")
if __name__ == '__main__':
    main()