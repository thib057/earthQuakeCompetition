import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.layers import CuDNNGRU


class NeuralNetv1():
	# create model
	def __init__(self, input_dim, **kwargs):
		model = Sequential()
		model.add(Dense(256, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
		model.add(Dense(128, kernel_initializer='normal', activation='relu'))
		model.add(Dense(1, kernel_initializer='normal'))
		# Compile model
		learning_rate = kwargs.get('learning_rate', 0.001)
		opt = Adam(lr=learning_rate , beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
		model.compile(loss='mean_absolute_error', optimizer=opt)
		self.model = model

	def train(self, X, y, **kwargs):

		epochs = kwargs.get('epochs', 15)
		batch_size = kwargs.get('batch_size', 1)


		history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)
		print(history.history.keys())
		# "Loss"
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.show()


	def predict(self, X):
		return self.model.predict(X)

class RNNNetv1():

	# create model
	def __init__(self, input_dim, **kwargs):
		model = Sequential()
		model.add(CuDNNGRU(48, input_shape=(None, input_dim)))
		model.add(Dense(10, activation='relu'))
		model.add(Dense(1))
		# Compile model
		learning_rate = kwargs.get('learning_rate', 0.001)
		opt = Adam(lr=learning_rate , beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
		model.compile(loss='mean_absolute_error', optimizer=opt)
		self.model = model

	def train(self, X, y, **kwargs):
		epochs = kwargs.get('epochs', 15)
		batch_size = kwargs.get('batch_size', 1)


		history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)
		print(history.history.keys())
		# "Loss"
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.show()


	def predict(self, X):
		return self.model.predict(X)


	def _extract_features_for_ts(self, ts):
			# Extract features for each time step
			return np.c_[ts.mean(axis=1),
						 ts.min(axis=1),
						 ts.max(axis=1),
						 ts.std(axis=1)]

