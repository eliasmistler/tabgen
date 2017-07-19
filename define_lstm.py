# BUILD MODEL

from tabgen.definitions import *
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM

batch_size = 128

# build a model
model = Sequential([
    LSTM(batch_size, return_sequences=False,
         input_shape=(FeatureConfiguration.max_depth, FeatureConfiguration.num_features_total)),
    Dense(128),
    Activation('tanh'),
    Dense(FeatureConfiguration.num_features_total),
    Activation('tanh'),
])

# save the model config
path_config = os.path.join(DATA_PATH, 'lstm_config.npy')
np.save(path_config, model.get_config())

# remove weights, so the model will be retrained
path_weights = os.path.join(DATA_PATH, 'lstm_weights.hdf')
if os.path.isfile(path_weights):
    os.remove(path_weights)
