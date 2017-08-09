"""
module tabgen.evaluation

Description:  Evaluator classes for tabgen
              All classes have to inherit from the base class ChordFrettingEvaluator

Contains:     ChordFrettingEvaluator
              RandomChordFrettingEvaluator
              BaselineChordFrettingEvaluator
              RegressionChordFrettingEvaluator

Author:       Elias Mistler
Institute:    The University of Edinburgh
Last changed: 2017-06
"""

from sklearn.exceptions import NotFittedError
from tabgen.modelling import ChordFretting
from .definitions import *


class DummyFrettingEvaluator(ChordFrettingEvaluatorBase):
    def __init__(self):
        ChordFrettingEvaluatorBase.__init__(self, 'dummy')

    def evaluate(self, fretting: ChordFretting) -> float:
        return 0.0

    def __str__(self) -> str:
        return 'DummyFrettingEvaluator()'

    __repr__ = __str__


class RandomChordFrettingEvaluator(ChordFrettingEvaluatorBase):
    """
    Simplest approach: just select a random fretting
    """

    def __init__(self, random_seed: int = None):
        ChordFrettingEvaluatorBase.__init__(self, 'random')
        np.random.seed(random_seed)

    def __str__(self) -> str:
        return 'RandomChordFrettingEvaluator()'

    __repr__ = __str__

    def evaluate(self, fretting: ChordFretting) -> float:
        return np.random.random()


class BaselineChordFrettingEvaluator(ChordFrettingEvaluatorBase):
    """
    Encapsulation of the Evaluation algorithm, i.e. cost function
    - Baseline model, using a simple heuristic:
        input weights relating to features are used to calculate
        cost = sum( weight[i] * feature[i] )
    - can use sub heuristics (heuristic features), e.g. heuristic_distance_move
    """

    def __init__(self, weights: dict):
        """
        :param weights: dictionary of weights associated with parameters
        :type weights: dict of string, float
        """
        ChordFrettingEvaluatorBase.__init__(self, 'baseline')
        assert type(weights) is dict
        self._weights = weights

    def __str__(self) -> str:
        return 'BaselineChordFrettingEvaluator({})'.format(self._weights)

    __repr__ = __str__

    def evaluate(self, fretting: ChordFretting) -> float:
        # using a simple heuristic
        cost = 0.0
        for feature_name in self._weights.keys():
            cost += self._weights[feature_name] * fretting.features[feature_name]
        return cost


class ProbabilityLookupEvaluator(ChordFrettingEvaluatorBase):
    def __init__(self):
        ChordFrettingEvaluatorBase.__init__(self, 'probability_lookup')
        self._probs = np.load(os.path.join(Path.DATA, 'probabilities_1.npy')).tolist()

    def evaluate(self, fretting: ChordFretting) -> float:
        # setting
        rounding_digits = 2

        columns = [col for col in pd.read_csv(Path.FEATURE_FILE, nrows=0).columns if not col.startswith('prob')]

        # mask to exclude lookahead features
        mask = [int(not col.startswith('next')) for col in columns]

        # TODO ProbabilityLookupEvaluator - find probability or set low (e.g. 0.01)
        data_flat = np.array([[fretting.features[ff] for ff in columns if not ff.startswith('next')]], np.float32)

        # key = row.


class RegressionChordFrettingEvaluator(ChordFrettingEvaluatorBase):
    """
    Machine Learning based evaluation of the chords
    General Idea: often played = good, try to regress the count/goodness
    """

    def __init__(self):
        ChordFrettingEvaluatorBase.__init__(self, 'regression')

        columns = pd.read_csv(Path.FEATURE_FILE, nrows=0).columns

        self._target_names = ['probs_{}'.format(ii) for ii in range(1, FeatureConfig.max_depth + 1)]
        self._source_names = [col for col in columns if not col.startswith('probs_') and not col.startswith('count')]

        print('Initialising Regression Model. \n\t{} source fields: {}\n\t{} target fields: {}'.format(
            len(self._source_names), self._source_names, len(self._target_names), self._target_names
        ))

        self._batch_size = 128

        from keras.models import Sequential
        from keras.layers import Dense, Activation, LSTM
        self._model = Sequential([
            LSTM(self._batch_size, return_sequences=False,
                 input_shape=(FeatureConfig.max_depth + 1, len(self._source_names))),
            # Dense(self._batch_size, input_shape=((FeatureConfig.max_depth + 1) * len(self._source_names),), ),
            Dense(1024),
            Activation('tanh'),
            Dense(512),
            Activation('tanh'),
            Dense(1)
        ])

        self._model.compile(
            optimizer='adam',
            loss='mse'
        )
        self._trained = False

        # set paths for storing weights and normalisation factors
        self._path_maxvals = os.path.join(Path.DATA, 'cost_maxvals.npy')
        self._path_weights = os.path.join(Path.DATA, 'cost_weights.hdf')

        # train if no weights / load existing weights
        if not (os.path.isfile(self._path_maxvals) and
                os.path.isfile(self._path_weights)):
            print('No weights found. Train model before use!')
        else:
            # load from files
            try:
                self._model.load_weights(self._path_weights)
                self._max_vals = np.load(self._path_maxvals)
                print('Weights found and loaded. Regression model ready.')
                self._trained = True
            except ValueError:
                warnings.warn('Model structure changed. Re-train!')
                self._trained = False

    def __str__(self) -> str:
        return 'RegressionChordFrettingEvaluator()'

    __repr__ = __str__

    @property
    def is_trained(self):
        return self._trained

    def load_training_data(self):
        # load the dataset
        print('Loading training data...', end='')
        dataframe = pd.read_csv(Path.FEATURE_FILE,
                                usecols=np.concatenate((self._source_names, self._target_names)))

        # make probabilities into neg. log likelihood
        yy = dataframe[self._target_names].values.astype('float32')
        likelihood = yy[:, 0]
        for context_length in range(1, FeatureConfig.max_depth):
            likelihood = np.multiply(likelihood, yy[:, context_length])
        yy = -np.log(likelihood)

        # shape xx array
        xx = dataframe[self._source_names].values.astype('float32')
        xx = xx.reshape((xx.shape[0], 1, xx.shape[1]))
        if FeatureConfig.max_depth > 0:
            xx_prev = np.concatenate([
                np.concatenate([np.zeros_like(xx[0:i]), xx[:-i, :, :]])
                for i in range(FeatureConfig.max_depth, 0, -1)
            ], axis=1)
            xx = np.concatenate([xx_prev, xx], axis=1)

        # scale & remember scaling factor
        self._max_vals = (
            [max(x, 1) for x in np.max(xx, axis=0).max(axis=0)],
            # [max(y, 1) for y in yy.max(axis=0)]
            max(max(yy), 1)
        )
        print('Max Vals: ', self._max_vals)
        np.save(self._path_maxvals, self._max_vals)  # save maxvals

        xx /= self._max_vals[0]
        yy /= self._max_vals[1]

        print('done')
        return xx, yy

    def train(self, num_epochs: int = 10, train_relative: float = 0.95) -> None:
        """
        Train the model -- only do this once, load the weights afterwards
        """

        xx, yy = self.load_training_data()

        # reshape for non-LSTM-like input
        # xx = xx.reshape((xx.shape[0], -1))

        # split into x and y, train and test sets
        train_size = int(len(xx) * train_relative)
        x_train = xx[:train_size]
        y_train = yy[:train_size]
        x_valid = xx[train_size:]
        y_valid = yy[train_size:]
        print('Train and validation sets:', x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

        # TRAIN
        self._model.fit(x_train, y_train, epochs=num_epochs, batch_size=self._batch_size, shuffle=True)
        score_train = self._model.evaluate(x_train, y_train)
        print('\nTraining loss: {} (sqrt: {}) / {} features = {} ({})'.format(
            score_train, np.sqrt(score_train), len(self._target_names),
            score_train / len(self._target_names),
            np.sqrt(score_train) / len(self._target_names)
        ))
        score_valid = self._model.evaluate(x_valid, y_valid)
        print('\nEvaluation loss: {} (sqrt: {}) / {} features = {} ({})'.format(
            score_valid, np.sqrt(score_valid), len(self._target_names),
            score_valid / len(self._target_names),
            np.sqrt(score_valid) / len(self._target_names)
        ))

        # save for later
        self._model.save_weights(self._path_weights)
        self._trained = True

    def evaluate(self, fretting: ChordFretting) -> float:
        """
        add up weighted cost over models
        """
        if not self.is_trained:
            raise NotFittedError()

        # get handles to relevant frettings (current and previous (FeatureConfig.max_depth))
        fretting_handles = [fretting]
        while len(fretting_handles) < FeatureConfig.max_depth + 1:
            fretting_handles.append(fretting_handles[-1].previous)
        fretting_handles = fretting_handles[::-1]

        # form input vector
        xx = np.array([[
            [handle.features[ff] for ff in self._source_names]
            for handle in fretting_handles
        ]])
        xx = (xx / self._max_vals[0])

        # reshape for non-LSTM-like input
        # xx = xx.reshape(1, -1)

        cost = self._model.predict(xx)[0][0]
        cost *= self._max_vals[1]

        return cost


class LSTMChordFrettingEvaluator(ChordFrettingEvaluatorBase):
    """
    Predicts the most likely next feature vector and determines cost as Euclidean distance in feature space
    """

    def __init__(self):
        super().__init__('lstm')

        # set the source and target field names
        self._source_names = np.array([
            col for col in pd.read_csv(Path.FEATURE_FILE, nrows=0).columns if
            not col.startswith('heuristic') and
            not col.startswith('prob')
        ])

        self._target_names = np.array([
            col for col in self._source_names if
            not col.startswith('next') and
            not col.startswith('pitch') and
            not col.startswith('part')
        ])

        print('Initialising LSTM. \n\t{} source fields: {}\n\t{} target fields: {}'.format(
            len(self._source_names), self._source_names, len(self._target_names), self._target_names
        ))

        self._batch_size = 128

        # build the model
        from keras.models import Sequential
        from keras.layers import Dense, Activation, LSTM
        self._model = Sequential([
            LSTM(self._batch_size, return_sequences=False,
                 input_shape=(FeatureConfig.max_depth, len(self._source_names))),
            # Dense(self._batch_size, input_shape=(FeatureConfig.max_depth * len(self._source_names),), ),
            Dense(1024),
            Activation('tanh'),
            Dense(512),
            Activation('tanh'),
            Dense(len(self._target_names)),
            Activation('tanh'),
        ])

        self._model.compile(
            optimizer='adam',
            loss='mse'
        )
        self._trained = False

        # set paths for storing weights and normalisation factors
        self._path_maxvals = os.path.join(Path.DATA, 'lstm_maxvals.npy')
        self._path_weights = os.path.join(Path.DATA, 'lstm_weights.hdf')

        # train if no weights / load existing weights
        if not (os.path.isfile(self._path_maxvals) and
                os.path.isfile(self._path_weights)):
            print('No weights found. Train LSTM before use!')
        else:
            # load from files
            try:
                self._model.load_weights(self._path_weights)
                self._max_vals = np.load(self._path_maxvals)
                print('Weights found and loaded. LSTM ready.')
                self._trained = True
            except ValueError:
                warnings.warn('Model structure changed. Re-train!')
                self._trained = False

    def load_training_data(self):
        # load the dataset
        print('Loading training data...')
        dataframe = pd.read_csv(Path.FEATURE_FILE)
        dataframe.drop(
            [col for col in dataframe.columns
             if col.startswith('counts') or col.startswith('cost')],
            axis=1, inplace=True
        )

        yy = dataframe[self._target_names].values.astype('float32')

        xx = dataframe[self._source_names].values.astype('float32')
        xx = xx.reshape((xx.shape[0], 1, xx.shape[1]))
        xx = np.concatenate([
            np.concatenate([np.zeros_like(xx[0:i]), xx[:-i, :, :]])
            for i in range(FeatureConfig.max_depth, 0, -1)
        ], axis=1)

        # scale & remember scaling factor
        self._max_vals = (
            [max(x, 1) for x in np.max(xx, axis=0).max(axis=0)],
            [max(y, 1) for y in yy.max(axis=0)]
        )
        np.save(self._path_maxvals, self._max_vals)  # save maxvals
        xx /= self._max_vals[0]
        yy /= self._max_vals[1]

        return xx, yy

    @property
    def is_trained(self):
        return self._trained

    def evaluate(self, fretting: ChordFretting) -> float:
        """
        Evaluate the fit by comparing against a prediction from an LSTM
        predict current by prev
        """

        if not self.is_trained:
            raise NotFittedError('LSTM has to be trained first.')

        # build xx from previous frettings
        prev_x = fretting.previous
        xx = []
        for ii in range(1, FeatureConfig.max_depth + 1):
            xx = [np.array(
                [prev_x.features[name] for name in self._source_names]
            )] + xx
            prev_x = prev_x.previous

        xx = np.array([xx]) / self._max_vals[0]

        # get yy
        yy = np.array([[fretting.features[ff] for ff in self._target_names]]) / self._max_vals[1]

        assert xx.shape == (1, FeatureConfig.max_depth, len(self._source_names))
        assert yy.shape == (1, len(self._target_names))

        # reshape for non-LSTM-like input
        # xx = xx.reshape(1, -1)

        # Get cost as prediction loss from model
        cost = self._model.evaluate(xx, yy, verbose=0)
        return cost

    def train(self, num_epochs: int = 10, train_relative: float = 0.95) -> None:
        """
        Train the model -- only do this once, load the weights afterwards
        """

        xx, yy = self.load_training_data()

        # reshape for non-LSTM-like input
        # xx = xx.reshape((xx.shape[0], -1))

        # split into x and y, train and test sets
        train_size = int(len(xx) * train_relative)
        x_train = xx[:train_size]
        y_train = yy[:train_size]
        x_valid = xx[train_size:]
        y_valid = yy[train_size:]
        print('Train and validation sets:', x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

        # TRAIN
        self._model.fit(x_train, y_train, epochs=num_epochs, batch_size=self._batch_size, shuffle=False)
        score_train = self._model.evaluate(x_train, y_train)
        print('\nTraining loss: {} (sqrt: {}) / {} features = {} ({})'.format(
            score_train, np.sqrt(score_train), len(self._target_names),
            score_train / len(self._target_names),
            np.sqrt(score_train) / len(self._target_names)
        ))
        score_valid = self._model.evaluate(x_valid, y_valid)
        print('\nEvaluation loss: {} (sqrt: {}) / {} features = {} ({})'.format(
            score_valid, np.sqrt(score_valid), len(self._target_names),
            score_valid / len(self._target_names),
            np.sqrt(score_valid) / len(self._target_names)
        ))

        # save for later
        self._model.save_weights(self._path_weights)
        self._trained = True

    def inverse_scale(self, xx: np.ndarray) -> np.ndarray:
        """
        Scale a dataset by the maxvals (inverted transform)
        :param xx: normalised numpy array
        :type xx: np.ndarray
        :return: xx in original scale
        :rtype: np.ndarray
        """
        return xx * self._max_vals[0]
