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

import sklearn.preprocessing
import sklearn.linear_model
import sklearn.base

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


# TODO: fix regression model
class RegressionChordFrettingEvaluator(ChordFrettingEvaluatorBase):
    """
    Machine Learning based evaluation of the chords
    General Idea: often played = good, try to regress the count/goodness
    """
    __eval_count__ = 0
    __eval_avg_time__ = 0

    def __init__(self, training_data_file: str, weights: dict,
                 model_type: type = sklearn.linear_model.LinearRegression, kwargs: typing.Optional[dict] = None):
        """
        :param training_data_file: Where to load the training data from
        :type training_data_file: str
        :param weights: weights for the different sub models (e.g. {'unigram': 5.0, 'delta1': 1.0})
        :type weights: dict of str, float
        :param model_type: A regression model, subclass of sklearn.base.RegressorMixin
        :type model_type: subclass of sklearn.base.RegressorMixin
        :param kwargs: params for the regression model
        :type kwargs: dict
        """
        ChordFrettingEvaluatorBase.__init__(self, 'regression')
        assert os.path.isfile(training_data_file), 'Could not read file: {}'.format(training_data_file)
        assert type(weights) is dict
        assert issubclass(model_type, sklearn.base.RegressorMixin), \
            'model_type must be a Regression Model (subclass of sklearn.base.RegressorMixin): {}'.format(model_type)
        if kwargs is None:
            self._kwargs = {}
        else:
            assert type(kwargs) is dict
            self._kwargs = kwargs

        data = pd.read_csv(training_data_file, index_col=0)
        self._data_file = training_data_file
        self._data = data
        self._model_weights = weights
        self._model_type = model_type
        self._train()

    def __str__(self) -> str:
        return 'RegressionChordFrettingEvaluator(\'{}\', {}, {}, {})'.format(
            self._data_file, self._model_weights, self._model_type, self._kwargs)

    __repr__ = __str__

    def _train(self) -> None:
        print('Training {}'.format(self))

        t_start = time()

        self._columns = {}
        self._scalers = {}

        # unigram model
        model_name = 'unigram'
        columns = [col for col in self._data if not (col.startswith('delta') or
                                                     col.startswith('prev') or
                                                     col.startswith('counts_') or
                                                     col.startswith('cost_'))]
        x = self._data[columns]
        self._columns[model_name] = columns
        self._scalers[model_name] = sklearn.preprocessing.StandardScaler().fit(x)
        x = self._scalers[model_name].transform(x)

        y = self._data['cost_{}'.format(model_name)].values.reshape(-1, 1)
        model = self._model_type(**self._kwargs).fit(x, y)  # actual training
        self._models = {model_name: model}

        if __debug__:
            print('Model "{}" trained. Training score: {}'.format(model_name, model.score(x, y)))

        # delta models
        if FeatureConfiguration.delta:
            columns = []
            for depth in range(1, FeatureConfiguration.max_depth + 1):
                columns = columns[:]
                model_name = 'delta{}'.format(depth)

                # keep previous columns!
                for col in self._data:
                    if col.startswith(model_name):
                        columns.append(col)

                if model_name in self._model_weights:

                    x = self._data[columns]
                    self._columns[model_name] = columns
                    self._scalers[model_name] = sklearn.preprocessing.StandardScaler().fit(x)
                    x = self._scalers[model_name].transform(x)

                    y = self.count2cost(self._data[['counts_{}'.format(model_name)]])
                    model = self._model_type(**self._kwargs).fit(x, y)
                    self._models[model_name] = model

                    if __debug__:
                        print('Model "{}" trained. Training score: {}'.format(model_name, model.score(x, y)))

        # prev models
        if FeatureConfiguration.prev:
            columns = self._columns['unigram'][:]  # copy unigram columns
            for depth in range(1, FeatureConfiguration.max_depth + 1):
                columns = columns[:]
                model_name = 'prev{}'.format(depth)

                # keep previous columns!
                for col in self._data:
                    if col.startswith(model_name):
                        columns.append(col)

                if model_name in self._model_weights:

                    x = self._data[columns]
                    self._columns[model_name] = columns
                    self._scalers[model_name] = sklearn.preprocessing.StandardScaler().fit(x)
                    x = self._scalers[model_name].transform(x)

                    y = self.count2cost(self._data[['counts_{}'.format(model_name)]])
                    model = self._model_type(**self._kwargs).fit(x, y)
                    self._models[model_name] = model

                    if __debug__:
                        print('Model "{}" trained. Training score: {}'.format(model_name, model.score(x, y)))

        t_end = time()
        if TRACK_PERFORMANCE:
            print('Model training time: {} s'.format(t_end - t_start))

    @staticmethod
    def count2cost(counts) -> float:
        """
        Transform counts to non-negative cost function
        """
        assert type(counts) is int or type(counts) is list or type(counts) is pd.DataFrame
        # use numpy to enable list handling
        return 1.0 / np.log(np.array(counts) + 2.0)

    def evaluate(self, fretting: ChordFretting) -> float:
        t_start = time()

        # add up weighted cost over models
        cost = 0
        for model_name, model in self._models.items():
            if model_name in self._model_weights:
                features_model = [[ff[1] for ff in fretting.features.items() if ff[0] in self._columns[model_name]]]
                x = self._scalers[model_name].transform(features_model)
                cost += float(model.predict(x))

        t_end = time()
        if TRACK_PERFORMANCE:
            t = (t_end - t_start) * 1000
            # print('ML single prediction time: {} ms'.format(t))
            self.__eval_count__ += 1
            if self.__eval_avg_time__ == 0:
                self.__eval_avg_time__ = t
            else:
                # moving average
                self.__eval_avg_time__ = \
                    (1.0 - 1.0 / self.__eval_count__) * self.__eval_avg_time__ \
                    + (1.0 / self.__eval_count__) * t

        return cost

    def __del__(self) -> None:
        if TRACK_PERFORMANCE:
            print('Calls of evaluate(): {}'.format(self.__eval_count__))
            print('Average runtime: {} ms'.format(self.__eval_avg_time__))
            print('Total time spent in evaluate(): {} s'.format(self.__eval_avg_time__ * self.__eval_count__ / 1000.0))


class LSTMChordFrettingEvaluator(ChordFrettingEvaluatorBase):
    """
    Predicts the most likely next feature vector and determines cost as Euclidean distance in feature space
    """

    def __init__(self):
        super().__init__('lstm')

        np.random.seed(1337)  # for reproducibility
        from keras.models import Sequential
        # from keras.backend.tensorflow_backend import tf
        # tf.device("/cpu:0")

        path_config = os.path.join(DATA_PATH, 'lstm_config.npy')
        self._path_maxvals = os.path.join(DATA_PATH, 'lstm_maxvals.npy')
        self._path_targets = os.path.join(DATA_PATH, 'lstm_targets.npy')
        self._path_weights = os.path.join(DATA_PATH, 'lstm_weights.hdf')

        # load model
        model_config = list(np.load(path_config).tolist())
        self._model = Sequential.from_config(model_config)
        self._batch_size = model_config[0]['config']['units']

        # train if no weights / load existing weights
        if not (
            os.path.isfile(self._path_maxvals) and
            os.path.isfile(self._path_targets) and
            os.path.isfile(self._path_weights)
        ):
            print('No weights found. Training LSTM.')
            self.train()
        else:
            # load from files
            self._max_vals = np.load(self._path_maxvals)
            self._target_names = np.load(self._path_targets)
            self._compile_model()
            self._model.load_weights(self._path_weights)
        assert len(self._target_names) == FeatureConfiguration.num_features_total, \
            'Targets: {} vs. num_features: {}'.format(
                len(self._target_names), FeatureConfiguration.num_features_total)

    def _compile_model(self):
        from keras.backend.tensorflow_backend import tf
        tf_mask = tf.constant(dtype=tf.float32, value=self._feature_mask)
        # loss = 'mse'
        self._model.compile(
            optimizer='adam',

            # custom loss function: apply mask to ignore some output features
            loss=lambda y_true, y_pred: tf.reduce_mean(tf.square(tf.multiply(tf.subtract(y_true, y_pred), tf_mask)))
        )

    def evaluate(self, fretting: ChordFretting) -> float:
        # Evaluate the fit by comparing against a prediction from an LSTM
        # predict current by prev

        prev_x = fretting
        dataset = []

        for ii in range(FeatureConfiguration.max_depth + 1):
            # concatenate previous feature vector
            dataset = [np.array(
                [prev_x.features[name] for name in self._target_names]
            )] + dataset

            # go back one step further
            prev_x = prev_x.previous

        dataset = np.array([dataset])
        xx = dataset[:, :-1, :]
        yy = dataset[:, -1, :] * self._feature_mask  # set NEXT features to 0 for prediction target

        assert dataset.shape == (1, FeatureConfiguration.max_depth + 1, FeatureConfiguration.num_features_total)

        # scale
        xx = self.divide_or_zero(xx, self._max_vals[0])
        yy = self.divide_or_zero(yy, self._max_vals[1])

        # Get cost as prediction loss from model
        cost = self._model.evaluate(xx, yy, verbose=0)

        return cost

    @property
    def _feature_mask(self):
        return np.array([int(not col.startswith('next') and not col.startswith('delta')) for col in self._target_names])

    def _get_training_data(self):

        # load the dataset
        print('Loading training data...')
        dataframe = pd.read_csv(TRAINING_DATA_FILE_PATH, index_col=0)
        dataframe.drop(
            [col for col in dataframe.columns
             if col.startswith('counts') or col.startswith('cost')],
            axis=1, inplace=True
        )

        # convert prev features to numpy dataset

        # prev0 = current (prediction  target)
        # remember feature names
        self._target_names = [col for col in dataframe.columns if not col.startswith('prev')]

        # mask to set the "next"-features to zero in the evaluation
        # i.e. do not try to predict the next_pitch stuff!
        yy = dataframe[self._target_names].values.astype('float32') * self._feature_mask

        # build array of prev features
        xx = []
        for prev_x in range(1, FeatureConfiguration.max_depth + 1):
            cols = [col for col in dataframe.columns if col.startswith('prev{}_'.format(prev_x))]
            xx.append(dataframe[cols].values.astype('float32'))

        # switch dimensions and invert prev 1,2,3 --> 3,2,1
        xx = np.array(xx).transpose((1, 0, 2))[:, ::-1, :]

        # scale & remember scaling factor
        self._max_vals = np.array([np.max(xx, axis=0), np.max(yy, axis=0)])
        xx = self.divide_or_zero(xx, self._max_vals[0])
        yy = self.divide_or_zero(yy, self._max_vals[1])

        # save
        np.save(self._path_maxvals, self._max_vals)
        np.save(self._path_targets, self._target_names)

        return xx, yy

    def train(self, num_epochs: int = 10, train_relative: float = 0.95):
        """
        Train the model -- only do this once, load the weights afterwards
        """

        xx, yy = self._get_training_data()

        print('Training LSTM on data of shape {}'.format(xx.shape))

        # split into x and y, train and test sets
        train_size = int(len(xx) * train_relative)
        x_train = xx[:train_size]
        y_train = yy[:train_size]
        x_valid = xx[train_size:]
        y_valid = yy[train_size:]
        print('Train and validation sets:', x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

        # TRAIN
        self._compile_model()
        self._model.fit(x_train, y_train, epochs=num_epochs, batch_size=self._batch_size, shuffle=False)
        score_train = self._model.evaluate(x_train, y_train)
        print('\nTraining loss: {} (sqrt: {}) / {} = {} ({})'.format(
            score_train, np.sqrt(score_train), FeatureConfiguration.num_features_total,
            score_train / FeatureConfiguration.num_features_total,
            np.sqrt(score_train) / FeatureConfiguration.num_features_total
        ))
        score_valid = self._model.evaluate(x_valid, y_valid)
        print('\nEvaluation loss: {} (sqrt: {}) / {} = {} ({})'.format(
            score_valid, np.sqrt(score_valid), FeatureConfiguration.num_features_total,
            score_valid / FeatureConfiguration.num_features_total,
            np.sqrt(score_valid) / FeatureConfiguration.num_features_total
        ))

        # save for later
        self._model.save_weights(self._path_weights)

    @staticmethod
    def divide_or_zero(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        divide x1/x2 and return zeroes if error
        :param x1: divident
        :type x1: np.ndarray
        :param x2: divisor
        :type x2: np.ndarray
        :return: x1/x2 or zeroes
        :rtype: np.ndarray
        """
        # division by 0 -> 0
        with np.errstate(divide='ignore', invalid='ignore'):
            ds = np.true_divide(x1, x2)
            ds[ds == np.inf] = 0
            ds = np.nan_to_num(ds)
            return ds

    def inverse_scale(self, xx: np.ndarray) -> np.ndarray:
        """
        Scale a dataset by the maxvals (inverted transform)
        :param xx: normalised numpy array
        :type xx: np.ndarray
        :return: xx in original scale
        :rtype: np.ndarray
        """
        return xx * self._max_vals[0]
