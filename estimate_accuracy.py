#!/usr/bin/env python

from tabgen.definitions import *
from tabgen import evaluation
from tabgen import processing
from tabgen import modelling

# from sklearn.neural_network import MLPRegressor

input_files = [os.path.join(VALIDATION_INPUT_TAB_PATH, input_file.strip())
               for input_file in os.listdir(VALIDATION_INPUT_TAB_PATH)]
# remove duplicates (gp + ms file)
input_files = [input_file for input_file in input_files if input_file + '.mscx' not in input_files]

# define models to compare
evaluators = {
    # accuracies over top 100 (92 files)
    # 'linear_regression': evaluation.RegressionChordFrettingEvaluator(
    #     TRAINING_DATA_FILE_PATH, {'unigram': 1.0, 'delta1': 1.0}),  # 63.25%
    # 'mlp_100': evaluation.RegressionChordFrettingEvaluator(
    #     TRAINING_DATA_FILE_PATH, {'unigram': 1.0, 'delta1': 1.0}, MLPRegressor  # 62.88%
    # ),
    # 'mlp_140_70_50': evaluation.RegressionChordFrettingEvaluator(
    #     TRAINING_DATA_FILE_PATH, {'unigram': 1.0, 'delta1': 1.0}, MLPRegressor,  # 64.13%
    #     dict(hidden_layer_sizes=(140, 70, 50))
    # ),
    # 'dummy': evaluation.DummyFrettingEvaluator(),  # 42.36%
    # 'random': evaluation.RandomChordFrettingEvaluator(),  # 53.84%
    # 65 - 70% for baseline
    # 'baseline': evaluation.BaselineChordFrettingEvaluator(dict(fret_range=1.0, fret_min=0.1, string_range=1.0)),
    'lstm': evaluation.LSTMChordFrettingEvaluator(),
}

pruning = modelling.PruningConfiguration(
    candidate_beam_width=0.5, max_candidates=2,
    sequence_beam_width=0.5, max_sequences=3,
)

# evaluate the different models
for eval_name, evaluator in evaluators.items():
    parser = processing.MuseScoreXMLParser(evaluator)
    solver = processing.FrettingGenerator(evaluator, pruning)
    solver.solve_multi(input_files, parser, True, 2)