#!/usr/bin/env python

from tabgen.definitions import *
from tabgen import evaluation
from tabgen import processing
from tabgen import modelling

# from sklearn.neural_network import MLPRegressor

input_files = [os.path.join(Path.VALIDATION_INPUT, input_file.strip())
               for input_file in os.listdir(Path.VALIDATION_INPUT)]
# remove duplicates (gp + ms file)
input_files = [input_file for input_file in input_files if input_file + '.mscx' not in input_files]

# define models to compare
evaluators = {
    # 'random': evaluation.RandomChordFrettingEvaluator(),
    # 'baseline_dense': evaluation.BaselineChordFrettingEvaluator(dict(fret_range=1.0, fret_min=0.1, string_range=1.0)),
    # 'baseline_heuristics': evaluation.BaselineChordFrettingEvaluator(dict(
    #     heuristic_distance_steady=0.2,
    #     heuristic_distance_move=0.7,
    #     heuristic_skipped_strings=3,
    #     fret_mean=0.01))
    # 'baseline_heuristic_distance_move': evaluation.BaselineChordFrettingEvaluator(dict(
    #     heuristic_distance_move=1.0)),
    'regression': evaluation.RegressionChordFrettingEvaluator(),
    # 'lstm': evaluation.LSTMChordFrettingEvaluator(),
}

to_train = ['regression', 'lstm']
for mm in to_train:
    if mm in evaluators and not evaluators[mm].is_trained:
        evaluators[mm].train()

pruning = modelling.PruningConfig(
    candidate_beam_width=0.5, max_candidates=2,
    sequence_beam_width=0.5, max_sequences=3,
)

# evaluate the different models
for eval_name, evaluator in evaluators.items():
    parser = processing.Parser(evaluator)
    solver = processing.Solver(evaluator, pruning)
    solver.solve_multi(input_files, parser, True, 2)
