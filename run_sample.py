#!/usr/bin/env python
# from tabgen.definitions import *
from tabgen import evaluation
from tabgen import processing
from tabgen import modelling
# import subprocess

input_files = ['./data/evaluation_input/215155 - Black Sabbath - Paranoid (guitar pro).gp5.mscx']

# evaluator = evaluation.LSTMChordFrettingEvaluator()
# if not evaluator.is_trained:
#     evaluator.train(10)

evaluator = evaluation.RegressionChordFrettingEvaluator()
if not evaluator.is_trained:
    evaluator.train()

# evaluator = evaluation.DummyFrettingEvaluator()

pruning = modelling.PruningConfig(
    candidate_beam_width=0.5, max_candidates=3,
    sequence_beam_width=0.5, max_sequences=5,
)
# pruning = None
parser = processing.Parser(evaluator)

solver = processing.Solver(evaluator, pruning)

# loop over multiple files
solver.solve_multi(input_files, parser, save_files=True, verbose=2)

# out_file = input_files[0].replace(Path.VALIDATION_INPUT, Path.VALIDATION_OUTPUT).replace('.mscx', '_lstm.mscx')
# subprocess.call('{} "{}" "{}"'.format(Path.MSCORE, input_files[0], out_file))
