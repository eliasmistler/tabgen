#!/usr/bin/env python
# from tabgen.definitions import *
from tabgen import evaluation
from tabgen import processing
from tabgen import modelling
# import subprocess

input_files = ['./data/evaluation_input/213511 - Aerosmith - Dream On (guitar pro).gp5.mscx']

# evaluator = evaluation.LSTMChordFrettingEvaluator()
# if not evaluator.is_trained:
#     evaluator.train(10)

evaluator = evaluation.RegressionChordFrettingEvaluator()
if not evaluator.is_trained:
    evaluator.train()

# evaluator = evaluation.RandomChordFrettingEvaluator(5)  # seed
# evaluator = evaluation.ProbabilityLookupEvaluator()

# pruning = modelling.PruningConfig(
#     candidate_beam_width=0.5, max_candidates=3,
#     sequence_beam_width=0.5, max_sequences=5,
# )

pruning = modelling.PruningConfig(
    candidate_beam_width=999.9, max_candidates=1,
    sequence_beam_width=999.9, max_sequences=1,
)

chord = modelling.Chord(1.0, [51, 58, 63, 65, 70], False)
chord_sequence = []
for pitch in chord.pitches:
    chord_sequence.append(modelling.Chord(1.0, [pitch], len(chord_sequence) > 0))

parser = processing.Parser(evaluator)
solver = processing.Solver(evaluator, pruning)

seq = solver.solve(chord_sequence, modelling.StringConfig.STANDARD_24_FRETS)
seq.to_ascii_tab(modelling.StringConfig.STANDARD_24_FRETS, 30, True)

# # loop over multiple files
# solver.solve_multi(input_files, parser, save_files=True, verbose=2)

# out_file = input_files[0].replace(Path.VALIDATION_INPUT, Path.VALIDATION_OUTPUT).replace('.mscx', '_lstm.mscx')
# subprocess.call('{} "{}" "{}"'.format(Path.MSCORE, input_files[0], out_file))
