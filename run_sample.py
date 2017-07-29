from tabgen.definitions import *
from tabgen import evaluation
from tabgen import processing
from tabgen import modelling
# import subprocess

input_files = [
    # 'Bach_Goes_to_Town.mscz',
    '220388 - Green Day - American Idiot (guitar pro).gpx.mscx',
    # 'riff.mscx'
]
input_files = [os.path.join(VALIDATION_INPUT_TAB_PATH, input_file) for input_file in input_files]

# config
# evaluator = evaluation.LSTMChordFrettingEvaluator()
# evaluator = evaluation.DummyFrettingEvaluator()
evaluator = evaluation.DistanceChordFrettingEvaluator(1)  # 2 = euclidean

pruning = modelling.PruningConfiguration(
    candidate_beam_width=0.5, max_candidates=3,
    sequence_beam_width=0.5, max_sequences=5,
)
# pruning = None
parser = processing.MuseScoreXMLParser(evaluator)
solver = processing.FrettingGenerator(evaluator, pruning)

# loop over multiple files
solver.solve_multi(input_files, parser, save_files=True, verbose=2)

# out_file = input_files[0].replace(VALIDATION_INPUT_TAB_PATH, OUTPUT_TAB_PATH).replace('.mscx', '_lstm.mscx')
# subprocess.call('{} "{}" "{}"'.format(MSCORE_PATH, input_files[0], out_file))


# --- some SANITY CHECKS to run ---
# test - run as %run run_sample.py in iPython
# parser.parse(input_files[0])
# seq_orig = parser.get_chord_fretting_sequence(1)
# solver.solve_multi(input_files, parser, save_files=True, verbose=2)
# seq_auto = parser.get_chord_fretting_sequence(1)
# seq_orig == seq_auto
# [o==a for o,a in zip(seq_auto, seq_orig)]
# [o.cost==a.cost for o,a in zip(seq_auto, seq_orig)]
# [o.features==a.features for o,a in zip(seq_auto, seq_orig)]
# [o._next_pitches==a.next_pitches for o,a in zip(seq_auto, seq_orig)]
