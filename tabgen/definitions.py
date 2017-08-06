import os

# noinspection PyUnresolvedReferences
from time import time
# noinspection PyUnresolvedReferences
import numpy as np
# noinspection PyUnresolvedReferences
import pandas as pd
# noinspection PyUnresolvedReferences
from tqdm import tqdm
# noinspection PyUnresolvedReferences
import warnings
# noinspection PyUnresolvedReferences
import typing
# noinspection PyUnresolvedReferences
from subprocess import call

# noinspection PyUnresolvedReferences
from .base import *

# DEBUG OPTIONS
TRACK_PERFORMANCE = False  # print runtime information to the console


# PATHS
class Path(object):
    ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir))
    DATA = os.path.join(ROOT, 'data')
    FEATURE_FILE = os.path.join(DATA, 'training_data.csv')
    FEATURE_FOLDER = os.path.join(DATA, 'training_features')
    if os.name == 'nt':  # windows config
        MSCORE = 'C:\\Program Files (x86)\\MuseScore 2\\bin\\MuseScore.exe'
    else:
        MSCORE = os.path.join(ROOT, 'mscore')
    TRAINING_INPUT = os.path.join(DATA, 'training_input')
    VALIDATION_INPUT = os.path.join(DATA, 'evaluation_input')
    VALIDATION_OUTPUT = os.path.join(DATA, 'evaluation_output')


# Core settings for feature generation
class FeatureConfig(object):
    HEURISTIC_PREFILTER = True
    HEURISTIC_MAX_FRETS = 4  # max fret range (finger spread) within a chord
    HEURISTIC_MAX_FINGERS = 4  # max number of different frets (i.e. fingers, considering barre)

    CHORDS_AS_NOTES = False  # scan through chords vertically instead of using
    DELTA_MODE = True

    max_depth = 1  # depth of features (i.e. go back (max_depth) chords)

    frettings_vectorised = False  # extract string{#}_played (bool) and string{#}_fret (int 0-24)?
    frettings_sparse = False  # string{#}_fret{#}_played --> full boolean maps
    basic = False  # duration, count, is_chord, ...
    num_strings = 6
    num_frets = 24
    frettings_desc = True  # extract general frettings_desc? (mean, min, max, ...)
    frettings_desc_corrcoef = True
    descriptors_functions = {
        'mean': np.mean,
        'std': np.std,  # if len(x) > 0 else 0,
        'min': min,
        '25%': lambda x: np.percentile(x, 25),
        '50%': np.median,
        '75%': lambda x: np.percentile(x, 75),
        'max': max,
        'range': lambda x: max(x) - min(x)
    }
    pitch = False  # include the pitches at the next time step (for RNN predictions)
    pitch_desc = True  # ... as frettings_desc
    pitch_sparse = False  # ... as sparse encoding
    pitch_sparse_min = 11
    pitch_sparse_max = 88
    heuristics = True

    num_features_total = (
        (frettings_vectorised * 2 * num_strings) +
        (frettings_sparse * num_strings * (num_frets + 1)) +
        (frettings_desc * (2 * len(descriptors_functions) + frettings_desc_corrcoef)) +
        (basic * 5) +  # count, duration, is_chord, is_rest, is_note
        (pitch * num_strings) +
        (pitch_desc * len(descriptors_functions)) +
        (pitch_sparse * (pitch_sparse_max - pitch_sparse_min + 1)) +
        (heuristics * 6) +
        (CHORDS_AS_NOTES * 1)  # for sequential handling of chords: part_of_previous
    )

# sanity check: directories
assert os.path.isdir(Path.ROOT)
assert os.path.isdir(Path.TRAINING_INPUT)
assert os.path.isdir(Path.VALIDATION_INPUT)

if not os.path.isdir(Path.DATA):
    os.mkdir(Path.DATA)
if not os.path.isdir(Path.FEATURE_FOLDER):
    os.mkdir(Path.FEATURE_FOLDER)
if not os.path.isdir(Path.VALIDATION_OUTPUT):
    os.mkdir(Path.VALIDATION_OUTPUT)

# sanity check: is MuseScore executable?
assert os.path.isfile(Path.MSCORE) and os.access(Path.MSCORE, os.X_OK), \
    'MuseScore not found in path: {}'.format(Path.MSCORE)

# sanity check: debug settings
assert type(TRACK_PERFORMANCE) is bool

# sanity check: FeatureConfig
assert type(FeatureConfig.max_depth) is int \
    and type(FeatureConfig.frettings_vectorised) is bool \
    and type(FeatureConfig.frettings_desc) is bool \
    and type(FeatureConfig.num_strings) is int
