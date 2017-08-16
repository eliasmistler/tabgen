"""
module tabgen.base

Description:  Configuration to run tabgen

Author:       Elias Mistler
Institute:    The University of Edinburgh
"""
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


# PATHS - make sure to adjust Path.MSCORE
# only touch the other paths if you are sure what you're doing
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

    # =================== #
    # SETTINGS START HERE #
    # =================== #

    # pre-filtering
    HEURISTIC_FILTER = True  # filter frettings before evaluating their cost
    HEURISTIC_MAX_FRETS = 4  # max fret range (finger spread) within a chord
    HEURISTIC_MAX_FINGERS = 4  # max number of different frets (i.e. fingers, considering barre)

    # special handling
    CHORDS_AS_NOTES = True  # scan through chords vertically instead of using explicit chord handling
    DELTA_MODE = False  # use first derivation of features instead of plain features

    basic = False  # some basic (useless?) features: duration, count, is_chord, ...

    # Fretting features - select one
    frettings_vectorised = False  # vector representation: string{#}_played (bool) and string{#}_fret (int 0-24)?
    frettings_sparse = False  # sparse matrix: string{#}_fret{#}_played --> full boolean maps
    frettings_desc = True  # describe frettings by descriptors (descriptors_functions)

    frettings_desc_corr_coef = True  # include correlation coefficient between string and fret distribution

    context_length = 1  # go back (context_length) chords when evaluating
    num_strings = 6  # max. number of strings to be considered for sparse representation
    num_frets = 24  # max. number of frets to be considered for sparse representation

    # descriptor functions to be used for frettings / pitches
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
    if CHORDS_AS_NOTES:
        # if only looking at one note, there is no point in all the descriptors
        frettings_desc_corr_coef = False
        descriptors_functions = {
            'mean': np.mean
        }

    pitch = False  # include the pitches at the next time step (for RNN predictions)
    pitch_desc = True  # describe pitch intention by descriptors (descriptors_functions)
    pitch_sparse = False  # sparse pitch representation
    pitch_sparse_min = 11  # minimum pitch to consider
    pitch_sparse_max = 88  # maximum pitch to consider

    heuristics = False  # pre-calculate some baseline heuristics

    # ================= #
    # SETTINGS END HERE
    # ================= #

    # always keep this in sync with any changes to features! has to count the features used!
    num_features_total = (
        (frettings_vectorised * 2 * num_strings) +
        (frettings_sparse * num_strings * (num_frets + 1)) +
        (frettings_desc * (2 * len(descriptors_functions) + frettings_desc_corr_coef)) +
        (basic * 5) +  # count, duration, is_chord, is_rest, is_note
        (pitch * num_strings) +
        (pitch_desc * len(descriptors_functions)) +
        (pitch_sparse * (pitch_sparse_max - pitch_sparse_min + 1)) +
        (heuristics * 6) +
        (CHORDS_AS_NOTES * 1)  # for sequential handling of chords: part_of_previous
    )

    # ============= #
    # sanity checks #
    # ============= #

    # sanity check: FeatureConfig
    assert \
        type(HEURISTIC_FILTER) is bool and \
        type(HEURISTIC_MAX_FRETS) is int and \
        type(HEURISTIC_MAX_FINGERS) is int and \
        \
        type(CHORDS_AS_NOTES) is bool and \
        type(DELTA_MODE) is bool and \
        \
        type(context_length) is int and \
        type(num_strings) is int and \
        type(num_frets) is int and \
        \
        type(basic) is bool and \
        \
        type(frettings_vectorised) is bool and \
        type(frettings_sparse) is bool and \
        type(frettings_desc) is bool and \
        type(frettings_desc_corr_coef) is bool and \
        \
        type(descriptors_functions) is dict and \
        \
        type(pitch) is bool and \
        type(pitch_desc) is bool and \
        type(pitch_sparse) is bool and \
        type(pitch_sparse_min) is int and \
        type(pitch_sparse_max) is int and \
        \
        type(heuristics) is bool and \
        \
        type(num_features_total) is int

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

# sanity check: is MuseScore available and executable?
assert os.path.isfile(Path.MSCORE) and os.access(Path.MSCORE, os.X_OK), \
    'MuseScore not found in path: {}'.format(Path.MSCORE)

# sanity check: debug settings
assert type(TRACK_PERFORMANCE) is bool
