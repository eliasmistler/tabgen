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


# PATHS
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
TRAINING_DATA_FILE_PATH = os.path.join(DATA_PATH, 'training_data.csv')
FEATURE_FOLDER_PATH = os.path.join(DATA_PATH, 'training_features')
if os.name == 'nt':  # windows config
    MSCORE_PATH = 'C:\\Program Files (x86)\\MuseScore 2\\bin\\MuseScore.exe'
else:
    MSCORE_PATH = os.path.join(ROOT_PATH, 'mscore')
TRAINING_INPUT_TAB_PATH = os.path.join(DATA_PATH, 'training_input')
VALIDATION_INPUT_TAB_PATH = os.path.join(DATA_PATH, 'evaluation_input')
OUTPUT_TAB_PATH = os.path.join(DATA_PATH, 'evaluation_output')

HEURISTIC_PREFILTER = True
HEURISTIC_MAX_FRETS = 4  # max fret range (finger spread) within a chord
HEURISTIC_MAX_FINGERS = 4  # max number of different frets (i.e. fingers, considering barre)

CHORDS_AS_NOTES = False  # scan through chords vertically instead of using

# DEBUG OPTIONS
TRACK_PERFORMANCE = False  # print runtime information to the console
CACHING = False  # activate caching for Chord.get_chord_frettings and Pitch.get_note_frettings


# Core settings for feature generation
class FeatureConfiguration(object):
    max_depth = 3  # depth of features (i.e. go back (max_depth) chords)
    prev = True  # extract prev{#}_{feature}?
    delta = False  # extract delta_{feature}?
    string_details = False  # extract string{#}_played (bool) and string{#}_fret (int 0-24)?
    fret_details = False  # extract fret{}_played (int 0-6)
    detail_matrix = False  # string{#}_fret{#}_played --> full boolean maps
    basic = False  # duration, count, is_chord, ...
    num_strings = 6
    num_frets = 24
    descriptors = True  # extract general descriptors? (mean, min, max, ...)
    pitch = False  # include the pitches at the next time step (for RNN predictions)
    pitch_descriptors = True  # ... as descriptors
    pitch_sparse = False  # ... as sparse encoding
    pitch_sparse_min = 11
    pitch_sparse_max = 88
    heuristics = True
    num_features_total = (
        (string_details * 2 * num_strings) +
        (fret_details * (num_frets + 1)) +
        (detail_matrix * num_strings * (num_frets + 1)) +
        (descriptors * 2 * 8) +
        (basic * 5) +  # count, duration, is_chord, is_rest, is_note
        (pitch * num_strings) +
        (pitch_descriptors * 8) +
        (pitch_sparse * (pitch_sparse_max - pitch_sparse_min + 1)) +
        (heuristics * 6)
    ) * (1 + delta)  # *2 if using delta features

# sanity check: directories
assert os.path.isdir(ROOT_PATH)
assert os.path.isdir(TRAINING_INPUT_TAB_PATH)
assert os.path.isdir(VALIDATION_INPUT_TAB_PATH)

if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)
if not os.path.isdir(FEATURE_FOLDER_PATH):
    os.mkdir(FEATURE_FOLDER_PATH)
if not os.path.isdir(OUTPUT_TAB_PATH):
    os.mkdir(OUTPUT_TAB_PATH)

# sanity check: is MuseScore executable?
assert os.path.isfile(MSCORE_PATH) and os.access(MSCORE_PATH, os.X_OK), \
    'MuseScore not found in path: {}'.format(MSCORE_PATH)

# sanity check: debug settings
assert type(TRACK_PERFORMANCE) is bool \
    and type(CACHING) is bool

# sanity check: FeatureConfiguration
assert type(FeatureConfiguration.max_depth) is int \
    and type(FeatureConfiguration.prev) is bool \
    and type(FeatureConfiguration.delta) is bool \
    and type(FeatureConfiguration.string_details) is bool \
    and type(FeatureConfiguration.descriptors) is bool \
    and type(FeatureConfiguration.num_strings) is int

# TODO: custom cost function as distance on the fretboard?? -- Hamming distance???
# TODO: only string / only fret as prediction target?

# TODO: do empty strings behave strange? i.e. pull notes down (?)
# TODO: local stuff normalised to fret 1
# TODO: add normalised features where -466--- = 466---
# --> do NOT include in prev_ and delta_ (?) as the distances will be messed up!

# TODO: transpose to one tuning / add tuning as feature (e.g. deviation from standard tuning? e.g. -2, 0,... for D