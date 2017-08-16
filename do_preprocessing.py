#!/usr/bin/env python

from tabgen import preprocessing

preprocessing.extract_features(force_overwrite=False, delete_mscx_afterwards=True)
preprocessing.merge_files()
preprocessing.add_probabilities()
