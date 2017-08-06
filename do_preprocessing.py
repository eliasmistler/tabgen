#!/usr/bin/env python

from tabgen import preprocessing

preprocessing.extract_features(force_overwrite=True, delete_mscx_afterwards=False)
preprocessing.merge_files()
preprocessing.add_probabilities()
