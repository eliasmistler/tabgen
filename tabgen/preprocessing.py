"""
module tabgen.base

Description:  Processing functions to turn input tablatures into training data

Author:       Elias Mistler
Institute:    The University of Edinburgh
"""
from .definitions import *
from tabgen import processing, evaluation
from tabgen.modelling import InvalidFrettingException
from tqdm import tqdm


def extract_features(force_overwrite: bool=False, delete_mscx_afterwards: bool=False) -> None:
    """
    extracts csv feature files from TAB_PATH to Path.FEATURE_FOLDER (only if in worklist file WORKLIST_PATH)
    using the settings from tabgen.definitions
    :param force_overwrite: Overwrites existing feature files
    :type force_overwrite: bool
    :param delete_mscx_afterwards: delete .mscx files after parsing
    :type delete_mscx_afterwards: bool
    """
    assert type(force_overwrite) is bool and type(delete_mscx_afterwards) is bool

    # don't need an evaluator for preprocessing - pass dummy
    parser = processing.Parser(evaluation.DummyFrettingEvaluator())

    # get list of input files
    input_files = os.listdir(Path.TRAINING_INPUT)
    input_files = [input_file for input_file in input_files if input_file + '.mscx' not in input_files]

    # exclude files already done
    if not force_overwrite:
        done_files = [ff.replace('.csv', '') for ff in os.listdir(Path.FEATURE_FOLDER)]
        for ff in done_files:
            if ff in input_files:
                input_files.remove(ff)

    # enrich with full path
    input_files = [os.path.join(Path.TRAINING_INPUT, input_file.strip()) for input_file in input_files]

    # extract features
    for input_file in tqdm(input_files, desc='Extracting tab features', unit='file'):

        try:
            parser.parse(input_file)
        except InvalidFrettingException as ee:
            warnings.warn(ee.message)
            warnings.warn('Deleting file: {}'.format(input_file))
            parser.delete_mscx_file()
            os.remove(input_file.replace('.mscx', ''))
            continue

        input_file = parser.mscx_file

        features = [chord_fretting.features
                    for instrument_id in parser.instrument_ids
                    for chord_fretting in parser.get_chord_fretting_sequence(instrument_id)]

        target_file = os.path.join(Path.FEATURE_FOLDER, os.path.relpath(input_file, Path.TRAINING_INPUT) + '.csv')
        pd.DataFrame(pd.DataFrame(features).astype(np.float)).to_csv(target_file, index=False)

        if delete_mscx_afterwards:
            parser.delete_mscx_file()

    print('Features extracted from {} files.'.format(len(input_files)))


def merge_files() -> None:
    """
    merges files from Path.FEATURE_FOLDER into file Path.FEATURE_FILE
    """
    with open(Path.FEATURE_FILE, 'w') as target:
        keys = None
        for csv_file in tqdm(os.listdir(Path.FEATURE_FOLDER), desc='Merging feature files'):
            if csv_file.endswith('.csv'):
                with open(os.path.join(Path.FEATURE_FOLDER, csv_file), 'r') as csv:

                    # read keys (first line) and check consistency
                    keys_new = csv.readline()
                    if keys is None:
                        keys = keys_new
                        target.write(keys)
                    empty_line = ','.join([str(0.0) for _ in range(keys.count(',') + 1)])+'\n'

                    if not keys == keys_new:
                        warnings.warn('File format not matching: {}'.format(csv_file))
                        warnings.warn('Deleting file.')
                        os.remove(os.path.join(Path.FEATURE_FOLDER, csv_file))
                        continue

                    # copy value lines to merged target file
                    for line in csv:
                        target.write(line)

                    # add empty lines to get context clean
                    for _ in range(FeatureConfig.context_length + 1):
                        target.write(empty_line)

                csv.close()
    target.close()
    print('File merged: {}'.format(Path.FEATURE_FILE))


def add_probabilities(rounding_digits=3) -> None:
    """
    Count feature vector occurrences
    """

    # get data
    print('CALCULATE PROBABILITIES')
    print('Loading data...', end='')
    dataframe = pd.read_csv(Path.FEATURE_FILE)
    dataframe.drop(
        [col for col in dataframe.columns if col.startswith('prob')], axis=1, inplace=True
    )
    print('done')

    data_flat = dataframe.values.astype(np.float32)
    n_features = len(dataframe.columns)

    # mask to exclude lookahead features
    mask = [int(not col.startswith('next')) for col in dataframe.columns]
    print('Targets: {}\nSource: {}'.format(
        [col for m, col in zip(mask, dataframe.columns) if m == 1.0], dataframe.columns)
    )

    # rounding, to ignore small differences
    if rounding_digits > -1:
        data_flat = data_flat.round(rounding_digits)

    # iterate through context lengths, find the respective counts and create a new column in dataframe
    for context_length in range(1, FeatureConfig.context_length + 1):
        print('Calculating probabilities for context length {}'.format(context_length))

        counts = {}

        # get data of the correct context length
        # order is [..., prev2, prev1, current]
        data = data_flat
        if context_length > 0:
            data = np.concatenate([
                data * mask,
                np.concatenate([
                    np.concatenate([np.zeros_like(data[0:i]), data[:-i, :]])
                    for i in range(1, context_length + 1)
                ], axis=1)
            ], axis=1)
        data = data[:, ::-1]

        # get counts from the data
        for row in tqdm(data, desc='Counting occurrences'):
            key = row.tobytes()
            if key in counts:
                counts[key] += 1
            else:
                counts[key] = 1

        # turn into probabilities
        counts_prev = {}
        for key, value in tqdm(counts.items(), desc='Building conditional counts'):
            row = np.fromstring(key, dtype=data.dtype)
            key_prev = row[:n_features].tobytes()
            if key_prev in counts_prev:
                counts_prev[key_prev] += value
            else:
                counts_prev[key_prev] = value

        probs = {}
        for key in tqdm(counts, desc='Calculating probabilities'):
            # P = counts(n-gram) / counts(previous (n-1)-gram
            row = np.fromstring(key, dtype=data.dtype)
            key_prev = row[:n_features].tobytes()
            probs[key] = counts[key] / counts_prev[key_prev]

        print('{} probabilities extracted.'.format(len(probs)))
        np.save(os.path.join(Path.DATA, 'probabilities_{}.npy'.format(context_length)), probs)

        dataframe['probs_{}'.format(context_length)] = [probs[row.tobytes()] for row in data]

    # save updated dataframe
    print('Saving data...', end='')
    dataframe.to_csv(Path.FEATURE_FILE, index=False)
    print('done')
