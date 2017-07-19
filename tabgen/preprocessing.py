from tabgen import processing, evaluation
from .definitions import *
from tabgen.modelling import InvalidFrettingException


def extract_features(force_overwrite: bool=False, delete_mscx_afterwards: bool=False) -> None:
    """
    extracts csv feature files from TAB_PATH to FEATURE_FOLDER_PATH (only if in worklist file WORKLIST_PATH)
    using the settings from tabgen.definitions
    :param force_overwrite: Overwrites existing feature files
    :type force_overwrite: bool
    :param delete_mscx_afterwards: delete .mscx files after parsing
    :type delete_mscx_afterwards: bool
    """
    assert type(force_overwrite) is bool and type(delete_mscx_afterwards) is bool

    # don't need an evaluator for preprocessing - pass dummy
    parser = processing.MuseScoreXMLParser(evaluation.DummyFrettingEvaluator())

    # get list of input files
    input_files = os.listdir(TRAINING_INPUT_TAB_PATH)
    input_files = [input_file for input_file in input_files if input_file + '.mscx' not in input_files]

    # exclude files already done
    if not force_overwrite:
        done_files = [ff.replace('.csv', '') for ff in os.listdir(FEATURE_FOLDER_PATH)]
        for ff in done_files:
            if ff in input_files:
                input_files.remove(ff)

    # enrich with full path
    input_files = [os.path.join(TRAINING_INPUT_TAB_PATH, input_file.strip()) for input_file in input_files]

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

        target_file = os.path.join(FEATURE_FOLDER_PATH, os.path.relpath(input_file, TRAINING_INPUT_TAB_PATH) + '.csv')
        pd.DataFrame(pd.DataFrame(features).astype(np.float)).to_csv(target_file)

        if delete_mscx_afterwards:
            parser.delete_mscx_file()

    print('Features extracted from {} files.'.format(len(input_files)))


def merge_files() -> None:
    """
    merges files from FEATURE_FOLDER_PATH into file TRAINING_DATA_FILE_PATH
    """
    with open(TRAINING_DATA_FILE_PATH, 'w') as target:
        keys = None
        for csv_file in tqdm(os.listdir(FEATURE_FOLDER_PATH), desc='Merging feature files'):
            if csv_file.endswith('.csv'):
                with open(os.path.join(FEATURE_FOLDER_PATH, csv_file), 'r') as csv:

                    # read keys (first line) and check consistency
                    keys_new = csv.readline()
                    if keys is None:
                        keys = keys_new
                        target.write(keys)

                    if not keys == keys_new:
                        warnings.warn('File format not matching: {}'.format(csv_file))
                        warnings.warn('Deleting file.')
                        os.remove(os.path.join(FEATURE_FOLDER_PATH, csv_file))
                        continue

                    # copy value lines to merged target file
                    for line in csv:
                        target.write(line)
                csv.close()
    target.close()
    print('File merged: {}'.format(TRAINING_DATA_FILE_PATH))


def count() -> None:
    """
    Count feature vector occurrences
    """

    # get available features
    with open(TRAINING_DATA_FILE_PATH, 'r') as data_file:
        keys = data_file.readline().strip().split(',')[1:]
    data_file.close()

    count_columns = {}

    # delta models
    if FeatureConfiguration.delta:
        usecols = []
        for depth in tqdm(range(1, FeatureConfiguration.max_depth + 1), desc='Counting occurrences: delta'):
            prefix = 'delta{}'.format(depth)

            # append! --> first, delta1, then [delta1, delta2] etc.
            for col in keys:
                if col.startswith(prefix):
                    usecols.append(col)

            count_columns[prefix] = _count_single(usecols)

    # unigram model
    usecols = [k for k in keys if not (k.startswith('delta') or k.startswith('prev'))]
    count_columns['unigram'] = _count_single(usecols)

    # prev models
    if FeatureConfiguration.prev:
        usecols = usecols[:]  # copy unigram columns
        for depth in tqdm(range(1, FeatureConfiguration.max_depth + 1), desc='Counting occurrences: prev'):
            prefix = 'prev{}'.format(depth)

            # append! --> first, [current, prev1], then [current, prev1, prev2] etc.
            for col in keys:
                if col.startswith(prefix):
                    usecols.append(col)

            count_columns[prefix] = _count_single(usecols)

    # save
    dataframe = pd.read_csv(TRAINING_DATA_FILE_PATH, index_col=0)
    for prefix, counts in count_columns.items():
        dataframe['counts_{}'.format(prefix)] = counts
        dataframe['cost_{}'.format(prefix)] = \
            [evaluation.RegressionChordFrettingEvaluator.count2cost(cc) for cc in counts]
    dataframe.to_csv(TRAINING_DATA_FILE_PATH)


def _count_single(usecols: list) -> list:
    """
    generates a counts column while respecting only the columns is usecols
    :param usecols: list of column names to use
    :type usecols: list
    """
    t_start = time()

    # load data and make lines comparable
    dataframe = pd.read_csv(TRAINING_DATA_FILE_PATH, index_col=0, usecols=usecols)
    stringified = ['-'.join([str(x) for x in row]) for row in dataframe.itertuples()]
    del dataframe  # free space

    t_end = time()
    if TRACK_PERFORMANCE:
        print('read file: {} ms'.format((t_end - t_start) * 1000))

    # count occurrences
    counts = {}
    for stringified_line in stringified:
        if stringified_line in counts.keys():
            counts[stringified_line] += 1
        else:
            counts[stringified_line] = 1

    # build count column
    return [counts[stringified_line] for stringified_line in stringified]
