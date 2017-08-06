"""
module tabgen.processing

Description:  Processing classes needed for file handling and tab generation
              The solver (FrettingGenerator) is generic, the actual scoring functions
                are implemented in tabgen.evaluation

Contains:     FrettingGenerator
              MuseScoreXMLParser

Author:       Elias Mistler
Institute:    The University of Edinburgh
Last changed: 2017-06
"""
import xml.etree.ElementTree as ElementTree

import tabgen.modelling

from .definitions import *


class Parser:
    """
    Parses MuseScoreXML and writes back to it
    Warning: not very robust, stick to files converted from Guitar Pro!
    """

    # static map for XML durations
    _duration_map = dict(long=4.0, breve=2.0, whole=1.0, measure=1.0, half=0.5, quarter=0.25, eighth=0.125)

    def __init__(self, evaluator):
        """
        :param evaluator: evaluation object for fretting scoring
        :type evaluator: ChordFrettingEvaluatorBase
        """
        assert isinstance(evaluator, ChordFrettingEvaluatorBase)

        self._evaluator = evaluator
        self._file_path = None
        self._xml_tree = None
        self._updated = set()
        self._ignored_chords = None

    def __str__(self) -> str:
        return 'MuseScoreXMLParser(\'{}\')'.format(self._evaluator)

    __repr__ = __str__

    def parse(self, file_path: str, force_reload: bool=False) -> None:
        """
        parses an mscx file - or a file convertible to mscx
        :param file_path: path of the file to parse
        :type file_path: str
        :param force_reload: reload even if (apparently) not neccessary?
        :type force_reload: bool
        """
        # sanity check
        assert type(file_path) is str
        assert os.path.isfile(file_path), \
            '{} is not a valid file!'.format(file_path)

        # reload necessary?
        if (file_path == self._file_path or file_path + '.mscx' == self._file_path) \
                and len(self._updated) == 0 and not force_reload:
            print('XML Already imported: {}'.format(self._file_path))

        else:
            # convert first?
            if file_path.endswith('.mscx'):
                self._file_path = file_path
                self._xml_tree = ElementTree.parse(file_path)
            else:
                mscx_file_path = file_path + '.mscx'
                if not os.path.isfile(mscx_file_path):
                    # CONVERT TO XML
                    # by calling MuseScore portable to convert file
                    call([Path.MSCORE, '-o', mscx_file_path, file_path])

                # conversion errors
                assert os.path.isfile(mscx_file_path), \
                    '{} could not be converted to MuseScore XML!'.format(file_path)
                self._file_path = mscx_file_path

            print('Importing XML: {}'.format(self._file_path))

            self._xml_tree = ElementTree.parse(self._file_path)
            self._find_instruments()
            self._extract_chords()
        self._updated = set()  # reset updates

    def _find_instruments(self) -> None:
        """
        find instruments (sub routine of parse)
        sets self._instruments as dict with string_config=... as StringConfig
        """
        xml_score = self._xml_tree.getroot().find('Score')
        xml_parts = xml_score.findall('Part')

        # find instrument configurations (strings)
        self._instruments = {}
        self._ignored_chords = {}
        for xml_part in xml_parts:
            instrument_id = int(xml_part.find('Staff').attrib['id'])

            # instrument name
            name = ""
            xml_name = xml_part.find('trackName')
            if xml_name is not None and type(xml_name.text) is str and xml_name.text.strip() != '':
                name = xml_name.text.strip()

            xml_instrument = xml_part.find('Instrument')
            xml_string_config = xml_instrument.find('StringData')
            if xml_string_config is not None:
                frets = int(xml_string_config.find('frets').text)
                strings = sorted([int(s.text) for s in xml_string_config.findall('string')])

                # exclude [0 0 0] etc. (e.g. drums)
                if min(strings) > 0:
                    self._instruments[instrument_id] = dict(
                        string_config=tabgen.modelling.StringConfig(strings, frets),
                        name=name
                    )
                    self._ignored_chords[instrument_id] = 0

    def remove_instrument(self, instrument_id: int) -> None:
        if instrument_id in self._updated:
            self._updated.remove(self._updated)
        if instrument_id in self._ignored_chords:
            del self._ignored_chords[instrument_id]
        if instrument_id in self._instruments:
            del self._instruments[instrument_id]

    def delete_mscx_file(self) -> None:
        assert type(self._file_path) is str and self._file_path.endswith('.mscx')
        if self._file_path is not None:
            if os.path.isfile(self._file_path):
                os.remove(str(self._file_path))

    @property
    def mscx_file(self) -> str:
        return self._file_path

    @property
    def instrument_ids(self) -> list:
        return list(self._instruments.keys())

    def get_string_config(self, instrument_id: int) -> StringConfigBase:
        return self._instruments[instrument_id]['string_config']

    def get_chord_fretting_sequence(self, instrument_id: int) -> tabgen.modelling.ChordFrettingSequence:
        return self._instruments[instrument_id]['chord_fretting_sequence']

    def get_chords(self, instrument_id: int) -> list:
        return self._instruments[instrument_id]['chords']

    def get_instrument_name(self, instrument_id: int) -> str:
        return self._instruments[instrument_id]['name']

    def _extract_chords(self) -> None:
        """
        extract chords and frettings from XML (sub routine of parse)
        """
        score = self._xml_tree.getroot().find('Score')
        staff = score.findall('Staff')

        # tuplets
        self._tuplets = {}

        # extract chords and chord frettings
        for st in tqdm(staff, desc='Reading XML data', unit='instrument', disable=True):
            instrument_id = int(st.attrib['id'])

            if instrument_id in self.instrument_ids:
                chords = []
                chord_fretting_sequence = tabgen.modelling.ChordFrettingSequence()

                # number of strings for string inversion (1-6 vs. 6-1)
                num_strings = self.get_string_config(instrument_id).num_strings

                for xml_chord in st.findall('Measure/*'):

                    # ignore appoggiatura:
                    if xml_chord.find('appoggiatura') is not None \
                            or xml_chord.find('acciaccatura') is not None:
                        self._ignored_chords[instrument_id] += 1
                        continue

                    # identify tuplets = duration multipliers
                    if xml_chord.tag == 'Tuplet':
                        self._tuplets[xml_chord.attrib['id']] = \
                            float(xml_chord.find('normalNotes').text) / \
                            float(xml_chord.find('actualNotes').text)

                    # evaluate chords / rests
                    if xml_chord.tag == 'Chord':  # or xml_chord.tag == 'Rest':

                        # duration...
                        duration = xml_chord.find('durationType').text
                        duration = self.duration2float(duration)

                        # ...with augmentation dots
                        dots = xml_chord.find('dots')
                        if dots is not None:
                            duration *= pow(1.5, int(dots.text))

                        # ... and tuplets
                        tuplet = xml_chord.find('Tuplet')
                        if tuplet is not None:
                            # fix for corrupted tuplets: use info from previous tuplet
                            if len(self._tuplets) > 0:
                                if tuplet.text not in self._tuplets.keys():
                                    tuplet.text = max(self._tuplets.keys())
                                duration *= self._tuplets[tuplet.text]

                        xml_notes = xml_chord.findall('Note')

                        # GET NOTES AND FRETTINGS FROM XML
                        pitches = []
                        note_frettings = []
                        for note in xml_notes:
                            pitch = int(note.find('pitch').text)
                            string = num_strings - int(note.find('string').text)
                            fret = int(note.find('fret').text)
                            fully_muted = note.find('ghost') is not None and note.find('ghost').text == '1'

                            pitches.append(tabgen.modelling.Pitch(pitch, fully_muted))
                            note_frettings.append(tabgen.modelling.NoteFretting(string, fret, fully_muted))

                        # CONCATENATE INTO SEQUENCE
                        if FeatureConfig.CHORDS_AS_NOTES:
                            # handle chord as sequences of notes, modelled as 1-note-chords to keep logic in place
                            # chord as sequence of pitches
                            for ii, pitch in enumerate(pitches):
                                chords.append(
                                    tabgen.modelling.Chord(duration, [pitch], ii != 0)
                                )
                            # chord fretting as sequence of note frettings
                            for ii, note_fretting in enumerate(note_frettings):
                                chord_fretting_sequence.append(
                                    tabgen.modelling.ChordFretting(
                                        duration, [note_fretting], self._evaluator, None,
                                        [], self.get_string_config(instrument_id), ii != 0
                                    )
                                )
                        else:
                            # handle chords explicitly as chords
                            # chords
                            chords.append(
                                tabgen.modelling.Chord(duration, pitches, False)
                            )
                            # chord frettings
                            chord_fretting_sequence.append(
                                tabgen.modelling.ChordFretting(
                                    duration, note_frettings, self._evaluator, None,
                                    [], self.get_string_config(instrument_id), False)
                            )

                for idx in range(len(chord_fretting_sequence) - 1):
                    chord_fretting_sequence[idx].next_pitches = chords[idx + 1].pitches

                self._instruments[instrument_id]['chord_fretting_sequence'] = chord_fretting_sequence
                self._instruments[instrument_id]['chords'] = chords

    def update_chord_fretting_sequence(self, instrument_id: int,
                                       chord_fretting_sequence: tabgen.modelling.ChordFrettingSequence) -> None:
        """
        Replace the frettings for one instrument
        :param instrument_id: ID of the instrument (should be int (?))
        :type instrument_id: int
        :param chord_fretting_sequence: the new fretting sequence
        :type chord_fretting_sequence: ChordFrettingSequence
        """
        assert instrument_id in self.instrument_ids, \
            'Instrument {} does not exist: {}'.format(instrument_id, self.instrument_ids)
        assert isinstance(chord_fretting_sequence, tabgen.modelling.ChordFrettingSequence)
        assert len(chord_fretting_sequence) == len(self.get_chord_fretting_sequence(instrument_id)), \
            'Sequence has wrong length {}, expected: {}'.format(
                len(chord_fretting_sequence), len(self.get_chord_fretting_sequence(instrument_id)))

        self._instruments[instrument_id]['chord_fretting_sequence'] = chord_fretting_sequence
        self._updated.add(instrument_id)

    def save(self, target_path: str) -> None:
        """
        write chords back to XML
        :param target_path: path to output file (*.mscx)
        :type target_path: str
        """
        # sanity check
        assert self._instruments is not None, 'Can only save based on original: parse() a file first!'
        assert type(target_path) is str \
            and target_path.endswith('.mscx'),\
            'target_file must be a str path, ending with .mscx: {}'.format(target_path)

        # touch file to make sure it exists
        with open(target_path, 'a'):
            os.utime(target_path, None)

        """
        XML structure (for Guitar Pro converted scores):
        root (<museScore>)
        >Score
        >>Part -- Instrument Information: <Staff id=...> (link), Instrument, Instrument/StringData
        >>Staff id=... -- Actual Notes: Staff/Measure/*

        >>Score -- contains the detailed score (incl. tab) for a single instrument
        >>>Part -- Info, same as before. different "local" Staff IDs - but Part/Staff/linkedTo/[CDATA] is original ID
        >>>Staff -- Actual Notes again -- with the "local" Staff ID
        """

        # get main staff
        xml_main_score = self._xml_tree.getroot().find('Score')
        xml_main_staff_all = xml_main_score.findall('Staff')
        xml_staff_per_instrument = dict([
            (int(xml_main_staff.attrib['id']), [xml_main_staff]) for xml_main_staff in xml_main_staff_all
        ])

        # get sub staff
        xml_sub_score_all = xml_main_score.findall('Score')
        for xml_sub_score in xml_sub_score_all:
            instrument_id = int(xml_sub_score.find('Part/Staff/linkedTo').text)

            for xml_sub_staff in xml_sub_score.findall('Staff'):
                xml_staff_per_instrument[instrument_id].append(xml_sub_staff)

        """
        Special handling for single page tabs (not multiple instruments)
        --> update all instruments (should be just 1 note line and 1 tab line)
        """
        if len(xml_sub_score_all) == 0:
            xml_staff_per_instrument[1] = xml_main_staff_all

        # only change actually changed instruments
        for instrument_id in tqdm(self._updated, desc='updating', unit='instrument'):

            # now change both the main and the sub score
            for xml_staff_single in xml_staff_per_instrument[instrument_id]:

                chord_frettings = self.get_chord_fretting_sequence(instrument_id)
                xml_chord_nodes = xml_staff_single.findall('Measure/Chord')

                # make sure we are looking at the same amount of chords (ignore rests!)
                if FeatureConfig.CHORDS_AS_NOTES:
                    expected_len = len(xml_staff_single.findall('Measure/Chord/Note'))
                else:
                    expected_len = len(xml_chord_nodes)
                actual_len = len([cf for cf in chord_frettings if len(cf) > 0])
                assert actual_len == expected_len - self._ignored_chords[instrument_id], \
                    'XML: Chords={} vs. {} new values. File manipulated?: {}, Instrument {} - {}'.format(
                        expected_len, len(chord_frettings), self._file_path,
                        instrument_id, self.get_instrument_name(instrument_id))

                string_config = self.get_string_config(instrument_id)

                idx = 0
                # update all chords...
                for xml_chord in xml_chord_nodes:

                    # ignore appoggiatura and acciaccatura:
                    if xml_chord.find('appoggiatura') is not None \
                            or xml_chord.find('acciaccatura') is not None:
                        continue

                    # skip all rests in chord_frettings
                    chord_fretting = chord_frettings[idx]
                    while len(chord_fretting.note_frettings) == 0:
                        idx += 1
                        chord_fretting = chord_frettings[idx]

                    xml_notes = xml_chord.findall('Note')

                    # update all notes in the chord
                    for subidx, xml_note in enumerate(xml_notes):
                        pitch = int(xml_note.find('pitch').text)

                        if FeatureConfig.CHORDS_AS_NOTES:
                            note_fretting = chord_frettings[idx + subidx].note_frettings[0]
                        else:
                            note_fretting = chord_fretting.note_frettings[subidx]

                        # note_fretting = [nf for nf in chord_fretting.note_frettings
                        #                  if nf.get_pitch(string_config) == pitch][0]
                        # make sure we're at the right position and the chord has not changed
                        if not note_fretting.get_pitch(string_config) == pitch:
                            assert (note_fretting.get_pitch(string_config) == pitch)

                        xml_note.find('string').text = str(string_config.num_strings - note_fretting.string)
                        xml_note.find('fret').text = str(note_fretting.fret)

                    # next step: advance one chord (or, all notes of that chord)
                    if FeatureConfig.CHORDS_AS_NOTES:
                        idx += len(xml_notes)
                    else:
                        idx += 1

        # now save
        self._xml_tree.write(target_path)

    @staticmethod
    def duration2float(duration_text: str) -> float:
        """
        conversion from MuseScoreXML time note durations to float
        :param duration_text: XML duration name
        :type duration_text: str
        :return: float duration
        :rtype: float
        """
        if duration_text in Parser._duration_map.keys():
            return Parser._duration_map[duration_text]
        return 1 / float(duration_text.replace('th', '').replace('nd', ''))


class Solver:
    """
    Solver for finding the best chord fretting sequence from a chord sequence
    based on a scoring function implemented  by a subclass of tabgen.modelling.ChordFrettingEvaluatorBase
    """

    __solve_count__ = 0
    __solve_avg_time__ = 0

    def __init__(self, evaluator: ChordFrettingEvaluatorBase, pruning_config: PruningConfig=None):
        """
        :param evaluator: evaluation object for fretting scoring
        :type evaluator: ChordFrettingEvaluatorBase
        :param pruning_config: pruning information
        :type pruning_config: PruningConfig
        """
        # sanity check
        assert isinstance(evaluator, ChordFrettingEvaluatorBase)
        assert isinstance(pruning_config, PruningConfig) or pruning_config is None

        self._evaluator = evaluator
        self._pruning_config = pruning_config

    def __str__(self) -> str:
        return 'FrettingGenerator({}, {})'.format(
            self._evaluator, self._pruning_config)
    __repr__ = __str__

    def solve_multi(self, input_files: list, parser: Parser, save_files=False, verbose=2) -> dict:
        """
        Solve multiple files (proxy for solve)
        :param input_files: absolute file paths to create new tabs for
        :type input_files: list of str
        :param parser: Parser to use to read the files
        :type parser: ParserBase
        :param save_files: write output to new files?
        :type save_files: bool
        :param verbose: how much to print on stdout (0: nothing, 1: some, 2: more, 3: all)
        :type verbose: int
        :return: solved chord fretting sequences as dictionary {'filename.mscx': tabgen.modelling.ChordFrettingSequence}
        :rtype: dict of str, tabgen.modelling.ChordFrettingSequence
        """
        assert type(input_files) is list

        if verbose >= 1:
            print('\n\n', 100 * "=", '\n', self._evaluator.name, self._evaluator, '\n', 100 * "=")

        accuracies = []
        sequences = {}

        # loop over multiple files
        for file_no, input_file in enumerate(input_files):
            sequences[input_file] = {}

            if verbose >= 1:
                print('File {}/{}'.format(file_no + 1, len(input_files)))
            parser.parse(input_file)

            # per file, evaluate every single instrument
            for instrument_id in parser.instrument_ids:
                sequence_original = parser.get_chord_fretting_sequence(instrument_id)
                string_config = parser.get_string_config(instrument_id)

                # print original tab
                if verbose >= 2:
                    print('=== INSTRUMENT:', parser.get_instrument_name(instrument_id), '===')
                    print('ORIGINAL TAB:')
                    sequence_original.to_ascii_tab(string_config, do_print=True, n_first=30)
                    print('Cost: {}'.format(sequence_original.cost / len(sequence_original)))

                # solve
                try:
                    sequence_generated = self.solve(parser.get_chords(instrument_id), string_config, verbose)
                except tabgen.modelling.NoValidFrettingException as ee:
                    warnings.warn(ee.message)
                    warnings.warn('Skipping instrument {} - "{}" for {}'.format(
                        instrument_id, parser.get_instrument_name(instrument_id), input_file
                    ))
                    parser.remove_instrument(instrument_id)
                    break  # skip this instrument

                # get accuracy score
                n_total = len(sequence_original)
                n_same = sum([sequence_original[ii] == sequence_generated[ii] for ii in range(n_total)])
                accuracy = float(n_same) / n_total * 100
                accuracies.append(accuracy)

                # print generated tab
                if verbose >= 2:
                    print('\nGENERATED TAB:')
                    sequence_generated.to_ascii_tab(string_config, do_print=True, n_first=30)
                    print('Cost: {}'.format(sequence_generated.cost / len(sequence_generated)))
                    print('Accuracy: {}'.format(accuracy))

                # write back to parser for writing new file
                parser.update_chord_fretting_sequence(instrument_id, sequence_generated)
                sequences[input_file][instrument_id] = sequence_generated

            if save_files:
                target_file = parser.mscx_file.replace('.mscx', '_{}.mscx'.format(self._evaluator.name))
                target_file = os.path.join(Path.VALIDATION_OUTPUT, os.path.relpath(target_file, Path.VALIDATION_INPUT))
                parser.save(target_file)

        if verbose >= 1:
            print(self._evaluator.name,
                  '\n\taccuracies:', accuracies,
                  '\n\tmean:', np.mean(accuracies))

        return sequences

    def solve(self, chord_sequence: list, string_config: StringConfigBase, verbose: int=2) \
            -> tabgen.modelling.ChordFrettingSequence:
        """
        finds the best ChordFrettingSequence for a given chord_sequence
        :param chord_sequence: the chords to be transferred to the instrument
        :type chord_sequence: list of Chord
        :param string_config: configuration of the target instrument
        :type string_config: StringConfig
        :param verbose: how much to print on stdout (0: nothing, 1: some, 2: all)
        :type verbose: int
        :return: ChordFrettingSequence
        """
        # sanity check
        assert type(chord_sequence) is list
        for chord in chord_sequence:
            assert isinstance(chord, tabgen.modelling.Chord)
        assert isinstance(string_config, tabgen.modelling.StringConfig), \
            '{} is not of type StringConfig!'.format(string_config)

        t_start = time()

        # initial chord candidates
        next_chord = tabgen.modelling.Chord(1.0, [], False)
        if len(chord_sequence) > 1:
            next_chord = chord_sequence[1].pitches

        candidate_sequences = [tabgen.modelling.ChordFrettingSequence([x]) for x in
                               chord_sequence[0].get_chord_frettings(
                                   string_config, self._evaluator, self._pruning_config, next_chord, None)]

        # subsequent chords
        for idx, chord in tqdm(enumerate(chord_sequence[1:]), desc='solving', unit='chord', initial=1,
                               total=len(chord_sequence), disable=verbose == 0):
            candidate_sequences_new = []

            # # get the next chord
            # next_chord = tabgen.modelling.Chord(1.0, [])
            # if len(chord_sequence) > idx + 2:
            #     next_chord = chord_sequence[idx + 2]

            # branch the  candidate sequences with the chord
            for candidate_sequence in candidate_sequences:
                candidate_sequences_new += candidate_sequence.branch(
                    chord, string_config, self._evaluator, self._pruning_config
                )

            # copy and prune
            candidate_sequences = self._prune(candidate_sequences_new)
            del candidate_sequences_new

        # find the best sequence(s)
        min_cost = min([seq.cost for seq in candidate_sequences])
        best_sequences = [seq for seq in candidate_sequences if seq.cost == min_cost]

        if len(best_sequences) > 1:
            warnings.warn('More than one "best" sequence!')
        best_sequence = best_sequences[0]

        t_end = time()
        if TRACK_PERFORMANCE:
            t = (t_end - t_start) * 1000
            self.__solve_count__ += 1
            if self.__solve_avg_time__ == 0:
                self.__solve_avg_time__ = t
            else:
                # moving average
                self.__solve_avg_time__ = \
                    (1.0 - 1.0 / self.__solve_count__) * self.__solve_avg_time__\
                    + (1.0 / self.__solve_count__) * t

        if verbose >= 3:
            for cf in best_sequence.raw():
                print('\t', cf)

        # tie break: just take the first sequence
        return best_sequence

    def __del__(self) -> None:
        if TRACK_PERFORMANCE:
            print('Calls of solve(): {}'.format(self.__solve_count__))
            print('Average runtime: {} ms'.format(self.__solve_avg_time__))
            print('Total time spent in solve(): {} s'.format(self.__solve_avg_time__ * self.__solve_count__ / 1000.0))

    def _prune(self, candidate_sequences: list) -> list:
        """
        pruning, if configured
        """

        if self._pruning_config is not None and len(candidate_sequences) > 1:

            # beam pruning
            costs = [seq.cost for seq in candidate_sequences]
            cost_limit = min(costs) + np.std(costs) + self._pruning_config.sequence_beam_width * np.std(costs)
            candidate_sequences = [seq for seq in candidate_sequences if seq.cost <= cost_limit]

            # max sequence pruning
            candidate_sequences = sorted(candidate_sequences, key=lambda cs: cs.cost)
            if len(candidate_sequences) > self._pruning_config.max_sequences > 0:
                candidate_sequences = candidate_sequences[:self._pruning_config.max_sequences]

        return candidate_sequences
