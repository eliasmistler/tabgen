"""
module tabgen.modelling

Description:  Objects modelling the structure of the transcription problem
              Generally, there are two focus views:
                1. pitch view: The musical content (Pitches and Chords)
                2. tab view:   A way of playing on a stringed instrument (Frettings)
              The two views are related through the instrument of interest,
                which is represented by the StringConfigBase object

Author:       Elias Mistler
Institute:    The University of Edinburgh
"""
from .definitions import *


class NoValidFrettingException(Exception):
    def __init__(self, chord_or_note: object, string_config: StringConfigBase):
        self.message = 'No fretting for {} on strings {} with {} frets!'.format(
            chord_or_note, string_config.string_pitches, string_config.num_frets)


class InvalidFrettingException(Exception):
    def __init__(self, reason: str, args):
        self.message = 'Invalid Fretting: {} - {}'.format(reason, args)


class Pitch:
    """
    Wrapper for integer Pitches (MIDI-Pitch)
    allows to retrieve possible frettings
    """

    def __init__(self, pitch: int, fully_muted: bool=False):
        """
        :param pitch: a MIDI pitch, int>0
        :type pitch: int
        """
        assert type(pitch) is int and pitch > 0, 'pitch has to be int>0: {}'.format(pitch)
        assert type(fully_muted) is bool, 'fully_muted has to be bool: {}'.format(fully_muted)
        self._pitch = pitch
        self._fully_muted = fully_muted

    def __str__(self) -> str:
        if self._fully_muted:
            return 'Pitch({}, fully_muted=True)'.format(self._pitch)
        else:
            return 'Pitch({})'.format(self._pitch)
    __repr__ = __str__

    def get_note_frettings(self, string_config: StringConfigBase) -> list:
        """
        gets all possible frettings for a single pitch
        :param string_config: string Configuration to be considered
        :type string_config: StringConfigBase
        :return: possible note frettings for the pitch
        :rtype: list of NoteFretting
        """
        # sanity check
        assert isinstance(string_config, StringConfigBase), \
            '{} is not of type StringConfigBase!'.format(string_config)

        # calculate frettings
        frettings = []
        for string, stringPitch in enumerate(string_config.string_pitches):
            if self._pitch >= stringPitch.pitch:
                fret = self._pitch - stringPitch.pitch

                # only possible frets
                if fret <= string_config.num_frets:

                    # only play fully muted notes on empty strings. EDIT: assumption does not hold (data quality?)
                    # if (self._fully_muted and fret == 0) or not self._fully_muted:
                    frettings.append(NoteFretting(string + 1, fret, fully_muted=self._fully_muted))

        # # special case handling for strange tabbing of muted noises
        # if self._fully_muted and len(frettings) == 0:
        #     for string, stringPitch in enumerate(string_config.string_pitches):
        #         if self._pitch >= stringPitch.pitch \
        #                 and ((string_config.num_strings > string+1
        #                       and not self._pitch >= string_config.string_pitches[string+1].pitch)
        #                      or (string_config.num_strings == string+1)):  # last string
        #             fret = self._pitch - stringPitch.pitch
        #             frettings.append(NoteFretting(string+1, fret, fully_muted=self._fully_muted))

        if len(frettings) == 0:
            raise NoValidFrettingException(self, string_config)

        return frettings

    @property
    def pitch(self) -> int:
        return self._pitch

    def get_note_name(self, latex: bool=False) -> str:
        pitch_sequence = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        name = str(pitch_sequence[self._pitch % 12])

        if latex:
            name = name.replace('#', '\sharp')
            octave = int(self._pitch / 12)
            return '${}_{}$'.format(name, octave)

        if self._fully_muted:
            name = '({})'.format(name)
        return name

    def __eq__(self, other) -> bool:
        assert type(other) is int or isinstance(other, Pitch)
        if type(other) is int:
            pitch = other
        else:
            pitch = other.pitch
        return pitch == self.pitch

    def __ne__(self, other) -> bool:
        return not self == other

    def __lt__(self, other) -> bool:
        assert type(other) is int or isinstance(other, Pitch)
        if type(other) is int:
            pitch = other
        else:
            pitch = other.pitch
        return self.pitch < pitch

    def __gt__(self, other) -> bool:
        assert type(other) is int or isinstance(other, Pitch)
        if type(other) is int:
            pitch = other
        else:
            pitch = other.pitch
        return self.pitch > pitch

    def __hash__(self) -> int:
        if self._fully_muted:
            return hash(-self._pitch)
        else:
            return hash(self._pitch)

    def __int__(self) -> int:
        return self._pitch


class StringConfig(StringConfigBase):
    """
    A configuration object describing an instrument (strings and frets)
    """
    STANDARD_24_FRETS = None
    DROP_D_24_FRETS = None

    def __init__(self, string_pitches: list, num_frets: int):
        """
            :param string_pitches: individual pitches of the strings, list of int; e.g. EADGBE: [40,45,50,55,59,64]
            :type string_pitches: list
            :param num_frets: Number of frets
            :type num_frets: int
            """
        super().__init__()
        assert type(string_pitches) is list
        assert len(string_pitches) > 0
        assert string_pitches == sorted(string_pitches)
        self._pitches = []
        for pitch in string_pitches:
            if type(pitch) is int:
                pitch = Pitch(pitch)
            assert isinstance(pitch, Pitch), 'pitch needs to be int or Pitch: {}'.format(pitch)
            self._pitches.append(pitch)
        assert type(num_frets) is int and num_frets > 0
        self._frets = num_frets

    def __str__(self) -> str:
        return 'StringConfigBase({}, {})'.format(self._pitches, self._frets)
    __repr__ = __str__

    @property
    def string_pitches(self) -> list:
        return self._pitches

    @property
    def num_frets(self) -> int:
        return self._frets

    @property
    def num_strings(self) -> int:
        return len(self._pitches)

    @property
    def min_pitch(self) -> int:
        return min(self._pitches).pitch

    @property
    def max_pitch(self) -> int:
        return max(self._pitches).pitch + self.num_frets

    def __eq__(self, other) -> bool:
        if not isinstance(other, StringConfig):
            return False
        return (self._pitches == other.string_pitches) \
            and (self._frets == other.num_frets)

    def __ne__(self, other) -> bool:
        return not self == other

StringConfig.STANDARD_24_FRETS = StringConfig([40, 45, 50, 55, 59, 64], 24)
StringConfig.DROP_D_24_FRETS = StringConfig([38, 45, 50, 55, 59, 64], 24)


class NoteFretting:
    """
    Fretting of a single note
    """
    def __init__(self, string: int, fret: int, fully_muted: bool=False):
        """
        :param string: number of string (usually 1-6)
        :type string: int
        :param fret: number of fret (usually 0-24)
        :type fret: int
        :param fully_muted:
        """
        assert type(string) is int and string > 0, 'string has to be int>0: {}'.format(string)
        assert type(fret) is int and fret >= 0, 'fret has to be int>=0: {}'.format(fret)
        assert type(fully_muted) is bool, 'fully_muted has to be bool: {}'.format(fully_muted)

        self._string = string
        self._fret = fret
        self._fully_muted = fully_muted

    def __str__(self) -> str:
        if self.fully_muted:
            return 'NoteFretting({}, {}, fully_muted={})'.format(self.string, self.fret, self.fully_muted)
        else:
            return 'NoteFretting({}, {})'.format(self.string, self.fret)
    __repr__ = __str__

    def __eq__(self, other) -> bool:
        if not isinstance(other, NoteFretting):
            return False
        return self.fret == other.fret and self.string == other.string and self.fully_muted == other.fully_muted

    def __ne__(self, other) -> bool:
        return not self == other

    def to_dict(self) -> dict:
        return {'string': self.string, 'fret': self.fret}

    def to_tuple(self) -> tuple:
        return self.string, self.fret

    @property
    def string(self) -> int:
        return self._string

    @property
    def fret(self) -> int:
        return self._fret

    @property
    def fully_muted(self) -> bool:
        return self._fully_muted

    def to_ascii_tab(self) -> str:
        if self.fully_muted:
            return 'x'
        else:
            return str(self.fret)

    def get_pitch(self, string_config: StringConfigBase) -> Pitch:
        """
        get Pitch by applying fretting to string_config
        :param string_config: the string configuration to use
        :type string_config: StringConfigBase
        :return: pitch
        :rtype: Pitch
        """
        # sanity check
        assert isinstance(string_config, StringConfigBase), \
            '{} is not of type StringConfigBase!'.format(string_config)
        assert len(string_config.string_pitches) >= self.string, \
            'String {} not in StringConfigBase {}!'.format(self.string, string_config)
        assert string_config.num_frets >= self.fret, \
            'Fret {} not reachable in StringConfigBase {}!'.format(self.fret, string_config)
        return Pitch(string_config.string_pitches[self.string - 1].pitch + self.fret, fully_muted=self.fully_muted)


class Chord:
    """
    A chord, i.e. an event in time, representing actual chord, single note or a rest,
    depending on the number of pitches (0, 1, 2+)
    """

    __fret_avg_time__ = 0
    __fret_count__ = 0

    def __init__(self, duration: float, pitches: list, part_of_previous: bool):
        """
        :param pitches: which pitches to play, list of int or Pitch
        :type pitches: list
        :param duration: note/chord/rest duration
        :type duration: float
        """
        # sanity check
        assert type(pitches) is list and (len(pitches) == 0 or min(pitches) > 0)
        for idx, pitch in enumerate(pitches):
            if type(pitch) is int:
                pitches[idx] = Pitch(pitch)
            assert isinstance(pitches[idx], Pitch)

        assert type(duration) is float and duration > 0
        assert type(part_of_previous) is bool

        self._pitches = sorted(pitches)
        self._duration = duration
        self._part_of_previous = part_of_previous

    def __str__(self) -> str:
        return 'Chord({}, {}, {})'.format(self._duration, self._pitches, self._part_of_previous)
    __repr__ = __str__

    def __len__(self) -> int:
        return len(self._pitches)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Chord):
            return False
        return self._pitches == other.pitches \
            and self._duration == other.duration

    def __ne__(self, other) -> bool:
        return not self == other

    @property
    def pitches(self):
        return self._pitches

    @property
    def duration(self):
        return self._duration

    @property
    def part_of_previous(self):
        return self._part_of_previous

    def get_chord_frettings(self, string_config: StringConfigBase,
                            evaluator: ChordFrettingEvaluatorBase,
                            pruning_config: PruningConfig=None,
                            next_pitches: typing.Optional[list]=None,
                            prev: 'ChordFretting'=None) -> list:
        """
        finds and evaluates fretting options
        only keeps frettings with cost <= min+beam_width*std
        :param string_config: string configuration object
        :type string_config: StringConfigBase
        :param pruning_config: pruning configuration
        :type pruning_config: PruningConfig
        :param evaluator: Evalutator for pruning purposes
        :type evaluator: ChordFrettingEvaluatorBase
        :param next_pitches: Next chord's pitches (for lookahead-prediction)
        :type next_pitches: list of Pitch
        :param prev: previous chord fretting to be passed on to ChordFretting constructor
        :type prev: ChordFretting
        :return: possible frettings of the chord
        :rtype: list of ChordFretting
        """
        # sanity check
        assert isinstance(string_config, StringConfigBase), \
            '{} is not an instance of StringConfigBase!'.format(string_config)
        assert isinstance(pruning_config, PruningConfig) or pruning_config is None,\
            '{} is not an instance of PruningConfig!'.format(pruning_config)
        assert isinstance(evaluator, ChordFrettingEvaluatorBase), \
            '{} is not an instance of ChordFrettingEvaluatorBase!'.format(evaluator)
        assert prev is None or isinstance(prev, ChordFretting)
        if next_pitches is None:
            next_pitches = []
        else:
            assert type(next_pitches) is list, 'Not a list: {}'.format(next_pitches)
            for pitch in next_pitches:
                assert type(pitch) is Pitch

        # no pitch -- empty fretting
        if len(self._pitches) == 0:
            return [ChordFretting(self._duration, [], evaluator, prev, next_pitches, string_config, False)]

        t_start = time()

        # too many pitches to play?
        assert len(string_config.string_pitches) >= len(self._pitches), \
            'Chord {} impossible to fret on strings {} with {} frets: too many pitches!'.format(
                self._pitches, string_config.string_pitches, string_config.num_frets)

        # first, get all possible frettings for the single notes
        note_frettings_dict = dict([(pitch, pitch.get_note_frettings(string_config)) for pitch in self._pitches])

        # generate all possible chord frettings
        # ... starting from all possibilities of the first pitch
        chord_frettings = [[ff] for ff in note_frettings_dict[self._pitches[0]]]

        # handle chords in sequential scenario
        if FeatureConfig.CHORDS_AS_NOTES and self._part_of_previous:
            strings_used = []
            prev_x = prev
            part = self._part_of_previous

            # remove already used strings from possible frettings
            while part:
                strings_used.append(prev_x.note_frettings[0].string)
                part = prev_x.is_part_of_previous
                prev_x = prev_x.previous

            chord_frettings = [cf for cf in chord_frettings if cf[0].string not in strings_used]

        if len(chord_frettings) == 0:
            raise NoValidFrettingException(self, string_config)

        # ... then adding all valid possibilities for the next pitches
        for pitch in self._pitches[1:]:

            # concatenate every possible next note fretting
            # to every fretting found for previous notes
            chord_frettings_new = []
            for note_fretting in note_frettings_dict[pitch]:
                for chord_fretting in chord_frettings:
                    # check if string already used
                    if note_fretting.string not in [nf.string for nf in chord_fretting]:
                        chord_frettings_new.append(chord_fretting + [note_fretting])
            chord_frettings = chord_frettings_new

        if len(chord_frettings) == 0:
            raise NoValidFrettingException(self, string_config)

        # heuristic filtering
        if FeatureConfig.HEURISTIC_PREFILTER:
            chord_frettings_filtered = chord_frettings

            if FeatureConfig.CHORDS_AS_NOTES:
                # sequence based pre-filtering (this is a simplified version):
                if self._part_of_previous:
                    chord_frettings_filtered = [
                        cf for cf in chord_frettings
                        if abs(cf[0].fret - prev.note_frettings[0].fret) <= FeatureConfig.HEURISTIC_MAX_FRETS
                    ]

            else:
                chord_frettings_filtered = [
                    cf for cf in chord_frettings
                    if max([nf.fret for nf in cf]) - min([nf.fret for nf in cf]) <= FeatureConfig.HEURISTIC_MAX_FRETS
                    and len(set([nf.fret for nf in cf])) <= FeatureConfig.HEURISTIC_MAX_FINGERS
                ]

            if len(chord_frettings_filtered) > 0:
                chord_frettings = chord_frettings_filtered
            else:
                warnings.warn('Pre-filtering removed all frettings! Defaulting to all frettings.')

        # wrap in class
        chord_frettings = [
            ChordFretting(self._duration, sorted(ff, key=lambda x: x.string),
                          evaluator, prev, next_pitches, string_config, self._part_of_previous)
            for ff in chord_frettings
        ]

        if len(chord_frettings) == 0:
            raise NoValidFrettingException(self, string_config)

        # if pruning requirements are given, prune
        if pruning_config is not None and evaluator is not None:

            # beam pruning
            costs = [fretting.cost for fretting in chord_frettings]

            cost_limit = min(costs) + np.std(costs) + pruning_config.candidate_beam_width * np.std(costs)
            chord_frettings = sorted([fretting for fretting in chord_frettings if fretting.cost <= cost_limit],
                                     key=lambda x: x.cost)

            # max candidate pruning
            if len(chord_frettings) > pruning_config.max_candidates > 0 and pruning_config.max_candidates > 0:
                chord_frettings = chord_frettings[:pruning_config.max_candidates]

        if len(chord_frettings) == 0:
            raise NoValidFrettingException(self, string_config)

        t_end = time()
        if TRACK_PERFORMANCE:
            t = (t_end - t_start) * 1000
            Chord.__fret_count__ += 1
            if Chord.__fret_avg_time__ == 0:
                Chord.__fret_avg_time__ = t
            else:
                # moving average
                Chord.__fret_avg_time__ = \
                    (1.0 - 1.0 / Chord.__fret_count__) * Chord.__fret_avg_time__\
                    + (1.0 / Chord.__fret_count__) * t

        return chord_frettings

    def __del__(self):
        if TRACK_PERFORMANCE:
            print('Calls of get_chord_frettings(): {}'.format(Chord.__fret_count__))
            print('Average runtime: {} ms'.format(Chord.__fret_avg_time__))
            print('Total time spent in get_chord_frettings(): {} s'.format(
                Chord.__fret_avg_time__ * Chord.__fret_count__ / 1000.0))


class ChordFretting:
    """
    Fretting of a Chord, represented by a list of single note frettings, context features and duration
    """
    _entity_names = {'string', 'fret'}
    _descriptor_features_zero = {}  # create in init
    _descriptor_pitches_zero = {}  # create in init
    _empty_fretting = None

    def __init__(self, duration: float, note_frettings: list, evaluator: ChordFrettingEvaluatorBase,
                 previous_chord_fretting: typing.Optional['ChordFretting'], next_pitches: list,
                 string_config: StringConfigBase, part_of_previous: bool):
        """
        :param duration: duration as fracture of whole notes (e.g. 0.25 for quarter note)
        :type duration: float
        :param note_frettings: list of dicts, i.e. [{'string': int, 'fret': int}, ...]
        :type note_frettings: list of NoteFretting
        :param evaluator: Evaluator object used to estimate the chord cost
        :type evaluator: ChordFrettingEvaluatorBase
        """
        # sanity check
        assert type(duration) is float and duration > 0.0, 'Duration must be float>0.0: {}'.format(duration)
        assert type(note_frettings) is list
        for note_fretting in note_frettings:
            assert isinstance(note_fretting, NoteFretting), 'Not a NoteFretting: {}'.format(note_fretting)
        # make sure no string is used multiple times
        strings = [nf.string for nf in note_frettings]
        frets = [nf.fret for nf in note_frettings]
        assert isinstance(evaluator, ChordFrettingEvaluatorBase)
        assert isinstance(string_config, StringConfigBase)
        assert previous_chord_fretting is None or isinstance(previous_chord_fretting, ChordFretting)
        assert type(part_of_previous) is bool

        self._features = {}
        self._features_delta = {}
        self._cost = None

        if sorted(strings) != sorted(list(set(strings))):
            raise InvalidFrettingException('duplicate use of string', note_frettings)
        if len(strings) > 0 and len(string_config.string_pitches) < max(strings):
            raise InvalidFrettingException(
                'String {} not in StringConfigBase {}!'.format(max(strings), string_config), max(strings))
        if len(frets) > 0 and string_config.num_frets < max(frets):
            raise InvalidFrettingException(
                'Fret {} not reachable in StringConfigBase {}!'.format(max(frets), string_config), max(frets))

        # create zero-vector for filling up later
        if ChordFretting._descriptor_features_zero == {}:
            ChordFretting._descriptor_features_zero = \
                dict([('{}_{}'.format(entity, descriptor), 0.0)
                      for descriptor in FeatureConfig.descriptors_functions
                      for entity in ChordFretting._entity_names])

        # create zero-vector for filling up later
        if ChordFretting._descriptor_pitches_zero == {}:
            ChordFretting._descriptor_pitches_zero = \
                dict([('next_pitches_{}'.format(descriptor), 0.0)
                      for descriptor in FeatureConfig.descriptors_functions])

        self._duration = duration
        self._note_frettings = note_frettings
        self._next_pitches = []  # create attribute ...
        self.next_pitches = next_pitches  # ... then call setter
        self._string_config = string_config
        self._evaluator = evaluator
        self._prev = previous_chord_fretting
        self._part_of_previous = part_of_previous

    @property
    def next_pitches(self):
        return self._next_pitches

    @next_pitches.setter
    def next_pitches(self, next_pitches: list):
        if next_pitches is None:
            self._next_pitches = []
        else:
            assert type(next_pitches) is list
            for pitch in next_pitches:
                if type(pitch) is int:
                    pitch = Pitch(pitch)
                assert type(pitch) is Pitch
            self._next_pitches = list(next_pitches)
        if self._features != {}:
            self._update_next_pitch_features()

    @property
    def is_part_of_previous(self):
        return self._part_of_previous

    def __str__(self) -> str:
        return 'ChordFretting({}, {}, {}, previous_chord_fretting set?: {})'.format(
            self._duration, self._note_frettings, self._evaluator, self.previous is not None)
    __repr__ = __str__

    def __len__(self) -> int:
        return len(self._note_frettings)

    def clone(self) -> 'ChordFretting':
        """
        returns a copy of the object itself, without copying previous note or cost
        :return: copy of self
        :rtype: ChordFretting
        """
        return ChordFretting(self.duration, self.note_frettings, self._evaluator,
                             self.previous, self._next_pitches, self._string_config, self._part_of_previous)

    @property
    def previous(self) -> 'ChordFretting':
        if self._prev is None:
            # create empty fretting as a "terminal state" (to use instead of None as previous)
            if type(self)._empty_fretting is None:
                type(self)._empty_fretting = \
                    ChordFretting(1.0, [], self._evaluator, None, [], self._string_config, False)
                type(self)._empty_fretting.previous = type(self)._empty_fretting

            # no previous chord frettings --> use 1 bar rest as default "previous"
            self._prev = ChordFretting(1.0, [], self._evaluator, type(self)._empty_fretting,
                                       self.get_chord().pitches, self._string_config, False)
        # noinspection PyTypeChecker
        return self._prev

    @previous.setter
    def previous(self, previous_chord_fretting: 'ChordFretting') -> None:
        assert previous_chord_fretting is None or isinstance(previous_chord_fretting, type(self))
        self._prev = previous_chord_fretting
        self._features = {}  # empty features --> has to recalculate next time

    @property
    def cost(self) -> float:
        self._update_cost()
        return self._cost

    def _update_cost(self, force: bool=False) -> None:
        if force or self._cost is None:
            self._cost = self._evaluator.evaluate(self)

            # zero join cost trick for repetition of chords
            # and zero cost for rests
            # sometimes, predictions < 0 -> only crop to 0 when > 0
            if (self == self.previous and self._cost > 0.0) \
                    or len(self._note_frettings) == 0:
                self._cost = 0.0

    def _extract_features(self) -> None:
        """
        extract the features
        """
        count = len(self._note_frettings)
        self._features = {}

        if FeatureConfig.basic:
            self._features.update(dict(
                duration=self._duration,
                is_chord=count > 1,
                is_note=count == 1,
                is_rest=count == 0,
                count=count
            ))

        if FeatureConfig.CHORDS_AS_NOTES:
            self._features.update(
                part_of_previous=self._part_of_previous
            )

        # set the next_pitch features
        self._update_next_pitch_features()

        # use entities X frettings_desc (pd.describe()) as features
        if FeatureConfig.frettings_desc:

            # empty (=rest) --> fill all zeroes
            if len(self._note_frettings) == 0:
                self._features.update(type(self)._descriptor_features_zero)

            else:
                for entity in ChordFretting._entity_names:
                    items = [nf.to_dict()[entity] for nf in self._note_frettings]
                    # run _descriptor_functions
                    for descriptor in FeatureConfig.descriptors_functions:
                        self._features['{}_{}'.format(entity, descriptor)] = \
                            FeatureConfig.descriptors_functions[descriptor](items)

            # calculate the correlation coefficient between frets and strings
            if FeatureConfig.frettings_desc_corrcoef:
                items = [list(nf.to_dict().values()) for nf in self._note_frettings]
                if len(items) <= 1 or min(np.std(items, axis=0)) == 0.0:
                    cc = 0.0
                else:
                    cc = np.corrcoef(items, rowvar=False)[1, 0]
                self._features['correlation_coefficient'] = cc

        if FeatureConfig.frettings_vectorised:
            # supporting 6 (?) strings, remember all details
            # create all empty first
            for string in range(1, FeatureConfig.num_strings+1):
                self._features['string{}_played'.format(string)] = False
                self._features['string{}_fret'.format(string)] = 0
            # then add actually played ones
            for note_fretting in self._note_frettings:
                if note_fretting.string <= FeatureConfig.num_strings:
                    self._features['string{}_played'.format(note_fretting.string)] = True
                    self._features['string{}_fret'.format(note_fretting.string)] = note_fretting.fret

        if FeatureConfig.frettings_sparse:
            # single fret features (complete boolean map)
            # create all empty first
            for string in range(1, FeatureConfig.num_strings+1):
                for fret in range(FeatureConfig.num_frets+1):
                    self._features['string{}_fret{:02n}_played'.format(string, fret)] = False
            # then add actually played ones
            for note_fretting in self._note_frettings:
                if note_fretting.string <= FeatureConfig.num_strings \
                        and note_fretting.fret <= FeatureConfig.num_frets:
                    self._features['string{}_fret{:02n}_played'.format(
                        note_fretting.string, note_fretting.fret)] = True

        self._add_heuristic_features()
        self._cost = None  # delete cost, so it has to be recalculated next time

    def _update_next_pitch_features(self):
        if FeatureConfig.pitch:
            self._features.update(dict(zip(
                ['next_pitch_{}'.format(ii) for ii in range(1, FeatureConfig.num_strings + 1)],
                [pitch.pitch for pitch in list(self._next_pitches)] +
                [0 for _ in range(FeatureConfig.num_strings - len(self._next_pitches))]
            )))

        if FeatureConfig.pitch_desc:
            # empty (next=rest) --> fill all zeroes
            if len(self._next_pitches) == 0:
                self._features.update(type(self)._descriptor_pitches_zero)

            else:
                pitches = [pp.pitch for pp in list(self._next_pitches)]
                # run _descriptor_functions
                for descriptor in FeatureConfig.descriptors_functions:
                    self._features['next_pitches_{}'.format(descriptor)] = \
                        FeatureConfig.descriptors_functions[descriptor](pitches)

        if FeatureConfig.pitch_sparse:
            # init all 0
            pitch_counts = dict([(pitch, 0)
                                 for pitch in range(FeatureConfig.pitch_sparse_min,
                                                    FeatureConfig.pitch_sparse_max + 1)])

            # set pitches
            for pitch in list(self._next_pitches):
                if FeatureConfig.pitch_sparse_min <= pitch.pitch <= FeatureConfig.pitch_sparse_max:
                    pitch_counts[pitch.pitch] += 1
                else:
                    warnings.warn('Pitch out of observed range: {} (Range {}-{})'.format(
                        pitch.pitch, FeatureConfig.pitch_sparse_min, FeatureConfig.pitch_sparse_max
                    ))

            pitch_counts = dict(zip(['next_has_pitch_{}'.format(kk)
                                     for kk in pitch_counts], pitch_counts.values()))
            self._features.update(pitch_counts)

    def _add_heuristic_features(self) -> None:
        if FeatureConfig.heuristics:
            self._features.update(dict(
                heuristic_distance_move=self.heuristic_distance_move(1),
                heuristic_distance_move_fret=self.heuristic_distance_move(1, True),
                heuristic_distance_steady=self.heuristic_distance_steady(),
                heuristic_distance_steady_fret=self.heuristic_distance_steady(1, True),
                heuristic_skipped_strings=self.heuristic_skipped_strings(),
                heuristic_all_zero=int(all([nf.fret == 0 for nf in self._note_frettings])),
            ))

    def heuristic_skipped_strings(self) -> int:
        strings = [nf.string for nf in self._note_frettings]
        if len(strings) == 0:
            return 0
        return max(strings) - min(strings) + 1 - len(strings)

    def heuristic_distance_steady(self, minkowski_order=2, fret_only=False):
        """
        sum of all combinations of note fretting distances
        """
        dd = 0.0
        for idx1, nf1 in enumerate(self._note_frettings):
            for nf2 in self._note_frettings[idx1 + 1:]:
                if fret_only:
                    dd += self.minkowski_distance(
                        nf1.fret, nf2.fret, minkowski_order=minkowski_order
                    )
                else:
                    dd += self.minkowski_distance(
                        nf1.fret, nf2.fret, nf1.string, nf2.string, minkowski_order=minkowski_order
                    )
        return dd

    def heuristic_distance_move(self, minkowski_order=2, fret_only=False):
        string = self._features['string_mean']
        prev_string = self.previous.features_raw['string_mean']
        fret = self._features['fret_mean']
        prev_fret = self.previous.features_raw['fret_mean']

        if fret_only:
            return self.minkowski_distance(fret, prev_fret, minkowski_order=minkowski_order)
        else:
            return self.minkowski_distance(fret, prev_fret, string, prev_string, minkowski_order=minkowski_order)

    @staticmethod
    def minkowski_distance(fret1, fret2, string1=0.0, string2=0.0, minkowski_order=2):
        # if coming from empty fret / rest, there is no cost!
        if fret1 == 0.0 or fret2 == 0.0:
            delta_fret = 0.0
        else:
            delta_fret = abs(fret1 - fret2)

        if string1 == 0.0 or string2 == 0.0:
            delta_string = 0.0
        else:
            delta_string = abs(string1 - string2)

        return pow(
            pow(delta_fret, minkowski_order) +
            pow(delta_string, minkowski_order)
            , 1.0 / minkowski_order)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ChordFretting):
            return False
        # ignore: duration, previous etc.
        return self._note_frettings == other.note_frettings

    def __ne__(self, other) -> bool:
        return not self == other

    @property
    def features_raw(self) -> dict:
        """
        returns the features of the current fretting
        """
        if self._features == {}:
            self._extract_features()
        return self._features

    @property
    def features(self) -> dict:
        """
        returns the features of the current fretting
        can be either the features or DELTA features
        """
        if FeatureConfig.DELTA_MODE:
            return self.features_delta
        else:
            return self.features_raw

    @property
    def features_delta(self) -> dict:
        if self._features_delta == {} or self._features == {}:
            assert len(self.previous.features_raw) == len(self.features_raw)
            for key in self.features_raw:
                # shift next_ features so they are in line with the rest
                if key.startswith('next'):
                    self._features_delta[key.replace('next_', '')] = \
                        self.previous.features_raw[key] - self.previous.previous.features_raw[key]
                else:
                    self._features_delta[key] = self.features_raw[key] - self.previous.features_raw[key]
        return self._features_delta

    @property
    def duration(self):
        return self._duration

    def to_ascii_tab(self, string_config: StringConfigBase=None) -> list:
        """
        converts the chord fretting to a simple text representation
        :param string_config: string configuration object
        :type string_config: StringConfigBase
        """
        # sanity check
        assert string_config is None or isinstance(string_config, StringConfigBase), \
            '{} is not of type StringConfigBase!'.format(string_config)

        # deal with number of strings
        if len(self.note_frettings) > 0:
            max_string = max([nf.string for nf in self.note_frettings])
        else:
            max_string = 0

        if string_config is None:
            num_strings = max_string
        else:
            num_strings = len(string_config.string_pitches)
        assert num_strings >= max_string, \
            'Not enough strings to convert: {}/{}!'.format(num_strings, max_string)

        # build the fret array
        frets = []
        for ss in range(1, num_strings + 1):
            fret = [nf.to_ascii_tab() for nf in self.note_frettings if nf.string == ss]
            if len(fret) > 0:
                assert len(fret) == 1, 'String {} used twice!'.format(ss)
                fret = fret[0]
                frets.append(fret)
            else:
                frets.append('-')
        frets = ['-{}-'.format(fret) for fret in frets][::-1]

        # adjust string length (e.g. -11- needs --9- in other string)
        max_len = max([len(ff) for ff in frets])
        for idx, fret in enumerate(frets):
            while len(fret) < max_len:
                fret = '-' + fret
            frets[idx] = fret

        return frets

    def to_tuple_list(self) -> list:
        return [note_fretting.to_tuple() for note_fretting in self.note_frettings]

    def get_chord(self) -> Chord:
        return Chord(
            self._duration, [fretting.get_pitch(self._string_config) for fretting in self.note_frettings],
            self._part_of_previous
        )

    @property
    def note_frettings(self) -> list:
        return self._note_frettings


class ChordFrettingSequence:
    """
    Sequence of chord frettings (e.g. candidate sequence for a song)
    """

    def __init__(self, chord_frettings: typing.Optional[list]=None):
        """
        :param chord_frettings: chord frettings to initialise the sequence with
        :type chord_frettings: list of ChordFretting
        """
        # sanity check
        if chord_frettings is None:
            self._frettings = []
        else:
            self._frettings = chord_frettings
        assert type(self._frettings) is list
        for idx, fretting in enumerate(self._frettings):
            assert isinstance(fretting, ChordFretting)
            assert idx == 0 or fretting.previous == self._frettings[idx-1], \
                'ChordFrettingSequence: "previous" mismatch: idx={} .previous={}, [idx-1]:{}'.format(
                    idx, fretting.previous, self._frettings[idx-1]
                )

    def __str__(self) -> str:
        # too long to print!
        return 'ChordFrettingSequence(len={})'.format(len(self))
    __repr__ = __str__

    def raw(self) -> list:
        return [(cf.to_tuple_list(), cf.cost) for cf in self._frettings]

    def __eq__(self, other) -> bool:
        if not isinstance(other, ChordFrettingSequence):
            return False
        if len(self) != len(other):
            return False
        for idx in range(len(self)):
            if self[idx] != other[idx]:
                return False
        return True

    def __ne__(self, other) -> bool:
        return not self == other

    @property
    def cost(self) -> float:
        return sum([fretting.cost for fretting in self._frettings])

    def append(self, chord_fretting: ChordFretting):
        """
        append a CLONE of the given chord fretting to the sequence
        :param chord_fretting: the fretting to append
        :type chord_fretting: ChordFretting
        """
        assert isinstance(chord_fretting, ChordFretting)

        cf = chord_fretting.clone()

        # get own last item and set as previous
        if len(self) > 0:
            cf.previous = self[len(self) - 1]
        # append the fretting
        self._frettings.append(cf)

    # def branch(self, chord_frettings: typing.List[ChordFretting]) -> typing.List['ChordFrettingSequence']:
    def branch(self, chord: Chord, string_config: StringConfigBase,
               evaluator: ChordFrettingEvaluatorBase, pruning_config: PruningConfig):
        """
        efficient clone and append:
        keep same ChordFretting instances up to now, but not new one --> save re-evaluation time
        :param chord: The chord which will be used to generate the frettings
        :type chord: Chord
        :param string_config: string configuration object
        :type string_config: StringConfigBase
        :param pruning_config: pruning_config configuration
        :type pruning_config: PruningConfig
        :param evaluator: Evalutator for pruning_config purposes
        :type evaluator: ChordFrettingEvaluatorBase
        :return: possible frettings of the chord
        :rtype: list of ChordFretting
        :return: branched sequences
        """

        assert isinstance(chord, Chord)
        assert isinstance(string_config, StringConfigBase)
        assert isinstance(evaluator, ChordFrettingEvaluatorBase)

        previous = None
        if len(self) > 0:
            previous = self[len(self) - 1]
            previous.next_pitches = chord.pitches

        try:
            chord_frettings = chord.get_chord_frettings(
                string_config, evaluator, pruning_config, prev=previous
            )
        except NoValidFrettingException:
            chord_frettings = []  # if no frettings, just pass back empty array (sequential scenario)

        sequences = []
        for chord_fretting in chord_frettings:
            seq = ChordFrettingSequence(self._frettings.copy())
            seq.append(chord_fretting)
            sequences.append(seq)

        return sequences

    def to_ascii_tab(self, string_config: StringConfigBase,
                     n_first: int=None, do_print: bool=False) -> list:
        """
        converts the chord fretting sequence to a simple text representation
        :param string_config: string configuration object
        :type string_config: StringConfigBase
        :param n_first: only show n first chords
        :type n_first: int
        :param do_print: print directly to console?
        :type do_print: bool
        """
        # sanity check
        assert isinstance(string_config, StringConfigBase), \
            '{} is not of type StringConfigBase!'.format(string_config)
        if n_first is None:
            n_first = len(self._frettings)
        assert type(n_first) is int, 'n_first must be int: {}'.format(n_first)

        # start line with the string name
        tab = ['{} '.format(pitch.get_note_name()) for pitch in string_config.string_pitches[::-1]]

        for chord_fretting in self._frettings[:n_first]:
            chord_tab = chord_fretting.to_ascii_tab(string_config)
            for string in range(len(tab)):
                tab[string] += chord_tab[string]

        if do_print:
            print('')
            for string in tab:
                print(string)
        else:
            return tab

    def __getitem__(self, item) -> ChordFretting:
        return self._frettings[item]

    def __len__(self) -> int:
        return len(self._frettings)
