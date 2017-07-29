"""
module tabgen.modelling

Description:  Objects modelling the structure of the transcription problem
              Generally, there are two focus views:
                1. pitch view: The musical content (Pitches and Chords)
                2. tab view:   A way of playing on a stringed instrument (Frettings)
              The two views are related through the instrument of interest,
                which is represented by the StringConfigurationBase object

Contains:     PruningConfiguration
              StringConfigurationBase
              Pitch
              NoteFretting
              Chord
              ChordFretting
              ChordFrettingSequence

Author:       Elias Mistler
Institute:    The University of Edinburgh
Last changed: 2017-06
"""
from .definitions import *


class NoValidFrettingException(Exception):
    def __init__(self, chord_or_note: object, string_config: StringConfigurationBase):
        self.message = 'No fretting for {} on strings {} with {} frets!'.format(
            chord_or_note, string_config.string_pitches, string_config.num_frets)


class InvalidFrettingException(Exception):
    def __init__(self, reason: str, args):
        self.message = 'Invalid Fretting: {} - {}'.format(reason, args)


class StringConfiguration(StringConfigurationBase):
    """
    A configuration object describing an instrument (strings and frets)
    """

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
        return 'StringConfigurationBase({}, {})'.format(self._pitches, self._frets)
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
        if not isinstance(other, StringConfiguration):
            return False
        return (self._pitches == other.string_pitches) \
            and (self._frets == other.num_frets)

    def __ne__(self, other) -> bool:
        return not self == other


class Pitch:
    """
    Wrapper for integer Pitches (MIDI-Pitch)
    allows to retrieve possible frettings
    """

    # cache dict: StringConfigurationBase --> Pitch --> NoteFretting
    _cache = {}

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

    def get_note_frettings(self, string_config: StringConfigurationBase) -> list:
        """
        gets all possible frettings for a single pitch
        :param string_config: string Configuration to be considered
        :type string_config: StringConfigurationBase
        :return: possible note frettings for the pitch
        :rtype: list of NoteFretting
        """
        # sanity check
        assert isinstance(string_config, StringConfigurationBase), \
            '{} is not of type StringConfigurationBase!'.format(string_config)

        s_self = str(self)
        s_conf = str(string_config)
        if CACHING:
            # use cache for possible frettings to speed up
            # cache dict: StringConfigurationBase --> Pitch --> NoteFretting
            if s_conf in Pitch._cache:
                if s_self in Pitch._cache[s_conf]:
                    return Pitch._cache[s_conf][s_self]  # need not clone (has no dependencies)

            # create a new dictionary for the cache if it does not exist yet
            else:
                Pitch._cache[s_conf] = {}

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
        # assert len(frettings) > 0, \
        #     '{} impossible to fret on strings {} with {} frets!'.format(
        #         str(self), string_config.string_pitches, string_config.num_frets
        #     )

        if CACHING:
            # add entry to cache
            Pitch._cache[s_conf][s_self] = frettings

        return frettings

    @property
    def pitch(self) -> int:
        return self._pitch

    @property
    def note_name(self, note_only: bool=True) -> str:
        pitch_sequence = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        name = str(pitch_sequence[self._pitch % 12])
        if not note_only:
            name += str(self._pitch / 12)
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

    def get_pitch(self, string_config: StringConfigurationBase) -> Pitch:
        """
        get Pitch by applying fretting to string_config
        :param string_config: the string configuration to use
        :type string_config: StringConfigurationBase
        :return: pitch
        :rtype: Pitch
        """
        # sanity check
        assert isinstance(string_config, StringConfigurationBase), \
            '{} is not of type StringConfigurationBase!'.format(string_config)
        assert len(string_config.string_pitches) >= self.string, \
            'String {} not in StringConfigurationBase {}!'.format(self.string, string_config)
        assert string_config.num_frets >= self.fret, \
            'Fret {} not reachable in StringConfigurationBase {}!'.format(self.fret, string_config)
        return Pitch(string_config.string_pitches[self.string - 1].pitch + self.fret, fully_muted=self.fully_muted)


class Chord:
    """
    A chord, i.e. an event in time, representing actual chord, single note or a rest,
    depending on the number of pitches (0, 1, 2+)
    """

    # cache dict: Evaluator --> StringConfigurationBase --> Chord --> ChordFretting
    _cache = {}

    __fret_avg_time__ = 0
    __fret_count__ = 0

    def __init__(self, duration: float, pitches: list):
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

        self._pitches = sorted(pitches)
        self._duration = duration

    def __str__(self) -> str:
        return 'Chord({}, {})'.format(self._duration, self._pitches)
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

    def get_chord_frettings(self, string_config: StringConfigurationBase,
                            evaluator: ChordFrettingEvaluatorBase,
                            pruning_config: PruningConfiguration=None,
                            next_pitches: typing.Optional[list]=None,
                            prev: 'ChordFretting'=None) -> list:
        """
        finds and evaluates fretting options
        only keeps frettings with cost <= min+beam_width*std
        :param string_config: string configuration object
        :type string_config: StringConfigurationBase
        :param pruning_config: pruning configuration
        :type pruning_config: PruningConfiguration
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
        assert isinstance(string_config, StringConfigurationBase), \
            '{} is not an instance of StringConfigurationBase!'.format(string_config)
        assert isinstance(pruning_config, PruningConfiguration) or pruning_config is None,\
            '{} is not an instance of PruningConfiguration!'.format(pruning_config)
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
            return [ChordFretting(self._duration, [], evaluator, prev, next_pitches, string_config)]

        t_start = time()

        chord_frettings = []
        # caching path
        s_eval = str(evaluator)
        s_config = str(string_config)
        s_chord = str(self)
        if CACHING:

            # use cache for possible frettings to speed up
            # cache dict: Evaluator --> StringConfigurationBase --> Chord --> ChordFretting
            if s_eval not in Chord._cache:
                Chord._cache[s_eval] = {}

            if s_config not in Chord._cache[s_eval]:
                Chord._cache[s_eval][s_config] = {}

            if s_chord not in Chord._cache[s_eval][s_config]:
                # Chord._cache[s_eval][s_config][s_chord] does not exists --> will be created!
                pass

            # found in cache
            else:
                # need to clone (depedencies!)
                chord_frettings = [ff.clone() for ff in Chord._cache[s_eval][s_config][s_chord]]

        # not in cache --> create and insert into cache
        if len(chord_frettings) == 0:

            # too many pitches to play
            assert len(string_config.string_pitches) >= len(self._pitches), \
                'Chord {} impossible to fret on strings {} with {} frets: too many pitches!'.format(
                    self._pitches, string_config.string_pitches, string_config.num_frets)

            # first, get all possible frettings for the single notes
            note_frettings_dict = dict([(pitch, pitch.get_note_frettings(string_config)) for pitch in self._pitches])

            # generate all possible chord frettings
            # ... starting from all possibilities of the first pitch
            chord_frettings = [[ff] for ff in note_frettings_dict[self._pitches[0]]]

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

            # heuristic filtering
            if HEURISTIC_PREFILTER:
                chord_frettings = [
                    cf for cf in chord_frettings
                    if max([nf.fret for nf in cf]) - min([nf.fret for nf in cf]) <= HEURISTIC_MAX_FRETS
                    and len(set([nf.fret for nf in cf])) <= HEURISTIC_MAX_FINGERS
                ]

            # wrap in class
            chord_frettings = [
                ChordFretting(self._duration, sorted(ff, key=lambda x: x.string),
                              evaluator, prev, next_pitches, string_config)
                for ff in chord_frettings
            ]

            # add entry to cache
            if CACHING:
                # need to clone (depedencies!)
                Chord._cache[s_eval][s_config][s_chord] = [ff.clone() for ff in chord_frettings]

        if len(chord_frettings) == 0:
            raise NoValidFrettingException(self, string_config)
        # assert len(frettings) > 0, \
        #   'No fretting could be found for {} with configuration {}'.format(self, string_config)

        # if pruning requirements are given, prune
        if pruning_config is not None and evaluator is not None:

            # beam pruning
            costs = [fretting.cost for fretting in chord_frettings]

            cost_limit = min(costs) + np.std(costs) + pruning_config.candidate_beam_width * np.std(costs)
            chord_frettings = sorted([fretting for fretting in chord_frettings if fretting.cost <= cost_limit],
                                     key=lambda x: x.cost)

            # max candidate pruning
            if len(chord_frettings) > pruning_config.max_candidates > 0:
                chord_frettings = chord_frettings[:pruning_config.max_candidates]

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
    _descriptor_functions = {
        'mean': np.mean,
        'std': np.std,  # if len(x) > 0 else 0,
        'min': min,
        '25%': lambda x: np.percentile(x, 25),
        '50%': np.median,
        '75%': lambda x: np.percentile(x, 75),
        'max': max,
        'range': lambda x: max(x) - min(x)
    }
    _descriptor_features_zero = {}  # create in init
    _descriptor_pitches_zero = {}  # create in init
    _empty_fretting = None

    def __init__(self, duration: float, note_frettings: list, evaluator: ChordFrettingEvaluatorBase,
                 previous_chord_fretting: typing.Optional['ChordFretting'], next_pitches: list,
                 string_config: StringConfigurationBase):
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
        assert isinstance(evaluator, ChordFrettingEvaluatorBase)
        assert isinstance(string_config, StringConfigurationBase)
        assert previous_chord_fretting is None or isinstance(previous_chord_fretting, ChordFretting)

        self._features = {}
        self._cost = None

        if sorted(strings) != sorted(list(set(strings))):
            raise InvalidFrettingException('duplicate use of string', note_frettings)

        # create zero-vector for filling up later
        if ChordFretting._descriptor_features_zero == {}:
            ChordFretting._descriptor_features_zero = \
                dict([('{}_{}'.format(entity, descriptor), 0.0)
                      for descriptor in ChordFretting._descriptor_functions
                      for entity in ChordFretting._entity_names])

        # create zero-vector for filling up later
        if ChordFretting._descriptor_pitches_zero == {}:
            ChordFretting._descriptor_pitches_zero = \
                dict([('next_pitches_{}'.format(descriptor), 0.0)
                      for descriptor in ChordFretting._descriptor_functions])

        self._duration = duration
        self._note_frettings = sorted(note_frettings, key=lambda x: x.string)
        self._next_pitches = []  # create attribute ...
        self.next_pitches = next_pitches  # ... then call setter
        self._string_config = string_config
        self._evaluator = evaluator
        self._prev = previous_chord_fretting

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
                             self.previous, self._next_pitches, self._string_config)

    @property
    def previous(self) -> 'ChordFretting':
        if self._prev is None:
            # create empty fretting as a "terminal state" (to use instead of None as previous)
            if type(self)._empty_fretting is None:
                type(self)._empty_fretting = ChordFretting(1.0, [], self._evaluator, None, [], self._string_config)
                type(self)._empty_fretting.previous = type(self)._empty_fretting

            # no previous chord frettings --> use 1 bar rest as default "previous"
            self._prev = ChordFretting(1.0, [], self._evaluator, type(self)._empty_fretting,
                                       self.get_chord().pitches, self._string_config)
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
            # zero join cost trick for repetition of chords
            # and zero cost for rests
            if self == self.previous \
                    or len(self._note_frettings) == 0:
                self._cost = 0.0
            else:

                # split chord into single notes if CHORDS_AS_NOTES=True
                if CHORDS_AS_NOTES and len(self) > 1:
                    self._cost = 0.0
                    prev = self._prev
                    for note_fretting in self._note_frettings:
                        chord_fretting = ChordFretting(self.duration, [note_fretting], self._evaluator,
                                                       prev, self._next_pitches, self._string_config)
                        self._cost += chord_fretting.cost
                        prev = chord_fretting
                else:
                    self._cost = self._evaluator.evaluate(self)

    def _extract_features(self) -> None:
        """
        extract the general features
        """
        count = len(self._note_frettings)
        self._features = {}

        if FeatureConfiguration.basic:
            self._features.update(dict(
                duration=self._duration,
                is_chord=count > 1,
                is_note=count == 1,
                is_rest=count == 0,
                count=count
            ))

        # set the next_pitch features
        self._update_next_pitch_features()

        # use entities X descriptors (pd.describe()) as features
        if FeatureConfiguration.descriptors:

            # empty (=rest) --> fill all zeroes
            if len(self._note_frettings) == 0:
                self._features.update(type(self)._descriptor_features_zero)

            else:
                for entity in ChordFretting._entity_names:
                    items = [nf.to_dict()[entity] for nf in self._note_frettings]
                    # run _descriptor_functions
                    for descriptor in self._descriptor_functions:
                        self._features['{}_{}'.format(entity, descriptor)] = \
                            self._descriptor_functions[descriptor](items)

        if FeatureConfiguration.string_details:
            # supporting 6 (?) strings, remember all details
            # create all empty first
            for string in range(1, FeatureConfiguration.num_strings+1):
                self._features['string{}_played'.format(string)] = False
                self._features['string{}_fret'.format(string)] = 0
            # then add actually played ones
            for note_fretting in self._note_frettings:
                if note_fretting.string <= FeatureConfiguration.num_strings:
                    self._features['string{}_played'.format(note_fretting.string)] = True
                    self._features['string{}_fret'.format(note_fretting.string)] = note_fretting.fret

        if FeatureConfiguration.fret_details:
            # supporting 24 (?) frets, count how often each fret is played
            # create all empty first
            for fret in range(FeatureConfiguration.num_frets+1):
                self._features['fret{}_played'.format(fret)] = 0
            # then add actually played ones
            for note_fretting in self._note_frettings:
                if note_fretting.fret <= FeatureConfiguration.num_frets:
                    self._features['fret{}_played'.format(note_fretting.fret)] += 1

        if FeatureConfiguration.detail_matrix:
            # single fret features (complete boolean map)
            # create all empty first
            for string in range(1, FeatureConfiguration.num_strings+1):
                for fret in range(FeatureConfiguration.num_frets+1):
                    self._features['string{}_fret{:02n}_played'.format(string, fret)] = False
            # then add actually played ones
            for note_fretting in self._note_frettings:
                if note_fretting.string <= FeatureConfiguration.num_strings \
                        and note_fretting.fret <= FeatureConfiguration.num_frets:
                    self._features['string{}_fret{:02n}_played'.format(note_fretting.string, note_fretting.fret)] = True

        self._add_previous_features()
        self._cost = None  # empty cost, so it has to be recalculated next time!

    def _update_next_pitch_features(self):
        if FeatureConfiguration.pitch:
            self._features.update(dict(zip(
                ['next_pitch_{}'.format(ii) for ii in range(1, FeatureConfiguration.num_strings + 1)],
                [pitch.pitch for pitch in list(self._next_pitches)] +
                [0 for _ in range(FeatureConfiguration.num_strings - len(self._next_pitches))]
            )))

        if FeatureConfiguration.pitch_descriptors:
            # empty (next=rest) --> fill all zeroes
            if len(self._next_pitches) == 0:
                self._features.update(type(self)._descriptor_pitches_zero)

            else:
                pitches = [pp.pitch for pp in list(self._next_pitches)]
                # run _descriptor_functions
                for descriptor in type(self)._descriptor_functions:
                    self._features['next_pitches_{}'.format(descriptor)] = \
                        self._descriptor_functions[descriptor](pitches)

        if FeatureConfiguration.pitch_sparse:
            # init all 0
            pitch_counts = dict([(pitch, 0)
                                 for pitch in range(FeatureConfiguration.pitch_sparse_min,
                                                    FeatureConfiguration.pitch_sparse_max+1)])

            # set pitches
            for pitch in list(self._next_pitches):
                if FeatureConfiguration.pitch_sparse_min <= pitch.pitch <= FeatureConfiguration.pitch_sparse_max:
                    pitch_counts[pitch.pitch] += 1
                else:
                    warnings.warn('Pitch out of observed range: {} (Range {}-{})'.format(
                        pitch.pitch, FeatureConfiguration.pitch_sparse_min, FeatureConfiguration.pitch_sparse_max
                    ))

            pitch_counts = dict(zip(['next_has_pitch_{}'.format(kk)
                                     for kk in pitch_counts], pitch_counts.values()))
            self._features.update(pitch_counts)

    def _add_previous_features(self) -> None:
        """
        add features from previous chord frettings
        """

        if self._features == {}:
            self._extract_features()

        # delta features: current - prev_1
        if FeatureConfiguration.delta:
            for feature_name in list(self._features.keys()):
                if not feature_name.startswith('prev') and not feature_name.startswith('delta'):
                    self._features['delta_{}'.format(feature_name)] = \
                        self._features[feature_name] - self.previous.features[feature_name]

        prev_x = self  # prev 0 is self

        # go back step by step
        for current_depth in range(1, FeatureConfiguration.max_depth + 1):

            # copy previous features
            prev_x = prev_x.previous  # go one step back

            for feature_name in list(self._features.keys()):
                if FeatureConfiguration.prev and not feature_name.startswith('prev'):
                    self._features['prev{}_{}'.format(current_depth, feature_name)] = \
                        prev_x.features[feature_name]

    def __eq__(self, other) -> bool:
        if not isinstance(other, ChordFretting):
            return False
        # ignore: duration, previous etc.
        return self._note_frettings == other.note_frettings

    def __ne__(self, other) -> bool:
        return not self == other

    @property
    def features(self) -> dict:
        if self._features == {}:
            self._extract_features()
        return self._features

    @property
    def duration(self):
        return self._duration

    def to_ascii_tab(self, string_config: StringConfigurationBase=None) -> list:
        """
        converts the chord fretting to a simple text representation
        :param string_config: string configuration object
        :type string_config: StringConfigurationBase
        """
        # sanity check
        assert string_config is None or isinstance(string_config, StringConfigurationBase), \
            '{} is not of type StringConfigurationBase!'.format(string_config)

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
        return Chord(self._duration, [fretting.get_pitch(self._string_config) for fretting in self.note_frettings])

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
    def branch(self, chord: Chord, string_config: StringConfigurationBase,
               evaluator: ChordFrettingEvaluatorBase, pruning_config: PruningConfiguration):
        """
        efficient clone and append:
        keep same ChordFretting instances up to now, but not new one --> save re-evaluation time
        :param chord: The chord which will be used to generate the frettings
        :type chord: Chord
        :param string_config: string configuration object
        :type string_config: StringConfigurationBase
        :param pruning_config: pruning_config configuration
        :type pruning_config: PruningConfiguration
        :param evaluator: Evalutator for pruning_config purposes
        :type evaluator: ChordFrettingEvaluatorBase
        :return: possible frettings of the chord
        :rtype: list of ChordFretting
        :return: branched sequences
        """

        assert isinstance(chord, Chord)
        assert isinstance(string_config, StringConfigurationBase)
        assert isinstance(evaluator, ChordFrettingEvaluatorBase)

        previous = None
        if len(self) > 0:
            previous = self[len(self) - 1]
            previous.next_pitches = chord.pitches

        chord_frettings = chord.get_chord_frettings(
            string_config, evaluator, pruning_config, prev=previous
        )

        sequences = []
        for chord_fretting in chord_frettings:
            seq = ChordFrettingSequence(self._frettings.copy())
            seq.append(chord_fretting)
            sequences.append(seq)
        return sequences

    def to_ascii_tab(self, string_config: StringConfigurationBase,
                     n_first: int=None, do_print: bool=False) -> list:
        """
        converts the chord fretting sequence to a simple text representation
        :param string_config: string configuration object
        :type string_config: StringConfigurationBase
        :param n_first: only show n first chords
        :type n_first: int
        :param do_print: print directly to console?
        :type do_print: bool
        """
        # sanity check
        assert isinstance(string_config, StringConfigurationBase), \
            '{} is not of type StringConfigurationBase!'.format(string_config)
        if n_first is None:
            n_first = len(self._frettings)
        assert type(n_first) is int, 'n_first must be int: {}'.format(n_first)

        # start line with the string name
        tab = ['{} '.format(pitch.note_name) for pitch in string_config.string_pitches[::-1]]

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
