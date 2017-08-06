"""
Base classes for polymorphism
"""


class StringConfigBase:
    """
    ABSTRACT
    A configuration object describing an instrument (strings and frets)
    """
    def __init__(self):
        pass

    @property
    def string_pitches(self) -> list:
        raise NotImplementedError()

    @property
    def num_frets(self) -> int:
        raise NotImplementedError()

    @property
    def num_strings(self) -> int:
        raise NotImplementedError()


class ChordFrettingEvaluatorBase:
    """
    ABSTRACT
    Encapsulation of the Evaluation algorithm, i.e. cost function
    """
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, fretting):
        """
        evaluates a single fretting
        :param fretting: the chord fretting to evaluate
        :type fretting: ChordFretting
        :return: a cost estimate for the fretting
        :rtype: float
        """
        raise NotImplementedError()


class PruningConfig:
    """
    A full set of pruning settings
    """
    def __init__(self,
                 candidate_beam_width: float, max_candidates: int,
                 sequence_beam_width: float, max_sequences: int):
        """
        :param candidate_beam_width: beam width in std for candidate pruning
        :type candidate_beam_width: float
        :param max_candidates: number of candidates to keep per step (after pruning)
        :type max_candidates: int
        :param sequence_beam_width: beam width in std for sequence pruning
        :type sequence_beam_width: float
        :param max_sequences: number of sequences to keep after every step
        :type max_sequences: int
        """
        assert type(candidate_beam_width) is float and candidate_beam_width >= 0.0
        assert type(sequence_beam_width) is float and sequence_beam_width >= 0.0
        assert type(max_candidates) is int and max_candidates >= 0
        assert type(max_sequences) is int and max_sequences >= 0
        self._candidate_beam_width = candidate_beam_width
        self._sequence_beam_width = sequence_beam_width
        self._max_candidates = max_candidates
        self._max_sequences = max_sequences

    def __str__(self) -> str:
        return 'PruningConfig({}, {}, {}, {})'.format(
            self._candidate_beam_width, self._max_candidates, self._sequence_beam_width, self._max_sequences
        )
    __repr__ = __str__

    @property
    def candidate_beam_width(self) -> float:
        return self._candidate_beam_width

    @property
    def sequence_beam_width(self) -> float:
        return self._sequence_beam_width

    @property
    def max_candidates(self) -> int:
        return self._max_candidates

    @property
    def max_sequences(self) -> int:
        return self._max_sequences
