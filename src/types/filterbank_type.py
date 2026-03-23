from enum import Enum


class FilterbankType(str, Enum):
    MEL = 'mel'
    GAMMATONE = 'gammatone'
