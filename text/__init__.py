
import json
import re
from pathlib import Path

from text import cleaners
from text.symbols import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)", re.DOTALL)

_VOCAB_CACHE = {}


def load_phone_map(language="English"):
    if language not in _VOCAB_CACHE:
        path = (
            Path(__file__).resolve().parent.parent
            / "data" / language / "phone_to_id.json"
        )
        with path.open("r", encoding="utf-8") as f:
            _VOCAB_CACHE[language] = json.load(f)
    return _VOCAB_CACHE[language]


def text_to_sequence(text, cleaner_names=None, language=None):
    if language in ("English", "Telugu"):
        vocab = load_phone_map(language)
        unknown_id = vocab["<unk>"]

        if language == "English":
            tokens = text.strip().split()
            sequence = [
                vocab.get(token, unknown_id)
                for token in tokens
            ]
        else:
            sequence = []
            for char in text:
                if char.isspace():
                    sequence.append(vocab["<space>"])
                else:
                    sequence.append(vocab.get(char, unknown_id))

        return sequence

    if cleaner_names is None:
        cleaner_names = []

    sequence = []

    while len(text):
        match = _curly_re.match(text)

        if not match:
            sequence += _symbols_to_sequence(
                _clean_text(text, cleaner_names)
            )
            break

        sequence += _symbols_to_sequence(
            _clean_text(match.group(1), cleaner_names)
        )
        sequence += _arpabet_to_sequence(match.group(2))
        text = match.group(3)

    return sequence


def sequence_to_text(sequence):
    result = ""

    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            symbol = _id_to_symbol[symbol_id]
            if len(symbol) > 1 and symbol[0] == "@":
                symbol = "{%s}" % symbol[1:]
            result += symbol

    return result.replace("}{", " ")


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbol_list):
    return [
        _symbol_to_id[s]
        for s in symbol_list
        if s in _symbol_to_id and s not in ("_", "~")
    ]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])
