from .registry import *
from .builders import *
from .configs import *

from typing import TYPE_CHECKING

from .utils.dummy_objects.framework import _LazyModule  # noqa
from .utils.dummy_objects import *
from .utils.integration_utils import are_backends_available
from .constants import Backends, DUMMY_PATH

__version__ = "0.31.3"

# no third-party python libraries are required for the following classes
_import_structure = {
    "utils.logging": ["Logger"],

    # Configs with no lib requirement
    "models.text_classification.distilbert.distilbert_text_classification_config": [
        "DistilBertTextClassificationConfig"],

    DUMMY_PATH: [],
}

if are_backends_available([Backends.TORCH, Backends.TOKENIZERS]):
    _import_structure['preprocessors.tokenizers.tokenizer'] = ["Tokenizer", 'TokenizerConfig']
    _import_structure['preprocessors.tokenizers.wordpiece'] = ['WordPieceConfig', 'WordPieceTokenizer']
else:
    _import_structure[DUMMY_PATH].extend(["Tokenizer", 'TokenizerConfig', 'WordPieceConfig', 'WordPieceTokenizer'])

if are_backends_available([Backends.TORCH]):
    _import_structure["models.model"] = ["Model"]
else:
    _import_structure[DUMMY_PATH].extend(["Model"])

if are_backends_available([Backends.TORCH, Backends.TRANSFORMERS]):
    _import_structure["models.text_classification.distilbert.distilbert_text_classification"] = [
        "DistilBertTextClassification"]
else:
    _import_structure[DUMMY_PATH].extend(["DistilBertTextClassification"])

if TYPE_CHECKING:
    from .models.model import Model
    from .models.text_classification.distilbert.distilbert_text_classification import DistilBertTextClassification
    from .models.text_classification.distilbert.distilbert_text_classification_config import (
        DistilBertTextClassificationConfig)
    from .preprocessors.tokenizers.tokenizer import Tokenizer, TokenizerConfig
    from .preprocessors.tokenizers.wordpiece import WordPieceTokenizerConfig, WordPieceTokenizer

    from .utils.logging import Logger
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
