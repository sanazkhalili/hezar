from .registry import *
from .builders import *
from .configs import *

from typing import TYPE_CHECKING

from .dummy_objects.framework import _LazyModule  # noqa
from .dummy_objects.framework import is_torch_available, is_transformers_available, is_tokenizers_available
from .dummy_objects import *

__version__ = "0.31.3"

DUMMY_PATH = "dummy_objects.dummy_objects"
# no third-party python libraries are required for the following classes
_import_structure = {
    "utils.logging": ["Logger"],
    DUMMY_PATH: []
}

if is_torch_available() and is_tokenizers_available():
    _import_structure['preprocessors.tokenizers.tokenizer'] = ["Tokenizer", 'TokenizerConfig']
    _import_structure['preprocessors.tokenizers.wordpiece'] = ['WordPieceConfig', 'WordPieceTokenizer']
else:
    _import_structure[DUMMY_PATH].extend(["Tokenizer", 'TokenizerConfig', 'WordPieceConfig', 'WordPieceTokenizer'])

if is_torch_available():
    _import_structure["models.model"] = ["Model"]
else:
    _import_structure[DUMMY_PATH].extend(["Model"])

if is_torch_available() and is_transformers_available():
    _import_structure["models.text_classification.distilbert.distilbert_text_classification"] = [
        "DistilBertTextClassification"]
else:
    _import_structure[DUMMY_PATH].extend(["DistilBertTextClassification"])

if TYPE_CHECKING:
    from .models.model import Model
    from .models.text_classification.distilbert.distilbert_text_classification import DistilBertTextClassification
    from .preprocessors.tokenizers.tokenizer import Tokenizer, TokenizerConfig
    from .preprocessors.tokenizers.wordpiece import WordPieceConfig, WordPieceTokenizer
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

# from .data import *
# from .embeddings import *
# from .metrics import *
# from .models import *
# from .preprocessors import *
# from .trainer import *
# from .utils import *
