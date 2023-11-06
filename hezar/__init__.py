from .registry import *
from .builders import *
from .configs import *

from typing import TYPE_CHECKING

from .utils import are_backends_available, LazyModule
from .utils.dummy_objects import *
from .constants import Backends, DUMMY_PATH

__version__ = "0.31.3"

# no third-party python libraries are required for the following classes
_import_structure = {
    "utils.logging": ["Logger"],

    # Configs with no lib requirement
    "models.text_classification.distilbert.distilbert_text_classification_config": [
        "DistilBertTextClassificationConfig"],

    "models.image2text.crnn.crnn_image2text_config": ["CRNNImage2TextConfig"],
    "models.image2text.vit_gpt2.vit_gpt2_image2text_config": ["ViTGPT2Image2TextConfig"],

    DUMMY_PATH: [],
}

if are_backends_available([Backends.TORCH, Backends.TRANSFORMERS, Backends.PILLOW]):
    _import_structure["models.image2text.vit_gpt2.vit_gpt2_image2text"] = ["ViTGPT2Image2Text"]
else:
    _import_structure[DUMMY_PATH].extend(["ViTGPT2Image2Text"])

if are_backends_available([Backends.TORCH, Backends.TOKENIZERS]):
    _import_structure['preprocessors.tokenizers.tokenizer'] = ["Tokenizer", 'TokenizerConfig']
    _import_structure['preprocessors.tokenizers.wordpiece'] = ['WordPieceTokenizerConfig', 'WordPieceTokenizer']
    _import_structure['preprocessors.tokenizers.bpe'] = ['BPETokenizerConfig', 'BPETokenizer']
else:
    _import_structure[DUMMY_PATH].extend(["Tokenizer", 'TokenizerConfig',
                                          'WordPieceTokenizerConfig', 'WordPieceTokenizer',
                                          'BPETokenizerConfig', 'BPETokenizer'])

if are_backends_available([Backends.TORCH, Backends.PILLOW]):
    _import_structure["preprocessors.image_processor"] = ["ImageProcessor", "ImageProcessorConfig"]
else:
    _import_structure[DUMMY_PATH].extend(["ImageProcessor", 'ImageProcessorConfig'])

if are_backends_available([Backends.TORCH]):
    _import_structure["models.model"] = ["Model"]
    _import_structure["models.image2text.crnn.crnn_image2text"] = ["CRNNImage2Text"]
else:
    _import_structure[DUMMY_PATH].extend(["Model"])
    _import_structure[DUMMY_PATH].extend(["CRNNImage2Text"])

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
    from .models.image2text.crnn.crnn_image2text import CRNNImage2Text
    from .models.image2text.crnn.crnn_image2text_config import CRNNImage2TextConfig
    from .models.image2text.vit_gpt2.vit_gpt2_image2text import ViTGPT2Image2Text
    from .models.image2text.vit_gpt2.vit_gpt2_image2text_config import ViTGPT2Image2TextConfig
    from .preprocessors.tokenizers.tokenizer import Tokenizer, TokenizerConfig
    from .preprocessors.tokenizers.wordpiece import WordPieceTokenizerConfig, WordPieceTokenizer
    from .preprocessors.image_processor import ImageProcessor, ImageProcessorConfig
    from .preprocessors.tokenizers.bpe import BPETokenizer, BPETokenizerConfig

    from .utils import Logger
else:
    import sys

    sys.modules[__name__] = LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
