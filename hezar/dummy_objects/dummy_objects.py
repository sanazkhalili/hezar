from ..preprocessors.tokenizers.wordpiece_config import WordPieceConfig
from ..registry import register_preprocessor
from .framework import DummyObject, requires_backends
from ..registry import register_model
from ..models.text_classification.distilbert.distilbert_text_classification_config import \
    DistilBertTextClassificationConfig
from ..constants import Backends


class Tokenizer(metaclass=DummyObject):
    _backend = [Backends.PYTORCH, Backends.TOKENIZERS]
    _module = "hezar.preprocessors.tokenizers.wordpiece"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)


@register_preprocessor("wordpiece_tokenizer", config_class=WordPieceConfig, dummy=True)
class WordPieceTokenizer(metaclass=DummyObject):
    _backend = [Backends.TOKENIZERS, Backends.PYTORCH]
    _module = "hezar.preprocessors.tokenizers.wordpiece"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)


class Model(metaclass=DummyObject):
    _backend = [Backends.PYTORCH]
    _module = "hezar.models.model"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)


@register_model(model_name="distilbert_text_classification", config_class=DistilBertTextClassificationConfig,
                dummy=True)
class DistilBertTextClassification(metaclass=DummyObject):
    _backend = [Backends.PYTORCH, Backends.TRANSFORMERS]
    _module = "hezar.models.text_classification.distilbert.distilbert_text_classification"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)
