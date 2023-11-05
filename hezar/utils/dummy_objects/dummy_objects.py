from hezar.registry import register_preprocessor
from .framework import DummyObject, requires_backends
from hezar.registry import register_model
from hezar.constants import Backends


class Tokenizer(metaclass=DummyObject):
    _backend = [Backends.TORCH, Backends.TOKENIZERS]
    _module = "hezar.preprocessors.tokenizers.tokenizer"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)


TokenizerConfig = Tokenizer


@register_preprocessor("wordpiece_tokenizer", dummy=True)
class WordPieceTokenizer(metaclass=DummyObject):
    _backend = [Backends.TOKENIZERS, Backends.TORCH]
    _module = "hezar.preprocessors.tokenizers.wordpiece"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)


WordPieceConfig = WordPieceTokenizer


class Model(metaclass=DummyObject):
    _backend = [Backends.TORCH]
    _module = "hezar.models.model"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)


@register_model(model_name="distilbert_text_classification", dummy=True)
class DistilBertTextClassification(metaclass=DummyObject):
    _backend = [Backends.TORCH, Backends.TRANSFORMERS]
    _module = "hezar.models.text_classification.distilbert.distilbert_text_classification"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)
