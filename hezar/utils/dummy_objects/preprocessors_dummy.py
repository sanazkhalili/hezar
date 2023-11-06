from ...constants import Backends
from ...registry import register_preprocessor
from .dummy_object import DummyObject


class Tokenizer(metaclass=DummyObject):
    _required_backends = [Backends.TOKENIZERS, Backends.TORCH]
    _module = "hezar.preprocessors.tokenizers.tokenizer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_preprocessor("wordpiece_tokenizer", config_class="WordPieceTokenizerConfig", dummy=True)
class WordPieceTokenizer(metaclass=DummyObject):
    _required_backends = [Backends.TOKENIZERS, Backends.TORCH]
    _module = "hezar.preprocessors.tokenizers.wordpiece"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
