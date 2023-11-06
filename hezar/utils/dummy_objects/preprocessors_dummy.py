from ...constants import Backends
from ...registry import register_preprocessor
from .dummy_object import DummyObject


class Tokenizer(metaclass=DummyObject):
    _required_backends = [Backends.TOKENIZERS, Backends.TORCH]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


TokenizerConfig = Tokenizer


@register_preprocessor("wordpiece_tokenizer", config_class="WordPieceTokenizerConfig", dummy=True)
class WordPieceTokenizer(metaclass=DummyObject):
    _required_backends = [Backends.TOKENIZERS, Backends.TORCH]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


WordPieceTokenizerConfig = WordPieceTokenizer


@register_preprocessor("image_processor",
                       config_class="ImageProcessorConfig",
                       dummy=True)
class ImageProcessor(metaclass=DummyObject):
    _required_backends = [Backends.PILLOW, Backends.TORCH]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


ImageProcessorConfig = ImageProcessor
