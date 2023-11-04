from typing import List

from .tokenizer import Tokenizer
from ...constants import DEFAULT_TOKENIZER_CONFIG_FILE, DEFAULT_TOKENIZER_FILE, Backends
from ...registry import register_preprocessor
from ...utils.integration_utils import is_backend_available
from .wordpiece_config import WordPieceConfig

if is_backend_available(Backends.TOKENIZERS):
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import decoders, models, trainers

_required_backends = [
    Backends.TOKENIZERS,
]


@register_preprocessor("wordpiece_tokenizer", config_class=WordPieceConfig)
class WordPieceTokenizer(Tokenizer):
    """
    A standard WordPiece tokenizer using 🤗HuggingFace Tokenizers

    Args:
        config: Preprocessor config for the tokenizer
        **kwargs: Extra/manual config parameters
    """

    required_backends = _required_backends

    tokenizer_filename = DEFAULT_TOKENIZER_FILE
    tokenizer_config_filename = DEFAULT_TOKENIZER_CONFIG_FILE
    token_ids_name = "token_ids"

    def __init__(self, config, tokenizer_file=None, **kwargs):
        super().__init__(config, tokenizer_file=tokenizer_file, **kwargs)

    def build(self):
        tokenizer = HFTokenizer(models.WordPiece(unk_token=self.config.unk_token))  # noqa
        tokenizer.decoder = decoders.WordPiece(self.config.wordpieces_prefix)  # noqa
        return tokenizer

    def train(self, files: List[str], **train_kwargs):
        """Train the model using the given files"""
        self.config.update(train_kwargs)

        trainer = trainers.WordPieceTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            limit_alphabet=self.config.limit_alphabet,
            initial_alphabet=self.config.initial_alphabet,
            special_tokens=self.config.special_tokens,
            show_progress=self.config.show_progress,
            continuing_subword_prefix=self.config.wordpieces_prefix,
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

    def train_from_iterator(self, dataset: List[str], **train_kwargs):
        """Train the model using the given files"""
        self.config.update(train_kwargs)

        trainer = trainers.WordPieceTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            limit_alphabet=self.config.limit_alphabet,
            initial_alphabet=self.config.initial_alphabet,
            special_tokens=self.config.special_tokens,
            show_progress=self.config.show_progress,
            continuing_subword_prefix=self.config.wordpieces_prefix,
        )
        self._tokenizer.train_from_iterator(dataset, trainer=trainer, length=len(dataset))
