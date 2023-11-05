from dataclasses import dataclass, field
from typing import List

from tokenizers import trainers

from .tokenizer import Tokenizer
from .tokenizer import TokenizerConfig
from ...constants import DEFAULT_TOKENIZER_CONFIG_FILE, DEFAULT_TOKENIZER_FILE
from ...registry import register_preprocessor


@dataclass
class WordPieceTokenizerConfig(TokenizerConfig):
    name = "wordpiece_tokenizer"
    max_length: int = 512
    truncation_strategy: str = "longest_first"
    truncation_direction: str = "right"
    stride: int = 0
    padding_strategy: str = "longest"
    padding_direction: str = "right"
    pad_to_multiple_of: int = 0
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    sep_token: str = "[SEP]"
    cls_token: str = "[CLS]"
    mask_token: str = "[MASK]"
    pad_token_type_id: int = 0
    additional_special_tokens: List[str] = None
    wordpieces_prefix: str = "##"
    vocab_size: int = 30000
    min_frequency: int = 2
    limit_alphabet: int = 1000
    initial_alphabet: list = field(default_factory=list)
    show_progress: bool = True


@register_preprocessor("wordpiece_tokenizer", config_class=WordPieceTokenizerConfig)
class WordPieceTokenizer(Tokenizer):
    """
    A standard WordPiece tokenizer using 🤗HuggingFace Tokenizers

    Args:
        config: Preprocessor config for the tokenizer
        **kwargs: Extra/manual config parameters
    """

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
