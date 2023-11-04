from dataclasses import dataclass, field
from typing import List

from .tokenizer_config import TokenizerConfig
from ...constants import Backends
from ...utils.integration_utils import is_backend_available

if is_backend_available(Backends.TOKENIZERS):
    pass

_required_backends = [
    Backends.TOKENIZERS,
]


@dataclass
class WordPieceConfig(TokenizerConfig):
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
