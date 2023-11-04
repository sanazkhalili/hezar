from dataclasses import dataclass
from typing import List

from ...configs import PreprocessorConfig
from ...constants import (
    Backends,
)
from ...utils.integration_utils import is_backend_available

if is_backend_available(Backends.TOKENIZERS):
    pass


@dataclass
class TokenizerConfig(PreprocessorConfig):
    name = "tokenizer"
    max_length: int = None
    truncation_strategy: str = None
    truncation_direction: str = None
    stride: int = None
    padding_strategy: str = None
    padding_direction: str = None
    pad_to_multiple_of: int = None
    pad_token_type_id: int = 0
    bos_token: str = None
    eos_token: str = None
    unk_token: str = None
    sep_token: str = None
    pad_token: str = None
    cls_token: str = None
    mask_token: str = None
    additional_special_tokens: List[str] = None
