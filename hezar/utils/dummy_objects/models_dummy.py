from ...constants import Backends
from ...registry import register_model
from .dummy_object import DummyObject


class Model(metaclass=DummyObject):
    _required_backends = [Backends.TORCH]
    _module = "hezar.models.model"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_model(
    model_name="distilbert_text_classification",
    config_class="DistilBertTextClassificationConfig",
    dummy=True,
)
class DistilBertTextClassification(metaclass=DummyObject):
    _required_backends = [Backends.TORCH, Backends.TRANSFORMERS]
    _module = "hezar.models.text_classification.distilbert.distilbert_text_classification"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
