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


@register_model(
    model_name="crnn_image2text",
    config_class="CRNNImage2TextConfig",
    dummy=True,
)
class CRNNImage2Text(metaclass=DummyObject):
    _required_backends = [Backends.TORCH]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_model(
    model_name="vit_gpt2_image2text",
    config_class="ViTGPT2Image2TextConfig",
    dummy=True,
)
class ViTGPT2Image2Text(metaclass=DummyObject):
    _required_backends = [Backends.TORCH, Backends.PILLOW, Backends.TRANSFORMERS]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
