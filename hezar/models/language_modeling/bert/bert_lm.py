"""
A BERT Language Model (HuggingFace Transformers) wrapped by a Hezar Model class
"""
from transformers import BertConfig, BertModel

from ....models import Model
from ....registry import register_model
from .bert_lm_config import BertLMConfig


@register_model("bert_lm", config_class=BertLMConfig)
class BertLM(Model):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.model = self._build()

    def _build(self):
        config = BertConfig(**self.config)
        model = BertModel(config)
        return model

    def forward(self, inputs, **kwargs):
        input_ids = inputs.get("token_ids")
        attention_mask = inputs.get("attention_mask", None)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return outputs

    def post_process(self, inputs, **kwargs):
        hidden_states = inputs.get("hidden_states", None)
        attentions = inputs.get("attentions", None)
        outputs = {
            "last_hidden_state": inputs.get("last_hidden_state"),
            "hidden_states": hidden_states,
            "attentions": attentions,
        }
        return outputs