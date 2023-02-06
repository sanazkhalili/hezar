from typing import *
from omegaconf import DictConfig

from hezar.models import models_registry, load_model


def test_build_distilbert_text_classification():
    model_name = 'distilbert_text_classification'
    model = load_model(model_name, num_labels2=10)
    print(model)


def test_load():
    from hezar.configs import ModelConfig
    config = ModelConfig.load('hezar-ai/distilbert-fa-sentiment-v1')
    print(config)


def test_distilbert_text_classification_prediction():
    model_name = 'distilbert_text_classification'
    model = models_registry[model_name]['model_class'].load('test')
    text = 'hello from Hezar!'
    print(model.predict(text))


if __name__ == '__main__':
    # test_load()
    # test_distilbert_save_model()
    test_build_distilbert_text_classification()
    # test_distilbert_text_classification_prediction()
