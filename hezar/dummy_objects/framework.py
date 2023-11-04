import importlib.util
import importlib.util
import os
import warnings
from collections import OrderedDict
from itertools import chain
from types import ModuleType
from typing import Any


def import_module(module_name: str, things_to_import):
    if not isinstance(module_name, str):
        raise ValueError("module_name is not string, make sure you have correctly defined it")
    try:
        new_module = importlib.import_module(module_name)
        return getattr(new_module, things_to_import)
    except ModuleNotFoundError as e:
        warnings.warn(
            f"\n{e}. If you don't use {things_to_import} ignore this message.",
            stacklevel=2,
        )
        return e


def is_available(module_name: str, lib_name: str = None):
    lib_name = module_name if lib_name is None else lib_name
    import_error = (
            "{0} "
            + f"""
    requires {module_name} library but it was not found in your environment. You can install it with the following instructions:
    ```
    pip install {lib_name}
    ```
    In a notebook or a colab, you can install it by executing a cell with
    ```
    !pip install {lib_name}
    ```
    """
    )
    return module_name, (
        lambda: importlib.util.find_spec(module_name) is not None,
        import_error,
    )


_is_torch_available = importlib.util.find_spec("torch") is not None
_is_torchvision_available = importlib.util.find_spec("torchvision") is not None
_is_torchaudio_available = importlib.util.find_spec("torchaudio") is not None
_is_pyannote_audio_available = importlib.util.find_spec("pyannote") is not None
_is_transformers_available = importlib.util.find_spec("transformers") is not None
_is_timm_available = importlib.util.find_spec("timm") is not None
_is_pillow_available = importlib.util.find_spec("PIL") is not None
_is_tokenizers_available = importlib.util.find_spec("tokenizers") is not None


def is_tokenizers_available():
    return _is_tokenizers_available


def is_pillow_available():
    return _is_pillow_available


def is_timm_available():
    return _is_timm_available


def is_transformers_available():
    return _is_transformers_available


def is_torch_available():
    return _is_torch_available


def is_torchvision_available():
    return _is_torchvision_available


def is_torchaudio_available():
    return _is_torchaudio_available


def is_pyannote_audio_available():
    return _is_pyannote_audio_available


CV2_IMPORT_ERROR = """
{0} requires OpenCV library but it was not found in your environment. You can install it with:
```
pip install opencv-python
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install opencv-python
```
"""

TENSORFLOW_IMPORT_ERROR = """
{0} requires TensorFlow library but it was not found in your environment. You can install it with the following instruction or check out the main webpage of tensorflow.org:
```
pip install tensorflow
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install tensorflow
```
"""

PYTORCH_IMPORT_ERROR = """
{0} requires PyTorch library but it was not found in your environment. You can install it with the following instruction or check out the main webpage of pytorch.org:
```
pip install torch
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install torch
```
"""

TORCHVISION_IMPORT_ERROR = """
{0} requires Torchvision library but it was not found in your environment. You can install it with the following instruction or check out the main webpage of pytorch.org:
```
pip install torchvision
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install torchvision
```
"""

TORCHAUDIO_IMPORT_ERROR = """
{0} requires Torchaudio library, but it was not found in your environment. You can install it with the following instruction or check out the main webpage of pytorch.org:
```
pip install torchaudio

```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install torchaudio

```
"""

PYANNOTE_AUDIO_IMPORT_ERROR = """
{0} requires PYANNOTE_AUDIO library, but it was not found in your environment. You can install it with the following instruction or check out the main repository page of pyannote(https://github.com/pyannote/pyannote-audio):
```
pip install pyannote.audio
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install pyannote.audio
```
"""


class BackendMapping(OrderedDict):
    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            raise KeyError(f"Library {item} is not available in BACKENDS_MAPPING. Contact the Library Maintainers.")


BACKENDS_MAPPING = BackendMapping(
    [
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("torchvision", (is_torchvision_available, TORCHVISION_IMPORT_ERROR)),
        ("torchaudio", (is_torchaudio_available, TORCHAUDIO_IMPORT_ERROR)),
        is_available("seaborn"),
        is_available("numpy"),
        is_available("albumentations"),
        is_available("sklearn"),
        is_available("PIL", "Pillow"),
        is_available("pyannote"),
        is_available("librosa"),
        is_available("transformers"),
        is_available("soundfile"),
        is_available("psutil"),
        is_available("yaml", "pyyaml"),
        is_available("ipython", "IPython"),
        is_available("monai"),
        is_available("glide_text2im"),
        is_available("groundingdino"),
        is_available("requests"),
        is_available("huggingface_hub"),
        is_available("qdrant_client"),
        is_available("tokenizers"),
    ]
)


def requires_backends(obj, backends, module_name: str = None, cls_name: str = None):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    checks = (BACKENDS_MAPPING[backend[0] if isinstance(backend, tuple) else backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        print("".join(failed))
    else:
        assert isinstance(module_name, str), f"module_name is not defined for obj: {name}"
        error = import_module(module_name, cls_name)
        raise error


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattr__(cls, key):
        cls()


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by huggingface/transformers
    # https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self):
        return self.__class__, (self._name, self.__file__, self._import_structure)