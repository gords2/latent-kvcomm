import importlib


def test_transformers_is_pinned_to_4_53_3():
    import transformers
    assert transformers.__version__ == "4.53.3", (
        "KVComm requires exactly 4.53.3 (model definition changed in 4.54.0); "
        f"got {transformers.__version__}"
    )


def test_lkv_imports():
    lkv = importlib.import_module("lkv")
    assert isinstance(lkv.__version__, str)


def test_torch_available_cpu():
    import torch
    assert float(torch.ones(2, 2).sum()) == 4.0
