"""laya.py's new checkpoint-loading env vars (LAYA_STATE_DICT / LAYA_HF_FILE). No real torch
or laya import happens here -- _resolve_state_dict_path() is pure env-var + huggingface_hub
plumbing, and torch/laya are stubbed in sys.modules so _load_state_dict()/_agent() can be
exercised without a model."""
import sys
import types

import pytest

import eval.adapters.laya as laya_adapter


def test_neither_env_set_returns_none(monkeypatch):
    monkeypatch.delenv("LAYA_STATE_DICT", raising=False)
    monkeypatch.delenv("LAYA_HF_FILE", raising=False)
    assert laya_adapter._resolve_state_dict_path() is None


def test_state_dict_env_returns_the_path_unchanged(monkeypatch):
    monkeypatch.delenv("LAYA_HF_FILE", raising=False)
    monkeypatch.setenv("LAYA_STATE_DICT", "/tmp/some/checkpoint.safetensors")
    assert laya_adapter._resolve_state_dict_path() == "/tmp/some/checkpoint.safetensors"


def test_hf_file_env_downloads_via_hf_hub_download(monkeypatch):
    calls = {}

    def fake_hf_hub_download(repo_id, filename):
        calls["repo_id"] = repo_id
        calls["filename"] = filename
        return "/fake/cache/pytorch_model.bin"

    monkeypatch.delenv("LAYA_STATE_DICT", raising=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_hf_hub_download))
    monkeypatch.setenv("LAYA_HF_FILE", "ShaunSpark/laya-mind2web-browser-agent:pytorch_model.bin")

    path = laya_adapter._resolve_state_dict_path()

    assert path == "/fake/cache/pytorch_model.bin"
    assert calls == {"repo_id": "ShaunSpark/laya-mind2web-browser-agent", "filename": "pytorch_model.bin"}


def test_hf_file_env_takes_priority_over_state_dict(monkeypatch):
    monkeypatch.setenv("LAYA_STATE_DICT", "/tmp/ignored.bin")
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=lambda repo_id, filename: "/fake/downloaded.bin")
    )
    monkeypatch.setenv("LAYA_HF_FILE", "some/repo:file.bin")
    assert laya_adapter._resolve_state_dict_path() == "/fake/downloaded.bin"


def test_hf_file_env_without_colon_raises(monkeypatch):
    monkeypatch.setenv("LAYA_HF_FILE", "no-colon-here")
    with pytest.raises(RuntimeError, match="LAYA_HF_FILE"):
        laya_adapter._resolve_state_dict_path()


def test_load_state_dict_reports_and_raises_on_mismatch(monkeypatch, tmp_path, capsys):
    class FakeModel:
        def load_state_dict(self, state_dict, strict):
            assert strict is True
            raise RuntimeError('Missing key(s) in state_dict: "foo.weight".')

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(load=lambda path, map_location=None: {"foo.weight": 1}))

    ckpt = tmp_path / "ckpt.bin"
    ckpt.write_bytes(b"not a real checkpoint")

    with pytest.raises(RuntimeError, match="Missing key"):
        laya_adapter._load_state_dict(FakeModel(), str(ckpt))
    assert "strict load_state_dict failed" in capsys.readouterr().err


def test_load_state_dict_succeeds_when_strict_passes(monkeypatch, tmp_path, capsys):
    class FakeModel:
        def load_state_dict(self, state_dict, strict):
            assert strict is True
            assert state_dict == {"foo.weight": 1}

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(load=lambda path, map_location=None: {"foo.weight": 1}))

    ckpt = tmp_path / "ckpt.bin"
    ckpt.write_bytes(b"not a real checkpoint")

    laya_adapter._load_state_dict(FakeModel(), str(ckpt))
    assert "loaded state dict" in capsys.readouterr().err


def test_agent_loads_state_dict_when_env_set(monkeypatch):
    laya_adapter._AGENT = None

    class FakeLayaModel:
        def __init__(self):
            self.loaded_with = None

        def load_state_dict(self, state_dict, strict):
            self.loaded_with = (state_dict, strict)

    class FakeAgent:
        def __init__(self):
            self.model = FakeLayaModel()

    fake_agent_instance = FakeAgent()
    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(load=lambda model_id, device: fake_agent_instance))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(load=lambda path, map_location=None: {"a": 1}))
    monkeypatch.delenv("LAYA_HF_FILE", raising=False)
    monkeypatch.setenv("LAYA_STATE_DICT", "/tmp/fake.bin")

    try:
        agent = laya_adapter._agent()
        assert agent is fake_agent_instance
        assert agent.model.loaded_with == ({"a": 1}, True)
    finally:
        laya_adapter._AGENT = None  # don't leak into other tests


def test_agent_skips_load_when_neither_env_set(monkeypatch):
    laya_adapter._AGENT = None

    class FakeLayaModel:
        def load_state_dict(self, *a, **k):
            raise AssertionError("load_state_dict must not be called when neither env var is set")

    class FakeAgent:
        def __init__(self):
            self.model = FakeLayaModel()

    fake_agent_instance = FakeAgent()
    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(load=lambda model_id, device: fake_agent_instance))
    monkeypatch.delenv("LAYA_STATE_DICT", raising=False)
    monkeypatch.delenv("LAYA_HF_FILE", raising=False)

    try:
        agent = laya_adapter._agent()
        assert agent is fake_agent_instance
    finally:
        laya_adapter._AGENT = None
