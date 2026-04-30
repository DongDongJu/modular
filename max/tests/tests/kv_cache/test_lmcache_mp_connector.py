# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Tests for the thin LMCache MP connector helpers."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest


def _install_fake_lmcache_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def package(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        return module

    monkeypatch.delitem(
        sys.modules,
        "max.kv_cache.connectors.lmcache_mp_connector",
        raising=False,
    )
    package("lmcache")
    package("lmcache.v1")
    package("lmcache.v1.multiprocess")
    package("lmcache.v1.multiprocess.protocols")

    utils_module: Any = types.ModuleType("lmcache.utils")

    class _EngineType:
        MAX = "max"

    utils_module.EngineType = _EngineType
    monkeypatch.setitem(sys.modules, "lmcache.utils", utils_module)

    custom_types_module: Any = types.ModuleType(
        "lmcache.v1.multiprocess.custom_types"
    )

    class _CudaIPCWrapper:
        @staticmethod
        def Serialize(_wrapper: object) -> bytes:
            return b""

    class _DeviceBufferDescriptor:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _IPCCacheEngineKey:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        @classmethod
        def from_token_ids(cls, **kwargs: object) -> _IPCCacheEngineKey:
            key = cls()
            key.kwargs = kwargs
            return key

    custom_types_module.CudaIPCWrapper = _CudaIPCWrapper
    custom_types_module.RawCudaIPCWrapper = _CudaIPCWrapper
    custom_types_module.DeviceBufferDescriptor = _DeviceBufferDescriptor
    custom_types_module.IPCCacheEngineKey = _IPCCacheEngineKey
    monkeypatch.setitem(
        sys.modules,
        "lmcache.v1.multiprocess.custom_types",
        custom_types_module,
    )

    mq_module: Any = types.ModuleType("lmcache.v1.multiprocess.mq")
    mq_module.MessageQueueClient = object
    monkeypatch.setitem(sys.modules, "lmcache.v1.multiprocess.mq", mq_module)

    protocol_module: Any = types.ModuleType("lmcache.v1.multiprocess.protocol")
    protocol_module.get_response_class = lambda _request_type: None
    monkeypatch.setitem(
        sys.modules, "lmcache.v1.multiprocess.protocol", protocol_module
    )

    base_module: Any = types.ModuleType(
        "lmcache.v1.multiprocess.protocols.base"
    )
    base_module.RequestType = types.SimpleNamespace(
        LOOKUP="LOOKUP",
        QUERY_PREFETCH_STATUS="QUERY_PREFETCH_STATUS",
        FREE_LOOKUP_LOCKS="FREE_LOOKUP_LOCKS",
        RETRIEVE="RETRIEVE",
        STORE="STORE",
        END_SESSION="END_SESSION",
    )
    monkeypatch.setitem(
        sys.modules, "lmcache.v1.multiprocess.protocols.base", base_module
    )

    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "zmq", types.ModuleType("zmq"))


def _load_mp_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    _install_fake_lmcache_dependencies(monkeypatch)
    module = importlib.import_module(
        "max.kv_cache.connectors.lmcache_mp_connector"
    )
    monkeypatch.delitem(
        sys.modules,
        "max.kv_cache.connectors.lmcache_mp_connector",
        raising=False,
    )
    connectors_package = importlib.import_module("max.kv_cache.connectors")
    if hasattr(connectors_package, "lmcache_mp_connector"):
        delattr(connectors_package, "lmcache_mp_connector")
    return module


def test_blocks_in_chunk_rejects_non_divisible_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_mp_module(monkeypatch)

    with pytest.raises(
        ValueError, match="chunk_size must be a multiple of MAX page_size"
    ):
        module._blocks_in_chunk(384, 256)


def test_model_name_validation_for_mp_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_mp_module(monkeypatch)

    with pytest.raises(ValueError, match="requires 'lmcache_model_name'"):
        module._model_name_from_config({})
    with pytest.raises(ValueError, match="must not contain '@'"):
        module._model_name_from_config({"lmcache_model_name": "bad@name"})
    assert (
        module._model_name_from_config({"lmcache_model_name": "meta/model"})
        == "meta/model"
    )


class _TokenBuffer:
    def __init__(self, tokens: list[int]) -> None:
        self._tokens = tokens

    def __len__(self) -> int:
        return len(self._tokens)

    @property
    def all(self) -> list[int]:
        return self._tokens


class _Future:
    def __init__(self, value: object = None) -> None:
        self.value = value

    def result(self, timeout: float | None = None) -> object:
        del timeout
        return self.value


def _bare_connector(module: Any) -> Any:
    connector = object.__new__(module.LMCacheMPConnector)
    connector._page_size = 128
    connector._chunk_size = 256
    connector._blocks_in_chunk = 2
    connector._world_size = 1
    connector._timeout_s = 1.0
    connector._model_name = "model"
    connector._pending_lookups = {}
    connector._pending_saves = []
    return connector


def test_save_with_tokens_queues_chunk_aligned_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_mp_module(monkeypatch)
    connector = _bare_connector(module)
    ctx = types.SimpleNamespace(
        request_id="req",
        tokens=_TokenBuffer(list(range(512))),
    )

    connector.save_with_tokens(
        ctx,
        block_ids=[10, 11, 12],
        block_hashes=[101, 102, 103],
        token_start=128,
    )

    [save] = connector._pending_saves
    assert save.start == 256
    assert save.end == 512
    assert save.block_ids == [11, 12]
    assert save.token_ids == list(range(512))


def test_save_with_tokens_rejects_multimodal_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_mp_module(monkeypatch)
    connector = _bare_connector(module)
    ctx = types.SimpleNamespace(
        request_id="req",
        tokens=_TokenBuffer(list(range(512))),
        images=[object()],
    )

    with pytest.raises(ValueError, match="text-only"):
        connector.save_with_tokens(
            ctx,
            block_ids=[10, 11],
            block_hashes=[101, 102],
            token_start=0,
        )


def test_lookup_with_tokens_uses_token_mode_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_mp_module(monkeypatch)
    connector = _bare_connector(module)
    requests: list[tuple[object, list[object]]] = []

    def request(request_type: object, payloads: list[object]) -> _Future:
        requests.append((request_type, payloads))
        return _Future()

    connector._request = request
    connector._poll_prefetch_status = lambda _request_id: 1
    ctx = types.SimpleNamespace(
        request_id="req",
        tokens=_TokenBuffer(list(range(512))),
    )

    assert connector.lookup_with_tokens(ctx, [101, 102, 103, 104], 0) == 256

    request_type, payloads = requests[0]
    assert request_type == module.RequestType.LOOKUP
    key: Any = payloads[0]
    assert key.kwargs["token_ids"] == list(range(512))
    assert key.kwargs["start"] == 0
    assert key.kwargs["end"] == 512
    state = connector._pending_lookups["req"]
    assert state.start == 0
    assert state.end == 256
    assert state.block_hashes == [101, 102]


def test_lookup_with_tokens_releases_already_cached_prefix_locks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_mp_module(monkeypatch)
    connector = _bare_connector(module)
    requests: list[tuple[object, list[object]]] = []

    def request(request_type: object, payloads: list[object]) -> _Future:
        requests.append((request_type, payloads))
        return _Future()

    connector._request = request
    connector._poll_prefetch_status = lambda _request_id: 2
    ctx = types.SimpleNamespace(
        request_id="req",
        tokens=_TokenBuffer(list(range(512))),
    )

    assert connector.lookup_with_tokens(ctx, [201, 202], 256) == 256

    free_request = requests[1]
    assert free_request[0] == module.RequestType.FREE_LOOKUP_LOCKS
    free_key: Any = free_request[1][0]
    assert free_key.kwargs["start"] == 0
    assert free_key.kwargs["end"] == 256
    state = connector._pending_lookups["req"]
    assert state.start == 256
    assert state.end == 512


def test_export_descriptor_rejects_rocm_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_mp_module(monkeypatch)
    connector = object.__new__(module.LMCacheMPConnector)

    with pytest.raises(RuntimeError, match="supports only CUDA IPC"):
        connector._export_buffer_descriptor(
            0, types.SimpleNamespace(api="rocm")
        )
