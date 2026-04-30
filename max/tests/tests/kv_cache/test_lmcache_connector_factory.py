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

"""Tests for LMCache connector factory selection."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.kv_cache.connectors import create_connector
from max.nn.kv_cache import KVCacheParams
from max.nn.kv_cache.cache_params import KVConnectorType


class _DummyConnectorConfig:
    def __init__(self, model_extra: dict[str, object] | None = None) -> None:
        self.model_extra = model_extra or {}


class _FakeLocalLMCacheConnector:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeMPLMCacheConnector:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _params(config: _DummyConnectorConfig | None = None) -> KVCacheParams:
    return KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        num_layers=4,
        devices=[DeviceRef.GPU()],
        enable_prefix_caching=True,
        kv_connector=KVConnectorType.lmcache,
        kv_connector_config=config,
    )


def _install_fake_lmcache_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    local_module: Any = types.ModuleType(
        "max.kv_cache.connectors.lmcache_connector"
    )
    local_module.LMCacheConnector = _FakeLocalLMCacheConnector
    mp_module: Any = types.ModuleType(
        "max.kv_cache.connectors.lmcache_mp_connector"
    )
    mp_module.LMCacheMPConnector = _FakeMPLMCacheConnector
    monkeypatch.setitem(
        sys.modules,
        "max.kv_cache.connectors.lmcache_connector",
        local_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "max.kv_cache.connectors.lmcache_mp_connector",
        mp_module,
    )


def test_lmcache_default_mode_selects_local_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lmcache_modules(monkeypatch)

    connector = create_connector(
        params=_params(),
        devices=[MagicMock()],
        device_buffer=MagicMock(),
        total_num_host_blocks=0,
        total_num_blocks=16,
    )

    assert isinstance(connector, _FakeLocalLMCacheConnector)


def test_lmcache_mp_mode_selects_mp_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lmcache_modules(monkeypatch)

    connector = create_connector(
        params=_params(_DummyConnectorConfig({"lmcache_mode": "mp"})),
        devices=[MagicMock()],
        device_buffer=MagicMock(),
        total_num_host_blocks=0,
        total_num_blocks=16,
    )

    assert isinstance(connector, _FakeMPLMCacheConnector)


def test_lmcache_rejects_unknown_mode() -> None:
    with pytest.raises(
        ValueError, match="lmcache_mode must be 'local' or 'mp'"
    ):
        create_connector(
            params=_params(_DummyConnectorConfig({"lmcache_mode": "remote"})),
            devices=[MagicMock()],
            device_buffer=MagicMock(),
            total_num_host_blocks=0,
            total_num_blocks=16,
        )
