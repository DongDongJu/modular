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

"""Gated integration test for the MAX LMCache MP connector."""

from __future__ import annotations

import os
import socket
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import closing, contextmanager
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LMCACHE_MAX_MP_E2E") != "1",
    reason="Set LMCACHE_MAX_MP_E2E=1 to run the LMCache MP e2e test.",
)

INTEGRATION_PAGE_SIZE = 16


def _require_hermetic_lmcache_environment() -> None:
    if os.environ.get("LMCACHE_SOURCE_PATH") or os.environ.get(
        "LMCACHE_SITE_PACKAGES"
    ):
        pytest.skip(
            "Do not inject a uv LMCache/Torch environment into this Bazel "
            "MAX test process. Run the full MP e2e with MAX and LMCache in "
            "separate processes that share one compatible Torch/CUDA runtime."
        )
    pytest.importorskip(
        "lmcache.v1.mp_observability.config",
        reason=(
            "The Bazel LMCache dependency does not provide the MP server API "
            "needed for this e2e target."
        ),
    )
    custom_types = pytest.importorskip(
        "lmcache.v1.multiprocess.custom_types",
        reason=(
            "The Bazel LMCache dependency does not provide the MAX MP buffer "
            "descriptor API needed for this e2e target."
        ),
    )
    required_attrs = (
        "DeviceBufferDescriptor",
        "RawCudaIPCWrapper",
    )
    if not all(hasattr(custom_types, attr) for attr in required_attrs):
        pytest.skip(
            "LMCache dependency does not include Modular MAX MP support."
        )


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _lmcache_mp_server(port: int) -> Generator[None, None, None]:
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.mp_observability.config import ObservabilityConfig
    from lmcache.v1.multiprocess.config import MPServerConfig
    from lmcache.v1.multiprocess.server import run_cache_server

    server, engine = run_cache_server(
        mp_config=MPServerConfig(
            host="127.0.0.1",
            port=port,
            chunk_size=INTEGRATION_PAGE_SIZE,
            max_workers=1,
            max_gpu_workers=1,
            max_cpu_workers=1,
        ),
        storage_manager_config=StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=64 * 1024 * 1024,
                    use_lazy=True,
                    init_size_in_bytes=64 * 1024 * 1024,
                )
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        ),
        obs_config=ObservabilityConfig(
            enabled=False,
            metrics_enabled=False,
            logging_enabled=False,
        ),
        return_engine=True,
    )
    try:
        yield
    finally:
        server.close()
        engine.close()


def _mp_config(port: int) -> Any:
    from max.pipelines.lib.config import KVConnectorConfig

    return KVConnectorConfig.model_validate(
        {
            "lmcache_mode": "mp",
            "lmcache_model_name": "modular-max-mp-e2e",
            "lmcache_mp_host": "127.0.0.1",
            "lmcache_mp_port": port,
            "lmcache_mp_timeout_s": 30,
        }
    )


def _make_kv_cache_manager(session: Any, connector_config: Any) -> Any:
    from max.dtype import DType
    from max.graph import DeviceRef
    from max.kv_cache.paged_kv_cache.cache_manager import PagedKVCacheManager
    from max.nn.kv_cache import KVCacheParams, KVConnectorType

    kv_params = KVCacheParams(
        dtype=DType.float32,
        num_layers=2,
        n_kv_heads=4,
        head_dim=64,
        enable_prefix_caching=True,
        kv_connector=KVConnectorType.lmcache,
        kv_connector_config=connector_config,
        page_size=INTEGRATION_PAGE_SIZE,
        devices=[DeviceRef.GPU()],
    )
    return PagedKVCacheManager(
        params=kv_params,
        session=session,
        total_num_pages=64,
        total_num_host_pages=0,
        max_batch_size=64,
    )


def _ctx(num_tokens: int) -> Any:
    from test_common.context_utils import create_text_context

    return create_text_context(np.arange(num_tokens, dtype=np.int64))


def _shutdown_connector_with_timeout(
    connector: Any, timeout: float = 15.0
) -> None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(connector.shutdown)
        try:
            future.result(timeout=timeout)
        except TimeoutError:
            pass


def test_lmcache_mp_connector_round_trip_with_max_buffer() -> None:
    _require_hermetic_lmcache_environment()

    from max.driver import Accelerator, Buffer, accelerator_count
    from max.engine import InferenceSession

    if accelerator_count() == 0:
        pytest.skip("No GPU available")

    port = _free_port()
    with _lmcache_mp_server(port):
        device = Accelerator()
        session = InferenceSession(devices=[device])
        manager = _make_kv_cache_manager(
            session, connector_config=_mp_config(port)
        )
        connector = manager._replica[0].connector
        assert connector.name == "LMCacheMPConnector"

        try:
            device_buffer = manager._replica[0].device_buffers[0].values[0]
            original = np.arange(
                device_buffer.to_numpy().size,
                dtype=np.float32,
            ).reshape(device_buffer.shape)
            device_buffer.inplace_copy_from(Buffer.from_numpy(original))

            ctx = _ctx(INTEGRATION_PAGE_SIZE * 3)
            source_blocks = [0, 1]
            target_blocks = [10, 11]
            block_hashes = [7001, 7002]

            connector.save_with_tokens(
                ctx,
                source_blocks,
                block_hashes,
                token_start=0,
            )
            connector.flush()
            connector.sync()

            device_buffer.inplace_copy_from(
                Buffer.zeros(
                    shape=device_buffer.shape,
                    dtype=device_buffer.dtype,
                    device=device_buffer.device,
                )
            )

            assert (
                connector.lookup_with_tokens(ctx, block_hashes, 0)
                == len(block_hashes) * INTEGRATION_PAGE_SIZE
            )
            assert connector.load(ctx, target_blocks) == block_hashes

            loaded = device_buffer.to_numpy()
            np.testing.assert_array_equal(
                loaded[target_blocks[0]],
                original[source_blocks[0]],
            )
            np.testing.assert_array_equal(
                loaded[target_blocks[1]],
                original[source_blocks[1]],
            )
        finally:
            _shutdown_connector_with_timeout(connector)
