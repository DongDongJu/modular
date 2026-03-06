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

"""KV cache connectors for external cache tiers.

- `NullConnector`: No-op connector when external caching is disabled
- `LocalConnector`: Host memory offloading
- `TieredConnector`: GPU <-> CPU <-> Disk offloading
- `LMCacheConnector`: LMCache integration for tiered external caching
- `LMCacheMPConnector`: LMCache multiprocess integration
- `create_connector()`: Factory function
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from max.driver import Device
from max.engine import InferenceSession
from max.kv_cache.kv_connector import KVConnector
from max.nn.kv_cache import KVCacheBuffer, KVCacheParams

from .local_connector import LocalConnector
from .null_connector import NullConnector
from .tiered_connector import TieredConnector

logger = logging.getLogger("max.pipelines")


def _load_optional_connector(
    module: str,
    class_name: str,
    install_hint: str,
) -> type[KVConnector]:
    """Load an optional connector class and surface a clear install hint."""
    try:
        connector_module = __import__(
            f"{__name__}.{module}",
            fromlist=[class_name],
        )
        return getattr(connector_module, class_name)
    except ImportError as exc:
        raise ImportError(install_hint) from exc


def create_connector(
    params: KVCacheParams,
    devices: Sequence[Device],
    device_buffer: KVCacheBuffer,
    total_num_host_blocks: int,
    total_num_blocks: int,
    session: InferenceSession | None = None,
) -> KVConnector:
    """Create a KV cache connector instance.

    Returns a connector appropriate for the configuration:
    - If `params.lmcache_mp_enabled` is set: LMCacheMPConnector
    - If `params.lmcache_config_file` is set: LMCacheConnector
    - If swapping enabled + disk_offload_dir set: TieredConnector
    - If swapping enabled (no disk): LocalConnector
    - Otherwise: NullConnector

    Args:
        params: KV cache parameters containing configuration.
        devices: Devices for the KV cache tensors.
        device_buffer: Device buffer for KV cache (owned by manager).
        total_num_host_blocks: Total number of host blocks for swapping.
        total_num_blocks: Total number of device blocks (required for LMCache).
        session: Optional inference session for loading custom kernels.

    Returns:
        A connector instance implementing KVConnectorProtocol.
    """
    # TODO: SERVOPT-1020
    # Check for LMCache MP configuration first.
    if params.lmcache_mp_enabled:
        LMCacheMPConnector = _load_optional_connector(
            module="lmcache_mp_connector",
            class_name="LMCacheMPConnector",
            install_hint=(
                "lmcache, torch, and zmq are required for LMCache MP integration. "
                "Install them with: pip install lmcache torch pyzmq"
            ),
        )

        return LMCacheMPConnector(
            params=params,
            devices=devices,
            device_buffer=device_buffer,
            total_num_blocks=total_num_blocks,
            session=session,
        )

    # Check for standard LMCache configuration.
    if params.lmcache_config_file:
        LMCacheConnector = _load_optional_connector(
            module="lmcache_connector",
            class_name="LMCacheConnector",
            install_hint=(
                "lmcache and torch are required for LMCache integration. "
                "Install them with: pip install lmcache torch"
            ),
        )

        return LMCacheConnector(
            params=params,
            devices=devices,
            device_buffer=device_buffer,
            total_num_blocks=total_num_blocks,
            session=session,
        )

    if not params.enable_kvcache_swapping_to_host or total_num_host_blocks == 0:
        logger.info(
            "Creating NullConnector: external KV cache swapping disabled or no host blocks allocated"
        )
        return NullConnector()

    if params.disk_offload_dir is not None:
        logger.info(
            "Creating TieredConnector: "
            f"host_blocks={total_num_host_blocks}, "
            f"disk_dir={params.disk_offload_dir}, "
            f"disk_max_gb={params.disk_offload_max_gb}"
        )
        return TieredConnector(
            params=params,
            devices=devices,
            device_buffer=device_buffer,
            total_num_host_blocks=total_num_host_blocks,
            disk_cache_dir=params.disk_offload_dir,
            max_disk_size_gb=params.disk_offload_max_gb,
            use_direct_io=params.disk_offload_direct_io,
        )

    logger.info(f"Creating LocalConnector: host_blocks={total_num_host_blocks}")
    return LocalConnector(
        params=params,
        device_buffer=device_buffer,
        total_num_host_blocks=total_num_host_blocks,
    )


__all__ = [
    "KVConnector",
    "LMCacheConnector",
    "LMCacheMPConnector",
    "LocalConnector",
    "NullConnector",
    "TieredConnector",
    "create_connector",
]
