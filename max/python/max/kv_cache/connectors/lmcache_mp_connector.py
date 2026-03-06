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

"""LMCache MP connector for external KV cache integration.

This connector uses LMCache's multiprocess server as an external KV tier,
with a hash-addressed bytes transport path dedicated for MAX paged KV cache.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch  # type: ignore[import-not-found]
import zmq  # type: ignore[import-not-found]


def _prepend_lmcache_source_path() -> None:
    """Optionally prepend a local LMCache source tree to sys.path.

    This allows MAX to run against a checked-out LMCache tree (for protocol
    compatibility experiments) instead of the pinned wheel bundled in Bazel
    runfiles.
    """
    lmcache_root = os.environ.get("MAX_LMCACHE_SOURCE_ROOT")
    if not lmcache_root:
        return

    root = Path(lmcache_root)
    if not (root / "lmcache").is_dir():
        return

    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_prepend_lmcache_source_path()

from lmcache.v1.memory_management import (  # type: ignore[import-not-found]
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.multiprocess.custom_types import (  # type: ignore[import-not-found]
    MAXHashKey,
)
from lmcache.v1.multiprocess.mq import (  # type: ignore[import-not-found]
    MessageQueueClient,
)
from lmcache.v1.multiprocess.protocol import (  # type: ignore[import-not-found]
    RequestType,
    get_response_class,
)
from max.driver import Device
from max.engine import InferenceSession
from max.interfaces import RequestID, TextGenerationContext
from max.kv_cache.connectors.lmcache_connector import MAXGPUConnector
from max.kv_cache.kv_connector import KVConnector
from max.nn.kv_cache import KVCacheBuffer, KVCacheParams
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.profiler import traced

logger = logging.getLogger("max.pipelines")


def _build_slot_mapping_tensor(
    block_ids: Sequence[int],
    block_size: int,
) -> torch.Tensor:
    """Return the flattened slot mapping for a block-aligned token span."""
    return torch.tensor(
        [
            block_id * block_size + offset
            for block_id in block_ids
            for offset in range(block_size)
        ],
        dtype=torch.long,
    )


def _send_lmcache_request(
    mq_client: MessageQueueClient,
    request_type: RequestType,
    payloads: list[Any],
    timeout_s: float | None = None,
) -> Any:
    """Send a synchronous request to LMCache MP and return the decoded response."""
    future = mq_client.submit_request(
        request_type,
        payloads,
        get_response_class(request_type),
    )
    return future.result(timeout=timeout_s)


class LMCacheMPConnector(KVConnector):
    """LMCache MP connector for MAX paged KV cache.

    The connector remains fail-open by default and falls back to local compute
    on MP lookup/retrieve/store failures.
    """

    @traced
    def __init__(
        self,
        params: KVCacheParams,
        devices: Sequence[Device],
        device_buffer: KVCacheBuffer,
        total_num_blocks: int,
        session: InferenceSession | None = None,
    ) -> None:
        """Initialize the LMCache MP connector for paged KV cache offload."""
        if not params.enable_prefix_caching:
            raise ValueError(
                "LMCacheMPConnector requires prefix caching to be enabled"
            )
        if session is None:
            raise ValueError(
                "LMCacheMPConnector requires an InferenceSession for KV transfers"
            )

        self.params = params
        self._devices = list(devices)
        self._device_buffer = device_buffer
        self._total_num_blocks = total_num_blocks
        self._block_size = params.page_size
        self._world_size = len(self._devices)
        self._fail_open = params.lmcache_mp_fail_open

        self._request_timeout_ms = params.lmcache_mp_request_timeout_ms
        self._register_timeout_ms = params.lmcache_mp_register_timeout_ms
        self._request_timeout_s = self._request_timeout_ms / 1000.0
        self._register_timeout_s = self._register_timeout_ms / 1000.0
        self._min_retrieve_tokens = (
            params.lmcache_mp_min_retrieve_tokens
            if params.lmcache_mp_min_retrieve_tokens is not None
            else self._block_size
        )

        self._model_name = (
            f"max-{params.dtype_shorthand}-{params.num_layers}l-"
            f"{params.n_kv_heads_per_device}h-{params.head_dim}d-"
            f"p{params.page_size}"
        )

        self._gpu_connector = MAXGPUConnector(
            params=params,
            device_buffer=device_buffer,
            devices=devices,
            total_num_blocks=total_num_blocks,
            session=session,
        )

        self._zmq_context = zmq.Context()
        server_url = f"{params.lmcache_mp_host}:{params.lmcache_mp_port}"
        self._mq_client = MessageQueueClient(server_url, self._zmq_context)

        self._pending_saves: list[tuple[int, int]] = []
        self._pending_loads: dict[str, list[int]] = {}

        self._h2d_blocks_copied = 0
        self._d2h_blocks_copied = 0

        self._degraded = False
        self._is_shutdown = False

        self._chunk_size = self._get_chunk_size()
        if self._chunk_size != self._block_size:
            raise ValueError(
                "LMCache MP chunk size must match MAX page size. "
                f"Got mp_chunk_size={self._chunk_size}, "
                f"page_size={self._block_size}."
            )

    @property
    def name(self) -> str:
        """Return the connector name used in metrics and logs."""
        return "LMCacheMPConnector"

    def _request_id(self, ctx: TextGenerationContext) -> str:
        return str(ctx.request_id)

    def _key_for(self, block_hash: int, worker_id: int) -> MAXHashKey:
        return MAXHashKey(
            model_name=self._model_name,
            world_size=self._world_size,
            worker_id=worker_id,
            block_hash=int(block_hash),
        )

    def _get_chunk_size(self) -> int:
        return int(
            _send_lmcache_request(
                self._mq_client,
                RequestType.GET_CHUNK_SIZE,
                [],
                timeout_s=self._register_timeout_s,
            )
        )

    def _make_memory_obj(self, num_tokens: int) -> TensorMemoryObj:
        shape = self._gpu_connector.get_shape(num_tokens)
        item_size = torch.empty(
            (),
            dtype=self._gpu_connector.kv_dtype,
        ).element_size()
        nbytes = math.prod(shape) * item_size
        raw_data = torch.empty(nbytes, dtype=torch.uint8)

        fmt = MemoryFormat.KV_MLA_FMT if self.params.is_mla else MemoryFormat.KV_2LTD
        metadata = MemoryObjMetadata(
            shape=shape,
            dtype=self._gpu_connector.kv_dtype,
            address=raw_data.data_ptr(),
            phy_size=nbytes,
            ref_count=1,
            fmt=fmt,
        )
        return TensorMemoryObj(
            raw_data=raw_data,
            metadata=metadata,
            parent_allocator=None,
        )

    def _make_memory_obj_from_payload(self, payload: bytes) -> TensorMemoryObj:
        obj = self._make_memory_obj(self._block_size)
        tensor = obj.tensor
        assert tensor is not None

        flat = tensor.reshape(-1).view(torch.uint8)
        src = torch.frombuffer(memoryview(payload), dtype=torch.uint8)
        if src.numel() != flat.numel():
            raise ValueError(
                "Invalid MP payload size. "
                f"Expected {flat.numel()} bytes, got {src.numel()}"
            )
        flat.copy_(src)
        return obj

    @traced
    def lookup(
        self,
        ctx: TextGenerationContext,
        block_hashes: list[int],
    ) -> int:
        """Return the number of prefix tokens available from LMCache MP."""
        if self._degraded or not block_hashes:
            return 0

        request_id = self._request_id(ctx)
        self._pending_loads.pop(request_id, None)

        try:
            min_hit_blocks = len(block_hashes)
            for tp_idx in range(self._world_size):
                keys = [self._key_for(h, tp_idx) for h in block_hashes]
                hit_blocks = int(
                    _send_lmcache_request(
                        self._mq_client,
                        RequestType.LOOKUP_MAX_HASHES,
                        [keys],
                        timeout_s=self._request_timeout_s,
                    )
                )
                min_hit_blocks = min(min_hit_blocks, hit_blocks)
                if min_hit_blocks == 0:
                    break

            if min_hit_blocks > 0:
                self._pending_loads[request_id] = block_hashes[:min_hit_blocks]

            hit_tokens = min_hit_blocks * self._block_size
            if hit_tokens < self._min_retrieve_tokens:
                # Keep semantics of "hit but skip retrieve": avoid re-store duplicates.
                return 0

            return hit_tokens
        except Exception:
            logger.exception("LMCache MP lookup failed")
            if self._fail_open:
                return 0
            raise

    @traced
    def load(
        self,
        ctx: TextGenerationContext,
        target_block_ids: list[int],
    ) -> list[int]:
        """Hydrate cached prefix blocks from LMCache MP into the local KV cache."""
        if self._degraded:
            return []

        request_id = self._request_id(ctx)
        pending_hashes = self._pending_loads.pop(request_id, None)
        if not pending_hashes or not target_block_ids:
            return []

        num_to_load = min(len(pending_hashes), len(target_block_ids))
        hashes_to_load = pending_hashes[:num_to_load]
        starts = [i * self._block_size for i in range(num_to_load)]
        ends = [(i + 1) * self._block_size for i in range(num_to_load)]
        slot_mapping_tensor = _build_slot_mapping_tensor(
            target_block_ids[:num_to_load],
            self._block_size,
        )

        loaded_hashes = hashes_to_load

        try:
            for tp_idx in range(self._world_size):
                keys = [self._key_for(h, tp_idx) for h in loaded_hashes]
                payloads = _send_lmcache_request(
                    self._mq_client,
                    RequestType.RETRIEVE_MAX_HASHES,
                    [keys],
                    timeout_s=self._request_timeout_s,
                )
                if not isinstance(payloads, list):
                    raise RuntimeError("Invalid retrieve response from LMCache MP")

                if len(payloads) < num_to_load:
                    num_to_load = len(payloads)
                    loaded_hashes = loaded_hashes[:num_to_load]
                    starts = starts[:num_to_load]
                    ends = ends[:num_to_load]
                    if num_to_load == 0:
                        return []

                memory_objs = [
                    self._make_memory_obj_from_payload(payloads[i])
                    for i in range(num_to_load)
                ]

                self._gpu_connector.set_tp_shard(tp_idx)
                self._gpu_connector.batched_to_gpu(
                    memory_objs,
                    starts,
                    ends,
                    slot_mapping=slot_mapping_tensor,
                )

            self._h2d_blocks_copied += num_to_load
            return loaded_hashes
        except Exception:
            logger.exception("LMCache MP load failed")
            if self._fail_open:
                return []
            raise

    @traced
    def save(self, block_ids: list[int], block_hashes: list[int]) -> None:
        """Queue newly computed blocks to be flushed to LMCache MP."""
        for block_id, block_hash in zip(block_ids, block_hashes, strict=True):
            self._pending_saves.append((block_id, block_hash))

    @traced
    def sync(self) -> None:
        """Keep the connector interface synchronous for the MP transport."""
        # MP transport requests are synchronous in this connector.
        return None

    @traced
    def flush(self) -> None:
        """Persist queued blocks to LMCache MP."""
        if self._degraded or not self._pending_saves:
            return

        block_ids = [block_id for block_id, _ in self._pending_saves]
        block_hashes = [block_hash for _, block_hash in self._pending_saves]
        slot_mapping_tensor = _build_slot_mapping_tensor(
            block_ids,
            self._block_size,
        )
        starts = [i * self._block_size for i in range(len(block_ids))]
        ends = [(i + 1) * self._block_size for i in range(len(block_ids))]

        try:
            for tp_idx in range(self._world_size):
                memory_objs = [
                    self._make_memory_obj(self._block_size)
                    for _ in block_ids
                ]

                self._gpu_connector.set_tp_shard(tp_idx)
                self._gpu_connector.batched_from_gpu(
                    memory_objs,
                    starts,
                    ends,
                    slot_mapping=slot_mapping_tensor,
                )

                keys: list[MAXHashKey] = []
                payloads: list[bytes] = []
                for block_hash, mem_obj in zip(block_hashes, memory_objs, strict=True):
                    tensor = mem_obj.tensor
                    assert tensor is not None
                    payload = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                    keys.append(self._key_for(block_hash, tp_idx))
                    payloads.append(payload)

                _send_lmcache_request(
                    self._mq_client,
                    RequestType.STORE_MAX_HASHES,
                    [keys, payloads],
                    timeout_s=self._request_timeout_s,
                )

            self._d2h_blocks_copied += len(self._pending_saves)
            self._pending_saves.clear()
        except Exception:
            logger.exception("LMCache MP flush failed")
            if self._fail_open:
                self._pending_saves.clear()
                return
            raise

    def on_request_complete(self, request_id: RequestID, block_ids: list[int]) -> None:
        """Drop request-local MP state once the MAX request finishes."""
        self._pending_loads.pop(str(request_id), None)

    def shutdown(self) -> None:
        """Flush pending work and close the MP client connection."""
        if self._is_shutdown:
            return

        try:
            self.flush()
        except Exception:
            logger.exception("LMCache MP flush failed during shutdown")

        try:
            self._mq_client.close()
        finally:
            self._zmq_context.term()

        self._pending_loads.clear()
        self._pending_saves.clear()
        self._is_shutdown = True

    @property
    def num_host_blocks(self) -> int:
        """Expose an effectively unbounded external block count for MAX."""
        return 2**31 - 1

    @property
    def num_used_host_blocks(self) -> int:
        """Report external host usage, which is not tracked locally in MP mode."""
        return 0

    def reset_prefix_cache(self) -> None:
        """Clear any pending LMCache MP request state."""
        self._pending_loads.clear()
        self._pending_saves.clear()

    @property
    def metrics(self) -> KVCacheMetrics:
        """Return connector transfer metrics tracked by MAX."""
        return KVCacheMetrics(
            h2d_blocks_copied=self._h2d_blocks_copied,
            d2h_blocks_copied=self._d2h_blocks_copied,
        )
