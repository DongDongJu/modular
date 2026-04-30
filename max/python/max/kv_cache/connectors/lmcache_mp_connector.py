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

"""LMCache multiprocess connector for Modular MAX."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch  # type: ignore[import-not-found]
import zmq  # type: ignore[import-not-found]
from lmcache.utils import EngineType  # type: ignore[import-not-found]
from lmcache.v1.multiprocess.custom_types import (  # type: ignore[import-not-found]
    CudaIPCWrapper,
    DeviceBufferDescriptor,
    IPCCacheEngineKey,
    RawCudaIPCWrapper,
)
from lmcache.v1.multiprocess.mq import (  # type: ignore[import-not-found]
    MessageQueueClient,
)
from lmcache.v1.multiprocess.protocol import (  # type: ignore[import-not-found]
    get_response_class,
)
from lmcache.v1.multiprocess.protocols.base import (  # type: ignore[import-not-found]
    RequestType,
)
from max.driver import Buffer, Device, accelerator_api
from max.engine import InferenceSession
from max.interfaces import RequestID, TextGenerationContext
from max.nn.kv_cache import KVCacheBuffer, KVCacheParams
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.profiler import traced

logger = logging.getLogger("max.pipelines")

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 5555
_DEFAULT_TIMEOUT_S = 30.0
_MAX_LAYOUT_FORMAT = "NB_KV_NL_BS_NH_HS"


@dataclass
class _LookupState:
    request_id: str
    token_ids: list[int]
    start: int
    end: int
    block_hashes: list[int]
    matched_chunks: int
    loaded: bool = False


@dataclass
class _PendingSave:
    token_ids: list[int]
    start: int
    end: int
    block_ids: list[int]


@dataclass
class _PendingFuture:
    request_id: str
    future: Any
    device_id: int | None


def _model_name_from_config(config: dict[str, object]) -> str:
    model_name = config.get("lmcache_model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(
            "LMCache MP mode requires 'lmcache_model_name' in "
            "kv_connector_config so cache keys use a stable model namespace."
        )
    if "@" in model_name:
        raise ValueError(
            f"LMCache model name must not contain '@' (got {model_name!r})"
        )
    return model_name


def _server_url_from_config(config: dict[str, object]) -> str:
    server_url = config.get("lmcache_mp_server_url")
    if isinstance(server_url, str) and server_url:
        return server_url
    host = config.get("lmcache_mp_host", _DEFAULT_HOST)
    port = config.get("lmcache_mp_port", _DEFAULT_PORT)
    if not isinstance(host, str) or not host:
        raise ValueError("lmcache_mp_host must be a non-empty string")
    if not isinstance(port, int):
        raise ValueError("lmcache_mp_port must be an integer")
    return f"tcp://{host}:{port}"


def _timeout_from_config(config: dict[str, object]) -> float:
    timeout = config.get("lmcache_mp_timeout_s", _DEFAULT_TIMEOUT_S)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("lmcache_mp_timeout_s must be a positive number")
    return float(timeout)


def _blocks_in_chunk(chunk_size: int, page_size: int) -> int:
    if chunk_size % page_size != 0:
        raise ValueError(
            "LMCache MP chunk_size must be a multiple of MAX page_size "
            "for token-mode MAX integration."
        )
    return chunk_size // page_size


def _chunk_aligned_floor(value: int, chunk_size: int) -> int:
    return (value // chunk_size) * chunk_size


def _chunk_aligned_ceil(value: int, chunk_size: int) -> int:
    return ((value + chunk_size - 1) // chunk_size) * chunk_size


def _max_backend_for_device(device: Device) -> str:
    api = getattr(device, "api", None)
    if api is None:
        api = accelerator_api()
    api_name = str(api).lower()
    if "cuda" in api_name:
        return "cuda"
    if "hip" in api_name or "rocm" in api_name:
        return "rocm"
    if "cpu" in api_name:
        return "cpu"
    return api_name


class LMCacheMPConnector:
    """Thin MAX client for LMCache's existing multiprocess protocol."""

    @traced
    def __init__(
        self,
        params: KVCacheParams,
        devices: Sequence[Device],
        device_buffer: KVCacheBuffer,
        total_num_blocks: int,
        session: InferenceSession | None = None,
    ) -> None:
        """Initialize the LMCache MP connector.

        Args:
            params: KV cache parameters containing connector config.
            devices: Devices that own KV cache shards.
            device_buffer: Device KV cache buffers owned by MAX.
            total_num_blocks: Total device blocks in the paged KV cache.
            session: Unused; accepted to match connector factory signature.

        Raises:
            ValueError: If prefix caching or MP config is invalid.
            RuntimeError: If the LMCache MP server is unreachable or the device
                backend is unsupported.
        """
        del session
        if not params.enable_prefix_caching:
            raise ValueError(
                "LMCache MP connector requires prefix caching to be enabled"
            )

        cfg = params.kv_connector_config
        config = cfg.as_lmcache_mp_config() if cfg else {}
        self.params = params
        self._devices = list(devices)
        self._device_buffer = device_buffer
        self._total_num_blocks = total_num_blocks
        self._world_size = len(self._devices)
        self._page_size = params.page_size
        self._model_name = _model_name_from_config(config)
        self._server_url = _server_url_from_config(config)
        self._timeout_s = _timeout_from_config(config)
        self._instance_ids = [
            os.getpid() * 1000 + idx for idx in range(self._world_size)
        ]

        self._ctx = zmq.Context()
        self._mq_client = MessageQueueClient(self._server_url, self._ctx)
        self._is_shutdown = False

        self._chunk_size = self._get_chunk_size()
        self._blocks_in_chunk = _blocks_in_chunk(
            self._chunk_size, self._page_size
        )

        self._pending_lookups: dict[str, _LookupState] = {}
        self._pending_saves: list[_PendingSave] = []
        self._pending_futures: list[_PendingFuture] = []
        self._pending_end_sessions: set[str] = set()
        self._h2d_blocks_copied = 0
        self._d2h_blocks_copied = 0
        self._save_counter = 0

        self._register_kv_caches()

    @property
    def name(self) -> str:
        """Connector name for logging/debugging."""
        return "LMCacheMPConnector"

    @traced
    def lookup(
        self,
        ctx: TextGenerationContext,
        block_hashes: list[int],
    ) -> int:
        """Look up a prefix from token position 0."""
        return self.lookup_with_tokens(ctx, block_hashes, token_start=0)

    @traced
    def lookup_with_tokens(
        self,
        ctx: TextGenerationContext,
        block_hashes: list[int],
        token_start: int,
    ) -> int:
        """Look up MAX blocks using LMCache's token-mode MP keys."""
        if not block_hashes:
            return 0
        if token_start % self._chunk_size != 0:
            logger.debug(
                "Skipping LMCache MP lookup from non-chunk-aligned token "
                "offset %d (chunk_size=%d)",
                token_start,
                self._chunk_size,
            )
            return 0

        request_id = str(ctx.request_id)
        self._pending_lookups.pop(request_id, None)

        lookup_end = _chunk_aligned_floor(
            token_start + len(block_hashes) * self._page_size,
            self._chunk_size,
        )
        if lookup_end <= token_start:
            return 0

        token_ids = self._token_prefix(ctx, lookup_end)
        key = self._key_for_tokens(
            token_ids,
            request_id=request_id,
            worker_id=None,
            start=0,
            end=lookup_end,
        )
        self._request(RequestType.LOOKUP, [key, self._world_size]).result(
            timeout=self._timeout_s
        )

        matched_chunks_from_root = self._poll_prefetch_status(request_id)
        start_chunk = token_start // self._chunk_size
        if start_chunk:
            overlap_chunks = min(matched_chunks_from_root, start_chunk)
            if overlap_chunks:
                self._free_lookup_locks(
                    request_id,
                    token_ids,
                    start=0,
                    end=overlap_chunks * self._chunk_size,
                )

        matched_chunks = 0
        if matched_chunks_from_root > start_chunk:
            max_lookup_chunks = (lookup_end - token_start) // self._chunk_size
            matched_chunks = min(
                matched_chunks_from_root - start_chunk,
                max_lookup_chunks,
            )

        if matched_chunks:
            matched_blocks = matched_chunks * self._blocks_in_chunk
            self._pending_lookups[request_id] = _LookupState(
                request_id=request_id,
                token_ids=token_ids,
                start=token_start,
                end=token_start + matched_chunks * self._chunk_size,
                block_hashes=block_hashes[:matched_blocks],
                matched_chunks=matched_chunks,
            )
        else:
            self._request(RequestType.END_SESSION, [request_id]).result(
                timeout=self._timeout_s
            )

        return min(
            matched_chunks * self._chunk_size,
            len(block_hashes) * self._page_size,
        )

    @traced
    def load(
        self,
        ctx: TextGenerationContext,
        target_block_ids: list[int],
    ) -> list[int]:
        """Retrieve matched chunks into MAX target blocks."""
        request_id = str(ctx.request_id)
        state = self._pending_lookups.get(request_id)
        if state is None or state.loaded or not target_block_ids:
            return []

        max_chunks_by_targets = len(target_block_ids) // self._blocks_in_chunk
        chunks_to_load = min(state.matched_chunks, max_chunks_by_targets)
        if chunks_to_load <= 0:
            return []

        loaded_blocks = chunks_to_load * self._blocks_in_chunk
        retrieve_end = state.start + chunks_to_load * self._chunk_size
        chunk_block_ids = target_block_ids[:loaded_blocks]
        for worker_id, instance_id in enumerate(self._instance_ids):
            event, device_id = self._record_cuda_event(worker_id)
            key = self._key_for_tokens(
                state.token_ids,
                request_id=state.request_id,
                worker_id=worker_id,
                start=state.start,
                end=retrieve_end,
            )
            future = self._request(
                RequestType.RETRIEVE,
                [key, instance_id, chunk_block_ids, event.ipc_handle(), 0],
            )
            self._wait_cuda_future(state.request_id, future, device_id)

        if retrieve_end < state.end:
            self._free_lookup_locks(
                state.request_id,
                state.token_ids,
                start=retrieve_end,
                end=state.end,
            )
        self._request(RequestType.END_SESSION, [state.request_id]).result(
            timeout=self._timeout_s
        )

        state.loaded = True
        self._pending_lookups.pop(request_id, None)
        self._h2d_blocks_copied += loaded_blocks
        return state.block_hashes[:loaded_blocks]

    @traced
    def save(
        self,
        block_ids: list[int],
        block_hashes: list[int],
        parent_seq_hash: int = 0,
    ) -> None:
        """Reject token-mode saves that do not include token context."""
        del block_ids, block_hashes, parent_seq_hash
        raise RuntimeError(
            "LMCache MP token mode requires MAX to call save_with_tokens() "
            "so STORE requests can use token-mode keys."
        )

    @traced
    def save_with_tokens(
        self,
        ctx: TextGenerationContext,
        block_ids: list[int],
        block_hashes: list[int],
        token_start: int,
        parent_seq_hash: int = 0,
    ) -> None:
        """Queue full MAX chunks for LMCache STORE in ``flush()``."""
        del parent_seq_hash
        if len(block_ids) != len(block_hashes):
            raise ValueError(
                "block_ids and block_hashes must have the same length"
            )
        if not block_ids:
            return
        if token_start < 0 or token_start % self._page_size != 0:
            raise ValueError(
                "LMCache MP token-mode save token_start must be non-negative "
                "and page-aligned."
            )

        raw_end = token_start + len(block_ids) * self._page_size
        aligned_start = _chunk_aligned_ceil(token_start, self._chunk_size)
        aligned_end = _chunk_aligned_floor(raw_end, self._chunk_size)
        if aligned_end <= aligned_start:
            return

        first_block_offset = (aligned_start - token_start) // self._page_size
        num_blocks = (aligned_end - aligned_start) // self._page_size
        chunk_block_ids = block_ids[
            first_block_offset : first_block_offset + num_blocks
        ]
        self._pending_saves.append(
            _PendingSave(
                token_ids=self._token_prefix(ctx, aligned_end),
                start=aligned_start,
                end=aligned_end,
                block_ids=list(chunk_block_ids),
            )
        )

    @traced
    def sync(self) -> None:
        """Wait for pending store futures and backend completion events."""
        futures = self._pending_futures
        self._pending_futures = []
        for pending in futures:
            self._wait_cuda_future(
                pending.request_id,
                pending.future,
                pending.device_id,
            )
        end_sessions = self._pending_end_sessions
        self._pending_end_sessions = set()
        for request_id in end_sessions:
            self._request(RequestType.END_SESSION, [request_id]).result(
                timeout=self._timeout_s
            )

    @traced
    def flush(self) -> None:
        """Submit queued full-chunk STORE requests to the LMCache MP server."""
        if not self._pending_saves:
            return

        save_specs = self._pending_saves
        self._pending_saves = []
        for spec in save_specs:
            request_id = f"max-mp-save-{os.getpid()}-{self._save_counter}"
            self._save_counter += 1
            for worker_id, instance_id in enumerate(self._instance_ids):
                event, device_id = self._record_cuda_event(worker_id)
                key = self._key_for_tokens(
                    spec.token_ids,
                    request_id=request_id,
                    worker_id=worker_id,
                    start=spec.start,
                    end=spec.end,
                )
                future = self._request(
                    RequestType.STORE,
                    [key, instance_id, spec.block_ids, event.ipc_handle()],
                )
                self._pending_futures.append(
                    _PendingFuture(
                        request_id=request_id,
                        future=future,
                        device_id=device_id,
                    )
                )
            self._pending_end_sessions.add(request_id)
        self._d2h_blocks_copied += sum(
            len(spec.block_ids) for spec in save_specs
        )

    def on_request_complete(
        self,
        request_id: RequestID,
        block_ids: list[int],
    ) -> None:
        """Release lookup locks and end MP sessions for abandoned lookups."""
        del block_ids
        state = self._pending_lookups.pop(str(request_id), None)
        if state is None or state.loaded:
            return
        self._free_lookup_locks(
            state.request_id,
            state.token_ids,
            start=state.start,
            end=state.end,
        )
        self._request(RequestType.END_SESSION, [state.request_id]).result(
            timeout=self._timeout_s
        )

    def shutdown(self) -> None:
        """Flush, unregister KV buffers, and close the MP client idempotently."""
        if self._is_shutdown:
            return
        try:
            self.flush()
            self.sync()
            for instance_id in self._instance_ids:
                self._request(
                    RequestType.UNREGISTER_KV_CACHE, [instance_id]
                ).result(timeout=self._timeout_s)
        finally:
            self._pending_saves.clear()
            self._pending_lookups.clear()
            self._pending_futures.clear()
            self._pending_end_sessions.clear()
            self._mq_client.close()
            self._ctx.term()
            self._is_shutdown = True

    @property
    def num_host_blocks(self) -> int:
        """Signal that an external tier is available."""
        return 2**31 - 1

    @property
    def num_used_host_blocks(self) -> int:
        """LMCache owns external block accounting."""
        return 0

    def reset_prefix_cache(self) -> None:
        """Clear local pending request state without clearing LMCache."""
        self._pending_lookups.clear()
        self._pending_saves.clear()
        self._pending_futures.clear()
        self._pending_end_sessions.clear()

    @property
    def metrics(self) -> KVCacheMetrics:
        """Transfer metrics for this connector."""
        return KVCacheMetrics(
            h2d_blocks_copied=self._h2d_blocks_copied,
            d2h_blocks_copied=self._d2h_blocks_copied,
        )

    def _request(self, request_type: RequestType, payloads: list[Any]) -> Any:
        return self._mq_client.submit_request(
            request_type,
            payloads,
            get_response_class(request_type),
        )

    def _get_chunk_size(self) -> int:
        try:
            return int(
                self._request(RequestType.GET_CHUNK_SIZE, []).result(
                    timeout=self._timeout_s
                )
            )
        except TimeoutError as e:
            raise RuntimeError(
                "LMCache MP server did not respond to GET_CHUNK_SIZE at "
                f"{self._server_url} within {self._timeout_s}s."
            ) from e

    def _register_kv_caches(self) -> None:
        for worker_id, (device, instance_id) in enumerate(
            zip(self._devices, self._instance_ids, strict=True)
        ):
            descriptor = self._export_buffer_descriptor(worker_id, device)
            layout_hints: dict[str, object] = {
                "layout_format": _MAX_LAYOUT_FORMAT,
                "num_layers": self.params.num_layers,
                "kv_dim": 1 if self.params.is_mla else 2,
                "tokens_per_block": self.params.page_size,
                "page_size": self.params.page_size,
                "num_kv_heads": self.params.n_kv_heads_per_device,
                "head_dim": self.params.head_dim,
                "total_num_blocks": self._total_num_blocks,
            }
            self._request(
                RequestType.REGISTER_KV_CACHE,
                [
                    instance_id,
                    [descriptor],
                    self._model_name,
                    self._world_size,
                    EngineType.MAX,
                    layout_hints,
                ],
            ).result(timeout=self._timeout_s)

    def _export_buffer_descriptor(
        self,
        worker_id: int,
        device: Device,
    ) -> DeviceBufferDescriptor:
        backend = _max_backend_for_device(device)
        if backend != "cuda":
            raise RuntimeError(
                "LMCache MP for Modular MAX currently supports only CUDA IPC; "
                f"got backend {backend!r}."
            )

        buffer = self._device_buffer.values[worker_id]
        if not isinstance(buffer, Buffer):
            raise TypeError(f"Expected MAX Buffer, got {type(buffer)!r}")
        tensor = torch.from_dlpack(buffer)
        if tensor.device.type != "cuda":
            raise RuntimeError(
                "LMCache MP CUDA registration expected a CUDA tensor view, got "
                f"{tensor.device}."
            )
        wrapper = RawCudaIPCWrapper(tensor)
        return DeviceBufferDescriptor(
            engine_type=EngineType.MAX,
            backend="cuda",
            handle_type="cuda_ipc",
            handle=CudaIPCWrapper.Serialize(wrapper),
            device_id=tensor.device.index or 0,
            dtype=str(tensor.dtype),
            shape=tuple(tensor.shape),
            strides=tuple(tensor.stride()),
            nbytes=tensor.untyped_storage().nbytes(),
            storage_offset_bytes=tensor.storage_offset()
            * tensor.element_size(),
            layout_format=_MAX_LAYOUT_FORMAT,
            layout_hints={
                "num_layers": self.params.num_layers,
                "kv_dim": 1 if self.params.is_mla else 2,
                "tokens_per_block": self.params.page_size,
                "num_kv_heads": self.params.n_kv_heads_per_device,
                "head_dim": self.params.head_dim,
                "total_num_blocks": self._total_num_blocks,
            },
        )

    def _token_prefix(
        self,
        ctx: TextGenerationContext,
        end: int,
    ) -> list[int]:
        images = getattr(ctx, "images", None)
        if images:
            raise ValueError(
                "LMCache MP token mode for Modular MAX currently supports "
                "text-only requests. Multimodal/image prompts require a "
                "cross-engine image-token contract."
            )
        if end < 0 or end > len(ctx.tokens):
            raise ValueError(
                f"Cannot build LMCache token key for [0, {end}); "
                f"context has {len(ctx.tokens)} tokens."
            )
        return [int(token) for token in ctx.tokens.all[:end]]

    def _key_for_tokens(
        self,
        token_ids: list[int],
        *,
        request_id: str,
        worker_id: int | None,
        start: int,
        end: int,
    ) -> IPCCacheEngineKey:
        return IPCCacheEngineKey.from_token_ids(
            model_name=self._model_name,
            world_size=self._world_size,
            worker_id=worker_id,
            token_ids=token_ids,
            start=start,
            end=end,
            request_id=request_id,
        )

    def _free_lookup_locks(
        self,
        request_id: str,
        token_ids: list[int],
        *,
        start: int,
        end: int,
    ) -> None:
        if end <= start:
            return
        key = self._key_for_tokens(
            token_ids,
            request_id=request_id,
            worker_id=None,
            start=start,
            end=end,
        )
        self._request(
            RequestType.FREE_LOOKUP_LOCKS, [key, self._world_size]
        ).result(timeout=self._timeout_s)

    def _poll_prefetch_status(self, request_id: str) -> int:
        deadline = time.monotonic() + self._timeout_s
        while True:
            result = self._request(
                RequestType.QUERY_PREFETCH_STATUS,
                [request_id],
            ).result(timeout=self._timeout_s)
            if result is not None:
                return int(result)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"LMCache MP prefetch for request {request_id!r} timed out"
                )
            time.sleep(0.001)

    def _record_cuda_event(
        self, worker_id: int
    ) -> tuple[torch.cuda.Event, int]:
        self._devices[worker_id].synchronize()
        tensor = torch.from_dlpack(self._device_buffer.values[worker_id])
        device_id = tensor.device.index or 0
        with torch.cuda.device(device_id):
            event = torch.cuda.Event(interprocess=True)
            event.record()
        return event, device_id

    def _wait_cuda_future(
        self,
        request_id: str,
        future: Any,
        device_id: int | None,
    ) -> bool:
        event_handle, succeeded = future.result(timeout=self._timeout_s)
        if not succeeded:
            logger.warning("LMCache MP request %s did not complete", request_id)
            return False
        if device_id is not None:
            event = torch.cuda.Event.from_ipc_handle(device_id, event_handle)
            event.synchronize()
        return True
