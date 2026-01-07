# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from .external_backend import (
    ExternalCacheLoadResult,
    ExternalCacheLookupResult,
    ExternalCacheStats,
    ExternalCacheStoreResult,
    ExternalKVCacheBackend,
    NullExternalKVCacheBackend,
    create_external_backend,
    register_external_backend,
)
from .null_cache_manager import NullKVCacheManager
from .paged_cache import (
    InsufficientBlocksError,
    KVTransferEngine,
    KVTransferEngineMetadata,
    PagedKVCacheManager,
    TransferReqData,
    available_port,
)
from .registry import (
    estimate_kv_cache_size,
    infer_optimal_batch_size,
    load_kv_manager,
)

__all__ = [
    # External backend
    "ExternalCacheLoadResult",
    "ExternalCacheLookupResult",
    "ExternalCacheStats",
    "ExternalCacheStoreResult",
    "ExternalKVCacheBackend",
    "NullExternalKVCacheBackend",
    "create_external_backend",
    "register_external_backend",
    # Existing exports
    "InsufficientBlocksError",
    "KVTransferEngine",
    "KVTransferEngineMetadata",
    "NullKVCacheManager",
    "PagedKVCacheManager",
    "TransferReqData",
    "available_port",
    "estimate_kv_cache_size",
    "infer_optimal_batch_size",
    "load_kv_manager",
]
