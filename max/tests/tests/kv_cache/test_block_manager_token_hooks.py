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

"""Tests for token-aware optional KV connector hooks."""

from __future__ import annotations

from typing import Any

from max.kv_cache.paged_kv_cache.block_manager import BlockManager


class _TokenAwareConnector:
    num_host_blocks = 1

    def __init__(self) -> None:
        self.lookup_calls: list[tuple[object, list[int], int]] = []
        self.save_calls: list[
            tuple[object, list[int], list[int], int, int]
        ] = []

    def lookup_with_tokens(
        self,
        ctx: object,
        block_hashes: list[int],
        token_start: int,
    ) -> int:
        self.lookup_calls.append((ctx, block_hashes, token_start))
        return 128

    def save_with_tokens(
        self,
        ctx: object,
        block_ids: list[int],
        block_hashes: list[int],
        token_start: int,
        parent_seq_hash: int = 0,
    ) -> None:
        self.save_calls.append(
            (ctx, block_ids, block_hashes, token_start, parent_seq_hash)
        )


class _LegacyConnector:
    num_host_blocks = 1

    def __init__(self) -> None:
        self.lookup_calls: list[tuple[object, list[int]]] = []
        self.save_calls: list[tuple[list[int], list[int], int]] = []

    def lookup(self, ctx: object, block_hashes: list[int]) -> int:
        self.lookup_calls.append((ctx, block_hashes))
        return 64

    def save(
        self,
        block_ids: list[int],
        block_hashes: list[int],
        parent_seq_hash: int = 0,
    ) -> None:
        self.save_calls.append((block_ids, block_hashes, parent_seq_hash))


def _manager(connector: object) -> Any:
    manager: Any = object.__new__(BlockManager)
    manager.connector = connector
    return manager


def test_connector_lookup_prefers_token_aware_hook() -> None:
    connector = _TokenAwareConnector()
    manager = _manager(connector)
    ctx = object()

    assert manager._connector_lookup(ctx, [1, 2], 256) == 128
    assert connector.lookup_calls == [(ctx, [1, 2], 256)]


def test_connector_lookup_falls_back_to_legacy_hook() -> None:
    connector = _LegacyConnector()
    manager = _manager(connector)
    ctx = object()

    assert manager._connector_lookup(ctx, [1, 2], 256) == 64
    assert connector.lookup_calls == [(ctx, [1, 2])]


def test_connector_save_prefers_token_aware_hook() -> None:
    connector = _TokenAwareConnector()
    manager = _manager(connector)
    ctx = object()

    manager._connector_save(ctx, [1], [2], 128, parent_seq_hash=7)

    assert connector.save_calls == [(ctx, [1], [2], 128, 7)]


def test_connector_save_falls_back_to_legacy_hook() -> None:
    connector = _LegacyConnector()
    manager = _manager(connector)
    ctx = object()

    manager._connector_save(ctx, [1], [2], 128, parent_seq_hash=7)

    assert connector.save_calls == [([1], [2], 7)]
