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

"""Tests for LMCache connector configuration filtering."""

from __future__ import annotations

from max.pipelines.lib.config import KVConnectorConfig


def test_as_lmcache_config_filters_mp_only_keys() -> None:
    cfg = KVConnectorConfig.model_validate(
        {
            "lmcache_mode": "mp",
            "lmcache_model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "lmcache_mp_host": "127.0.0.1",
            "lmcache_mp_port": 5555,
            "local_cpu": True,
            "max_local_cpu_size": 4,
        }
    )

    assert cfg.as_lmcache_config() == {
        "local_cpu": True,
        "max_local_cpu_size": 4,
    }
    assert cfg.as_lmcache_mp_config()["lmcache_mode"] == "mp"
