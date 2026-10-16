# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.naive import NaiveRewardLoopManager
from verl.workers.reward_manager.rllm_skywork import rllmSkyworkRewardManager


@register("rllm_skywork")
class RllmSkyworkRewardLoopManager(NaiveRewardLoopManager):
    """Adapter that wraps rllmSkyworkRewardManager into the experimental reward loop system."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        reward_kwargs = dict(config.reward_model.get("reward_kwargs", {}))
        skywork_manager = rllmSkyworkRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            **reward_kwargs,
        )
        super().__init__(
            config,
            tokenizer,
            compute_score=skywork_manager.compute_score,
            reward_router_address=reward_router_address,
            reward_model_tokenizer=reward_model_tokenizer,
        )
