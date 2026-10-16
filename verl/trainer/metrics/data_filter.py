import numpy as np
import torch
from datasets import Dataset
from verl.utils.dataset.rl_dataset import RLHFDataset


class DataFilter:
    """
    按 (prompt_idx, epoch) 记录每个 prompt 的 step/acc/reward/entropy/ppl/response_length 以及可选的 ratio_seq/ratio_seq_geo。

    触发条件（训过程中）：
    - 基础记录（step, acc, reward, entropy, ppl, response_length）：每次训练 step 只要走到 compute_data_metrics
      → process_data_filter，就会对当前 batch 的 prompt 写入，即正常训练路径下百分百触发。
    - ratio 记录（ratio_seq, ratio_seq_geo）：仅当 batch 中带有 rollout_log_probs（及 old_log_probs）时才会
      计算并写入，取决于是否启用 rollout correction 或上游是否写入 rollout_log_probs，不是百分百触发。
    """

    def __init__(self, dataset: RLHFDataset, config):
        self.data = {}
        self.data_num = dataset.ori_data_num
        self.rollout_n = config.actor_rollout_ref.rollout.n
        self.c = -2.0
        self.reset(dataset)

    def format_data_key(self, data_source: str, index: int) -> str:
        return f"{data_source}--{index}"

    def reset(self, dataset=None):
        self.data = {
            "acc": torch.full((self.data_num, 1), self.c, dtype=torch.float32),
            "step": torch.full((self.data_num, 1), self.c, dtype=torch.int64),
            "selected": torch.full((self.data_num, 1), self.c, dtype=torch.int64),
            "reward": torch.full((self.data_num, 1, self.rollout_n), self.c, dtype=torch.float32),
            "entropy": torch.full((self.data_num, 1, self.rollout_n), self.c, dtype=torch.float32),
            "ppl": torch.full((self.data_num, 1, self.rollout_n), self.c, dtype=torch.float32),
            "response_length": torch.full((self.data_num, 1, self.rollout_n), self.c, dtype=torch.float32),
            # train/infer ratio（与 rollout_corr 一致）：sequence-level 与 seq-geo，用于轨迹层面 mismatch 统计
            "ratio_seq": torch.full((self.data_num, 1, self.rollout_n), self.c, dtype=torch.float32),
            "ratio_seq_geo": torch.full((self.data_num, 1, self.rollout_n), self.c, dtype=torch.float32),
        }

    def preprocess_epoch(self, epoch: int, dataset=None):
        if epoch == 0 or self.data["acc"].shape[1] > epoch:
            return
        for key in list(self.data.keys()):
            if key in ["acc", "step", "selected"]:
                self.data[key] = torch.cat([self.data[key], torch.full((self.data_num, 1), self.c, dtype=self.data[key].dtype)], axis=1)
            else:
                self.data[key] = torch.cat([self.data[key], torch.full((self.data_num, 1, self.rollout_n), self.c, dtype=self.data[key].dtype)], axis=1)

    def get_prompt_data_at_key(self, key: str, epoch: int, prompt_idx: int):
        return self.data[key][prompt_idx, epoch]

    def add_prompt_data_at_key(self, key: str, epoch: int, prompt_idx: int, prompt_data):
        self.data[key][prompt_idx, epoch] = prompt_data

    def add_weight(self, key: str, weight: torch.Tensor):
        self.data[key] = weight

    def add_data(self, epoch: int, key: str, data_batch):
        for k, item in self.data.items():
            self.data[k][key, epoch] = data_batch[key]
        # self.accs[key, epoch] = avg_reward
        # self.cnts[key, epoch] = cnt

    def add_reward_batch(self, epoch, unique_data_indices, avg_accs, unique_data_cnts):
        self.accs[unique_data_indices, epoch] = avg_accs
        self.cnts[unique_data_indices, epoch] = unique_data_cnts

    def get_reward_trace(self, key: str):
        return self.data.get(key, [])

    def compute_rollout_mismatch_stats(self, epoch: int, prompt_indices, eps: float = 1e-8):
        """
        ratio 本身即 mismatch 指标（train/infer π_train/π_rollout）。
        先对每个 prompt 把 n 条轨迹聚合成一个代表该 prompt 的 ratio，再对这几个 prompt 的 ratio 求均值和方差。

        - ratio 含义：与 rollout_corr 一致。sequence-level：ρ_seq = exp(sum_t log_ratio_t)；seq-geo：ρ_geo = exp(mean_t log_ratio_t)。
        - epoch：训练轮次，用于定位 data 中 (prompt_idx, epoch) 那一页。
        - 每个 prompt：n 条轨迹的 n 个 ratio → 聚合成一个值（取 mean，无效用 self.c 排除），作为该 prompt 的 mismatch。
        - 返回：这 P 个 prompt 的（聚合后）ratio 的均值与方差，ratio_seq 与 ratio_seq_geo 各两个标量。
        """
        if "ratio_seq" not in self.data or "ratio_seq_geo" not in self.data:
            return {}
        if epoch >= self.data["ratio_seq"].shape[1]:
            return {}
        if isinstance(prompt_indices, (list, tuple)):
            prompt_indices = torch.as_tensor(prompt_indices, device=self.data["ratio_seq"].device)
        elif not isinstance(prompt_indices, torch.Tensor):
            prompt_indices = torch.as_tensor(prompt_indices, device=self.data["ratio_seq"].device)

        ratio_seq = self.data["ratio_seq"][prompt_indices, epoch]      # (P, rollout_n)
        ratio_seq_geo = self.data["ratio_seq_geo"][prompt_indices, epoch]

        valid_seq = ratio_seq != self.c
        valid_geo = ratio_seq_geo != self.c

        def per_prompt_aggregated_ratio(vals, valid):
            # 每个 prompt：n 条轨迹的 ratio 聚合成一个值（mean），代表该 prompt 的 mismatch
            agg = torch.full((vals.shape[0],), float("nan"), device=vals.device, dtype=torch.float32)
            for i in range(vals.shape[0]):
                v = vals[i][valid[i]]
                if v.numel() > 0:
                    agg[i] = v.mean()
            return agg

        agg_seq = per_prompt_aggregated_ratio(ratio_seq, valid_seq)
        agg_geo = per_prompt_aggregated_ratio(ratio_seq_geo, valid_geo)

        def mean_var(x):
            x = x[~torch.isnan(x)]
            if x.numel() == 0:
                return float("nan"), float("nan")
            return x.mean().item(), x.var().item()

        seq_mean, seq_var = mean_var(agg_seq)
        geo_mean, geo_var = mean_var(agg_geo)

        return {
            "rollout_mismatch/ratio_seq_mean": seq_mean,
            "rollout_mismatch/ratio_seq_var": seq_var,
            "rollout_mismatch/ratio_seq_geo_mean": geo_mean,
            "rollout_mismatch/ratio_seq_geo_var": geo_var,
        }

    def get_all_data_keys(self):
        return list(self.data.keys())

    def save(self, path: str):
        torch.save(self.data, path)

    def load(self, path: str):
        obj = torch.load(path, weights_only=False)
        self.data = obj
        # 兼容旧 checkpoint：若无 ratio_seq/ratio_seq_geo 则补全，避免后续写入报错
        if "ratio_seq" not in self.data:
            self.data["ratio_seq"] = torch.full(
                (self.data_num, self.data["reward"].shape[1], self.rollout_n), self.c, dtype=torch.float32
            )
        if "ratio_seq_geo" not in self.data:
            self.data["ratio_seq_geo"] = torch.full(
                (self.data_num, self.data["reward"].shape[1], self.rollout_n), self.c, dtype=torch.float32
            )

    def filter_all_dataset(self, epoch, dataset, mode="grpo"):
        if epoch == 0:
            print("Skipping dataset filtering in epoch 0")
            return dataset, {}

        data_num = len(dataset.dataframe_ori)
        assert hasattr(dataset, "dataframe_ori"), "Dataset must have a dataframe_ori attribute"
        dataset.dataframe = dataset.dataframe_ori.filter(
            lambda item: self.filter_easy_item(epoch, item, mode), num_proc=dataset.num_workers,
            desc=f"Filtering easy examples (mode: {mode})",
        )
        data_num_after_filter1 = len(dataset)

        dataset.dataframe = dataset.dataframe.filter(
            lambda item: self.filter_hard_item(epoch, item, mode), num_proc=dataset.num_workers,
            desc=f"Filtering hard examples (mode: {mode})",
        )
        data_num_after_filter2 = len(dataset)

        easy_filtered = data_num - data_num_after_filter1
        hard_filtered = data_num_after_filter1 - data_num_after_filter2
        print(f"Epoch {epoch}: {data_num} -> {data_num_after_filter1} -> {data_num_after_filter2}, "
              f"filtering {easy_filtered} easy examples, {hard_filtered} hard examples")

        info = {"batch/original": data_num, "batch/easy_filtered": easy_filtered, "batch/hard_filtered": hard_filtered}
        return dataset, info

    def filter_easy_item(self, epoch, data_item, mode="grpo", epsilon=0.1):
        key = data_item["extra_info"]["index"]
        assert self.accs.shape[1] == epoch
        if mode == "grpo":
            idxs = np.where(self.cnts[key, :epoch] > 0)[0]
            if idxs.size and self.accs[key, idxs].mean() == 1.0:
                return False
        elif mode == "all" or mode == "all_correct":
            idxs = np.where(self.cnts_all[key, :epoch] > 0)[0]
            if idxs.size and self.accs_all[key, idxs].mean() == 1.0:
                return False
        return True

    def filter_hard_item(self, epoch, data_item, mode="grpo", epsilon=0.1):
        key = data_item["extra_info"]["index"]
        assert self.accs.shape[1] == epoch
        if mode == "grpo":
            idxs = np.where(self.cnts[key, :epoch] > 0)[0]
            if idxs.size and self.accs[key, idxs].mean() == 0.0:
                return False
        elif mode == "all":
            idxs = np.where(self.cnts_all[key, :epoch] > 0)[0]
            if idxs.size and self.accs_all[key, idxs].mean() == 0.0:
                return False
        return True

    def __len__(self):
        return len(self.data["accs"])
