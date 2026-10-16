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
"""
A Ray logger will receive logging info from different processes.
"""

import datetime
import logging
import math
import numbers
import pprint

import torch


_CONSOLE_HIDDEN_METRIC_PREFIXES = (
    "rollout_corr", "rollout_corr_actor",
    "token_p_diff", "seq_p_diff", "seq_sum_p_diff",
    "token_logp_diff", "seq_logp_diff", "seq_sum_logp_diff",
    "token_p_ratio", "seq_p_ratio", "seq_sum_p_ratio",
    "token_frac_cnt", "seq_frac_cnt",
)
_CONSOLE_SCI_NOTATION_THRESHOLD = 1e-5


def _should_hide_console_metric(metric_name: str) -> bool:
    return metric_name.startswith(_CONSOLE_HIDDEN_METRIC_PREFIXES)


def _format_scientific_notation(value: float) -> str:
    significand, exponent = f"{value:.2e}".split("e")
    significand = significand.rstrip("0").rstrip(".")
    return f"{significand}e{exponent}"


def _format_number_for_console(value: numbers.Number) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, numbers.Integral):
        return str(value)
    if not isinstance(value, numbers.Real):
        return pprint.pformat(value)

    value = float(value)
    if not math.isfinite(value):
        return str(value)

    original_text = str(value)
    if "e" not in original_text.lower():
        decimal_part = original_text.partition(".")[2].rstrip("0")
        if len(decimal_part) <= 2:
            return original_text

    abs_value = abs(value)
    if abs_value >= 0.1:
        decimals = 2
    elif abs_value >= 0.01:
        decimals = 3
    elif abs_value >= 0.001:
        decimals = 4
    elif abs_value >= 0.0001:
        decimals = 5
    elif abs_value >= _CONSOLE_SCI_NOTATION_THRESHOLD:
        decimals = 6
    else:
        return _format_scientific_notation(value)

    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def concat_dict_to_str(dict: dict, step):
    output = [f"step:{step}"]
    for k, v in dict.items():
        if _should_hide_console_metric(k):
            continue
        if isinstance(v, numbers.Number):
            output.append(f"{k}:{_format_number_for_console(v)}")
    output_str = " - ".join(output)
    return output_str


class LocalLogger:
    """
    A local logger that logs messages to the console.

    Args:
        print_to_console (bool): Whether to print to the console.
    """

    def __init__(self, print_to_console=True):
        self.print_to_console = print_to_console

    def flush(self):
        pass

    def log(self, data, step):
        if self.print_to_console:
            print(concat_dict_to_str(data, step=step), flush=True)


class DecoratorLoggerBase:
    """
    Base class for all decorators that log messages.

    Args:
        role (str): The role (the name) of the logger.
        logger (logging.Logger): The logger instance to use for logging.
        level (int): The logging level.
        rank (int): The rank of the process.
        log_only_rank_0 (bool): If True, only log for rank 0.
    """

    def __init__(
        self, role: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0, log_only_rank_0: bool = True
    ):
        self.role = role
        self.logger = logger
        self.level = level
        self.rank = rank
        self.log_only_rank_0 = log_only_rank_0
        self.logging_function = self.log_by_logging
        if logger is None:
            self.logging_function = self.log_by_print

    def log_by_print(self, log_str):
        if not self.log_only_rank_0 or self.rank == 0:
            print(f"{self.role} {log_str}", flush=True)

    def log_by_logging(self, log_str):
        if self.logger is None:
            raise ValueError("Logger is not initialized")
        if not self.log_only_rank_0 or self.rank == 0:
            self.logger.log(self.level, f"{self.role} {log_str}")


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_with_rank(message: str, rank: int = 0, log_only_rank_0: bool = False):
    """_summary_
    Print a message with rank information.
    This function prints the message only if `log_only_rank_0` is False or if the rank is 0.

    Args:
        message (str): _description_
        rank (int, optional): _description_. Defaults to 0.
        log_only_rank_0 (bool, optional): _description_. Defaults to False.
    """
    if not log_only_rank_0 or rank == 0:
        print(f"[Rank {rank}] {message}", flush=True)


def print_with_rank_and_timer(message: str, rank: int = 0, log_only_rank_0: bool = False):
    """_summary_
    Print a message with rank information and a timestamp.
    This function prints the message only if `log_only_rank_0` is False or if the rank is 0.

    Args:
        message (str): _description_
        rank (int, optional): _description_. Defaults to 0.
        log_only_rank_0 (bool, optional): _description_. Defaults to False.
    """
    now = datetime.datetime.now()
    message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] [Rank {rank}] {message}"
    if not log_only_rank_0 or rank == 0:
        print(message, flush=True)


def log_with_rank(message: str, rank, logger: logging.Logger, level=logging.INFO, log_only_rank_0: bool = False):
    """_summary_
    Log a message with rank information using a logger.
    This function logs the message only if `log_only_rank_0` is False or if the rank is 0.
    Args:
        message (str): The message to log.
        rank (int): The rank of the process.
        logger (logging.Logger): The logger instance to use for logging.
        level (int, optional): The logging level. Defaults to logging.INFO.
        log_only_rank_0 (bool, optional): If True, only log for rank 0. Defaults to False.
    """
    if not log_only_rank_0 or rank == 0:
        logger.log(level, f"[Rank {rank}] {message}")
