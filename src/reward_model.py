from typing import List, Optional, Union
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer
import datasets
import textstat
import networkx
from amr_utils.amr_readers import PENMAN_Wrapper, Matedata_Parser
import re
from amr_utils.alignments import AMR_Alignment
from amr_utils.amr import AMR

from amr_utils.graph_utils import get_node_alignment

from transformer_rl.src.amr_utils import AMR_Reader


class TextClassifierRewardModel:
    """
    The reward is a classification score returned by a pretrained transformer classifier
    """

    def __init__(
        self, pretrained_weights: str, score_class: int = 0, device: Optional[torch.device] = torch.device("cpu")
    ) -> None:
        """
        Args:
            pretrained_weights (str): Transformer classifier pretrained weights
            score_class (int, optional): The index of the class whose logits will be used as reward. Defaults to 0.
            device (torch.device, optional): The device on which to run inference. Defaults to torch.device("cpu").
        """
        self.device = device
        self.score_class = score_class
        self.pretrained_weights = pretrained_weights
        self.classifier = AutoModelForSequenceClassification.from_pretrained(self.pretrained_weights).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_weights)

    def get_reward(
        self, input: Union[List, torch.Tensor], output: Union[List, torch.Tensor], target=None
    ) -> torch.Tensor:
        with torch.no_grad():
            encoded = self.tokenizer.batch_encode_plus(output, return_tensors="pt", padding=True, truncation=True).to(
                self.device
            )
            scores = self.classifier(**encoded).logits
        return scores[:, self.score_class]


class LetterCounterRewardModel:
    """
    Assings the reward based on the number of occurencies of certain letter in the output
    """

    def __init__(self, letter_to_maximize: str) -> None:
        """
        Args:
            letter_to_maximize (str): The letter which will be used to compute the reward
        """
        self.letter_to_maximize = letter_to_maximize

    def get_reward(self, input, output, target=None) -> torch.Tensor:
        return torch.Tensor([self.__count_letter_in_phrase(o) for o in output])

    def __count_letter_in_phrase(self, phrase: str):
        c = 0
        for letter in phrase.lower():
            if letter == self.letter_to_maximize:
                c = c + 1
        return c


class ResponseLengthModel:
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def get_reward(self, input, output, target=None) -> torch.Tensor:
        return torch.Tensor([len(self.tokenizer(o)) for o in output])


class RougeRewardModel:
    """
    Maximize Rouge between input text and summary
    """

    def __init__(self) -> None:
        self.rouge = datasets.load_metric("rouge")

    def get_reward(self, input, output, target=None):
        rouge_output = self.rouge.compute(references=target, predictions=output, use_aggregator=False)
        return torch.Tensor([sum([v[i].fmeasure for v in rouge_output.values()]) for i in range(len(input))])
        # rewards = torch.Tensor([rouge_output["rouge2"][i].fmeasure for i in range(len(input))])
        # return rewards


class ReadablityRewardModel:
    """
    Maximize Readability of the summary
    """

    MAX_FLESH = 100
    MIN_FLESH = 0
    DALE_CHALL_MAX = 15
    DALE_CHALL_MIN = 0

    def __init__(self) -> None:
        pass

    def get_reward(self, input, output, target):
        return torch.Tensor(
            [
                self.__convert_range(
                    textstat.flesch_reading_ease(summary),
                    ReadablityRewardModel.MAX_FLESH,
                    ReadablityRewardModel.MIN_FLESH,
                    1,
                    -1,
                )
                for summary in output
            ]
        )

    def __convert_range(self, old_value, old_max, old_min, new_max, new_min):
        return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
