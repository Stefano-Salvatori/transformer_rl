# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02-ppo.ipynb (unless otherwise specified).

__all__ = ["AdaptiveKLController", "FixedKLController", "PPOTrainer"]

# Cell
import numpy as np
from torch.optim import Adam, AdamW
import torch
import time
import random
import torch.nn.functional as F
import wandb
from transformers import DataCollatorForLanguageModeling
from ..train_utils import WarmupCosineLR

from .core import (
    logprobs_from_logits,
    whiten,
    clip_by_value,
    entropy_from_logits,
    flatten_dict,
    average_torch_dicts,
    stats_to_np,
    stack_dicts,
    add_suffix,
    WANDB_PADDING,
)

# Cell


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


# Cell


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


# Cell


class PPOTrainer:
    """
    The PPO_trainer uses Proximal Policy Optimization to optimise language models.
    """

    default_params = {
        "steps": 10000,
        "lr": 1.41e-5,
        "critic_lr": 1.41e-6,
        "weight_decay": 1e-4,
        "adap_kl_ctrl": True,
        "init_kl_coef": 0.2,
        "target": 6,
        "horizon": 10000,
        "gamma": 1,
        "lam": 0.95,
        "cliprange": 0.2,
        "cliprange_value": 0.2,
        "vf_coef": 0.1,
        "entropy_beta": 0.001,
        "batch_size": 256,
        "forward_batch_size": 16,
        "ppo_epochs": 4,
        "clip_gradients": True,
        "clip_gradient_value": 0.5,
    }

    def __init__(self, model, ref_model, critic, tokenizer, **ppo_params):
        """
        Initialize PPOTrainer.

        Args:
            model (torch.model): Hugging Face transformer GPT2 model with value head
            ref_model (torch.model): Hugging Face transformer GPT2 refrence model used for KL penalty
            tokenizer (tokenizer): Hugging Face tokenizer
            ppo_params (dict or None): PPO parameters for training. Can include following keys:
                'lr' (float): Adam learning rate, default: 1.41e-5
                'batch_size' (int): Number of samples per optimisation step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time, default: 16
                'ppo_epochs' (int): Number of optimisation epochs per batch of samples, default: 4
                'gamma' (float)): Gamma parameter for advantage calculation, default: 1.
                'lam' (float): Lambda parameter for advantage calcualation, default: 0.95
                'cliprange_value' (float): Range for clipping values in loss calculation, default: 0.2
                'cliprange' (float): Range for clipping in PPO policy gradient loss, default: 0.2
                'vf_coef' (float): Scaling factor for value loss, default: 0.1
                'adap_kl_ctrl' (bool): Use adaptive KL control, otherwise linear, default: True
                'init_kl_coef' (float): Initial KL penalty coefficient (used for adaptive and linear control), default: 0.2
                'target' (float): Target KL value for adaptive KL control, default: 6.0
                'horizon' (float): Horizon for adaptive KL control, default: 10000

        """
        self.ppo_params = self.default_params
        self.ppo_params.update(ppo_params)

        self.ref_model = ref_model
        self.actor = model
        self.critic = critic
        self.tokenizer = tokenizer

        self.model_optimizer = AdamW(
            self.actor.parameters(),
            lr=self.ppo_params["lr"],
            eps=1e-5,
            weight_decay=self.ppo_params["weight_decay"],
        )
        self.critic_optimizer = AdamW(
            self.critic.parameters(),
            lr=self.ppo_params["critic_lr"],
            eps=1e-5,
            weight_decay=self.ppo_params["weight_decay"],
        )
        self.model_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.model_optimizer, gamma=0.999)
        self.critic_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.999)

        self.total_steps = int(np.ceil(self.ppo_params["steps"] / self.ppo_params["batch_size"]))

        if self.ppo_params["adap_kl_ctrl"]:
            self.kl_ctl = AdaptiveKLController(
                self.ppo_params["init_kl_coef"], self.ppo_params["target"], self.ppo_params["horizon"]
            )
        else:
            self.kl_ctl = FixedKLController(self.ppo_params["init_kl_coef"])

    def step(self, input_ids, attention_mask, decoder_output_ids, scores, **kwargs):
        """
        Run a PPO optimisation step.

        args:
            input_ids (List): List of tensors containing the encoded input, shape [query_length]
            attention_mask (List): List of tensors containing the input attention mask, shape [query_length]
            decoder_output_ids (List): List of tensors containing the encoded responses, shape [response_length]
            scores (List): tensor containing the scores, shape [batch_size]

        returns:
            train_stats (dict): a summary of the training statistics
        """

        bs = self.ppo_params["batch_size"]
        assert bs == len(input_ids), f"Batch size ({bs}) does not match number of examples ({len(input_ids)})"

        timing = dict()
        t0 = time.time()

        # response_lengths = [len(r) for r in responses]

        t = time.time()
        logprobs, ref_logprobs, values = self.batched_forward_pass(
            input_ids, attention_mask, decoder_output_ids, **kwargs
        )
        timing["time/ppo/forward_pass"] = time.time() - t

        t = time.time()
        rewards, non_score_reward = self.compute_rewards(scores, logprobs, ref_logprobs)
        timing["time/ppo/compute_rewards"] = time.time() - t

        t = time.time()
        all_stats = []
        idxs = list(range(bs))
        for _ in range(self.ppo_params["ppo_epochs"]):
            random.shuffle(idxs)
            for i in range(bs):
                idx = idxs[i]
                train_stats = self.train_minibatch(
                    logprobs[idx].unsqueeze(0),
                    values[idx].unsqueeze(0),
                    rewards[idx].unsqueeze(0),
                    input_ids[idx].unsqueeze(0),
                    attention_mask[idx].unsqueeze(0),
                    decoder_output_ids[idx].unsqueeze(0),
                    **kwargs,
                )
                all_stats.append(train_stats)
        timing["time/ppo/optimize_step"] = time.time() - t

        self.model_lr_scheduler.step()
        self.critic_lr_scheduler.step()
        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
        )
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t

        self.kl_ctl.update(stats["objective/kl"], self.ppo_params["batch_size"])

        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)
        return stats

    def batched_forward_pass(self, input_ids, attention_mask, decoder_output_ids, **kwargs):
        """Calculate model outputs in multiple batches."""
        bs = self.ppo_params["batch_size"]
        fbs = self.ppo_params["forward_batch_size"]

        all_values = []
        all_logprobs = []
        all_ref_logprobs = []
        for i in range(int(bs / fbs)):
            input_ids_batch = input_ids[i * fbs : (i + 1) * fbs]
            attn_mask_batch = attention_mask[i * fbs : (i + 1) * fbs]
            decoder_output_ids_batch = decoder_output_ids[i * fbs : (i + 1) * fbs]
            # decoder_attn_mask_batch = (decoder_output_ids_batch != self.tokenizer.pad_token_id).int()

            with torch.no_grad():
                actor_output = self.actor(
                    input_ids=input_ids_batch,
                    attention_mask=attn_mask_batch,
                    decoder_input_ids=decoder_output_ids_batch,
                    return_dict=True,
                    **kwargs
                    # decoder_attention_mask=decoder_attn_mask_batch,
                )
                logits, decoder_last_hidden_states = actor_output.logits, actor_output.decoder_last_hidden_state
                v = self.critic(decoder_last_hidden_states).squeeze(-1)
                ref_logits = self.ref_model(
                    input_ids=input_ids_batch,
                    attention_mask=attn_mask_batch,
                    decoder_input_ids=decoder_output_ids_batch,
                    return_dict=True,
                    **kwargs
                    # decoder_attention_mask=decoder_attn_mask_batch,
                ).logits
            logprobs = logprobs_from_logits(logits[:, :-1, :], decoder_output_ids_batch[:, 1:])
            ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], decoder_output_ids_batch[:, 1:])
            for j in range(fbs):
                gen_length = (decoder_output_ids_batch[j] != self.tokenizer.pad_token_id).int().sum()
                all_values.append(v[j, :-1])
                all_logprobs.append(logprobs[j, :])
                all_ref_logprobs.append(ref_logprobs[j, :])

        return all_logprobs, all_ref_logprobs, all_values

    def train_minibatch(self, logprobs, values, rewards, input_ids, attention_mask, decoder_output_ids, **kwargs):
        """Train one PPO minibatch"""
        total_loss, actor_loss, critic_loss, entropy_loss, train_stats = self.loss(
            logprobs, values, rewards, input_ids, attention_mask, decoder_output_ids, **kwargs
        )
        # l = loss_p + loss_v
        self.model_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        (self.ppo_params["vf_coef"] * critic_loss).backward(retain_graph=True)
        if self.ppo_params["clip_gradients"]:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.ppo_params["clip_gradient_value"])
        self.critic_optimizer.step()
        (actor_loss - self.ppo_params["entropy_beta"] * entropy_loss).backward()
        if self.ppo_params["clip_gradients"]:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.ppo_params["clip_gradient_value"])
        self.model_optimizer.step()
        return train_stats

    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """Compute per token rewards from scores and KL-penalty."""
        rewards, non_score_rewards = [], []
        for score, logprob, ref_logprob in zip(scores, logprobs, ref_logprobs):
            kl = logprob - ref_logprob
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            reward[-1] += score
            rewards.append(reward)
        return rewards, non_score_rewards

    def loss(self, old_logprobs, values, rewards, encoder_ids, attention_mask, decoder_ids, **kwargs):
        """Calculate policy and value losses."""
        lastgaelam = 0
        advantages_reversed = []
        # decoder_attn_mask = (decoder_ids != self.tokenizer.pad_token_id).int()
        for t in reversed(range(values.shape[1])):
            nextvalues = values[:, t + 1] if t < values.shape[1] - 1 else 0.0
            delta = rewards[:, t] + self.ppo_params["gamma"] * nextvalues - values[:, t]
            lastgaelam = delta + self.ppo_params["gamma"] * self.ppo_params["lam"] * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = whiten(advantages)
        advantages = advantages.detach()

        actor_output = self.actor(
            input_ids=encoder_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_ids,
            return_dict=True,
            **kwargs
            # decoder_attention_mask=decoder_attn_mask,
        )
        logits, decoder_last_hidden_states = actor_output.logits, actor_output.decoder_last_hidden_state
        vpred = self.critic(decoder_last_hidden_states).squeeze(-1)
        # logprob = logprobs_from_logits(logits[:, :-1, :], model_input[:, 1:])
        logprob = logprobs_from_logits(logits[:, :-1, :], decoder_ids[:, 1:])

        # only the generation part of the values/logprobs is needed
        # logprob, vpred = logprob[:, -gen_len:], vpred[:, -gen_len - 1 : -1]
        gen_length = (decoder_ids != self.tokenizer.pad_token_id).int().sum()
        logprob, vpred = logprob[:, :], vpred[:, :-1]

        # critic_loss = 0.5 * torch.mean(torch.square(vpred - returns))
        # vf_clipfrac = 0.0
        vpredclipped = clip_by_value(
            vpred, values - self.ppo_params["cliprange_value"], values + self.ppo_params["cliprange_value"]
        )
        critic_loss_1 = torch.square(vpred - returns)
        critic_loss_clipped = torch.square(vpredclipped - returns)
        vf_losses2 = torch.square(vpredclipped - returns)
        critic_loss = 0.5 * torch.mean(torch.max(critic_loss_1, critic_loss_clipped))
        vf_clipfrac = torch.mean(torch.gt(vf_losses2, critic_loss_1).double())

        ratio = torch.exp(logprob - old_logprobs)
        actor_losses = -advantages * ratio
        actor_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - self.ppo_params["cliprange"], 1.0 + self.ppo_params["cliprange"]
        )
        actor_loss = torch.mean(torch.max(actor_losses, actor_losses2))
        pg_clipfrac = torch.mean(torch.gt(actor_losses2, actor_losses).double())

        # Adding an entropy term is optional, but it encourages our actor model to explore different policies and the degree
        # to which we want to experiment can be controlled by an entropy beta parameter.
        # entropy_loss = torch.mean(-(torch.exp(logprob) * logprob))
        entropy_loss = torch.mean(entropy_from_logits(logits))

        total_loss = (
            actor_loss + self.ppo_params["vf_coef"] * critic_loss - self.ppo_params["entropy_beta"] * entropy_loss
        )

        approxkl = 0.5 * torch.mean((logprob - old_logprobs) ** 2)
        policykl = torch.mean(logprob - old_logprobs)
        return_mean, return_var = torch.mean(returns), torch.var(returns)
        value_mean, value_var = torch.mean(values), torch.var(values)

        stats = dict(
            loss=dict(
                policy=actor_loss,
                value=self.ppo_params["vf_coef"] * critic_loss,
                entropy_loss=self.ppo_params["entropy_beta"] * entropy_loss,
                total=total_loss,
            ),
            policy=dict(
                entropy=entropy_loss,
                approxkl=approxkl,
                policykl=policykl,
                clipfrac=pg_clipfrac,
                advantages=advantages,
                advantages_mean=torch.mean(advantages),
                ratio=ratio,
            ),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(
                vpred=torch.mean(vpred),
                error=torch.mean((vpred - returns) ** 2),
                clipfrac=vf_clipfrac,
                mean=value_mean,
                var=value_var,
            ),
        )
        return (
            total_loss,
            actor_loss,
            critic_loss,
            entropy_loss,
            flatten_dict(stats),
        )

    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        kl_list = [logprobs - ref_logprobs for logprobs, ref_logprobs in zip(data["logprobs"], data["ref_logprobs"])]
        mean_kl = torch.mean(torch.stack([torch.sum(kl) for kl in kl_list]))
        mean_entropy = torch.mean(torch.stack([torch.sum(-log_probs) for log_probs in data["logprobs"]]))
        mean_non_score_reward = torch.mean(
            torch.stack([torch.sum(non_score_reward) for non_score_reward in data["non_score_reward"]])
        )
        stats = {
            "objective/kl": mean_kl,
            # "objective/kl_dist": wandb.Histogram(kl_list),
            # "objective/logprobs": wandb.Histogram(data["logprobs"]),
            # "objective/ref_logprobs": wandb.Histogram(data["ref_logprobs"]),
            "objective/kl_coef": kl_coef,
            "objective/entropy": mean_entropy,
            "ppo/mean_non_score_reward": mean_non_score_reward,
            "train/model_lr": self.model_lr_scheduler.get_last_lr()[0],
            "train/critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
        }

        for k, v in data["train_stats"].items():
            stats[f"ppo/{k}"] = torch.mean(v, axis=0)
        stats["ppo/val/var_explained"] = 1 - stats["ppo/val/error"] / stats["ppo/returns/var"]
        return stats
