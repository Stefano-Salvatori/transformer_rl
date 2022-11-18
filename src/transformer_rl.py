from dataclasses import dataclass
import dataclasses
import time
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
import pandas as pd
import torch
import wandb
import math
import torch
from typing import Callable
from src.train_utils import respond_to_batch
from src.trl.ppo import PPOTrainer



@dataclass
class ExperimentConfig:
    project_name: str
    experiment_name: str
    team: str
    main_model_pretrained_weights: str
    tokenizer_pretrained: str
    save_folder: str
    save_every: int
    steps: int
    batch_size: int
    forward_batch_size: int
    minibatch_size: int
    ppo_epochs: int
    txt_in_len: int
    max_txt_out_len: int
    min_txt_out_len: int
    log_reference_output: 10
    lr: float = 1.41e-5
    critic_lr: float = 1.41e-6
    lr_decay_gamma: float = 0.99
    weight_decay: float = 1e-4
    adap_kl_ctrl: bool = True
    init_kl_coef: float = 0.2
    target: int = 6
    kl_save_factor: float = 2.0
    horizon: int = 10000
    gamma: int = 1
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    entropy_beta: float = 0.001
    seed: int = 42
    clip_gradients: bool = True
    clip_gradient_value: float = 0.5
    reward_multiplier: float = 1.0

    def save(self, path):
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)
        with open(os.path.join(self.save_folder, path), "w") as f:
            f.write(json_string)


class TransformerReinforcementLearning:
    INPUT_COLUMN_NAME = "input"
    OUTPUT_COLUMN_NAME = "output"
    TARGET_COLUMN_NAME = "target"
    REWARD_COLUMN_NAME = "reward"
    REFERENCE_OUTPUT_COLUMN_NAME = "reference_output"
    REFERENCE_REWARD_COLUMN_NAME = "reference_reward"

    def __init__(
        self,
        transformer_model: PreTrainedModel,
        reference_model: PreTrainedModel,
        critic: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        reward_model,
        dataset: pd.DataFrame,
        config: ExperimentConfig,
        generate_addargs: Callable = lambda _: {},
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.transformer_model = transformer_model.to(device)
        self.reference_model = reference_model.to(device)
        self.critic = critic.to(device)
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.dataset = dataset
        self.generate_addargs = generate_addargs
        self.config = config
        self.device = device

        # np.random.seed(config.seed)

        # init wandb
        wandb.init(name=config.experiment_name, project=config.project_name, entity=config.team, config=config)
        wandb.watch((self.transformer_model, self.critic), log="all")

        assert (
            TransformerReinforcementLearning.INPUT_COLUMN_NAME in self.dataset
        ), f"There must be a column named {TransformerReinforcementLearning.INPUT_COLUMN_NAME}"

        # Initialize trainer
        self.ppo_trainer = PPOTrainer(
            model=self.transformer_model,
            ref_model=self.reference_model,
            critic=self.critic,
            tokenizer=self.tokenizer,
            **dataclasses.asdict(config),
        )

    def run(self) -> None:
        for i in tqdm(range(int(np.ceil(self.config.steps / self.config.batch_size)))):
            # torch.cuda.empty_cache()
            logs = dict()
            data = dict()
            timing = dict()
            t0 = time.time()

            #### get a batch from the dataset
            df_batch = self.dataset.sample(self.config.batch_size)
            data[TransformerReinforcementLearning.INPUT_COLUMN_NAME] = df_batch[
                TransformerReinforcementLearning.INPUT_COLUMN_NAME
            ]

            data[TransformerReinforcementLearning.TARGET_COLUMN_NAME] = df_batch[
                TransformerReinforcementLearning.TARGET_COLUMN_NAME
            ]

            kwargs = self.generate_addargs(df_batch)

            #### get response from the model
            t = time.time()
            # encode input
            encoded = self.tokenizer.batch_encode_plus(
                data[TransformerReinforcementLearning.INPUT_COLUMN_NAME].to_list(),
                padding=True,
                truncation=True,
                max_length=self.config.txt_in_len,
                return_tensors="pt",
            ).to(self.device)
            self.transformer_model.eval()
            response_tensors = self.__get_model_response(
                encoded.input_ids, encoded.attention_mask, model=self.transformer_model, **kwargs
            )
            self.transformer_model.train()
            response_lengths = (response_tensors != self.tokenizer.pad_token_id).int().sum(dim=1)

            data[TransformerReinforcementLearning.OUTPUT_COLUMN_NAME] = self.tokenizer.batch_decode(
                response_tensors, skip_special_tokens=True
            )
            timing["time/get_response"] = time.time() - t

            ### Compute Reward
            t = time.time()
            rewards = self.reward_model.get_reward(
                data[TransformerReinforcementLearning.INPUT_COLUMN_NAME],
                data[TransformerReinforcementLearning.OUTPUT_COLUMN_NAME],
                data[TransformerReinforcementLearning.TARGET_COLUMN_NAME],
            )
            # normalize rewards
            # normalized_rewards = (rewards - rewards.mean()) / rewards.std()
            timing["time/get_reward"] = time.time() - t

            #### Run PPO training
            t = time.time()
            stats = self.ppo_trainer.step(
                encoded.input_ids, encoded.attention_mask, response_tensors, rewards, **kwargs
            )
            timing["time/optimization"] = time.time() - t

            #### Log everything
            if i % self.config.log_reference_output == 0:
                reference_response_tensors = self.__get_model_response(
                    encoded.input_ids, encoded.attention_mask, model=self.reference_model, **kwargs
                )
                reference_response_lengths = (
                    (reference_response_tensors != self.tokenizer.pad_token_id).int().sum(dim=1)
                )
                reference_response_text = self.tokenizer.batch_decode(
                    reference_response_tensors, skip_special_tokens=True
                )
                reference_rewards = self.reward_model.get_reward(
                    data[TransformerReinforcementLearning.INPUT_COLUMN_NAME],
                    reference_response_text,
                    data[TransformerReinforcementLearning.TARGET_COLUMN_NAME],
                )

                logs["misc/reference_response_length"] = wandb.Histogram(
                    reference_response_lengths.cpu().numpy().tolist()
                )
                logs["env/reference_reward_mean"] = torch.mean(reference_rewards).cpu().numpy()
                logs.update(
                    {
                        "reference_output_log": wandb.Table(
                            dataframe=pd.DataFrame(
                                {
                                    TransformerReinforcementLearning.INPUT_COLUMN_NAME: data[
                                        TransformerReinforcementLearning.INPUT_COLUMN_NAME
                                    ],
                                    TransformerReinforcementLearning.OUTPUT_COLUMN_NAME: data[
                                        TransformerReinforcementLearning.OUTPUT_COLUMN_NAME
                                    ],
                                    TransformerReinforcementLearning.TARGET_COLUMN_NAME: data[
                                        TransformerReinforcementLearning.TARGET_COLUMN_NAME
                                    ],
                                    TransformerReinforcementLearning.REFERENCE_OUTPUT_COLUMN_NAME: reference_response_text,
                                    TransformerReinforcementLearning.REFERENCE_REWARD_COLUMN_NAME: reference_rewards.cpu().tolist(),
                                    TransformerReinforcementLearning.REWARD_COLUMN_NAME: rewards.cpu().tolist(),
                                }
                            )
                        )
                    }
                )

            timing["time/epoch"] = time.time() - t0

            logs.update(
                {
                    "game_log": wandb.Table(
                        dataframe=pd.DataFrame(
                            {
                                TransformerReinforcementLearning.INPUT_COLUMN_NAME: data[
                                    TransformerReinforcementLearning.INPUT_COLUMN_NAME
                                ],
                                TransformerReinforcementLearning.OUTPUT_COLUMN_NAME: data[
                                    TransformerReinforcementLearning.OUTPUT_COLUMN_NAME
                                ],
                                TransformerReinforcementLearning.REWARD_COLUMN_NAME: rewards.cpu().tolist(),
                            }
                        )
                    )
                }
            )
            logs.update(timing)
            logs.update(stats)
            logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy()
            logs["env/reward_std"] = torch.std(rewards).cpu().numpy()
            logs["env/reward_dist"] = wandb.Histogram(rewards.cpu().numpy().tolist())
            logs["misc/response_length"] = wandb.Histogram(response_lengths.cpu().numpy().tolist())
            wandb.log(logs)

    def evaluate_sample(self, sample_size: int, log: bool = False):
        output_data = dict()
        df_batch = self.dataset.sample(sample_size)
        # encode input
        encoded = self.tokenizer.batch_encode_plus(
            df_batch[TransformerReinforcementLearning.INPUT_COLUMN_NAME].to_list(),
            padding=True,
            truncation=True,
            max_length=self.config.txt_in_len,
            return_tensors="pt",
        ).to(self.device)
        kwargs = self.generate_addargs(df_batch)
        with torch.no_grad():
            trained_model_output = self.__get_model_response(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                model=self.transformer_model,
                **kwargs,
            )
            reference_model_output = self.__get_model_response(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                model=self.reference_model,
                **kwargs,
            )
        #### decode responses
        output_data["response_before"] = self.tokenizer.batch_decode(
            reference_model_output.cpu().numpy(), skip_special_tokens=True
        )
        output_data["response_after"] = self.tokenizer.batch_decode(
            trained_model_output.cpu().numpy(), skip_special_tokens=True
        )

        #### reward query/response pairs before/after
        output_data["reward_before"] = (
            self.reward_model.get_reward(df_batch["input"], output_data["response_before"]).cpu().numpy()
        )
        output_data["reward_after"] = (
            self.reward_model.get_reward(df_batch["input"], output_data["response_after"]).cpu().numpy()
        )

        table = pd.DataFrame(output_data)
        if log:
            wandb.log({"evaluation_table": wandb.Table(dataframe=table)})

        # store results in a dataframe
        return table

    def __get_model_response(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, model: PreTrainedModel, **kwargs
    ):
        response_tensors = []
        batch_size = input_ids.shape[0]

        # makes sure that we do at least 1 forward step (if the batch has less items than the forward batch size)
        num_forward_steps = max(1, math.ceil(batch_size / self.config.forward_batch_size))

        for i in range(num_forward_steps):
            start = i * self.config.forward_batch_size
            end = (i + 1) * self.config.forward_batch_size
            with torch.no_grad():
                response = respond_to_batch(
                    model,
                    input_ids=input_ids[start:end],
                    attention_mask=attention_mask[start:end],
                    max_text_length=self.config.max_txt_out_len,
                    **kwargs,
                )
            # response = model.generate(
            #     input_ids=input_ids[start:end],
            #     attention_mask=attention_mask[start:end],
            #     bos_token_id=model.config.bos_token_id,
            #     eos_token_id=model.config.eos_token_id,
            #     pad_token_id=model.config.pad_token_id,
            #     do_sample=True,
            #     max_length=self.config.max_txt_out_len,
            #     min_length=self.config.min_txt_out_len,
            #     top_p=1,
            #     top_k=0,
            # )
            # pad response to max length
            # missing = self.config.max_txt_out_len - response.shape[1]
            # if missing > 0:
            #    response = torch.nn.functional.pad(response, (0, missing), value=model.config.pad_token_id)
            response_tensors.append(response)
        # concatenate minibatches and return
        return torch.cat(response_tensors)
