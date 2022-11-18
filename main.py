from src.critic import CriticModel
from src.reward_model import (
    AMRGraphRewardModel,
    LetterCounterRewardModel,
    ReadablityRewardModel,
    RougeRewardModel,
    TextClassifierRewardModel,
)
from src.transformer_rl import ExperimentConfig, TransformerReinforcementLearning
from src.actor_critic_bart import ActorBartModel
import torch
from tqdm import tqdm

tqdm.pandas()

from datasets import load_dataset
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers.utils import logging

# logging.set_verbosity_info()


def main():
    config = ExperimentConfig(
        project_name="bart-ctrl",
        experiment_name="transformer_rl",
        team="team-uni",
        main_model_pretrained_weights="ainize/bart-base-cnn",
        tokenizer_pretrained="ainize/bart-base-cnn",
        steps=8192,
        log_reference_output=15,
        batch_size=1,
        forward_batch_size=1,
        ppo_epochs=4,
        txt_in_len=1024,
        max_txt_out_len=256,
        min_txt_out_len=None,
        lr=9.07e-6,  # 1.41e-5, 7.07e-6
        critic_lr=9.07e-6,
        weight_decay=1e-4,
        adap_kl_ctrl=False,
        init_kl_coef=0.15,  # 0.2
        target=6.0,  # 6
        horizon=10000,
        gamma=0.5,
        lam=0.95,
        cliprange=0.3,  # 0.2
        cliprange_value=0.1,  # 0.2
        vf_coef=0.2,
        entropy_beta=0.0,
        seed=42,
        clip_gradients=True,
        clip_gradient_value=0.5,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load cnn  dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train", cache_dir="/home/salvatori/datasets/CNN")
    dataset.set_format("pandas")
    df_dataset = dataset[:]
    # keep only a small fraction of the dataset
    # df_dataset = df_dataset.sample(frac=1.0, random_state=config.seed)
    df_dataset = df_dataset.rename(columns={"article": TransformerReinforcementLearning.INPUT_COLUMN_NAME})
    df_dataset = df_dataset.rename(columns={"highlights": TransformerReinforcementLearning.TARGET_COLUMN_NAME})

    # Initialize reward model
    # reward_model = TextClassifierRewardModel(
    #    pretrained_weights="jy46604790/Fake-News-Bert-Detect", score_class=1, device=device
    # )
    reward_model = LetterCounterRewardModel("k")
    # reward_model = RougeRewardModel()
    # reward_model = ReadablityRewardModel()
    # reward_model = ResponseLengthModel()
    reward_model = AMRGraphRewardModel(
        checkpoint="/home/salvatori/transformer_rl/spring/AMR3.parsing.pt", device=device
    )
    # Create main model
    bart_model = ActorBartModel.from_pretrained(config.main_model_pretrained_weights)
    bart_model_ref = BartForConditionalGeneration.from_pretrained(config.main_model_pretrained_weights)
    bart_model_ref.eval()
    critic = CriticModel(bart_model.config)
    bart_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_pretrained)

    transformer_rl = TransformerReinforcementLearning(
        transformer_model=bart_model,
        reference_model=bart_model_ref,
        critic=critic,
        tokenizer=bart_tokenizer,
        reward_model=reward_model,
        dataset=df_dataset,
        config=config,
        device=device,
    )
    transformer_rl.run()

    print(transformer_rl.evaluate_sample(config.batch_size, log=True))


if __name__ == "__main__":
    main()
