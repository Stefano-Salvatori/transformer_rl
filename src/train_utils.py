import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from transformers import top_k_top_p_filtering
import torch
import torch.nn.functional as F


class WarmupCosineLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, warm_up=0, T_max=10, start_ratio=0.1):
        """Description: - get warmup consine lr scheduler
        Arguments:
            - optimizer: (torch.optim.*), torch optimizer
            - lr_min: (float), minimum learning rate
            - lr_max: (float), maximum learning rate
            - warm_up: (int), warm_up epoch or iteration
            - T_max: (int), maximum epoch or iteration
            - start_ratio: (float), to control epoch 0 lr, if ratio=0, then epoch 0 lr is lr_min
        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warm_up = warm_up
        self.T_max = T_max
        self.start_ratio = start_ratio
        self.cur = 0  # current epoch or iteration

        super().__init__(optimizer, -1)

    def get_lr(self):
        if (self.warm_up == 0) & (self.cur == 0):
            lr = self.lr_max
        elif (self.warm_up != 0) & (self.cur <= self.warm_up):
            if self.cur == 0:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur + self.start_ratio) / self.warm_up
            else:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur) / self.warm_up
                # print(f'{self.cur} -> {lr}')
        else:
            # this works fine
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (
                np.cos((self.cur - self.warm_up) / (self.T_max - self.warm_up) * np.pi) + 1
            )

        self.cur += 1

        return [lr for base_lr in self.base_lrs]


def respond_to_batch(model, input_ids, attention_mask, max_text_length, top_k=0, top_p=1.0, **kwargs):
    """Sample text from language model."""
    batch_size = input_ids.shape[0]
    device = input_ids.device
    decoder_start_token_id = model.config.decoder_start_token_id
    decoder_eos_token_id = model.config.eos_token_id
    decoder_pad_token_id = model.config.pad_token_id
    decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id
    encoder_outputs = None
    for _ in range(max_text_length - 1):
        # Get Logits
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            return_dict=True,
            use_cache=False,
            **kwargs
        )
        # Save encoder outputs that are fixed and can be reused
        if encoder_outputs is None:
            encoder_outputs = (
                outputs.encoder_last_hidden_state,
                outputs.encoder_hidden_states,
                outputs.encoder_attentions,
            )
        next_token_logits = outputs.logits[:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(-1)], dim=-1)
        # if ((next_token == decoder_eos_token_id) | (next_token == decoder_pad_token_id)).all():
        #    break

    return decoder_input_ids  # [:, -txt_len:]
