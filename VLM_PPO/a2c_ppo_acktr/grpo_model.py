import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate, grpo_llava_generate, grpo_llava_evaluate
import torch.nn.init as init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)



class GRPOVLMPolicy(nn.Module):
    def __init__(self, tokenizer,
                image_processor,
                base,
                args,
                INPUT_IDS,
                projection_f,
                num_samples, #number of times we run inference
                base_kwargs=None):
        """
        projection_f: the postprocessing function to parse text action
        """
        super(GRPOVLMPolicy, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.base = base
        self.INPUT_IDS = INPUT_IDS
        self.projection_f = projection_f # this is the postprocessing function to parse text action
        self.num_samples = num_samples

    def process_obs(self, obs):
        #process the observation with the image processor
        processed_images = obs
        return self.image_processor.preprocess(processed_images, return_tensors='pt')['pixel_values'].to(dtype=self.base.dtype)

    def act(self, inputs, deterministic=False, INPUT_IDS=None):
        image_tensor = self.process_obs(inputs)
        if INPUT_IDS is None:
            INPUT_IDS = self.INPUT_IDS
        output_ids_list, text_action_list, action_tokens_log_prob_list = grpo_llava_generate(base = self.base,
                                                    tokenizer = self.tokenizer,
                                        input_ids = INPUT_IDS,
                                        image_tensor = image_tensor,
                                        args = self.args,
                                        num_samples = self.num_samples)
        action_list = []
        for text_action in text_action_list:
            action_list.append(self.projection_f(text_action))
        return output_ids_list, action_list, action_tokens_log_prob_list #return the log probs of every token

    #returns just the action log probs
    def evaluate_actions(self, inputs, output_ids, INPUT_IDS=None):
        image_tensor = self.process_obs(inputs)
        if INPUT_IDS is None:
            INPUT_IDS = self.INPUT_IDS
        action_log_prob= grpo_llava_evaluate(base = self.base,
                                        input_ids = INPUT_IDS,
                                        output_ids = output_ids,
                                        image_tensor = image_tensor,
                                        temperature = self.args.temperature,
                                        thought_prob_coef = self.args.thought_prob_coef)
        return action_log_prob
