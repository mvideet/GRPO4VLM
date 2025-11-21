import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class GRPORolloutStorage(object):
    def __init__(self, num_processes, obs_shape, action_space, max_new_tokens, num_generations=1):
        self.num_generations = num_generations
        self.num_processes = num_processes
        self.num_steps = 1
        self.obs = torch.zeros(1, num_processes, *obs_shape)
        
        self.output_ids = torch.zeros(
            1, num_processes, num_generations, 2*max_new_tokens).long()
        self.rewards = torch.zeros(1, num_processes, num_generations, 1)
    
        max_tokens = 2 * max_new_tokens - 2
        self.action_log_probs = torch.zeros(1, num_processes, num_generations, max_tokens)
        
        self.token_masks = torch.zeros(1, num_processes, num_generations, max_tokens)
        # Note: In episodic GRPO, actions are not really used for training (we train on token sequences).
        # But if needed for compatibility, we store one action per generation.
        # Each completion is a full action sequence (as tokens in output_ids), so this is just a placeholder.
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(1, num_processes, num_generations, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        
        self.masks = torch.ones(1, num_processes, 1)
        self.bad_masks = torch.ones(1, num_processes, 1)
        
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.output_ids = self.output_ids.to(device)
        self.rewards = self.rewards.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.token_masks = self.token_masks.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, output_ids, actions, action_log_probs, rewards, masks=None, bad_masks=None, token_masks=None):
               #obs is the initial observation, output_ids is the generated token sequence
               #actions are the actions that we took and so oon
        """
        Insert data into storage for episodic GRPO.
        
        Args:
            obs: [num_processes, *obs_shape] - initial observations
            output_ids: [num_processes, num_generations, 2*max_new_tokens] - generated token sequences
                       Each sequence is a FULL action sequence for the entire episode
            actions: [num_processes, num_generations, action_shape] - placeholder, not used for training
                     (we train on token sequences in output_ids directly)
            action_log_probs: [num_processes, num_generations, max_tokens] - per-token log probs
            rewards: [num_processes, num_generations, 1] - FINAL episode rewards per completion
                     (after executing the full action sequence)
            masks: [num_processes, 1] - not used for episodic GRPO (always 1), kept for compatibility
            bad_masks: [num_processes, 1] - not used for episodic GRPO, kept for compatibility
            token_masks: [num_processes, num_generations, max_tokens] - token padding masks
        """
        self.obs[0].copy_(obs)
        self.output_ids[0].copy_(output_ids)
        if actions is not None:
            self.actions[0].copy_(actions)
        self.action_log_probs[0].copy_(action_log_probs)
        self.rewards[0].copy_(rewards)
        if masks is not None:
            self.masks[0].copy_(masks)
        if bad_masks is not None:
            self.bad_masks[0].copy_(bad_masks)
        if token_masks is not None:
            self.token_masks[0].copy_(token_masks)
        self.step = 0

    def compute_group_relative_advantages(self):
        rewards_flat = self.rewards[0]
        mean_grouped_rewards = rewards_flat.mean(dim=1, keepdim=True)
        std_grouped_rewards = rewards_flat.std(dim=1, keepdim=True)
        #getting the advantages
        advantages = (rewards_flat - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        advantages = advantages.unsqueeze(0)
        
        return advantages

    def feed_forward_generator(self,
                               advantages=None,
                               mini_batch_size=None,
                               use_group_relative=True):
        """
        Generate mini-batches for training (episodic GRPO).
        
        IMPORTANT: For GRPO, we need ALL G generations for each question together.
        The objective function averages over all G: 1/G * Σ_{i=1}^G.
        Group-relative advantages are computed per-group (per question).
        
        This method batches by QUESTION (process), keeping all G generations together.
        
        Args:
            advantages: Pre-computed advantages [1, num_processes, num_generations, 1]
                       If None and use_group_relative=True, computes group-relative advantages
            mini_batch_size: Number of QUESTIONS (not completions) per mini-batch
                           Each question contributes G completions, so actual batch size = mini_batch_size * G
            use_group_relative: If True, use group-relative advantages (GRPO style)
                               If False, use standard advantages (PPO style)
        
        Yields:
            Batches of data for training, where each batch contains all G generations for each question
        """
        num_processes = self.num_processes
        
        if advantages is None:
            if use_group_relative:
                advantages = self.compute_group_relative_advantages()
            else:
                advantages = self.rewards[0]  # [num_processes, num_generations, 1]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                advantages = advantages.unsqueeze(0)  # [1, num_processes, num_generations, 1]
        
        if mini_batch_size is None:
            mini_batch_size = num_processes  # Process all questions at once
        
        question_indices = torch.randperm(num_processes).tolist()
        
        for i in range(0, num_processes, mini_batch_size):
            batch_question_indices = question_indices[i:i+mini_batch_size]
        
            obs_batch = self.obs[0][batch_question_indices]  # [num_questions, *obs_shape]
            
            output_ids_batch = self.output_ids[0, batch_question_indices, :]  # [num_questions, G, 2*max_new_tokens]
            actions_batch = self.actions[0, batch_question_indices, :]  # [num_questions, G, action_shape]
            old_action_log_probs_batch = self.action_log_probs[0, batch_question_indices, :]  # [num_questions, G, max_tokens]
            token_masks_batch = self.token_masks[0, batch_question_indices, :]  # [num_questions, G, max_tokens]
            masks_batch = self.masks[0, batch_question_indices]
            adv_targ = advantages[0, batch_question_indices, :]  # [num_questions, G, 1]
            
            yield obs_batch, output_ids_batch, actions_batch, \
                masks_batch, old_action_log_probs_batch, \
                token_masks_batch, adv_targ
