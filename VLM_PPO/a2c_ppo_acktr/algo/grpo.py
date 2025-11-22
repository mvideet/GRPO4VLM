import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import accelerate


class GRPO():
    def __init__(self,
                 policy,
                 reference_model,
                 optimizer,
                 accelerator,
                 clip_param,
                 ppo_epoch,
                 mini_batch_size,
                 value_loss_coef,
                 entropy_coef,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 num_generations = 10,
                 num_iterations = 5,
                 beta = 0.1):

        #all GRPO parameters
        self.policy = policy
        self.num_generations = num_generations
        self.beta = beta
        self.num_iterations = num_iterations
        if self.beta == 0.0:
            self.ref_model = None
        else:
            self.ref_model = reference_model
        self.mini_batch_size = mini_batch_size

        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param

        self.ppo_epoch = ppo_epoch
        self.optimizer = optimizer
        self.accelerator = accelerator

    def update(self, rollouts):
        """
        Update policy using GRPO (Group Relative Policy Optimization).
        
        Args:
            rollouts: GRPORolloutStorage with shapes:
                - obs: [1, num_processes, *obs_shape]
                - output_ids: [1, num_processes, num_generations, 2*max_new_tokens]
                - rewards: [1, num_processes, num_generations, 1]
                - action_log_probs: [1, num_processes, num_generations, max_tokens] (per-token)
                - token_masks: [1, num_processes, num_generations, max_tokens]
        """
        # Compute group-relative advantages
        # Shape: [1, num_processes, num_generations, 1]
        advantages = rollouts.compute_group_relative_advantages()
        
        # Normalize advantages globally (optional, but helps with training stability)
        advantages_flat = advantages.view(-1)
        advantages_normalized = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-5)
        advantages = advantages_normalized.view(advantages.shape)

        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_step = 0
        self.policy.train()
        
        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                    advantages, self.mini_batch_size)
            for sample in data_generator:
                with self.accelerator.accumulate(self.policy):
                    grad_step += 1
                   
                    obs_batch, output_ids_batch, actions_batch, \
                    masks_batch, old_action_log_probs_batch, \
                    token_masks_batch, adv_targ = sample
                    
                    # Reshape for batch processing: flatten questions and generations
                    # [num_questions * G, ...]
                    num_questions, G, max_tokens = old_action_log_probs_batch.shape
                    obs_batch_flat = obs_batch.unsqueeze(1).expand(-1, G, *obs_batch.shape[1:]).contiguous()
                    obs_batch_flat = obs_batch_flat.view(num_questions * G, *obs_batch.shape[1:])
                    output_ids_batch_flat = output_ids_batch.view(num_questions * G, -1)

                    action_log_probs = self.policy.evaluate_actions(
                        obs_batch_flat, output_ids_batch_flat)
                    
                    if torch.isnan(action_log_probs).any():
                        continue
                    
                    action_log_probs = action_log_probs.view(num_questions, G, max_tokens)
                    
                    old_action_log_probs_batch = old_action_log_probs_batch.to(action_log_probs.device)
                    adv_targ = adv_targ.to(action_log_probs.device)
                    token_masks_batch = token_masks_batch.to(action_log_probs.device)
                    
                    # Sequence-level clipping: sum log probs over tokens to get sequence-level log probs
                    # Shape: [num_questions, G, max_tokens] -> [num_questions, G]
                    masked_new_log_probs = (action_log_probs * token_masks_batch).sum(dim=2)
                    masked_old_log_probs = (old_action_log_probs_batch * token_masks_batch).sum(dim=2)
                    
                    # Compute sequence-level policy ratio
                    # Shape: [num_questions, G]
                    ratio_seq = torch.exp(masked_new_log_probs - masked_old_log_probs)
                    
                    # Clip at sequence level
                    # Shape: [num_questions, G]
                    ratio_clipped = torch.clamp(ratio_seq, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    
                    # PPO clipped objective at sequence level
                    # adv_targ: [num_questions, G, 1] -> [num_questions, G]
                    adv_targ_squeezed = adv_targ.squeeze(-1)
                    surr1 = ratio_seq * adv_targ_squeezed
                    surr2 = ratio_clipped * adv_targ_squeezed
                    
                    # Sequence-level loss (one per completion)
                    # Shape: [num_questions, G]
                    seq_loss = -torch.min(surr1, surr2)
                    
                    # Average over completions and questions
                    action_loss = seq_loss.mean()
                    
                    # Add KL divergence penalty if beta > 0
                    if self.beta > 0 and self.ref_model is not None:
                        # Get reference model's log probabilities
                        with torch.no_grad():
                            ref_action_log_probs = self.ref_model.evaluate_actions(
                                obs_batch_flat, output_ids_batch_flat)
                            ref_action_log_probs = ref_action_log_probs.view(num_questions, G, max_tokens)
                            ref_action_log_probs = ref_action_log_probs.to(action_log_probs.device)
                        
                        # Compute KL divergence per token unbiased estimator
                        # KL(p||q) = exp(log_p - log_q) - (log_p - log_q) - 1
                        per_token_kl = torch.exp(ref_action_log_probs - action_log_probs) - \
                                      (ref_action_log_probs - action_log_probs) - 1
                        
                        # Average KL over tokens and completions
                        masked_kl = (per_token_kl * token_masks_batch).sum(dim=2) / (token_masks_batch.sum(dim=2) + 1e-8)
                        kl_penalty = self.beta * masked_kl.mean()
                        action_loss = action_loss + kl_penalty
                        dist_entropy_epoch += kl_penalty.item()

                    try:
                        assert not torch.isnan(action_loss), "action_loss is nan"
                    except:
                        print("action loss is nan")
                        exit(1)
                    
                    loss = action_loss
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.policy.parameters(),
                            self.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    action_loss_epoch += action_loss.item()

        if grad_step > 0:
            action_loss_epoch /= grad_step
            dist_entropy_epoch /= grad_step
        else:
            action_loss_epoch = 0
            dist_entropy_epoch = 0

        return 0.0, action_loss_epoch, dist_entropy_epoch  # No value loss for GRPO
