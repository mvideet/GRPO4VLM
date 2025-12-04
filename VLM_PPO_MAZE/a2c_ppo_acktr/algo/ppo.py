import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import accelerate


class PPO():
    def __init__(self,
                 actor_critic,
                 optimizer,
                 accelerator,
                 clip_param,
                 ppo_epoch,
                 mini_batch_size,
                 value_loss_coef,
                 entropy_coef,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.mini_batch_size = mini_batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param

        self.ppo_epoch = ppo_epoch

        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optimizer
        self.accelerator = accelerator

    def update(self, rollouts):
        # Check for NaN/Inf in returns and value_preds BEFORE computing advantages
        if torch.isnan(rollouts.returns[:-1]).any() or torch.isinf(rollouts.returns[:-1]).any():
            print(f"ERROR: NaN/Inf detected in rollouts.returns[:-1]")
            print(f"Returns stats: min={rollouts.returns[:-1].min()}, max={rollouts.returns[:-1].max()}, mean={rollouts.returns[:-1].mean()}")
            print(f"NaN count: {torch.isnan(rollouts.returns[:-1]).sum()}, Inf count: {torch.isinf(rollouts.returns[:-1]).sum()}")
        
        if torch.isnan(rollouts.value_preds[:-1]).any() or torch.isinf(rollouts.value_preds[:-1]).any():
            print(f"ERROR: NaN/Inf detected in rollouts.value_preds[:-1]")
            print(f"Value_preds stats: min={rollouts.value_preds[:-1].min()}, max={rollouts.value_preds[:-1].max()}, mean={rollouts.value_preds[:-1].mean()}")
            print(f"NaN count: {torch.isnan(rollouts.value_preds[:-1]).sum()}, Inf count: {torch.isinf(rollouts.value_preds[:-1]).sum()}")
        
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = advantages.detach()  # Detach to free computation graph
        
        # Check for NaN/Inf in advantages AFTER computation
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            print(f"ERROR: NaN/Inf detected in advantages after computation")
            print(f"Advantages stats: min={advantages.min()}, max={advantages.max()}, mean={advantages.mean()}, std={advantages.std()}")
            print(f"NaN count: {torch.isnan(advantages).sum()}, Inf count: {torch.isinf(advantages).sum()}")
            # Replace NaN/Inf with zeros to prevent training crash
            advantages = torch.where(torch.isnan(advantages) | torch.isinf(advantages), 
                                    torch.zeros_like(advantages), advantages)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_step = 0
        self.actor_critic.train()
        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                    advantages, self.mini_batch_size)
            for sample in data_generator:
                with self.accelerator.accumulate(self.actor_critic):
                    obs_batch, output_ids_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                            adv_targ = sample
                    
                    # Normalize advantages per batch (subtract mean, divide by std)
                    if adv_targ is not None:
                        adv_mean = adv_targ.mean()
                        # Use unbiased=False to handle single-element batches correctly
                        # For single element, std should be 0, not NaN
                        adv_std = adv_targ.std(unbiased=False)
                        
                        # Check for NaN/Inf in advantages before normalization
                        if torch.isnan(adv_mean) or torch.isinf(adv_mean):
                            print(f"Warning: NaN/Inf in advantage mean, skipping batch. Mean: {adv_mean}")
                            continue
                        
                        # Handle case where std is NaN (single element or all same values)
                        if torch.isnan(adv_std) or adv_std < 1e-5:
                            # If std is NaN or very small, don't normalize (just center)
                            # This happens when batch size is 1 or all advantages are identical
                            adv_targ = adv_targ - adv_mean
                        else:
                            # Normal case: normalize by std
                            adv_targ = (adv_targ - adv_mean) / adv_std
                    
                    # Check for NaN/Inf in advantages after normalization
                    if adv_targ is not None and (torch.isnan(adv_targ).any() or torch.isinf(adv_targ).any()):
                        print("Warning: NaN/Inf in normalized advantages, skipping batch")
                        continue
                    
                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs = self.actor_critic.evaluate_actions(
                        obs_batch, output_ids_batch)
                    #values and action_log_probs on two different devices!! because they come from two llava
                    
                    # Check for NaN/Inf in model outputs
                    if torch.isnan(action_log_probs).any() or torch.isinf(action_log_probs).any():
                        print("Warning: NaN/Inf in action_log_probs, skipping batch")
                        continue
                    if torch.isnan(values).any() or torch.isinf(values).any():
                        print("Warning: NaN/Inf in values, skipping batch")
                        continue
                    
                    grad_step += 1  # Only increment after NaN check
                    old_action_log_probs_batch = old_action_log_probs_batch.to(action_log_probs.device).view(-1)
                    adv_targ = adv_targ.to(action_log_probs.device)
                    value_preds_batch = value_preds_batch.to(values.device)
                    return_batch = return_batch.to(values.device)
                    
                    # Check for NaN/Inf in returns and value_preds
                    if torch.isnan(return_batch).any() or torch.isinf(return_batch).any():
                        print("Warning: NaN/Inf in return_batch, skipping batch")
                        continue
                    if torch.isnan(value_preds_batch).any() or torch.isinf(value_preds_batch).any():
                        print("Warning: NaN/Inf in value_preds_batch, skipping batch")
                        continue

                    # Clamp action_log_probs difference to avoid extreme ratios
                    log_prob_diff = action_log_probs - old_action_log_probs_batch
                    log_prob_diff = torch.clamp(log_prob_diff, min=-10.0, max=10.0)
                    ratio = torch.exp(log_prob_diff)
                    
                    # Clamp ratio to prevent extreme values
                    ratio = torch.clamp(ratio, min=1e-8, max=100.0)
                    
                    # Clamp advantages to prevent extreme values
                    adv_targ = torch.clamp(adv_targ, min=-100.0, max=100.0)

                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_targ
                    ## adding a ratio clip, inspired by https://github.com/huggingface/trl/blob/5a233546ee48532eaeb24b89b8d0042147574688/trl/trainer/ppo_trainer.py#L1199
                    if torch.any(ratio > 10):
                        action_loss = -surr2.mean()
                    else:
                        action_loss = -torch.min(surr1, surr2).mean()
                    
                    # Check for NaN/Inf in surr1, surr2, action_loss
                    if torch.isnan(surr1).any() or torch.isinf(surr1).any() or \
                       torch.isnan(surr2).any() or torch.isinf(surr2).any() or \
                       torch.isnan(action_loss) or torch.isinf(action_loss):
                        print(f"Warning: NaN/Inf in action loss computation, skipping batch")
                        self.optimizer.zero_grad()
                        continue
                    # Clamp values and returns to prevent extreme differences
                    values = torch.clamp(values, min=-1e6, max=1e6)
                    return_batch = torch.clamp(return_batch, min=-1e6, max=1e6)
                    value_preds_batch = torch.clamp(value_preds_batch, min=-1e6, max=1e6)
                    
                    if self.use_clipped_value_loss:
                        value_pred_clipped = value_preds_batch + \
                            (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                        value_loss = 0.5 * torch.max(value_losses,
                                                    value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * (return_batch - values).pow(2).mean()
                    
                    # Check for NaN/Inf in value_loss
                    if torch.isnan(value_loss) or torch.isinf(value_loss):
                        print(f"Warning: NaN/Inf in value_loss computation, skipping batch")
                        self.optimizer.zero_grad()
                        continue

                    # Check for NaN/Inf in losses before proceeding
                    if torch.isnan(value_loss) or torch.isinf(value_loss) or torch.isnan(action_loss) or torch.isinf(action_loss):
                        print(f"Warning: NaN/Inf in losses (value_loss: {value_loss}, action_loss: {action_loss}), skipping batch")
                        # Reset gradients and continue to next batch
                        self.optimizer.zero_grad()
                        continue
                    loss = value_loss * self.value_loss_coef+action_loss
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.actor_critic.parameters(),
                            self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Compute entropy from action log_probs
                    # For discrete actions, entropy = -sum(p_i * log(p_i))
                    # Since we have log_probs of selected actions: p = exp(log_prob)
                    # entropy ≈ mean(-exp(log_prob) * log_prob) = mean(-p * log(p))
                    # This gives proper entropy: higher when uncertain, lower when confident
                    probs = torch.exp(action_log_probs)
                    # Clamp probs to avoid numerical issues
                    probs = torch.clamp(probs, min=1e-8, max=1.0)
                    # Compute entropy: -p * log(p) = -p * log_prob (since log(p) = log_prob)
                    batch_entropy = (-probs * action_log_probs).mean().item()
                    # Ensure entropy is non-negative
                    batch_entropy = max(0.0, batch_entropy)
                    dist_entropy_epoch += batch_entropy
                    
                    # Extract scalar values before deleting tensors
                    value_loss_epoch += value_loss.item()
                    action_loss_epoch += action_loss.item()
                    
                    # Clear intermediate tensors to free memory (after extracting values)
                    del obs_batch, output_ids_batch, actions_batch
                    del value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch
                    del adv_targ, values, action_log_probs, ratio, surr1, surr2
                    del action_loss, value_loss, loss
                    if self.use_clipped_value_loss:
                        del value_pred_clipped, value_losses, value_losses_clipped

        # Avoid division by zero if no valid gradient steps were taken
        if grad_step > 0:
            value_loss_epoch /= grad_step
            action_loss_epoch /= grad_step
            dist_entropy_epoch /= grad_step
        else:
            # If no valid steps, return zeros (shouldn't happen in normal training)
            print("Warning: No valid gradient steps taken in PPO update!")
            value_loss_epoch = 0.0
            action_loss_epoch = 0.0
            dist_entropy_epoch = 0.0
        
        # Clear advantages and GPU cache after PPO update
        del advantages
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
