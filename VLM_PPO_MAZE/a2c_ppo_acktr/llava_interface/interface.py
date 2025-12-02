import torch
import math
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def llava_generate(value_model, tokenizer, input_ids, image_tensor, args):
    base = value_model.base
    image_tensor = image_tensor.to(base.device, dtype = base.dtype)
    _, _, _, _, inputs_embeds, _ = base.prepare_inputs_labels_for_multimodal(input_ids.to(base.device), None, None, None, None, image_tensor)
    inputs_embeds = inputs_embeds.to(base.device, dtype = base.dtype)
    with torch.inference_mode():
        outputs = base.generate(
        inputs_embeds = inputs_embeds,
        do_sample=True,
        temperature=args.temperature,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        output_scores=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,)
        output_ids = outputs['sequences']
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    padded_output_ids = torch.zeros(output_ids.size(0), 2*args.max_new_tokens).to(dtype=output_ids.dtype, device = output_ids.device)
    padded_output_ids[:, :output_ids.size(1)] = output_ids
    with torch.no_grad():
        values, sum_log_probs, action_tokens_log_prob = llava_evaluate(value_model, input_ids, padded_output_ids, image_tensor, args.temperature, args.thought_prob_coef, tokenizer)
    return values, padded_output_ids, outputs, sum_log_probs, action_tokens_log_prob

def llava_evaluate(value_model, input_ids, output_ids, image_tensor, temperature, thought_prob_coef, tokenizer = None):
    if output_ids.size(0) != 1:
        input_ids = input_ids.broadcast_to(output_ids.size(0), input_ids.size(-1))
    base = value_model.base
    image_tensor = image_tensor.to(base.device, dtype=base.dtype)
    output_ids = output_ids.to(base.device)
    input_ids = input_ids.to(base.device)
    _, _, _, _, inputs_embeds, _ = base.prepare_inputs_labels_for_multimodal(torch.cat([input_ids, output_ids], dim = 1), None, None, None, None, image_tensor)

    #calling the model
    inputs_embeds = inputs_embeds.to(base.device, dtype = base.dtype)
    #omit the first output token
    outputs = base(
        inputs_embeds = inputs_embeds,
        output_hidden_states = True,
        )
    scores = outputs.logits

    input_token_len = inputs_embeds.shape[1] - output_ids.shape[1]
    hidden_states = outputs.hidden_states[-1][:, input_token_len-1]
    values = value_model.value_head(hidden_states)
    scores = scores * (1/temperature)
    scores = scores.to(torch.float32)
    log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
    log_probs = log_probs.to(torch.bfloat16)
    # omit the first outputted id which is decoder start token
    output_ids_mask = (output_ids != 0)[:, 1:]
    ## selected_log_probs counts the log prob of the first token
    selected_log_probs = output_ids_mask*torch.take_along_dim(log_probs[:, input_token_len:-1], output_ids[:,1:].unsqueeze(2), dim = 2).squeeze(2)
    unfolded = output_ids.unfold(dimension=-1, size=3, step=1)
    # the text string '"action":' corresponts to this sequence of tokens: (torch.tensor([[29908,2467,1115]]))
    target = torch.tensor([29908,2467,1115]).to(base.device)
    matches = (unfolded == target).all(dim = -1)
    match_index = matches.nonzero(as_tuple=True)[-1]
    if match_index.shape[0] >= 1:
        ## if we find multuple patterns, we will take the last one, and make it size torch.Size([1])
        match_index = match_index[-1].unsqueeze(0)
    else:
        ## if we don't find any pattern, we will take the last 4 tokens, as "action tokens"
        try:
            match_index = output_ids_mask.nonzero(as_tuple=False)[-4,1]
        except:
            sum_log_prob = torch.tensor([-2]).to(base.device)
            action_tokens_log_prob = torch.tensor([-1]).to(base.device)
            return values, sum_log_prob, action_tokens_log_prob
    ## omitting the second token for calculating log prob, because its logprb is very very small
    thought_log_prob = torch.sum(selected_log_probs[:,1:match_index-1], dim = 1)

    action_tokens_log_prob = torch.sum(selected_log_probs[:,match_index-1:], dim = 1)
    sum_log_prob = thought_prob_coef*thought_log_prob + action_tokens_log_prob
    return values, sum_log_prob, action_tokens_log_prob

def grpo_llava_generate(base, tokenizer, input_ids, image_tensor, args, num_samples):
    """
    Generate multiple completions for GRPO (Group Relative Policy Optimization).
    Returns lists of output_ids, text actions, and action token log probs.
    
    Args:
        base: Base model (LLaVA)
        tokenizer: Tokenizer
        input_ids: Input token IDs [1, seq_len]
        image_tensor: Image tensor [batch_size, *image_shape]
        args: Arguments with generation parameters
        num_samples: Number of generations per observation
    
    Returns:
        output_ids_list: List of [num_processes, 2*max_new_tokens] tensors, one per sample
        text_action_list: List of [num_processes] lists of strings
        action_tokens_log_prob_list: List of [num_processes, max_tokens] tensors
    """
    batch_size = image_tensor.size(0)
    output_ids_list = []
    text_action_list = []
    action_tokens_log_prob_list = []
    
    image_tensor = image_tensor.to(base.device, dtype=base.dtype)
    _, _, _, _, inputs_embeds, _ = base.prepare_inputs_labels_for_multimodal(
        input_ids.to(base.device), None, None, None, None, image_tensor)
    inputs_embeds = inputs_embeds.to(base.device, dtype=base.dtype)
    
    # Get input length for extracting only generated tokens
    input_length = input_ids.size(1)
    
    # Generate num_samples completions for each observation
    for _ in range(num_samples):
        with torch.inference_mode():
            outputs = base.generate(
                inputs_embeds=inputs_embeds,
                do_sample=True,
                temperature=args.temperature,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_scores=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            output_ids = outputs['sequences']  # [batch_size, seq_len]
        
        # Extract only the generated tokens (exclude the prompt)
        # When using inputs_embeds, the output includes the full sequence
        # We need to extract only the generated portion
        generated_ids = output_ids[:, input_length:]  # [batch_size, generated_len]
        
        # Decode only the generated portion to text
        text_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Pad output_ids
        padded_output_ids = torch.zeros(
            batch_size, 2*args.max_new_tokens,
            dtype=output_ids.dtype, device=output_ids.device
        )
        padded_output_ids[:, :output_ids.size(1)] = output_ids
        
        # Compute per-token log probs
        with torch.no_grad():
            action_tokens_log_prob = grpo_llava_evaluate_token_log_probs(
                base, input_ids, padded_output_ids, image_tensor, args.temperature, args.thought_prob_coef
            )
        
        output_ids_list.append(padded_output_ids)
        text_action_list.append(text_outputs)
        action_tokens_log_prob_list.append(action_tokens_log_prob)
    
    return output_ids_list, text_action_list, action_tokens_log_prob_list

def grpo_llava_evaluate_token_log_probs(base, input_ids, output_ids, image_tensor, temperature, thought_prob_coef):
    """
    Compute per-token log probabilities for GRPO evaluation.
    
    Returns:
        action_tokens_log_prob: [batch_size, max_tokens] - per-token log probs
    """
    if output_ids.size(0) != input_ids.size(0):
        input_ids = input_ids.broadcast_to(output_ids.size(0), input_ids.size(-1))
    
    image_tensor = image_tensor.to(base.device, dtype=base.dtype)
    output_ids = output_ids.to(base.device)
    input_ids = input_ids.to(base.device)
    
    _, _, _, _, inputs_embeds, _ = base.prepare_inputs_labels_for_multimodal(
        torch.cat([input_ids, output_ids], dim=1), None, None, None, None, image_tensor
    )
    inputs_embeds = inputs_embeds.to(base.device, dtype=base.dtype)
    
    outputs = base(inputs_embeds=inputs_embeds, output_hidden_states=True)
    scores = outputs.logits
    
    input_token_len = inputs_embeds.shape[1] - output_ids.shape[1]
    scores = scores * (1/temperature)
    scores = scores.to(torch.float32)
    log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
    log_probs = log_probs.to(torch.bfloat16)
    
    # Get log probs for output tokens
    output_ids_mask = (output_ids != 0)[:, 1:]  # Skip first token
    max_tokens = output_ids_mask.size(1)
    
    # Extract log probs for each output token
    selected_log_probs = output_ids_mask * torch.take_along_dim(
        log_probs[:, input_token_len:-1],
        output_ids[:, 1:].unsqueeze(2),
        dim=2
    ).squeeze(2)
    
    # Pad or truncate to max_tokens
    if selected_log_probs.size(1) < max_tokens:
        padding = torch.zeros(
            selected_log_probs.size(0), max_tokens - selected_log_probs.size(1),
            dtype=selected_log_probs.dtype, device=selected_log_probs.device
        )
        selected_log_probs = torch.cat([selected_log_probs, padding], dim=1)
    elif selected_log_probs.size(1) > max_tokens:
        selected_log_probs = selected_log_probs[:, :max_tokens]
    
    return selected_log_probs

def grpo_llava_evaluate(base, input_ids, output_ids, image_tensor, temperature, thought_prob_coef):
    """
    Evaluate action log probabilities for GRPO.
    Returns per-token log probabilities.
    """
    return grpo_llava_evaluate_token_log_probs(
        base, input_ids, output_ids, image_tensor, temperature, thought_prob_coef
    )
