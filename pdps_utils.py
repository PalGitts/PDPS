import torch
import torch.nn.functional as F
from transformers import DynamicCache
import numpy as np


'''
def get_attention_basedOn_input(ip_ids, eos_id, device):
    ip_attn_masks = torch.ones_like(ip_ids).to(device) 
    eos_positions = (ip_ids == eos_id).to(device)

    if eos_positions.any():
        first_eos = eos_positions.to(torch.int).argmax(dim=-1)
        B, T = ip_ids.shape
        no_eos_mask = (~eos_positions.any(dim=1))                # [B] bool
        first_eos[no_eos_mask] = T # # Replace 0s for "no EOS" rows with T (sequence length

        for b, idx in enumerate(first_eos):
            ip_attn_masks[b, idx:] = 0

    return ip_attn_masks.to(device)
'''


def calculate_angular_distance(u, V):
    if V.dim() == 2:
        u = u.unsqueeze(0)
    cosine_similarity_score = F.cosine_similarity(u, V, dim=-1)
    cosine_similarity_score = torch.clamp(cosine_similarity_score, -1.0, 1.0)

    angular_distance = torch.arccos(cosine_similarity_score)
    return angular_distance

    
def max_avg_dp_greedy(
        hyp_lambda, 
        subset_size, 
        embeddings, # [bs, h_dim]
        inv_perplexity, # [sc_0 ... sc_bs]
        device
    ):
    
    n_elem = embeddings.shape[0]
    
    assert n_elem >= subset_size

    if n_elem == subset_size:
        return list(range(n_elem)), 0.0, 0.0
    
    embeddings = embeddings.to(device)
    inv_perplexity = inv_perplexity.to(device)
    # -----------------------------
    # First use singletone set to get 'best'-diverse sequence
    # -----------------------------
    current_subset = []
    current_remaining = torch.ones((n_elem), dtype=float, device=device)
    current_remaining_ids = set(range(n_elem))
    current_marginal_gain = inv_perplexity / subset_size

    quality_objective_value = 0
    total_objective_value = 0
    
    while True:
        # Select the element with highest current marginal gain
        max_elem = int(torch.argmax(current_marginal_gain * current_remaining))
        current_subset.append(max_elem)
        current_remaining[max_elem] = 0
        current_remaining_ids.remove(max_elem)
        quality_objective_value += inv_perplexity[max_elem] / subset_size
        total_objective_value += current_marginal_gain[max_elem]

        # if subset_size elems have been laready selected, return it
        if len(current_subset) == subset_size:
            diversity_objective_value = (total_objective_value - quality_objective_value) / hyp_lambda
            return sorted(current_subset), quality_objective_value.cpu().numpy(), diversity_objective_value.cpu().numpy()

        # else update the marginal gains for all remaining elements by adding their distances to the newly added elements
        remaining_elem_embeddings = embeddings[sorted(list(current_remaining_ids))]
        max_elem_embedding = embeddings[max_elem]
        new_distances = calculate_angular_distance(max_elem_embedding, remaining_elem_embeddings)
        current_marginal_gain[sorted(list(current_remaining_ids))] += hyp_lambda * new_distances / subset_size / (subset_size-1)
    

def pad_and_concat(generated_sequences, pad_value=0):
    # Find maximum sequence length (dimension 1)
    max_len = max(t.size(1) for t in generated_sequences)

    padded_tensors = []
    for t in generated_sequences:
        pad_len = max_len - t.size(1)

        # Pad on the right side of dimension 1
        # F.pad format for 2D: (left, right, top, bottom)
        padded = F.pad(t, (0, pad_len, 0, 0), value=pad_value)
        padded_tensors.append(padded)

    return torch.cat(padded_tensors, dim=0)


def get_new_tokens_and_adjusted_attn_masks(input_ids, eos_id, generated_sequence, past_attn_masks, device):
    input_len = input_ids.shape[1]
    new_token_ids = generated_sequence[:, input_len:] 
    
    new_token_attn_masks = torch.ones_like(new_token_ids).to(device) 
    eos_positions = (new_token_ids == eos_id)
    
    if eos_positions.any():
        first_eos = eos_positions.to(torch.long).argmax(dim=-1)
        B, T = new_token_ids.shape
        no_eos_mask = (~eos_positions.all(dim=-1))                # [B] bool
        first_eos[no_eos_mask] = T # # Replace 0s for "no EOS" rows with T (sequence length

        for b, idx in enumerate(first_eos):
            new_token_attn_masks[b, idx:] = 0

    if past_attn_masks == None:
        attn_masks = new_token_attn_masks
    else:
        past_attn_masks = past_attn_masks.to(device)
        B, _ = past_attn_masks.shape
        for b in range(B):
            if past_attn_masks[b, -1] == 0:
                new_token_attn_masks[b, :] = 0
        new_token_attn_masks = new_token_attn_masks.to(dtype=past_attn_masks.dtype)
        attn_masks = torch.concat([past_attn_masks, new_token_attn_masks], dim=-1)

    return new_token_ids, new_token_attn_masks, attn_masks


def get_sum_hidden_state(hidden_states_list, curr_attn_mask, past_hidden_state_sum, past_hidden_state_count):
    hidden_state_sum_list = []
    batch_start = 0
    for batch, hidden_states in enumerate(hidden_states_list):
        batch_end = batch_start + hidden_states[0].shape[0]
        hs_sum = None
        for hs_idx, hs in enumerate(hidden_states):
            if hs_idx == 0:
                continue
            temp_hs = hs.squeeze(1) * curr_attn_mask[batch_start: batch_end, hs_idx].unsqueeze(1) # [bs, h_dim]
  
            if hs_sum == None:
                hs_sum = temp_hs
            else:
                hs_sum = hs_sum + temp_hs
        hidden_state_sum_list.append(hs_sum)
        batch_start = batch_end
    hidden_state_sum = torch.concat(hidden_state_sum_list, dim=0)

    if past_hidden_state_sum != None:
        hidden_state_sum = past_hidden_state_sum.to(hidden_state_sum.device) + hidden_state_sum
    hidden_state_count = curr_attn_mask.sum(dim=-1)

    return hidden_state_sum, hidden_state_count


def get_sum_log_prob(scores_list, new_token_attn_masks, past_sum_log_prob):
    sum_log_prob_list = []
    batch_start = 0
    for batch, scores in enumerate(scores_list):
        batch_end = batch_start + scores[0].shape[0]
        sum_log_prob = 0
        for sc_idx, sc in enumerate(scores):
            sum_log_prob += sc * new_token_attn_masks[batch_start: batch_end, sc_idx]
        sum_log_prob_list.append(sum_log_prob)
    sum_log_prob = torch.concat(sum_log_prob_list, dim=0)
        
    if past_sum_log_prob != None:
        sum_log_prob += past_sum_log_prob
    return sum_log_prob


def get_batch_past_key_values(model, past_key_values, b_start, b_end):
    if past_key_values == None:
        return None
    batch_past_key_values = DynamicCache(config=model.config)
    for i, layer in enumerate(past_key_values.layers):
        k = layer.keys
        v = layer.values
        batch_past_key_values.layers[i].keys = k[b_start: b_end]
        batch_past_key_values.layers[i].values = v[b_start: b_end]
    return batch_past_key_values


def select_past_key_values(model, past_key_values_list, best_subset, device):
    past_key_values = DynamicCache(config=model.config)
    for i in range(len(past_key_values.layers)):
        k_list = []
        v_list = []
        b_start = 0
        for past_kv in past_key_values_list:
            b_end = b_start + past_kv.layers[i].keys.shape[0]
            selected_idx = [e - b_start for e in best_subset if e in range(b_start, b_end)]
            
            if len(selected_idx) == 0:
                b_start = b_end
                continue

            selected_idx = torch.tensor(
                selected_idx,
                device=device,
                dtype=torch.long
            )

            k_list.append(
                past_kv.layers[i].keys.index_select(dim=0, index=selected_idx)
            )
            v_list.append(
                past_kv.layers[i].values.index_select(dim=0, index=selected_idx)
            )

            b_start = b_end

        past_key_values.layers[i].keys = torch.concat(k_list, dim=0)
        past_key_values.layers[i].values = torch.concat(v_list, dim=0)
    return past_key_values


