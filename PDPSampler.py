import torch
import numpy as np
from pdps_utils import *
import gc
import time


class PDPSampler:
    def __init__(self, model):
        self.model = model

    def generate(
        self, 
        input_ids, 
        attn_masks,
        hyp_lambda,
        seq_len_scheduling,
        num_seq_scheduling,
        eos_id,
        temperature = 1.0,
        top_k = 50,
        top_p = 1.0,
        max_batch_size = 256,
        kv_cache_save_stage = 0,
        device = None
    ):
        past_key_values = None  # KV cache to persist across chunks
        past_log_probs_sum = None
        past_hidden_state_sum = None
        past_hidden_state_count = 0

        input_ids = input_ids.repeat(num_seq_scheduling[0], 1)
        attn_masks = attn_masks.repeat(num_seq_scheduling[0], 1)

        num_seq_scheduling += [num_seq_scheduling[-1]]
        for it_count, (subset_size, seq_len) in enumerate(zip(num_seq_scheduling[1:], seq_len_scheduling)):
            current_input_ids = input_ids
            current_attn_masks = attn_masks

            output_scores = []
            generated_sequences = []
            new_past_key_values = []
            hidden_states = []
            n_inputs = current_input_ids.shape[0]
            for b_start in range(0, n_inputs, max_batch_size):
                b_end = min(b_start + max_batch_size, n_inputs)

                batch_input_ids = current_input_ids[b_start: b_end]
                batch_attn_masks = current_attn_masks[b_start: b_end]
                batch_past_key_values = get_batch_past_key_values(self.model, past_key_values, b_start, b_end)

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attn_masks,
                        past_key_values=batch_past_key_values,
                        max_new_tokens=seq_len,  
                        num_return_sequences=1,
                        do_sample=True,
                        use_cache=True,
                        return_dict_in_generate=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        output_hidden_states=True,
                        output_scores=True,
                    )

                output_scores.append(outputs.scores)
                generated_sequences.append(outputs.sequences)
                hidden_states.append(outputs.hidden_states)
                if it_count >= kv_cache_save_stage and it_count < len(seq_len_scheduling) - 1:
                    new_past_key_values.append(outputs.past_key_values)
                del outputs

            generated_sequences = pad_and_concat(generated_sequences, pad_value=eos_id)
            new_token_ids, new_token_attn_masks, attn_masks = get_new_tokens_and_adjusted_attn_masks(current_input_ids, eos_id, generated_sequences, attn_masks, device)
            
            hidden_state_sum, hidden_state_count = get_sum_hidden_state(hidden_states, new_token_attn_masks, past_hidden_state_sum, past_hidden_state_count)
            hidden_state_mean = hidden_state_sum / hidden_state_count.unsqueeze(-1)

            log_probs_sum = get_sum_log_prob(output_scores, new_token_attn_masks, past_log_probs_sum)
            log_probs_mean = log_probs_sum / hidden_state_count
            inv_perlexity = torch.exp(log_probs_mean)

            best_subset, quality_obj_value, diversity_obj_value = max_avg_dp_greedy(
                hyp_lambda, 
                subset_size, 
                hidden_state_mean, 
                inv_perlexity,
                device
            )
            best_subset = list(best_subset)
            
            attn_masks = attn_masks[torch.tensor(best_subset), :]

            past_hidden_state_sum = hidden_state_sum[best_subset, :]
            past_hidden_state_count = hidden_state_count[best_subset]
            past_log_probs_sum = log_probs_sum[best_subset]
            
            input_ids = generated_sequences[best_subset, :]
            
            if it_count >= kv_cache_save_stage and it_count < len(seq_len_scheduling) - 1:
                del past_key_values
                past_key_values = select_past_key_values(self.model, new_past_key_values, best_subset, device)
                del new_past_key_values
            else:
                past_key_values = None
            gc.collect()

            if it_count < 2:
                max_batch_size = max_batch_size // 4

        output_dic = {'output_ids': input_ids}

        return output_dic
            
