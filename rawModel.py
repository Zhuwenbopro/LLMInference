from typing import List
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, LlamaDecoderLayer
import torch
from torch import nn
import os
import json
import time

device = "cuda:1"

class LLM(nn.Module):
    def __init__(self, config):
        super(LLM, self).__init__()
        config._attn_implementation == "sdpa"

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self, config):
                super(Model, self).__init__()
                self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
                self.rotary_emb = LlamaRotaryEmbedding(config=config)
                self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                
                self.layers = nn.ModuleList()
                for i in range(config.num_hidden_layers):
                    layer = LlamaDecoderLayer(config, i)
                    self.layers.append(layer)
                

            def forward(self, input_ids, attention_mask):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                inputs_embeds = self.embed_tokens(input_ids)
                hidden_states = inputs_embeds
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                
                for decoder_layer in self.layers:
                    layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=None,
                            position_ids=position_ids,
                            past_key_value=None,
                            output_attentions=False,
                            use_cache=False,
                            cache_position=None,
                            position_embeddings=position_embeddings,
                        )
                        
                    hidden_states = layer_outputs[0]
                return self.norm(hidden_states)

        self.model = Model(config)

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        next_token_scores = logits[:, -1, :].clone()
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        # next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        return input_ids



class Pipeline:
    def __init__(self, model, task="text-generation"):
        self.config = AutoConfig.from_pretrained(model)
        self.config._attn_implementation == "sdpa"
        self.tokenizer = AutoTokenizer.from_pretrained(model, _from_pipeline=task)
        print("Loading model...")
        self.model = LLM(self.config).to(device)
        # 得到 checkpoint 文件
        archive_file = self._get_checkpoint_shard_files(model)
        state_dict = dict()
        for shard_file in archive_file:
            state_dict.update(self._load_state_dict(shard_file))

        self.model.load_state_dict(state_dict)
        del state_dict
        print("Model loaded.")

    def _decode_list(self, input_ids: List[int]):
        return self.tokenizer.decode(input_ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True,)

    def _model_inputs(self, prompt, params={}, return_tensors="pt"):
        return self.tokenizer(prompt, return_tensors=return_tensors, **params)
    
    def forward(self, prompt, params={}):
        max_length = params.get("max_length", 1024)
        start = time.time()
        with torch.no_grad():
            for _ in range(max_length):
                model_inputs = self._model_inputs(prompt, params)
                input_ids = self.model(**model_inputs)
                prompt = self._decode_list(input_ids[0])
        end = time.time()
        print(f"Time: {end - start}")
        return prompt

    def _get_checkpoint_shard_files(self, pretrained_model_name_or_path):
        index_filename = "model.safetensors.index.json"
        index_filename = os.path.join(pretrained_model_name_or_path, index_filename)
        
        if not os.path.isfile(index_filename):
            raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")

        with open(index_filename, "r") as f:
            index = json.loads(f.read())

        shard_filenames = sorted(set(index["weight_map"].values()))

        # First, let's deal with local folder.
        if os.path.isdir(pretrained_model_name_or_path):
            shard_filenames = [os.path.join(pretrained_model_name_or_path, f) for f in shard_filenames]
            return shard_filenames
        else: return []

    def _load_state_dict(self, filename):
        from safetensors import safe_open
        result = {}
        with safe_open(filename, framework="pt") as f:
            for k in f.keys():
                result[k] = f.get_tensor(k)
        return result


prompt = "Once upon a time, in a land far, far away, there was a"
# params = {"max_length": 100, }
model_id = "./Meta-Llama-3-8B-Instruct"


pipeline = Pipeline(model_id)
generation = pipeline.forward(prompt, params = {"max_length": 100, })
print(generation)
