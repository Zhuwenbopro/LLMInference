from typing import List
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, LlamaDecoderLayer
import torch
from torch import nn
import os
import json
import threading
import time

device1 = "cuda:0"
device2 = "cuda:1"

class LLM(nn.Module):
    def __init__(self, config):
        super(LLM, self).__init__()

        config._attn_implementation == "sdpa"

        class Head(nn.Module):
            def __init__(self, config, num_stage_layers):
                super(Head, self).__init__()
                self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
                self.rotary_emb = LlamaRotaryEmbedding(config=config)
                self.layers = nn.ModuleList()
                for i in range(num_stage_layers):
                    layer = LlamaDecoderLayer(config, i)
                    self.layers.append(layer)
                
            def forward(self, input_ids):
                inputs_embeds = self.embed_tokens(input_ids)
                hidden_states = inputs_embeds
                position_ids = torch.arange(input_ids.size(1), device=input_ids.device).expand_as(input_ids)
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
                return (position_embeddings, position_ids, hidden_states, input_ids)
            
            def reset_parameters(self):
                # 覆盖这个方法以避免自动初始化
                pass
        
        class Tail(nn.Module):
            def __init__(self, config, num_stage_layers):
                super(Tail, self).__init__()
                self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

                self.layers = nn.ModuleList()
                for i in range(num_stage_layers):
                    layer = LlamaDecoderLayer(config, i)
                    self.layers.append(layer)
                
            def forward(self, states):
                (position_embeddings, position_ids, hidden_states, input_ids) = states
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
                hidden_states = self.norm(hidden_states)
                logits = self.lm_head(hidden_states)
                logits = logits.float()
                next_token_scores = logits[:, -1, :].clone()
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                # next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                return input_ids
            
            def reset_parameters(self):
                # 覆盖这个方法以避免自动初始化
                pass

        
        self.head = Head(config, 16)
        self.tail = Tail(config, 16)


    def forward(self, input_ids):
        states = self.head(input_ids)
        input_ids = self.tail(states)
        return input_ids
    
    def reset_parameters(self):
        # 覆盖这个方法以避免自动初始化
        pass


def write_list_to_file_line_by_line(my_list, filename):
    with open(filename, 'w') as file:
        for item in my_list:
            file.write(f"{item}\n")


class Utils:
    pipeline_num = 2
    start = 0
    plan_len = 200
    count = 0
    lock = threading.Lock()
    
    def recursive_to(self, obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, tuple):
            return tuple(self.recursive_to(item, device) for item in obj)
        else:
            return obj

    def model_thread_func(self, stage):
        time_list_begin = [[],[]]
        time_list_end = [[],[]]
        print(f"worker {stage} is running")
        while self._semaphores[stage].acquire():
            states = self._transfer_states[stage].pop()
            states = self.recursive_to(states, self._devices[stage])
                
            time_list_begin[stage].append(time.time())
            states = self._worker_model[stage](states)
            time_list_end[stage].append(time.time())

            self.lock.acquire()
            self.count += 1
            self.lock.release()

            if self.count == self.plan_len + stage:
                write_list_to_file_line_by_line(time_list_begin[stage], f"worker{stage}_begin.txt")
                write_list_to_file_line_by_line(time_list_end[stage], f"worker{stage}_end.txt")
                if stage == 1:
                    print(self._decode_list(states.tolist()[0]))
                break
            self._transfer_states[(stage+1)%self.pipeline_num].append(states)
            self._semaphores[(stage+1)%self.pipeline_num].release()
        

    def __init__(self, model, task="text-generation"):
        self.config = AutoConfig.from_pretrained(model)
        self.config._attn_implementation == "sdpa"
        self.tokenizer = AutoTokenizer.from_pretrained(model, _from_pipeline=task)
        print("init model...")
        _s = time.time()
        self.model = LLM(self.config)
        print(f"Model init in {time.time() - _s} seconds.")
        print("Loading model...")
        _s = time.time()
        # 得到 checkpoint 文件
        archive_file = self._get_checkpoint_shard_files(model)
        state_dict = dict()
        for shard_file in archive_file:
            state_dict.update(self._load_state_dict(shard_file))

        self._state_dict_map(state_dict)
        self.model.load_state_dict(state_dict)
        print(f"Model load in {time.time() - _s} seconds.")
        del state_dict
        print("Model loaded.")

        self._worker_model = [self.model.head.to("cuda:0").eval(), 
                              self.model.tail.to("cuda:1").eval()]
        # pre stage1 stage2 post
        self._transfer_states = [[] for _ in range(self.pipeline_num)]
        self._semaphores = [threading.Semaphore(0) for _ in range(self.pipeline_num)]
        self._workers = []
        for i in range(self.pipeline_num):
            thread = threading.Thread(target=self.model_thread_func, args=(i, ))
            thread.start()
            self._workers.append(thread)
        
        self._devices = [torch.device("cuda:0"), torch.device("cuda:1")]


    def _decode_list(self, input_ids: List[int]):
        return self.tokenizer.decode(input_ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True,)

    def _model_inputs(self, prompt, params={}, return_tensors="pt"):
        return self.tokenizer(prompt, return_tensors=return_tensors, **params)
    
    def forward(self, prompts, params={}):
        self.start = time.time()
        for prompt in prompts:
            model_inputs = self._model_inputs(prompt, params={})
            self._transfer_states[0].append(model_inputs['input_ids'])
            self._semaphores[0].release()
        

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
    
    def _state_dict_map(self, state_dict):
        # 获取所有的键
        keys = list(state_dict.keys())

        # 遍历所有的键并修改
        for key in keys:
            # 处理层的转换 (假设原来的层是 0-31)
            if key.startswith('model.layers.'):
                layer_num = int(key.split('.')[2])  # 获取层号
                # 根据层号决定是 head 还是 tail
                if layer_num < 16:
                    new_key = key.replace(f'model.layers.{layer_num}', f'head.layers.{layer_num}')
                else:
                    new_key = key.replace(f'model.layers.{layer_num}', f'tail.layers.{layer_num - 16}')
                
                # 在字典中删除旧的键，并添加新的键
                state_dict[new_key] = state_dict.pop(key)
            
            if key.startswith('model.norm.'):
                new_key = key.replace('model.', 'tail.')
                state_dict[new_key] = state_dict.pop(key)

            if key.startswith('model.embed_tokens.'):
                new_key = key.replace('model.', 'head.')
                state_dict[new_key] = state_dict.pop(key)

            if key.startswith('lm_head.'):
                new_key = key.replace('lm_head.', 'tail.lm_head.')
                state_dict[new_key] = state_dict.pop(key)


prompts = ["A young girl named Alice lived in a small village,",
           "Alice was intrigued by the talking animals and decided to",]
# params = {"max_length": 100, }
model_id = "./Meta-Llama-3-8B-Instruct"


utils = Utils(model_id)
utils.forward(prompts, params = {})
