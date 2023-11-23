import os
import os.path as osp
import torch
import torch.cuda
import torch.backends.cudnn
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from typing import List, Dict
import time
import concurrent.futures
from ..utils.log import get_logger

logger = get_logger("INFO", "llama")

def _divide_list_into_sublists(input_list, num_sublists, batch_size):
    avg_len = int(len(input_list) / num_sublists) + 1
    if avg_len >= batch_size:
        sublists = [input_list[min(i * avg_len, len(input_list)): min((i + 1) * avg_len, len(input_list))] for i in range(num_sublists)]
    else:
        sublists = []
        i = 0
        while i < len(input_list):
            sublists.append(input_list[i:min(i + batch_size, len(input_list))])
            i += batch_size
        if len(sublists) < num_sublists:
            sublists += [[] for _ in range(num_sublists - len(sublists))]
    return sublists

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


class Llama2Wrapper:
    def __init__(
        self,
        model_name,
        is_chat_model,
        debug_mode=False,
        load_4bit=False,
        batch_size=40,
    ):
        self.model_name = model_name
        self.no_cuda = (os.environ["CUDA_VISIBLE_DEVICES"] == "")
        self.use_cuda = not self.no_cuda
        START_TIME = time.perf_counter()
        logger.info("Start loading {}...".format(model_name))
        if self.use_cuda:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_4bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            quantization_config = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )
        device_map = {
            'model.embed_tokens': "nan", 
            'model.norm': "nan", 
            'lm_head': 'nan'
        }
        for i in range(40):
            device_map[f'model.layers.{i}'] = "nan"
        self.device_count = torch.cuda.device_count()
        logger.info(f"Using {self.device_count} GPUs")
        self.model = []
        for item in range(self.device_count):
            for layer in device_map:
                device_map[layer] = item
            self.model.append(
                AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device_map,
                    quantization_config=quantization_config,
                )
            )
            self.model[-1].eval()
        logger.info("Done with {:.2f} seconds.".format(time.perf_counter() - START_TIME))
        self.debug_mode = debug_mode
        self.is_chat_model = is_chat_model
        if load_4bit:
            self.pipeline = [
                pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=self.tokenizer,
                    batch_size=batch_size,
                ) for device, model in enumerate(self.model)
            ]
        else:
            self.pipeline = [
                pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=self.tokenizer,
                    device=device,
                    batch_size=batch_size,
                ) for device, model in enumerate(self.model)
            ]
        for pipe, model in zip(self.pipeline, self.model):
            pipe.tokenizer.pad_token_id = model.config.eos_token_id

    @torch.no_grad()
    def chat_completion(
        self, dialogs, max_gen_len,
        temperature, top_p,
        return_prob=False,
        calc_str=None,
        batch_size=40,
    ) -> List[
        List[Dict[str, str]]
    ]:  
        assert self.is_chat_model
        divided_dialogs = _divide_list_into_sublists(dialogs, self.device_count, batch_size)
        dialog_input = []
        for dialogs in divided_dialogs:
            prompt_tokens = []
            for dialog in dialogs:
                if dialog[0]["role"] != "system":
                    dialog = [
                        {
                            "role": "system",
                            "content": DEFAULT_SYSTEM_PROMPT,
                        }
                    ] + dialog
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
                dialog_tokens: str = "".join(
                    [
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
                        for prompt, answer in zip(
                            dialog[::2],
                            dialog[1::2],
                        )
                    ]
                )
                dialog_tokens += f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"
                assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                    [msg["role"] == "assistant" for msg in dialog[1::2]]
                ), (
                    "model only supports 'system', 'user' and 'assistant' roles, "
                    "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
                )
                assert (
                    dialog[-1]["role"] == "user"
                ), f"Last message must be from user, got {dialog[-1]['role']}"
                prompt_tokens.append(dialog_tokens)
                logger.debug(dialog_tokens)
            dialog_input.append(prompt_tokens)
        if return_prob:
            raise("Not implemented yet")
            # assert calc_str is not None
            # input_ids = self.tokenizer.encode(
            #     prompt_tokens[0] + " " + calc_str,
            #     return_tensors="pt")
            # str_encoded = self.tokenizer.encode(
            #     calc_str, return_tensors="pt")[0]
            # if self.use_cuda:
            #     input_ids = input_ids.cuda(0)
            #     str_encoded = str_encoded.cuda(0)
            # res = self.model(input_ids).logits[0][-1-len(str_encoded):-1]
            # res = torch.gather(torch.softmax(res, dim=-1), 1, str_encoded.unsqueeze(1))
            # res = torch.sum(torch.log(res)) / len(str_encoded)
            # return res.cpu().item()
        else:
            generated_results = []
            import time
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.device_count) as executor:
                futures = []
                for i in range(self.device_count):
                    if len(dialog_input[i]) > 0:
                        futures.append(
                            executor.submit(
                                self.pipeline[i], 
                                dialog_input[i], 
                                temperature=temperature, 
                                top_p=top_p, 
                                max_length=max_gen_len, 
                                return_full_text=False,
                                batch_size=batch_size,
                            )
                        )
                for future in futures:
                    generated_results += future.result()
            for pipeline in self.pipeline:
                pipeline.call_count=0
            return generated_results

# class Llama2Wrapper:
#     def __init__(
#         self,
#         model_name,
#         is_chat_model,
#         debug_mode=False,
#         load_4bit=False,
#         batch_size=40
#     ):
#         self.model_name = model_name
#         self.no_cuda = (os.environ["CUDA_VISIBLE_DEVICES"] == "")
#         self.use_cuda = not self.no_cuda
#         START_TIME = time.perf_counter()
#         print("Start loading {}...".format(model_name))
#         if self.use_cuda:
#             from transformers import BitsAndBytesConfig
#             quantization_config = BitsAndBytesConfig(
#                 load_in_4bit=load_4bit,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_compute_dtype=torch.bfloat16,
#             )
#         else:
#             quantization_config = None
#         access_token = "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"
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_name,
#             cache_dir="./hf-models-cache/", token=access_token)
#         print(quantization_config)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             cache_dir="./hf-models-cache/",
#             device_map="auto" if self.use_cuda else None,
#             quantization_config=quantization_config)
#         self.model.eval()
#         print("Done with {:.2f} seconds.".format(time.perf_counter() - START_TIME))
#         self.debug_mode = debug_mode
#         self.is_chat_model = is_chat_model
#         self.pipeline = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#         )
#         self.pipeline.tokenizer.pad_token_id = self.model.config.eos_token_id


#     @torch.no_grad()
#     def completion(
#         self, prompts, max_gen_len,
#         temperature, top_p,
#         return_prob=False,
#         calc_str=None
#     ) -> List[
#         List[Dict[str, str]]
#     ]:
#         assert not self.is_chat_model
#         if return_prob:
#             assert calc_str is not None
#             input_ids = self.tokenizer.encode(
#                 prompts[0] + " " + calc_str,
#                 return_tensors="pt")
#             str_encoded = self.tokenizer.encode(
#                 calc_str, return_tensors="pt")[0]
#             if self.use_cuda:
#                 input_ids = input_ids.cuda(0)
#                 str_encoded = str_encoded.cuda(0)
#             res = self.model(input_ids).logits[0][-1-len(str_encoded):-1]
#             res = torch.gather(torch.softmax(res, dim=-1), 1, str_encoded.unsqueeze(1))
#             res = torch.sum(torch.log(res)) / len(str_encoded)
#             return res.cpu().item()
#         else:
#             generated_results = self.pipeline(
#                 prompts,
#                 temperature=temperature,
#                 top_p=top_p,
#                 max_new_tokens=max_gen_len,
#                 return_full_text=False,
#                 do_sample=True
#             )
#             return generated_results

#     @torch.no_grad()
#     def chat_completion(
#         self, dialogs, max_gen_len,
#         temperature, top_p,
#         return_prob=False,
#         calc_str=None,
#         batch_size=40
#     ) -> List[
#         List[Dict[str, str]]
#     ]:  # [batch_size, sampled_num], {"generated_text": "xxx"}
#         assert self.is_chat_model
#         prompt_tokens = []
#         for dialog in dialogs:
#             if dialog[0]["role"] != "system":
#                 dialog = [
#                     {
#                         "role": "system",
#                         "content": DEFAULT_SYSTEM_PROMPT,
#                     }
#                 ] + dialog
#             dialog = [
#                 {
#                     "role": dialog[1]["role"],
#                     "content": B_SYS
#                     + dialog[0]["content"]
#                     + E_SYS
#                     + dialog[1]["content"],
#                 }
#             ] + dialog[2:]
#             dialog_tokens: str = "".join(
#                 [
#                     f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
#                     for prompt, answer in zip(
#                         dialog[::2],
#                         dialog[1::2],
#                     )
#                 ]
#             )
#             dialog_tokens += f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"
#             assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
#                 [msg["role"] == "assistant" for msg in dialog[1::2]]
#             ), (
#                 "model only supports 'system', 'user' and 'assistant' roles, "
#                 "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
#             )
#             assert (
#                 dialog[-1]["role"] == "user"
#             ), f"Last message must be from user, got {dialog[-1]['role']}"
#             prompt_tokens.append(dialog_tokens)
#             logger.debug(dialog_tokens)
#         if return_prob:
#             assert calc_str is not None
#             input_ids = self.tokenizer.encode(
#                 prompt_tokens[0] + " " + calc_str,
#                 return_tensors="pt")
#             str_encoded = self.tokenizer.encode(
#                 calc_str, return_tensors="pt")[0]
#             if self.use_cuda:
#                 input_ids = input_ids.cuda(0)
#                 str_encoded = str_encoded.cuda(0)
#             res = self.model(input_ids).logits[0][-1-len(str_encoded):-1]
#             res = torch.gather(torch.softmax(res, dim=-1), 1, str_encoded.unsqueeze(1))
#             res = torch.sum(torch.log(res)) / len(str_encoded)
#             return res.cpu().item()
#         else:
#             logger.info(f"processing len {len(prompt_tokens)} prompts")
#             generated_results = self.pipeline(
#                 prompt_tokens,
#                 temperature=temperature,
#                 top_p=top_p,
#                 max_new_tokens=max_gen_len,
#                 return_full_text=False,
#                 batch_size=batch_size,
#                 do_sample=True
#             )
#             return generated_results