from text_generation import Client
from llama import Llama2Wrapper
from t5 import FlanT5Wrapper
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline


class Generator:

    def __init__(self, model_name, model_size):
        self.model_name = model_name
        self.model_size = model_size
        if model_name == 'llama2':
            try:
                self.generator = Llama2Wrapper(
                    "meta-llama/Llama-2-{}-hf".format(model_size),
                    is_chat_model=True,
                    load_4bit=True,
                    batch_size=10
                )
            except:
                assert False
        if model_name == 'flan-t5':
            self.generator = FlanT5Wrapper(
                    f"google/flan-t5-{model_size}".format(model_size),
                    is_chat_model=True,
                    load_4bit=True,
                    batch_size=10
            )

    def _send_request(
        self,
        dialogs,
        max_gen_len=1024,
        temperature=0.01,
        top_p=0.9,
        batch_size=40,
        return_probs=False,
        labels=None
    ):
        '''
        example for dialogs:[[{"role": "user", "content": "what is the recipe of mayonnaise?"}]]
        '''
        results = []
        if self.model_name == 'llama2':
            results = self.generator.chat_completion(
                dialogs,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                batch_size=batch_size
            )
            return [result[0]['generated_text'].strip() for result in results]
        if self.model_name == 'flan-t5':
            results, probs = self.generator.completion(
                dialogs, 
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                batch_size=batch_size,
                return_prob=return_probs,
                labels=labels
            )
            return results, probs

        return results


# def init():
#     # load model
#     model_size = "13b-chat"
#     # get current models and pick the first one
#     # models = Client.list_from_central()
#     # model_name, model_addr = models[0]["name"], models[0]["address"]
#     # print(f"Using model {model_name} at {model_addr}")

#     # generator = Client("http://" + model_addr, timeout=60)
#     # print(generator.generate("What is Deep Learning?", max_new_tokens=20).generated_text)

#     try:
#         generator = Llama2Wrapper(
#             "hf-models-cache/models--meta-llama--Llama-2-13b-chat-hf/snapshots/0ba94ac9b9e1d5a0037780667e8b219adde1908c",
#             # "./llama-2-{}-hf".format(model_size),
#             is_chat_model=True,
#             load_4bit=True,
#             batch_size=10
#         )
#     except:
#         assert False
#         logger.info(
#             "Loading from ./llama-2-{}-hf failed. Using huggingface hub.".format(
#                 model_size
#             )
#         )
#         generator = Llama2Wrapper(
#             "meta-llama/Llama-2-{}-hf".format(model_size),
#             is_chat_model=True,
#             load_4bit=True,
#             batch_size=10
#         )

#     return generator

# def _send_request(
#     dialogs,
#     max_gen_len=1024,
#     temperature=0.01,
#     top_p=0.9,
#     batch_size=40
# ):
#     '''
#     example for dialogs:[[{"role": "user", "content": "what is the recipe of mayonnaise?"}]]
#     '''
#     results = generator.chat_completion(
#         dialogs,
#         max_gen_len=max_gen_len,
#         temperature=temperature,
#         top_p=top_p,
#         batch_size=batch_size
#     )
#     return [result[0]['generated_text'].strip() for result in results]

#     # results = [generator.generate(
#     #     dialog[1]['content'],
#     #     max_new_tokens=100,
#     #     temperature=temperature,
#     #     top_p=top_p,
#     #     # batch_size=batch_size
#     # ) for dialog in dialogs]
#     # print(len(results))
#     # return [result.generated_text.strip() for result in results]


# def init():
#     size = 'xxl'
#     tokenizer = T5Tokenizer.from_pretrained(f"google/flan-t5-{size}")
#     model = T5ForConditionalGeneration.from_pretrained(f"google/flan-t5-{size}", device_map="auto", load_in_8bit=True)
#     generator = pipeline("text2text-generation",
#                     model=model,
#                     tokenizer=tokenizer)
    
#     return generator

# def _send_request(
#     dialogs,
#     max_gen_len=1024,
#     temperature=0.01,
#     top_p=0.9,
#     batch_size=40
# ):
#     '''
#     example for dialogs:[[{"role": "user", "content": "what is the recipe of mayonnaise?"}]]
#     '''
#     texts = [dialog[1]["content"] for dialog in dialogs]
#     print(texts[0])
#     results = generator(texts, batch_size=batch_size)
#     print(results[0])
#     return ['My answer is ' + result['generated_text'].strip() for result in results]

# generator = init()