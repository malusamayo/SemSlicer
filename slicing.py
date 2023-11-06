from utils.load_config import read_yaml_config
from utils.file import read_txt_file, read_csv_file
import pandas as pd
import re
from llama import Llama2Wrapper
from utils.log import get_logger
import spacy
import en_core_web_sm
import nltk
from multi_rake import Rake
import torch
import random
from llm_server import Generator

logger = get_logger("INFO", "slicing")
generator = Generator("flan-t5", "xxl")
# generator = Generator("llama2", "13b-chat")

PROMPT = '''# Text
{passage}

# Question
{question} Your answer is yes or no.

# Answer
My answer is '''

SYSTEM_PROMPT = '''In each round of the conversation, you will receive a text and a question. \
The question is about the text. Answer the question according to the text.
Please first answer the question with "My answer is yes" \
or "My answer is no", then explain your reason. Try your best.'''

def select_few_shot_examples(dialogs, results, probs, nums):
    num_per_class = int(nums / 2)
    _, indices_a = torch.topk(probs[:, 0], num_per_class)
    _, indices_b = torch.topk(probs[:, 1], num_per_class)

    # selected_idx = indices_a.tolist() + indices_b.tolist()
    selected_idx = [v for p in zip(indices_a.tolist(), indices_b.tolist()) for v in p] #interleave two lists
    few_shot_examples = [dialogs[idx][1]["content"] + results[idx] + '.' for idx in selected_idx]
    # few_shot_examples = few_shot_examples[::-1]
    
    few_shot_str = '\n\n'.join(few_shot_examples)
    few_shot_str += '\n\n'
    return few_shot_str

def slicing(args):
    # config
    config = read_yaml_config("./config.yaml", args)
    logger.info(args)
    logger.info(config)

    # load keyword file
    keywords = read_txt_file(config["SLICING"]["KEYWORDS_PATH"])

    # load dataset
    df = read_csv_file(config["SLICING"]["DATASET_PATH"])
    logger.info("loaded dataset")
    logger.info(df.info())

    # random select data
    test_data = df.sample(n=config["SLICING"]["SAMPLE_SIZE"], random_state=42)

    # process keywords
    for key_idx, keyword in enumerate(keywords):
        logger.info("processing keyword: {keyword}".format(keyword=keyword))

        prompt_df = pd.read_csv(config["SLICING"]["PROMPT_PATH"] + "prompt_result_" + str(key_idx) + ".csv")
        prompts = prompt_df["{keyword}_prompt".format(keyword=keyword)].tolist()

        test_data["{}_result".format(keyword)] = 0.0
        for index, prompt in enumerate(prompts):
            logger.info("processing prompt: {prompt}".format(prompt=prompt.split("\n")[0]))
            
            # generate dialogs
            dialogs = [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": PROMPT.format(question=prompt, passage=row)}
                ]
                for row in test_data['context']
            ]

            # generate results
            results, probs = generator._send_request(dialogs, temperature=0.2, batch_size=20, return_probs=True, labels=['yes', 'no'])

            # regenerate with few-shot examples
            if config["SLICING"]["FEW_SHOT"]:
                few_shot_str = select_few_shot_examples(dialogs, results, probs, 4)

                # regenerate dialogs
                dialogs = [
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": few_shot_str + PROMPT.format(question=prompt, passage=row)}
                    ]
                    for row in test_data['context']
                ]

                # regenerate results
                results = generator._send_request(dialogs, temperature=0.2, batch_size=20)
            
            # save raw data
            test_data["{keyword}_prompt{id}_meta".format(keyword=keyword, id=index)] = [result for result in results]
            
            # save classification result
            test_data["{keyword}_prompt{id}".format(keyword=keyword, id=index)] = test_data["{keyword}_prompt{id}_meta".format(keyword=keyword, id=index)].apply(
                lambda x: 1 if x.lower().find("yes") != -1 and x.lower().find("no") == -1 else 0
            )
            test_data["{}_result".format(keyword)] += test_data["{keyword}_prompt{id}".format(keyword=keyword, id=index)]
        
            ## save test data
            test_data.to_csv(config["SLICING"]["OUTPUT_PATH"], index=False)

    # save as file
    logger.info(test_data.info())
    test_data.to_csv(config["SLICING"]["OUTPUT_PATH"], index=False)