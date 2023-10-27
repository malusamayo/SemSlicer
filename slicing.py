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
from llm_server import Generator

logger = get_logger("INFO", "slicing")
generator = Generator("flan-t5", "xxl")
# generator = Generator("llama2", "13b-chat")

PROMPT = '''
# Text
{passage}

# Question
{question} Your answer is yes or no.

# Answer

My answer is
'''

SYSTEM_PROMPT = '''In each round of the conversation, you will receive a text and a question. \
The question is about the text. Answer the question according to the text.
Please first answer the question with "My answer is yes" \
or "My answer is no", then explain your reason. Try your best.'''

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
            results = generator._send_request(dialogs, temperature=0.2, batch_size=20)
            
            # save raw data
            test_data["{keyword}_prompt{id}_meta".format(keyword=keyword, id=index)] = [result for result in results]
            
            # save classification result
            test_data["{keyword}_prompt{id}".format(keyword=keyword, id=index)] = test_data["{keyword}_prompt{id}_meta".format(keyword=keyword, id=index)].apply(
                lambda x: 1 if x.lower().find("my answer is yes") != -1 and x.lower().find("my answer is no") == -1 else 0
            )
            test_data["{}_result".format(keyword)] += test_data["{keyword}_prompt{id}".format(keyword=keyword, id=index)]
        
            ## save test data
            test_data.to_csv(config["SLICING"]["OUTPUT_PATH"], index=False)

    # save as file
    logger.info(test_data.info())
    test_data.to_csv(config["SLICING"]["OUTPUT_PATH"], index=False)