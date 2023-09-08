from utils.log import get_logger
from utils.file import read_txt_file, read_csv_file
from utils.load_config import read_yaml_config
from llama import Llama2Wrapper
from datasets import load_dataset

logger = get_logger("INFO", "label")
generator = None

SYSTEM_PROMPT =  '''In each round of the conversation, you will receive a text and a question. \
The question is about the text. Answer the question according to the text.
Please first answer the question with "My answer is yes" \
or "My answer is no", then explain your reason. Try your best.'''
PROMPT = '''
# Text
{passage}

# Question
{question}'''
# EXAMPLE_PASSAGE = "The pancake I bought was terrible. It was burnt and tasted like rubber. I will never buy food from that place again."

# EXAMPLE_PROMPT = PROMPT.format(question="Does this text explicitly talks about nature topics?", passage=EXAMPLE_PASSAGE)

# EXAMPLE_ANSWER = '''My answer is no, the text does not talks about nature topics. The text indicates that the speaker bought a pancake and it was terrible. The speaker will never buy food from that place again. It does not mention any nature topics.'''


def init():
    model_size = "7b-chat"
    global generator
    try:
        generator = Llama2Wrapper(
            "/home/yiningho/workspace/datadisk/llama/llama-2-{}".format(model_size),
            is_chat_model=True,
            load_4bit=True,
            batch_size=40,
        )
    except:
        logger.info(
            "Loading from /home/yiningho/workspace/datadisk/llama/llama-2-{} failed. Using huggingface hub.".format(
                model_size
            )
        )
        generator = Llama2Wrapper(
            "meta-llama/Llama-2-{}-hf".format(model_size),
            is_chat_model=True,
            load_4bit=True,
            batch_size=40,
        )

def _send_request(
    dialogs,
    max_gen_len=1024,
    temperature=0.01,
    top_p=0.9,
    batch_size=40
):
    '''
    example for prompt:[{"role": "user", "content": "what is the recipe of mayonnaise?"}]
    '''
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size
    )
    return [result[0]['generated_text'].strip() for result in results]

def label(args):
    config = read_yaml_config("./config.yaml")
    logger.info(args)
    logger.info(config)
    init()
    df = read_csv_file(config["LABEL"]["RESULT_PATH"])
    logger.info(df.info())
    keywords = read_txt_file(config["LABEL"]["KEYWORDS_PATH"])
    for index, keyword in enumerate(keywords):
        logger.info("processing keyword: {key}".format(key=keyword))
        prompt_df = read_csv_file(config["LABEL"]["PROMPT_PATH"] + "prompt_final_result_" + str(index) + ".csv")
        prompt_df['sigma'] = prompt_df['sigma'].apply(lambda x: x if x > 0 else -x)
        min_sigma_idx = prompt_df['sigma'].idxmin()
        prompt = prompt_df.at[min_sigma_idx, '{keyword}_prompt'.format(keyword=keyword)]
        logger.info("prompt = {prompt}".format(prompt=prompt.split('\n')[0]))
        dialogs = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                # {"role": "user", "content": EXAMPLE_PROMPT},
                # {"role": "assistant", "content": EXAMPLE_ANSWER},
                {"role": "user", "content": PROMPT.format(question=prompt, passage=passage)}
            ]
            for passage in df['text']
        ]
        logger.info("generated dialogs")
        results = _send_request(dialogs, temperature=0.2)
        logger.info("generated results")
        # output 3 results
        for i, result in enumerate(results):
            logger.info(result)
            i += 1
            if i == 3:
                break
        df['label_{keyword}_meta'.format(keyword=keyword)] = "Nan"
        i = 0
        for idx, row in df.iterrows():
            df.at[idx, 'label_{keyword}_meta'.format(keyword=keyword)] = results[i]
            i += 1
        df['label_{keyword}'.format(keyword=keyword)] = df['label_{keyword}_meta'.format(keyword=keyword)].apply(
            lambda x: 1 if x.lower().find("my answer is yes") != -1 and x.lower().find("my answer is no") == -1 else 0
        )
        df.to_csv(config["LABEL"]["OUTPUT_PATH"], index=False)
