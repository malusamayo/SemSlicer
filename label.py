from utils.log import get_logger
from utils.file import read_txt_file, read_csv_file
from utils.load_config import read_yaml_config
from llama import Llama2Wrapper
from llm_server import Generator

logger = get_logger("INFO", "label")
generator = Generator("flan-t5", "xxl")
# generator = Generator("llama2", "13b-chat")

SYSTEM_PROMPT =  '''In each round of the conversation, you will receive a text and a question. \
The question is about the text. Answer the question according to the text.
Please first answer the question with "My answer is yes" \
or "My answer is no", then explain your reason. Try your best.'''

PROMPT = '''
# Text
{passage}

# Question
{question}'''

def select_prompt(prompt_df, keyword, criteria):
    if criteria == 'min_noise':
        prompt_df['sigma'] = prompt_df['sigma'].apply(lambda x: x if x > 0 else -x)
        min_sigma_idx = prompt_df['sigma'].idxmin()
        prompt = prompt_df.at[min_sigma_idx, '{keyword}_prompt'.format(keyword=keyword)]
    elif criteria == 'default':
        prompt = prompt_df.at[0, '{keyword}_prompt'.format(keyword=keyword)]
    elif criteria == 'maj_vote':
        max_acc_idx = prompt_df['pseudo_acc'].idxmax()
        prompt = prompt_df.at[max_acc_idx, '{keyword}_prompt'.format(keyword=keyword)]
    return prompt

def label(args):
    # read config
    config = read_yaml_config("./config.yaml", args)
    logger.info(args)
    logger.info(config)


    # read data
    df = read_csv_file(config["LABEL"]["RESULT_PATH"])
    logger.info(df.info())

    # read keywords
    keywords = read_txt_file(config["LABEL"]["KEYWORDS_PATH"])

    # label
    for index, keyword in enumerate(keywords):
        logger.info("processing keyword: {key}".format(key=keyword))
        # read prompt
        prompt_df = read_csv_file(config["LABEL"]["PROMPT_PATH"] + "prompt_final_result_" + str(index) + ".csv")

        # select prompt
        prompt = select_prompt(prompt_df, keyword, 'maj_vote')
        logger.info("prompt = {prompt}".format(prompt=prompt.split('\n')[0]))

        # generate dialogs
        dialogs = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT.format(question=prompt, passage=passage)}
            ]
            for passage in df['context']
        ]
        logger.info("generated dialogs")

        # generate results
        results = generator._send_request(dialogs, temperature=0.2, batch_size=10)
        logger.info("generated results")
        
        # save results
        df['label_{keyword}_meta'.format(keyword=keyword)] = [result for result in results]
        df['label_{keyword}'.format(keyword=keyword)] = df['label_{keyword}_meta'.format(keyword=keyword)].apply(
            lambda x: 1 if x.lower().find("yes") != -1 and x.lower().find("no") == -1 else 0
        )
        df.to_csv(config["LABEL"]["OUTPUT_PATH"], index=False)
