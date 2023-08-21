from utils.load_config import read_yaml_config
from utils.file import read_txt_file
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree
import language_tool_python
from datasets import load_dataset
import pandas as pd
from multiprocessing.connection import Client
import time
import re
from llama import Llama2Wrapper
from utils.log import get_logger

logger = get_logger("INFO", "slicing")

nlp = StanfordCoreNLP('../stanford-corenlp-4.5.4')
language_tool = language_tool_python.LanguageTool('en-US')
generator = None
# Please reply the question with a single word "No" or "Yes". If you are unsure, answer "Unsure" instead of "Yes" or "No".
PROMPT_SUFFIX = ''' Please first simply state your reason, then answer the question with "My Answer is No." or "My Answer is Yes." in a new line. If you are unsure, answer with "My Answer is Unsure." instead of "My Answeris No." or "My Answer is Yes.".

[Passage]
{passage}

'''
EXAMPLE_PASSAGE1 = "I saw the movie last night. It was really good. The plot was interesting and the action scenes were awesome. The acting was okay, but could have been better. Overall, I enjoyed it."

EXAMPLE_PROMPT1 = "Does this passage primarily make comments on movies?" + PROMPT_SUFFIX.format(passage=EXAMPLE_PASSAGE1)

EXAMPLE_ANSWER1 = '''The passage indicates that the speaker went for a movie and enjoyed it. The speaker praised the plot and action scenes while criticizing the acting, which means the speaker make comments on the movie. 
My Answer is Yes.'''

EXAMPLE_PASSAGE2 = "The pancake I bought was terrible. It was burnt and tasted like rubber. I will never buy food from that place again."

EXAMPLE_PROMPT2 = "Does this passage explicitly mentioned nature topics?" + PROMPT_SUFFIX.format(passage=EXAMPLE_PASSAGE1)

EXAMPLE_ANSWER2 = '''The passage indicates that the speaker bought a pancake and it was terrible. The speaker will never buy food from that place again. It does not mention any nature topics.
My Answer is No.'''

FIND_PROMPT = '''Can you strictly paraphrase the following question in 10 ways without changing the meaning of the question? \
List your paraphrases below. Do not include additional information in your answers.'''

def init():
    model_size = "7b-chat"
    global generator
    try:
        generator = Llama2Wrapper(
            "/home/yiningho/workspace/datadisk/llama/llama-2-{}".format(model_size),
            is_chat_model=True,
            load_4bit=False,
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
            load_4bit=False,
        )


def _send_request(
    dialogs,
    max_gen_len=512,
    temperature=0.01,
    top_p=0.9,
):
    '''
    example for prompt:[{"role": "user", "content": "what is the recipe of mayonnaise?"}]
    '''
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    return [result[0]['generated_text'] for result in results]

def _find_prompts(keyword, prompt_templates):
    '''
    find prompts for a keyword
    '''
    # parse to find pos (label)
    tree = nlp.parse(keyword)
    tree = Tree.fromstring(tree)
    label = tree[0].label()
    if label == 'S' or label == 'FRAG':
        label = tree[0][0].label()
    logger.info("{keyword}: {label}".format(keyword=keyword, label=label))

    # invalid label
    if label not in prompt_templates:
        logger.info(tree)
        return []
    
    # # get prompts using templates
    # prompts = [item.format(keyword=keyword) + PROMPT_SUFFIX for item in prompt_templates[label]]

    # prompts = [language_tool.correct(prompt) for prompt in prompts]

    # get prompts using templates
    prompts = [item.format(keyword=keyword) for item in prompt_templates[label]][0]

    #correct grammars
    prompts = language_tool.correct(prompts)

    # request for prompts
    results = _send_request([[{"role": "system", "content": FIND_PROMPT}, {"role": "user", "content": prompts}]], temperature=0)[0]

    # clean results
    results = results.split("\n")
    index_pattern = r'^\d+\.\s'
    results = [s for s in results if re.match(index_pattern, s)]
    remove_index_pattern = r'^\d+\.\s'
    results = [re.sub(remove_index_pattern, '', s) for s in results]

    prompts = [item + PROMPT_SUFFIX for item in results]
    # logger.info(prompts)
    logger.info("successfully found prompts for {keyword}".format(keyword=keyword))
    return prompts

def slicing(args):
    # config
    config = read_yaml_config("./config.yaml")
    logger.info(args)
    logger.info(config)
    init()
    # load keyword file
    keywords = read_txt_file(config["SLICING"]["KEYWORDS_PATH"])

    # load dataset
    dataset = load_dataset("tweet_eval", "emotion")
    df = pd.DataFrame(dataset['train'])
    logger.info("loaded dataset")
    logger.info(df.info())
    # random select data
    test_data = df.sample(n=config["SLICING"]["SAMPLE_SIZE"])

    # save prompt
    prompt_df = pd.DataFrame()
    # process keywords
    for keyword in keywords:
        logger.info("processing keyword: {keyword}".format(keyword=keyword))

        # get prompts
        prompts = _find_prompts(keyword, config["SLICING"]["PROMPT_TEMPLATES"])
        prompt_df["{keyword}_prompt".format(keyword=keyword)] = prompts
        # logger.info(test_data)
        test_data["{}_result".format(keyword)] = 0.0
        for index, prompt in enumerate(prompts):
            logger.info("processing prompt: {prompt}".format(prompt=prompt))
            # get meta output data
            dialogs = [
                [
                    # {"role": "user", "content": EXAMPLE_PROMPT1},
                    # {"role": "assistant", "content": EXAMPLE_ANSWER1},
                    {"role": "user", "content": EXAMPLE_PROMPT2},
                    {"role": "assistant", "content": EXAMPLE_ANSWER2},
                    {"role": "user", "content": prompt.format(passage=x)}
                ]
                for x in test_data["text"]
            ]
            results = _send_request(dialogs, temperature=0.5)
            # logger.info(results)
            test_data["{keyword}_prompt{id}_meta".format(keyword=keyword, id=index)] = "no implemented"
            i = 0
            for idx, row in test_data.iterrows():
                test_data.at[idx, "{keyword}_prompt{id}_meta".format(keyword=keyword, id=index)] = results[i]
                i += 1
            # logger.info(test_data)
            test_data["{keyword}_prompt{id}".format(keyword=keyword, id=index)] = test_data["{keyword}_prompt{id}_meta".format(keyword=keyword, id=index)].apply(
                lambda x: 1 if x.find("My Answer is Yes") != -1 else 0
            )
            # logger.info(test_data)
            test_data["{}_result".format(keyword)] += test_data["{keyword}_prompt{id}".format(keyword=keyword, id=index)]
        test_data["{}_result".format(keyword)] = test_data["{}_result".format(keyword)] / len(prompts)
        # test_data["hypothesis"] = test_data["result"].apply(lambda x: 1 if x >= config["SLICING"]["THRESHOLD"] else 0)
        # # logger.info(test_data[['label', 'result']])
        # count_true_positive = len(test_data[(test_data['hypothesis'] == 1) & (test_data['label'] == 0)])
        # count_true_negative = len(test_data[(test_data['hypothesis'] == 0) & (test_data['label'] != 0)])
        # count_false_positive = len(test_data[(test_data['hypothesis'] == 1) & (test_data['label'] != 0)])
        # count_false_negative = len(test_data[(test_data['hypothesis'] == 0) & (test_data['label'] == 0)])
        # logger.info("true positive: {count}".format(count=count_true_positive))
        # logger.info("true negative: {count}".format(count=count_true_negative))
        # logger.info("false positive: {count}".format(count=count_false_positive))
        # logger.info("false negative: {count}".format(count=count_false_negative))
    # save as csv
    test_data.to_csv(config["SLICING"]["OUTPUT_PATH"], index=False)
    prompt_df.to_csv(config["SLICING"]["PROMPT_PATH"], index=False)
    nlp.close()
    language_tool.close()