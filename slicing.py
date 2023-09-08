from utils.load_config import read_yaml_config
from utils.file import read_txt_file, read_csv_file
import pandas as pd
import re
from llama import Llama2Wrapper
from utils.log import get_logger
import spacy
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from multi_rake import Rake
from datasets import load_from_disk
import random

logger = get_logger("INFO", "slicing")
generator = None
# Please reply the question with a single word "No" or "Yes". If you are unsure, answer "Unsure" instead of "Yes" or "No".
PROMPT = '''
# Text
{passage}

# Question
{question}'''
# EXAMPLE_PASSAGE = "The pancake I bought was terrible. It was burnt and tasted like rubber. I will never buy food from that place again."

# EXAMPLE_PROMPT = PROMPT.format(question="Does this text explicitly talks about nature topics?", passage=EXAMPLE_PASSAGE)

# EXAMPLE_ANSWER = '''My answer is no, the text does not talks about nature topics. The text indicates that the speaker bought a pancake and it was terrible. The speaker will never buy food from that place again. It does not mention any nature topics.'''

FIND_PROMPT = '''Strictly paraphrase the following question without changing the meaning of the question. \
List 8 strict paraphrases below. Do not include additional information that the question does not mention in your answers. Be concise. \
Always include exactly the words "{keyword}" in your answers. Try your best not to change these words "{keyword}".'''

SYSTEM_PROMPT = '''In each round of the conversation, you will receive a text and a question. \
The question is about the text. Answer the question according to the text.
Please first answer the question with "My answer is yes" \
or "My answer is no", then explain your reason. Try your best.'''

def init():
    model_size = "7b-chat"
    global generator
    try:
        generator = Llama2Wrapper(
            "/home/yiningho/workspace/datadisk/llama/llama-2-{}".format(model_size),
            is_chat_model=True,
            load_4bit=True,
            batch_size=40
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
            batch_size=40
        )
    nltk.download("punkt")

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

def _find_prompts(keyword, prompt_templates):
    '''
    find prompts for a keyword
    '''
    # parse to find pos (label)
    nlp = spacy.load("en_core_web_md")
    doc = nlp(keyword)
    root = [token for token in doc if token.head == token][0]
    label = root.pos_
    logger.info("{keyword}: {label}".format(keyword=keyword, label=label))
    if label == "PROPN":
        label = "NOUN"
    # invalid label
    if label not in prompt_templates:
        logger.info(keyword, label)
        return []

    # get prompts using templates
    prompt = prompt_templates[label].format(keyword=keyword)

    # request for prompts
    logger.info(prompt)
    results = _send_request(
        [[
            {"role": "system", "content": FIND_PROMPT.format(keyword=keyword)}, 
            {"role": "user", "content": "{}".format(prompt)}
        ]], 
        temperature=0.1
    )[0]
    if results.find("I cannot") == 0:
        logger.info(results)
        return []
    logger.info(results)
    # clean results
    results = results.split("\n")
    index_pattern = r'^\d+\.\s'
    results = [s for s in results if re.match(index_pattern, s)]
    remove_index_pattern = r'^\d+\.\s'
    results = [re.sub(remove_index_pattern, '', s) for s in results]
    rake = Rake()
    # ps = PorterStemmer()

    required_words = rake.apply(keyword)
    required_words = set(' '.join([word[0] for word in required_words]).split())
    # keyword_no_punct = keyword.translate(str.maketrans('', '', string.punctuation))
    # logger.info(required_words)
    from difflib import SequenceMatcher
    s = SequenceMatcher(None)
    logger.info("required_words: {required_words}".format(required_words=required_words))
    prompts = []
    for result in results:
        contain_words = set(result.split(" "))
        match_cnt = 0
        for require_word in required_words:
            s.set_seq2(require_word)
            for contain_word in contain_words:
                s.set_seq1(contain_word)
                if s.ratio() >= 0.5:
                    match_cnt += 1
                    break
        # result_no_punct = result.translate(str.maketrans('', '', string.punctuation))
        # contain_words = [ps.stem(word) for word in word_tokenize(result_no_punct)]
        # logger.info("{result}, {contain_words}".format(result=result, contain_words=contain_words))
        # count = len(set(required_words).intersection(set(contain_words)))
        if (match_cnt >= int(len(required_words) * 0.5) and match_cnt > 0) or int(len(required_words)) == 0 or required_words == set(['']):
            prompts.append(result)

    for index, prompt in enumerate(prompts):
        logger.info("{index}: {prompt}".format(index=index, prompt=prompt))
    prompt = [item for item in prompts]
    logger.info("successfully found prompts for {keyword}".format(keyword=keyword))
    return prompt

def slicing(args):
    # config
    config = read_yaml_config("./config.yaml")
    logger.info(args)
    logger.info(config)
    init()
    # load keyword file
    keywords = read_txt_file(config["SLICING"]["KEYWORDS_PATH"])

    df = read_csv_file(config["SLICING"]["DATASET_PATH"])
    logger.info("loaded dataset")
    logger.info(df.info())
    # random select data
    test_data = df.sample(n=config["SLICING"]["SAMPLE_SIZE"])


    unsuccessful_keywords = []
    save_prompt_idx = 0
    # process keywords
    for key_idx, keyword in enumerate(keywords):
        logger.info("processing keyword: {keyword}".format(keyword=keyword))

        # save prompt
        prompt_df = pd.DataFrame()
        # get prompts
        prompts = _find_prompts(keyword, config["SLICING"]["PROMPT_TEMPLATES"])
        # continue
        if len(prompts) == 0:
            prompts = _find_prompts(keyword, config["SLICING"]["PROMPT_TEMPLATES"])
            if len(prompts) == 0:
                logger.info("no prompts found for {keyword}".format(keyword=keyword))
                unsuccessful_keywords.append(keyword)
                continue
        # continue
        prompt_df["{keyword}_prompt".format(keyword=keyword)] = prompts
        test_data["{}_result".format(keyword)] = 0.0
        for index, prompt in enumerate(prompts):
            logger.info("processing prompt: {prompt}".format(prompt=prompt.split("\n")[0]))
            # get meta output data
            dialogs = [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    # {"role": "user", "content": EXAMPLE_PROMPT},
                    # {"role": "assistant", "content": EXAMPLE_ANSWER},
                    {"role": "user", "content": PROMPT.format(question=prompt, passage=row)}
                ]
                for row in test_data['text']
            ]
            results = _send_request(dialogs, temperature=0.2)
            for i, result in enumerate(results):
                logger.info(result)
                i += 1
                if i == 3:
                    break
            # logger.info(results)
            # test_data.add_column("{keyword}_prompt{id}_meta".format(keyword=keyword, id=index), results[i])
            test_data["{keyword}_prompt{id}_meta".format(keyword=keyword, id=index)] = "not implemented"
            i = 0
            for idx, row in test_data.iterrows():
                test_data.at[idx, "{keyword}_prompt{id}_meta".format(keyword=keyword, id=index)] = results[i]
                i += 1
            test_data["{keyword}_prompt{id}".format(keyword=keyword, id=index)] = test_data["{keyword}_prompt{id}_meta".format(keyword=keyword, id=index)].apply(
                lambda x: 1 if x.lower().find("my answer is yes") != -1 and x.lower().find("my answer is no") == -1 else 0
            )
            test_data["{}_result".format(keyword)] += test_data["{keyword}_prompt{id}".format(keyword=keyword, id=index)]
        # test_data["{}_result".format(keyword)] = test_data["{}_result".format(keyword)] / len(prompts)
        prompt_df.to_csv(config["SLICING"]["PROMPT_PATH"] + "prompt_result_" + str(save_prompt_idx) + ".csv", index=False)
        save_prompt_idx += 1
    logger.info("unsuccessful keywords: {keywords}".format(keywords=unsuccessful_keywords))
    # save as csv
    logger.info(test_data.info())
    test_data.to_csv(config["SLICING"]["OUTPUT_PATH"], index=False)
    # test_data.save_to_disk(config["SLICING"]["OUTPUT_PATH"])
    # test_data.to_csv(config["SLICING"]["OUTPUT_PATH"] + ".csv", index=False)