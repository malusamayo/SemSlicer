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
generator = Generator("llama2", "13b-chat")

FIND_PROMPT = '''Strictly paraphrase the following question without changing the meaning of the question. \
List 8 strict paraphrases below. Do not include additional information that the question does not mention in your answers. Be concise. \
Always include exactly the words "{keyword}" in your answers. Try your best not to change these words "{keyword}".'''


def _find_prompts(keyword, prompt_templates):
    # find prompts for a keyword

    # parse to find pos (label)
    nlp = en_core_web_sm.load()
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
    results = generator._send_request(
        [[
            {"role": "system", "content": FIND_PROMPT.format(keyword=keyword)}, 
            {"role": "user", "content": "{}".format(prompt)}
        ]], 
        temperature=0.1
    )[0]


    # clean results
    if results.find("I cannot") == 0:
        logger.info(results)
        return []
    logger.info(results)
    results = results.split("\n")
    index_pattern = r'^\d+\.\s'
    results = [s for s in results if re.match(index_pattern, s)]
    remove_index_pattern = r'^\d+\.\s'
    results = [re.sub(remove_index_pattern, '', s) for s in results]

    # keyword extraction
    rake = Rake()
    required_words = rake.apply(keyword)
    required_words = set(' '.join([word[0] for word in required_words]).split())

    # similarity analysis
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
        if (match_cnt >= int(len(required_words) * 0.5) and match_cnt > 0) or int(len(required_words)) == 0 or required_words == set(['']):
            prompts.append(result)

    # print prompts
    for index, prompt in enumerate(prompts):
        logger.info("{index}: {prompt}".format(index=index, prompt=prompt))
    prompt = [item for item in prompts]
    logger.info("successfully found prompts for {keyword}".format(keyword=keyword))
    return prompt

def find_prompts(args):
    config = read_yaml_config("./config.yaml", args)
    logger.info(args)
    logger.info(config)

    keywords = read_txt_file(config["SLICING"]["KEYWORDS_PATH"])
    unsuccessful_keywords = []

    for key_idx, keyword in enumerate(keywords):
        logger.info("processing keyword: {keyword}".format(keyword=keyword))

        # prompt dataframe
        prompt_df = pd.DataFrame()

        # get prompts
        prompts = _find_prompts(keyword, config["SLICING"]["PROMPT_TEMPLATES"])
        if len(prompts) == 0:
            prompts = _find_prompts(keyword, config["SLICING"]["PROMPT_TEMPLATES"])
            if len(prompts) == 0:
                logger.info("no prompts found for {keyword}".format(keyword=keyword))
                unsuccessful_keywords.append(keyword)
                continue
        prompt_df["{keyword}_prompt".format(keyword=keyword)] = prompts
        prompt_df.to_csv(config["SLICING"]["PROMPT_PATH"] + f"prompt_result_{key_idx}.csv", index=False)

    logger.info("unsuccessful keywords: {keywords}".format(keywords=unsuccessful_keywords))