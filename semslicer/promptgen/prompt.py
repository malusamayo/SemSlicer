from ..utils.file import read_txt_file, read_csv_file
import pandas as pd
import re
from ..utils.log import get_logger
import spacy
import en_core_web_sm
import nltk
from multi_rake import Rake
from ..utils.config import config
from .paraphraser import Paraphraser
from difflib import SequenceMatcher

logger = get_logger("INFO", "prompt")

class PromptGenerator:

    def __init__(self, model_name="llama2", model_size="13b-chat"):
        self.prompt_templates = {
            "NOUN": "Does the text contain anything related to {keyword}?",
            "ADJ": "Does the text contain anything related to {keyword}?",
            "VERB": "Does the text contain anything related to {keyword}?",
            "ADP": "Does the text contain anything related to {keyword}??"
        }
        self.paraphraser = Paraphraser(model_name, model_size)
        self.validate_flag = False


    def find_template_prompt(self, keyword):
        '''
        find template prompt for a keyword
        '''

        # parse to find pos (label)
        nlp = en_core_web_sm.load()
        doc = nlp(keyword)
        root = [token for token in doc if token.head == token][0]
        label = root.pos_
        logger.info("{keyword}: {label}".format(keyword=keyword, label=label))
        if label == "PROPN":
            label = "NOUN"

        # invalid label
        if label not in self.prompt_templates:
            logger.info(keyword, label)
            return None

        # get prompts using templates
        prompt = self.prompt_templates[label].format(keyword=keyword)
        return prompt

    def filter_prompts(self, prompts, keyword):
        '''
        filter prompts using keyword extraction and similarity analysis
        '''
        # keyword extraction
        rake = Rake()
        required_words = rake.apply(keyword)
        required_words = set(' '.join([word[0] for word in required_words]).split())

        # similarity analysis
        s = SequenceMatcher(None)
        logger.info("required_words: {required_words}".format(required_words=required_words))

        def contain_keywords(prompt, required_words):
            contain_words = set(prompt.split(" "))
            match_cnt = 0
            for require_word in required_words:
                s.set_seq2(require_word)
                for contain_word in contain_words:
                    s.set_seq1(contain_word)
                    if s.ratio() >= 0.5:
                        match_cnt += 1
                        break
            return (match_cnt >= int(len(required_words) * 0.5) and match_cnt > 0) or int(len(required_words)) == 0 or required_words == set([''])
        
        prompts = [prompt for prompt in prompts if contain_keywords(prompt, required_words)]

        # print prompts
        for index, prompt in enumerate(prompts):
            logger.info("{index}: {prompt}".format(index=index, prompt=prompt))
        return prompts

    def find_prompts(self, keyword):
        logger.info("processing keyword: {keyword}".format(keyword=keyword))

        prompt = self.find_template_prompt(keyword)
        if prompt is None:
            return []

        prompts = self.paraphraser.paraphrase_prompt(prompt, keyword)
        prompts = [prompt] + prompts
        prompts = self.filter_prompts(prompts, keyword)
        if self.validate_flag:
            self.paraphraser.evaluate_prompts(prompts)
            ## [TODO] remove invalid prompts (and re-generate)

        logger.info("successfully found prompts for {keyword}".format(keyword=keyword))
        return prompts

        
    def find_prompts_list(self, keyword_list):
        '''
        find prompts for a list of keywords
        '''
        unsuccessful_keywords = []
        for key_idx, keyword in enumerate(keyword_list):
            # prompt dataframe
            prompt_df = pd.DataFrame()

            prompts = self.find_prompts(keyword)
            if len(prompts) == 0:
                logger.info("no prompts found for {keyword}".format(keyword=keyword))
                unsuccessful_keywords.append(keyword)
                continue

            prompt_df["{keyword}_prompt".format(keyword=keyword)] = prompts
            prompt_df = prompt_df.drop_duplicates()
            prompt_df.to_csv(config["EXPERIMENT"]["PROMPT_PATH"].format(key_idx=key_idx), index=False)

        logger.info("unsuccessful keywords: {keywords}".format(keywords=unsuccessful_keywords))



if __name__ == '__main__':
    promptGen = PromptGenerator()
    promptGen.find_prompts_list(["negation", "sarcasm"])