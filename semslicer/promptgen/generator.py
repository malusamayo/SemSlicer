from ..utils.file import read_txt_file, read_csv_file
import pandas as pd
import re
import os
from ..utils.log import get_logger
import spacy
import en_core_web_sm
import nltk
from multi_rake import Rake
from ..utils.config import config
from .paraphraser import Paraphraser
from ..model.llm_server import Generator
from difflib import SequenceMatcher

logger = get_logger("INFO", "prompt")

class PromptGeneratorV0:

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

GEN_PROMPT = '''A user is exploring a dataset to identify a subset that matches their goal. Your job is to help the user craft a classification question. Answer ONLY with the question.

Example 1
User goal: age
Question: Does the text mention a person's age?

Example 2
User goal: slang
Question: Does the text use any slang?

Example 3
User goal: music
Question: Is the text related to music?

Following the same format above from the examples, craft a classification question with the following goal.
'''

EXAMPLE_GEN_PROMPT = '''Write {n} examples with a '{label}' answer to the question above, following the format below.

Text: {{text}}
Answer: {{answer}}'''

class PromptGenerator:

    def __init__(self, model_name="gpt-4-turbo-preview", num_prompts=1, instruction_source="template", refine_flag=False):
        self.paraphraser = Paraphraser(model_name)
        self.generator = Generator(model_name)
        self.validate_flag = False
        self.num_prompts = num_prompts
        self.instruction_source = instruction_source
        self.refine_flag = refine_flag

    def generate_prompts(self, queries):
        if self.instruction_source == "template":
            results = [f"Is the text related to {query.lower()}?" for query in queries]
        elif self.instruction_source == "model":
            results = self.generator._send_request(
                [[
                    {"role": "system", "content": GEN_PROMPT}, 
                    {"role": "user", "content": f'User goal: {query}'},
                    {"role": "assistant", "content": f'Question: '}
                ] for query in queries], 
                temperature=1,
            )
        logger.info(results)
        return results

    def find_prompts_list(self, keyword_df):
        '''
        find prompts for a list of keywords
        '''
        keyword_list = keyword_df["keyword"].tolist()
        keyword_df["description"] = keyword_df["description"].fillna('').str.strip()
        description_list = keyword_df["description"].tolist()
        print(keyword_list)
        print(description_list)


        # prompt dataframe
        prompt_df = pd.DataFrame()
        for key_idx, (keyword, descrp) in enumerate(zip(keyword_list, description_list)):

            prompts = self.generate_prompts([keyword if descrp == '' else descrp])
            if self.num_prompts > 1:
                paraphrased_prompts = self.paraphraser.paraphrase_prompt(prompts[0], keyword, self.num_prompts - 1)
                prompts = prompts + paraphrased_prompts

            # deduplicate
            prompts = list(set(prompts))

            prompt_df["{keyword}_prompt".format(keyword=keyword)] = prompts
            prompt_df.to_csv(config["EXPERIMENT"]["PROMPT_PATH"], index=False)


class ExampleGenerator:

    def __init__(self, model_name="gpt-4-turbo-preview"):
        self.generator = Generator(model_name)

    def generate_examples(self, question_prompt, label, num_examples=5):
        results = self.generator._send_request(
            [[
                {"role": "user", "content": question_prompt},
                {"role": "user", "content": EXAMPLE_GEN_PROMPT.format(n=num_examples, label=label)},
            ]], 
            temperature=1,
        )
        examples_str = results[0]
        logger.info(examples_str)
        return examples_str

if __name__ == '__main__':
    result_path = os.path.join("result", 'testbed')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    config.update_path('testbed')

    promptGen = PromptGenerator()
    test_keywords = [
        "pronouns",
        "incorrect grammar",
        "malicious intent",
        "text with user instructions",
        "sarcasm"
    ]
    promptGen.find_prompts_list(test_keywords)

    exampleGen = ExampleGenerator()
    results = exampleGen.generate_examples("Does the text contain anything related to pronouns?", "yes")
    print(results)
    # promptGen.find_prompts_list(["negation", "sarcasm"])