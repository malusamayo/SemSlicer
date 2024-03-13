import pandas as pd
import re
from ..utils.log import get_logger
import spacy
import en_core_web_sm
import nltk
from multi_rake import Rake
from ..model.llm_server import Generator

logger = get_logger("INFO", "prompt")

FIND_PROMPT = '''Strictly paraphrase the following question without changing the meaning of the question. \
List {n} strict paraphrases below. Do not include additional information that the question does not mention in your answers. Be concise. \
Always include exactly the words "{keyword}" in your answers. Try your best not to change these words "{keyword}".'''

EVAL_PROMPT = '''Are the following two questions asking the same thing? Answer yes or no.
Question 1: {question1}
Question 2: {question2}'''

class Paraphraser:

    def __init__(self, model_name="gpt-4-turbo-preview", model_size=""):
        self.generator = Generator(model_name, model_size)

    def evaluate_prompts(self, prompts):
        default_prompt = prompts[0]
        prompts = prompts[1:]
        results = generator._send_request(
            [[
                {"role": "system", "content": ""}, 
                {"role": "user", "content": EVAL_PROMPT.format(question1=default_prompt, question2=prompt)}
            ] for prompt in prompts], 
            temperature=0.1
        )
        logger.info(results)


    def paraphrase_prompt(self, prompt, keyword="", n=8):
        
        logger.info(prompt)
        results = self.generator._send_request(
            [[
                {"role": "system", "content": FIND_PROMPT.format(n=n, keyword=keyword)}, 
                {"role": "user", "content": "{}".format(prompt)}
            ]], 
            temperature=0.1
        )[0]

        logger.info(results)

        results = results.split("\n")
        index_pattern = r'^\d+\.\s'
        results = [s for s in results if re.match(index_pattern, s)]
        remove_index_pattern = r'^\d+\.\s'
        results = [re.sub(remove_index_pattern, '', s) for s in results]
        
        return results