import os
import torch
import pandas as pd
from .utils.parseArgument import parseArg
from .utils.log import get_logger
from .utils.file import read_txt_file, read_csv_file
from .utils.config import config
from .model.llm_server import Generator
from .promptgen.prompt import PromptGenerator
from .promptgen.selector import PromptSelector


logger = get_logger("INFO", "label")

SYSTEM_PROMPT =  '''In each round of the conversation, you will receive a text and a question. \
The question is about the text. Answer the question according to the text.
Please first answer the question with "My answer is yes" \
or "My answer is no", then explain your reason. Try your best.'''

PROMPT = '''# Text
{passage}

# Question
{question} Your answer is yes or no.

# Answer
My answer is '''


def select_usp_examples(dialogs, results, probs, nums):
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

def select_boundary_examples(dialogs, results, probs, nums):

    max_probs = torch.maximum(probs[:, 0], probs[:, 1])
    _, selected_idx = torch.topk(-max_probs, nums)

    logger.info(max_probs[selected_idx])
    few_shot_examples = [dialogs[idx][1]["content"] + '<to-be-filled>.' for idx in selected_idx]
    # few_shot_examples = few_shot_examples[::-1]
    
    few_shot_str = '\n\n'.join(few_shot_examples)
    few_shot_str += '\n\n'
    return few_shot_str


def to_dialog(data, prompt, few_shot_str=""):
    dialogs = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": few_shot_str + PROMPT.format(question=prompt, passage=passage)}
        ]
        for passage in data['context']
    ]
    return dialogs

class Slicer(object):

    def __init__(self, model_name="flan-t5", model_size="xxl"):
        self.generator = Generator(model_name, model_size)
        self.prompt_selector = PromptSelector()


    def annotate(self, data, prompt, return_probs=False, few_shot_str=""):
        """Annotate data."""
        logger.info("prompt = {prompt}".format(prompt=prompt))
        logger.info("few_shot_str = {few_shot_str}".format(few_shot_str=few_shot_str))

        # generate dialogs
        dialogs = to_dialog(data, prompt, few_shot_str=few_shot_str)
        logger.info("generated dialogs")

        # generate results
        if return_probs:
            results, probs = self.generator._send_request(dialogs, batch_size=10, return_probs=True)
            return results, probs
        else:
            results = self.generator._send_request(dialogs, batch_size=10)
            logger.info("generated results")
            
            meta_result = [result for result in results]
            binary_result = [1 if x.lower().find("yes") != -1 and x.lower().find("no") == -1 else 0 for x in meta_result]

            return meta_result, binary_result

    def generate_few_shot_example(self, data, prompt, method="usp"):
        # generate results
        results, probs = self.annotate(data, prompt, return_probs=True)

        logger.info("generating few-shot examples with {method}: {prompt}".format(prompt=prompt, method=method))
        dialogs = to_dialog(data, prompt)
        if method == "usp":
            few_shot_str = select_usp_examples(dialogs, results, probs, 4)
        elif method == "active_learning":
            few_shot_str = select_boundary_examples(dialogs, results, probs, 4)
        logger.info(few_shot_str)

        return few_shot_str

    def generate_few_shot_example_batch(self, data, keywords, method="usp"):
        few_shot_str_df = pd.DataFrame(columns=keywords)
        for key_idx, keyword in enumerate(keywords):
            logger.info("processing keyword: {key}".format(key=keyword))

            # read prompt
            result_path = config["EXPERIMENT"]["FINAL_PROMPT_PATH"].format(key_idx=key_idx)
            if not os.path.exists(result_path):
                self.prompt_selector.analyze(keywords)
            prompt_df = read_csv_file(result_path)

            # select prompt
            prompt = self.prompt_selector.select_prompt(prompt_df, keyword, criteria='maj_vote')
            few_shot_str = self.generate_few_shot_example(data, prompt, method=method)
            few_shot_str_df.at[0, keyword] = few_shot_str
        few_shot_str_df.to_csv(config["EXPERIMENT"]["FEW_SHOT_PATH"], index=False)

    def annotate_batch(self, data, keywords, prompt_existed=False, add_few_shot=False):
        """Annotate data in batch.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to be annotated.
            The data should contain a column named "context" which contains the text to be annotated.
        keywords : list
            The list of keywords.
        prompt_existed : bool
            Whether the prompt has been generated.
        """
        
        if add_few_shot:
            few_shot_str_df = pd.read_csv(config["EXPERIMENT"]["FEW_SHOT_PATH"])

        for key_idx, keyword in enumerate(keywords):
            logger.info("processing keyword: {key}".format(key=keyword))
            
            if not prompt_existed:
                prompt_df = read_csv_file(config["EXPERIMENT"]["PROMPT_PATH"].format(key_idx=key_idx))
                prompts = prompt_df["{keyword}_prompt".format(keyword=keyword)].tolist()

                data["{}_result".format(keyword)] = 0.0
                for index, prompt in enumerate(prompts):
                    logger.info("processing prompt: {prompt}".format(prompt=prompt.split("\n")[0]))
                    
                    meta_result, binary_result = self.annotate(data, prompt)

                    data["{keyword}_prompt{id}_meta".format(keyword=keyword, id=index)] = meta_result
                    data["{keyword}_prompt{id}".format(keyword=keyword, id=index)] = binary_result
                    data["{}_result".format(keyword)] += data["{keyword}_prompt{id}".format(keyword=keyword, id=index)]
                
                    ## save test data
                    data.to_csv(config["EXPERIMENT"]["SLICE_RESULT_PATH"], index=False)
                
                logger.info(data.info())
                data.to_csv(config["EXPERIMENT"]["SLICE_RESULT_PATH"], index=False)
            else:
                # read prompt
                result_path = config["EXPERIMENT"]["FINAL_PROMPT_PATH"].format(key_idx=key_idx)
                if not os.path.exists(result_path):
                    self.prompt_selector.analyze(keywords)
                prompt_df = read_csv_file(result_path)
                # select prompt
                prompt = self.prompt_selector.select_prompt(prompt_df, keyword, criteria='maj_vote')
                
                few_shot_str = ""
                if add_few_shot:
                    few_shot_str = few_shot_str_df.at[0, keyword]
                meta_result, binary_result = self.annotate(data, prompt, few_shot_str=few_shot_str)
                
                data['label_{keyword}_meta'.format(keyword=keyword)] = meta_result
                data['label_{keyword}'.format(keyword=keyword)] = binary_result
                data.to_csv(config["EXPERIMENT"]["FINAL_RESULT_PATH"], index=False)
            

if __name__ == "__main__":
    args = parseArg()
    result_path = os.path.join("result", args.exp_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    config.update_path(args.exp_name)

    # read data and keywords
    keywords = read_txt_file(config["EXPERIMENT"]["KEYWORDS_PATH"])
    df = read_csv_file(config["EXPERIMENT"]["DATA_PATH"])

    # label
    slicer = Slicer(model_name="flan-t5", model_size="xxl")
    # random select data
    test_data = df.sample(n=config["SLICING"]["SAMPLE_SIZE"], random_state=42)
    # slicer.generate_few_shot_example_batch(df, keywords, method="usp")
    # slicer.annotate_batch(test_data, keywords, prompt_existed=False)
    slicer.annotate_batch(df, keywords, prompt_existed=True, add_few_shot=True)