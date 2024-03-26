import os
import re
import torch
import pandas as pd
from typing import List, Dict
from .utils.parseArgument import parseArg
from .utils.log import get_logger
from .utils.file import read_txt_file, read_csv_file
from .utils.config import config
from .model.llm_server import Generator
from .promptgen.generator import ExampleGenerator
from .promptgen.selector import PromptSelector, select_usp_examples, select_boundary_examples, select_random_examples

logger = get_logger("INFO", "label")

# SYSTEM_PROMPT =  '''In each round of the conversation, you will receive a text and a question. \
# The question is about the text. Answer the question according to the text.
# Please first answer the question with "My answer is yes" \
# or "My answer is no", then explain your reason. Try your best.'''

# PROMPT = '''# Text
# {passage}

# # Question
# {question} Your answer is yes or no.

# # Answer
# My answer is '''


SYSTEM_PROMPT =  '''{question} Answer ONLY yes or no.''' # Do NOT explain your answer.'''
PROMPT = '''Text: {passage}
Answer: '''

def from_few_shot_str(few_shot_str):
    texts = re.split(r'Text: |Answer: ', few_shot_str)
    texts = texts[1:]
    dialogs = texts[::2]
    results = texts[1::2]
    dialogs = [[
        {"role": "system", "content": ""},
        {"role": "user", "content": "Text: " + dialog + "Answer: "},
    ] for dialog in dialogs]
    results = [x.strip() for x in results]
    return dialogs, results

def to_few_shot_str(dialogs, results):
    few_shot_examples = [dialog[1]["content"] + result for dialog, result in zip(dialogs, results)]
    few_shot_str = '\n\n'.join(few_shot_examples)
    few_shot_str += '\n\n'
    return few_shot_str

def to_dialog(data, prompt, few_shot_str=""):
    dialogs = [
        [
            {"role": "system", "content": SYSTEM_PROMPT.format(question=prompt) + few_shot_str},
            {"role": "user", "content": PROMPT.format(question=prompt, passage=passage)}
        ]
        for passage in data['context']
    ]
    return dialogs

class Slicer(object):

    def __init__(self, model_name="dummy"):
        self.generator = Generator(model_name)
        self.prompt_selector = PromptSelector()
        self.example_generator = ExampleGenerator()
        self.teacher = Generator(model_name='gpt-4-turbo-preview')

        logger.info("Slicer initialized, model_name = {model_name}".format(model_name=model_name))

    def calibrate_prob(self, prompt: str, probs: torch.Tensor, labels: List[str], few_shot_str: str=""):
        """Calibrate the probability."""
        
        logger.info("calibrating probability")
        dialogs = [
            [
                {"role": "system", "content": SYSTEM_PROMPT.format(question=prompt) + few_shot_str},
                {"role": "user", "content": PROMPT.format(question=prompt, passage="")}
            ]
        ]
        _, base_probs = self.generator._send_request(dialogs, return_probs=True, labels=labels)
        logger.info("base_probs = {base_probs}".format(base_probs=base_probs))

        # calibrate the probability for n*2 tensor
        calibrated_probs = probs / base_probs
        results = [labels[prob.argmax()] for prob in calibrated_probs]
        return results, calibrated_probs

    def annotate(
        self, 
        data: pd.DataFrame, 
        prompt: str, 
        return_probs: bool=False, 
        use_calibrate: bool=False, 
        few_shot_str: str="",
        labels: List[str] = ["yes", "no"],
        label_map: Dict[str, int] = {"yes": 1, "no": 0}
    ):
        """Annotate data."""
        logger.info("prompt = {prompt}".format(prompt=prompt))
        logger.info("few_shot_str = {few_shot_str}".format(few_shot_str=few_shot_str))

        # generate dialogs
        dialogs = to_dialog(data, prompt, few_shot_str=few_shot_str)
        logger.info("generated dialogs")

        probs = None
        # generate results
        if return_probs:
            results, probs = self.generator._send_request(dialogs, batch_size=10, return_probs=True)
            if use_calibrate:
                meta_result, probs = self.calibrate_prob(prompt, probs, labels, few_shot_str=few_shot_str)
        else:
            results = self.generator._send_request(dialogs, batch_size=10)
            meta_result = [result for result in results]
        
        logger.info("generated results")
        logger.info(f"total_tokens = {self.generator.compute_total_tokens(dialogs)}")
        
        binary_result = [label_map['yes'] if x.lower().find("yes") != -1 and x.lower().find("no") == -1 else label_map['no'] for x in meta_result]
        return meta_result, binary_result, probs

    def synthesize_examples(self,
        prompt,
        selected_dialogs, 
        selected_results, 
        labels, 
        num
    ):
        simplified_result = ['yes' if x.lower().find("yes") != -1 and x.lower().find("no") == -1 else 'no' for x in selected_results]
        counts = {label: simplified_result.count(label) for label in labels}
        underrepresented_label = min(counts, key=counts.get)
        target_num = int(num/2)

        # generate extra dialogs if results are imbalanced
        if counts[underrepresented_label] < target_num:
            few_shot_str = to_few_shot_str(selected_dialogs, selected_results)
            context_prompt = SYSTEM_PROMPT.format(question=prompt) + few_shot_str

            extra_examples_str = self.example_generator.generate_examples(context_prompt, underrepresented_label, target_num - counts[underrepresented_label])
            extra_dialogs, extra_results = from_few_shot_str(extra_examples_str)

            selected_dialogs += extra_dialogs
            selected_results += extra_results
            simplified_result = ['yes' if x.lower().find("yes") != -1 and x.lower().find("no") == -1 else 'no' for x in selected_results]

        # interleave positive and negative examples
        positive_examples = [(dialog, result) for dialog, result, label in zip(selected_dialogs, selected_results, simplified_result) if label == 'yes'][:target_num]
        negative_examples = [(dialog, result) for dialog, result, label in zip(selected_dialogs, selected_results, simplified_result) if label == 'no'][:target_num]
        interleaved_examples = [v for p in zip(negative_examples, positive_examples) for v in p]
        selected_dialogs, selected_results = zip(*interleaved_examples)
        return selected_dialogs, selected_results


    def generate_few_shot_example(self, 
        data, 
        prompt, 
        num=8, 
        input_sampling_strategy="random",
        output_label_source="self",
        synthesize=False,
        labels: List[str] = ["yes", "no"],
        clusters: List[int] = None
    ):
        """Generate few-shot examples."""

        logger.info("generating few-shot examples for: {prompt}".format(prompt=prompt))
        logger.info("args: num = {num}, input_sampling_strategy = {input_sampling_strategy}, output_label_source = {output_label_source}, synthesize = {synthesize}".format(
            num=num, input_sampling_strategy=input_sampling_strategy, output_label_source=output_label_source, synthesize=synthesize))
        
        dialogs = to_dialog(data, prompt)

        if input_sampling_strategy == "random":
            selected_dialogs, _ = select_random_examples(dialogs, num)
        elif input_sampling_strategy == "diversity":
            # by default, the dataset should contain a column named "cluster"
            # we use SentenceTransformers+KMeans to cluster the data
            clusters = data['cluster'].tolist() 
            selected_dialogs, _ = select_random_examples(dialogs, num, clusters=clusters)
        elif input_sampling_strategy == "human":
            # for now we simulate human with gold labels
            # clusters store the gold slicing labels and we use them to select examples
            selected_dialogs, selected_idx = select_random_examples(dialogs, num, clusters=clusters)
            selected_results = ['yes' if clusters[i] else 'no' for i in selected_idx]
        elif input_sampling_strategy == "usp":
            data = data.sample(n=num*4, random_state=42)
            results, _, probs = self.annotate(data, prompt, return_probs=True)
            selected_dialogs, selected_results = select_usp_examples(dialogs, results, probs, num)
        elif input_sampling_strategy == "active_learning":
            data = data.sample(n=num*4, random_state=42)
            results, _, probs = self.annotate(data, prompt, return_probs=True)
            selected_dialogs = select_boundary_examples(dialogs, probs, num)
        else:
            raise NotImplementedError("input_sampling_strategy = {input_sampling_strategy} is not implemented".format(input_sampling_strategy=input_sampling_strategy))
        
        if output_label_source == "self":
            selected_results = self.generator._send_request(selected_dialogs, temperature=0)
        elif output_label_source == "teacher":
            selected_results = self.teacher._send_request(selected_dialogs, temperature=0)
        elif output_label_source == "human":
            pass
        else:
            raise NotImplementedError("output_label_source = {output_label_source} is not implemented".format(output_label_source=output_label_source))
            
        if synthesize:
            selected_dialogs, selected_results = self.synthesize_examples(prompt, selected_dialogs, selected_results, labels, num)
        
        few_shot_str = to_few_shot_str(selected_dialogs, selected_results)
        
        logger.info(few_shot_str)

        return few_shot_str

    def generate_few_shot_example_batch(self, 
        data, 
        keywords, 
        num=8, 
        input_sampling_strategy="random",
        output_label_source="self",
        synthesize=False,
        select_prompt=False
    ):
        few_shot_str_df = pd.DataFrame(columns=keywords)
        for keyword in keywords:
            logger.info("processing keyword: {key}".format(key=keyword))

            # read prompt
            if select_prompt:
                result_path = config["EXPERIMENT"]["FINAL_PROMPT_PATH"]
                if not os.path.exists(result_path):
                    self.prompt_selector.analyze(keywords)
                prompt_df = read_csv_file(result_path)
                prompt = self.prompt_selector.select_prompt(prompt_df, keyword, criteria='maj_vote')
            else:
                # use the default prompt
                prompt_df = read_csv_file(config["EXPERIMENT"]["PROMPT_PATH"])
                prompt = self.prompt_selector.select_prompt(prompt_df, keyword, criteria='default')

            clusters = None
            # select prompt
            few_shot_str = self.generate_few_shot_example(data, prompt, num=num, 
                input_sampling_strategy=input_sampling_strategy, output_label_source=output_label_source, 
                synthesize=synthesize, clusters=clusters)
            few_shot_str_df.at[0, keyword] = few_shot_str
            few_shot_str_df.to_csv(config["EXPERIMENT"]["FEW_SHOT_PATH"], index=False)

    def annotate_batch(self, data, keywords, select_prompt=False, use_calibrate=False, add_few_shot=False, use_cache=False):
        """Annotate data in batch.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to be annotated.
            The data should contain a column named "context" which contains the text to be annotated.
        keywords : list
            The list of keywords.
        select_prompt : bool
            Whether the prompt has been generated.
        use_calibrate : bool
            Whether to calibrate the probability.
        add_few_shot : bool
            Whether to add few-shot examples.
        """
        
        if add_few_shot:
            if not os.path.exists(config["EXPERIMENT"]["FEW_SHOT_PATH"]):
                self.generate_few_shot_example_batch(data, keywords, 
                    num=config["EXAMPLES"]["FEW_SHOT_SIZE"], 
                    input_sampling_strategy=config["EXAMPLES"]["SAMPLE_STRATEGY"],
                    output_label_source=config["EXAMPLES"]["LABEL_SOURCE"], 
                    synthesize=config["EXAMPLES"]["SYNTHESIZE"],
                    select_prompt=select_prompt)
            few_shot_str_df = pd.read_csv(config["EXPERIMENT"]["FEW_SHOT_PATH"])


        if select_prompt:
            if not os.path.exists(config["EXPERIMENT"]["FINAL_PROMPT_PATH"]):
                self.prompt_selector.analyze(keywords)
            prompt_df = read_csv_file(config["EXPERIMENT"]["FINAL_PROMPT_PATH"])
        else:
            prompt_df = read_csv_file(config["EXPERIMENT"]["PROMPT_PATH"])

        # partial results exist
        if use_cache and os.path.exists(config["EXPERIMENT"]["SLICE_RESULT_PATH"]):
            data = pd.read_csv(config["EXPERIMENT"]["SLICE_RESULT_PATH"])
            
        for keyword in keywords:
            logger.info("processing keyword: {key}".format(key=keyword))

            few_shot_str = ""
            if add_few_shot:
                few_shot_str = few_shot_str_df.at[0, keyword]
            
            prompts = prompt_df["{keyword}_prompt".format(keyword=keyword)].tolist()

            # only use the selected prompt
            if select_prompt:
                prompts = [self.prompt_selector.select_prompt(prompt_df, keyword, criteria='maj_vote')]
            
            # skip if exists
            if use_cache and "{keyword}_result".format(keyword=keyword) in data.columns:
                continue

            data["{}_result".format(keyword)] = 0.0
            for index, prompt in enumerate(prompts):
                logger.info("processing prompt: {prompt}".format(prompt=prompt.split("\n")[0]))
                
                meta_result, binary_result, _ = self.annotate(data, prompt, return_probs=use_calibrate, use_calibrate=use_calibrate, few_shot_str=few_shot_str)

                data["{keyword}_prompt{id}_meta".format(keyword=keyword, id=index)] = meta_result
                data["{keyword}_prompt{id}".format(keyword=keyword, id=index)] = binary_result
                data["{}_result".format(keyword)] += data["{keyword}_prompt{id}".format(keyword=keyword, id=index)]
            
                ## save test data
                data.to_csv(config["EXPERIMENT"]["SLICE_RESULT_PATH"], index=False)
            
            logger.info(data.info())
            data.to_csv(config["EXPERIMENT"]["SLICE_RESULT_PATH"], index=False)     
                
            

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
    slicer = Slicer(model_name="flan-t5-xxl")
    # random select data
    test_data = df.sample(n=config["SLICING"]["SAMPLE_SIZE"], random_state=42)
    slicer.generate_few_shot_example_batch(df, keywords)
    # slicer.annotate_batch(test_data, keywords, prompt_existed=False)
    # slicer.annotate_batch(df, keywords, prompt_existed=True, add_few_shot=True)