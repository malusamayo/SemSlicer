import os
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


def select_few_shot_examples(dialogs, results, probs, nums):
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

class Slicer(object):

    def __init__(self, model_name="flan-t5", model_size="xxl"):
        self.generator = Generator(model_name, model_size)
        self.prompt_selector = PromptSelector()

    def __call__(self, data, condition, condition_type='keyword'):
        """Slice data on condition."""
        pass

    def annotate(self, data, prompt):
        """Annotate data."""
        logger.info("prompt = {prompt}".format(prompt=prompt))

        # generate dialogs
        dialogs = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT.format(question=prompt, passage=passage)}
            ]
            for passage in data['context']
        ]
        logger.info("generated dialogs")

        # generate results
        results = self.generator._send_request(dialogs, temperature=0.2, batch_size=10)
        logger.info("generated results")
        
        meta_result = [result for result in results]
        binary_result = [1 if x.lower().find("yes") != -1 and x.lower().find("no") == -1 else 0 for x in meta_result]

        return meta_result, binary_result

    def annotate_with_few_shot_examples(self, data, prompt):
        logger.info("prompt = {prompt}".format(prompt=prompt))

        # generate dialogs
        dialogs = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT.format(question=prompt, passage=passage)}
            ]
            for passage in data['context']
        ]
        logger.info("generated dialogs")

        # generate results
        results, probs = self.generator._send_request(dialogs, temperature=0.2, batch_size=20, return_probs=True, labels=['yes', 'no'])

        logger.info("generating few-shot examples")
        few_shot_str = select_few_shot_examples(dialogs, results, probs, 4)

        # regenerate dialogs
        dialogs = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": few_shot_str + PROMPT.format(question=prompt, passage=row)}
            ]
            for row in data['context']
        ]

        # regenerate results
        results = self.generator._send_request(dialogs, temperature=0.2, batch_size=20)
        logger.info("generated results")
        
        meta_result = [result for result in results]
        binary_result = [1 if x.lower().find("yes") != -1 and x.lower().find("no") == -1 else 0 for x in meta_result]

        return meta_result, binary_result

    def annotate_batch(self, data, keywords, prompt_existed=False):
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
        for key_idx, keyword in enumerate(keywords):
            logger.info("processing keyword: {key}".format(key=keyword))
            
            if not prompt_existed:
                prompt_df = read_csv_file(config["SLICING"]["PROMPT_PATH"] + "prompt_result_" + str(key_idx) + ".csv")
                prompts = prompt_df["{keyword}_prompt".format(keyword=keyword)].tolist()

                data["{}_result".format(keyword)] = 0.0
                for index, prompt in enumerate(prompts):
                    logger.info("processing prompt: {prompt}".format(prompt=prompt.split("\n")[0]))
                    
                    meta_result, binary_result = self.annotate(data, prompt)

                    data["{keyword}_prompt{id}_meta".format(keyword=keyword, id=index)] = meta_result
                    data["{keyword}_prompt{id}".format(keyword=keyword, id=index)] = binary_result
                    data["{}_result".format(keyword)] += data["{keyword}_prompt{id}".format(keyword=keyword, id=index)]
                
                    ## save test data
                    data.to_csv(config["SLICING"]["OUTPUT_PATH"], index=False)
                
                logger.info(data.info())
                data.to_csv(config["SLICING"]["OUTPUT_PATH"], index=False)
            else:
                # read prompt
                result_path = config["LABEL"]["PROMPT_PATH"] + "prompt_final_result_" + str(key_idx) + ".csv"
                if not os.path.exists(result_path):
                    self.prompt_selector.analyze()
                prompt_df = read_csv_file(result_path)
                # select prompt
                prompt = self.prompt_selector.select_prompt(prompt_df, keyword, criteria='maj_vote')
                
                meta_result, binary_result = self.annotate(data, prompt)
                
                data['label_{keyword}_meta'.format(keyword=keyword)] = meta_result
                data['label_{keyword}'.format(keyword=keyword)] = binary_result
                data.to_csv(config["LABEL"]["OUTPUT_PATH"], index=False)
            

if __name__ == "__main__":
    # read data
    df = read_csv_file(config["LABEL"]["RESULT_PATH"])
    logger.info(df.info())

    # read keywords
    keywords = read_txt_file(config["EXPERIMENT"]["KEYWORDS_PATH"])

    # label
    slicer = Slicer(model_name="flan-t5", model_size="large")

    # random select data
    test_data = df.sample(n=config["SLICING"]["SAMPLE_SIZE"], random_state=42)
    # slicer.annotate_batch(test_data, keywords, prompt_existed=False)
    slicer.annotate_batch(df, keywords, prompt_existed=True)