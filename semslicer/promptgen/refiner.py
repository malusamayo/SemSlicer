import pandas as pd
from semslicer.slicer import Slicer

class Refiner:
    def __init__(self, config):
        self.config = config
        self.few_shot_examples = pd.read_csv(config["EXPERIMENT"]["FEW_SHOT_PATH"]) if config["SLICING"]["FEW_SHOT"] else None
        self.prompts = pd.read_csv(config["EXPERIMENT"]["PROMPT_PATH"])
        self.annotated_examples = pd.read_csv(config["EXPERIMENT"]["SLICE_RESULT_PATH"])
        self.slicer = Slicer(model_name="dummy")

    def sample_examples(self, keyword, label, sample_size):
        '''
        Sample examples from the data
        '''
        return self.annotated_examples[self.annotated_examples[f"{keyword}_result"] == label].sample(sample_size)[['context', f"{keyword}_prompt0_meta"]]

    def inspect(self, keyword, sample_size=5):
        '''
        Inspect the slicing result for a keyword
        '''
        print('Prompt:', self.prompts[f"{keyword}_prompt"])
        if self.few_shot_examples is not None:
            print('Few shot examples:', self.few_shot_examples[keyword])
        
        positive_examples = self.sample_examples(keyword, label=1, sample_size=sample_size)
        print('Sampled positive examples:')
        for index, row in positive_examples.iterrows():
            print('Text:', row['context'])
            print('Answer:', row[f"{keyword}_prompt0_meta"])
            print('-------------------')

        negative_examples = self.sample_examples(keyword, label=0, sample_size=sample_size)
        print('Sampled negative examples:')
        for index, row in negative_examples.iterrows():
            print('Text:', row['context'])
            print('Answer:', row[f"{keyword}_prompt0_meta"])
            print('-------------------')
        

    def refine(self, keyword, prompt, sample_examples=None):
        '''
        Refine the slicing result for a keyword
        '''
        self.prompts[f"{keyword}_prompt"] = prompt
        data = self.annotated_examples if sample_examples is None else sample_examples
        if self.few_shot_examples is not None:
            self.few_shot_examples[keyword] = self.slicer.generate_few_shot_example(data, prompt, method="random")
    
    def save(self):
        '''
        Save the changes
        '''
        self.prompts.to_csv(self.config["EXPERIMENT"]["PROMPT_PATH"], index=False)
        if self.few_shot_examples is not None:
            self.few_shot_examples.to_csv(self.config["EXPERIMENT"]["FEW_SHOT_PATH"], index=False)



