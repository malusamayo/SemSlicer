from utils.load_config import read_yaml_config
from utils.file import read_txt_file
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree
import language_tool_python
from datasets import load_dataset
import pandas as pd

nlp = StanfordCoreNLP('../stanford-corenlp-4.5.4')
language_tool = language_tool_python.LanguageTool('en-US')

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
    print("{keyword}: {label}".format(keyword=keyword, label=label))

    # invalid label
    if label not in prompt_templates:
        print(tree)
        return []
    
    # get prompts using templates
    prompts = [item.format(keyword=keyword) for item in prompt_templates[label]]

    #correct grammars
    prompts = [language_tool.correct(item) for item in prompts]

    print("successfully found prompts for {keyword}".format(keyword=keyword))
    return prompts

def slicing(args):
    # config
    config = read_yaml_config("./config.yaml")
    print(args)
    print(config)

    # load keyword file
    keywords = read_txt_file(config["SLICING"]["KEYWORDS_PATH"])

    # load dataset
    dataset = load_dataset("tweet_eval", "emotion")
    df = pd.DataFrame(dataset['train'])
    print("loaded dataset")
    print(df.info())

    # process keywords
    for keyword in keywords:
        # get prompts
        prompts = _find_prompts(keyword, config["SLICING"]["PROMPT_TEMPLATES"])

        # random select data
        test_data = df.sample(n=config["SLICING"]["SAMPLE_SIZE"])

        for prompt in prompts:
            pass
    nlp.close()
    language_tool.close()