from llama import Llama2Wrapper
from utils.load_config import read_yaml_config
from utils.log import get_logger
from datasets import load_dataset, load_from_disk
import random
import pandas as pd

logger = get_logger("INFO", "run model")
generator = None

SYSTEM_PROMPT='''You will receive several paragraphs with their title. The you will receive a question. Answer the question according to the paragraphs. Answer in the following format:

# Answer
your answer here

Your answer shoud be a short phrase less than 10 words. You must not type anything except the answer phrase.'''

# 1-shot example
EXAMPLE_QUESTION='''What is the name of the first person to walk on the moon?'''
EXAMPLE_ANSWER='''Neil Armstrong'''
EXAMPLE_TITLE=['''Apollo 11''']
EXAMPLE_PASSAGE=['''Apollo 11 (July 16–24, 1969) was the American spaceflight that first landed humans on the Moon. \
Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969, at 20:17 UTC, \
and Armstrong became the first person to step onto the Moon's surface six hours and 39 minutes later, on July 21 at 02:56 UTC. \
Aldrin joined him 19 minutes later, and they spent about two and a quarter hours together exploring the site they had named \
Tranquility Base upon landing. Armstrong and Aldrin collected 47.5 pounds (21.5 kg) of lunar material to bring back to Earth \
as pilot Michael Collins flew the Command Module Columbia in lunar orbit, and were on the Moon's surface for 21 hours, \
36 minutes before lifting off to rejoin Columbia.''']
                 
def init():
    # load model
    model_size = "13b-chat"
    global generator
    try:
        generator = Llama2Wrapper(
            "/home/yiningho/workspace/datadisk/llama/llama-2-{}".format(model_size),
            is_chat_model=True,
            load_4bit=True,
            batch_size=10
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
            batch_size=10
        )

def _send_request(
    dialogs,
    max_gen_len=1024,
    temperature=0.01,
    top_p=0.9,
    batch_size=40
):
    '''
    example for dialogs:[[{"role": "user", "content": "what is the recipe of mayonnaise?"}]]
    '''
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size
    )
    return [result[0]['generated_text'].strip() for result in results]

def run_model(args):
    # config
    config = read_yaml_config("./config.yaml")
    logger.info(args)
    logger.info(config)
    init()

    # load dataset
    dataset = load_dataset("hotpot_qa", "distractor")['train']
    logger.info("loaded dataset")
    logger.info(dataset.column_names)
    logger.info(len(dataset))

    # filter dataset
    dataset = dataset.filter(
        lambda example: len(
            ' '.join(
                [
                    ' '.join(sentences) 
                    for sentences in example["context"]["sentences"]
                ]
                + [
                    title
                    for title in example["context"]["title"]
                ]
                + [example["question"]]
                + [SYSTEM_PROMPT]
            ).split()
        ) < 650, 
        with_indices=False
    )
    logger.info(len(dataset))

    # random sample dataset
    dataset = dataset.shuffle(seed=random.randint(0, 1000)).select(range(config["RUN"]["SAMPLE_SIZE"]))

    # generate dialogs
    dialogs = [
        [
            {
                "role": "system", 
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user", 
                "content": '\n\n'.join(
                    [
                        "# Title\n{title}\n# Passage\n{passage}".format(
                            title=title,
                            passage=' '.join(sentences)
                        ) 
                        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"])
                    ]
                    + [
                        '''# Question
{question} Answer in the following format:

# Answer
your answer here

Your answer shoud be a short phrase strictly less than 10 words. You must not type anything except the answer phrase.'''.format(question=row["question"])
                    ]
                )
            }
        ]
        for row in dataset
    ]
    
    logger.info("generated dialogs")

    # generate results
    results = _send_request(dialogs=dialogs, max_gen_len=1700, temperature=0.02, batch_size=10)
    logger.info("generated results")
    
    # save results to datasets
    dataset = dataset.add_column("generated_answer", results)
    logger.info("added column")

    # save to disk
    dataset.save_to_disk(config["RUN"]["OUTPUT_PATH"])

    # save to csv
    transform_data(config)

def transform_data(config):
    # save to csv
    dataset = load_from_disk(config["RUN"]["OUTPUT_PATH"])
    df = pd.DataFrame()
    df['full_text'] = [
        '\n\n'.join(
                    [
                        "# Title\n{title}\n# Passage\n{passage}".format(
                            title=title,
                            passage=' '.join(sentences)
                        ) 
                        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"])
                    ]
                    + [
                        '# Question\n{question}'.format(question=row["question"])
                    ]
                )
        for row in dataset
    ]

    # text is the part that we use to slice data
    df['text'] = [
        row["question"]
        for row in dataset
    ]
    df['answer'] = [
        row['answer']
        for row in dataset
    ]
    df['generated_answer'] = [
        row['generated_answer']
        for row in dataset
    ]
    df['level'] = [
        row['level']
        for row in dataset
    ]
    df['supporting_facts'] = [
        str(row['supporting_facts'])
        for row in dataset
    ]
    df['id'] = [
        row['id']
        for row in dataset
    ]
    df.to_csv(config["RUN"]["CSV_PATH"], index=False)