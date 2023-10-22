from llama import Llama2Wrapper
from utils.load_config import read_yaml_config
from utils.log import get_logger
from datasets import load_dataset, load_from_disk, concatenate_datasets
import random
import pandas as pd
from llm_server import Generator

logger = get_logger("INFO", "run model")
generator = Generator("llama2", "13b-chat")

SYSTEM_PROMPT=''''''

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

def row_to_dialog(row):
    return [
        {
            "role": "system", 
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user", 
            "content": '''
# Context
{context}

# Question
{question} 

Select the correct answer from the following options:
1. {ans0}
2. {ans1}
3. {ans2}

Reply only the answer phrase. Do not write anything else except the answer.
# Answer
'''.format(context=row["context"], question=row["question"], ans0=row["ans0"], ans1=row["ans1"], ans2=row["ans2"])
        }
    ]

#     return [
#         {
#             "role": "system", 
#             "content": SYSTEM_PROMPT
#         },
#         {
#             "role": "user", 
#             "content": '\n\n'.join(
#                 [
#                     "# Title\n{title}\n# Passage\n{passage}".format(
#                         title=title,
#                         passage=' '.join(sentences)
#                     ) 
#                     for title, sentences in zip(row["context"]["title"], row["context"]["sentences"])
#                 ]
#                 + [
#                     '''# Question
# {question} Answer in the following format:

# Your answer shoud be a short phrase strictly less than 10 words. You must not type anything except the answer phrase.

# # Answer
# '''.format(question=row["question"])
#                 ]
#             )
#         }
#     ]

def load_and_filter_dataset(task, cols, split):
    # load dataset
    datasets = []
    for col in cols:
        datasets.append(load_dataset(task, col)[split])
    dataset = concatenate_datasets(datasets)
    logger.info("loaded dataset")
    logger.info(dataset.column_names)
    logger.info(len(dataset))

    # # filter dataset
    # dataset = dataset.filter(
    #     lambda example: len(
    #         ' '.join(
    #             [
    #                 ' '.join(sentences) 
    #                 for sentences in example["context"]["sentences"]
    #             ]
    #             + [
    #                 title
    #                 for title in example["context"]["title"]
    #             ]
    #             + [example["question"]]
    #             + [SYSTEM_PROMPT]
    #         ).split()
    #     ) < 450, 
    #     with_indices=False
    # )

    logger.info(len(dataset))

    # random sample dataset
    dataset = dataset.shuffle(seed=42).select(range(config["RUN"]["SAMPLE_SIZE"]))
    logger.info(len(dataset))

    return dataset

def run_model(args):
    # config
    global config
    config = read_yaml_config("./config.yaml")
    logger.info(args)
    logger.info(config)

    dataset = load_and_filter_dataset("heegyu/bbq", ["Age", "Gender_identity"], 'test')

    # generate dialogs
    dialogs = [ row_to_dialog(row) for row in dataset ]
    
    logger.info("generated dialogs")

    # generate results
    results = generator._send_request(dialogs=dialogs, max_gen_len=1700, temperature=0.02, batch_size=10)
    logger.info("generated results")
    
    # save results to datasets
    dataset = dataset.add_column("generated_answer", results)
    logger.info("added column")

    # save to disk
    dataset.save_to_disk(config["RUN"]["OUTPUT_PATH"])

    # save to csv
    dataset.to_csv(config["RUN"]["CSV_PATH"])
    # transform_data(config)



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