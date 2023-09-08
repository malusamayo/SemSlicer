import pandas as pd
from utils.file import read_csv_file
from utils.load_config import read_yaml_config
from utils.log import get_logger
from cubam_py.Binary1dSignalModel import Binary1dSignalModel

logger = get_logger("INFO", "dataprocess")

def dataprocess(args):
    # config
    config = read_yaml_config("./config.yaml")
    logger.info(args)
    logger.info(config)
    df = read_csv_file(config["DATA_PROCESS"]["RESULT_PATH"])
    print(df.info())
    column_names = df.columns.tolist()
    for index in range(len(config["DATA_PROCESS"]["KEYWORDS"])):
        key = config["DATA_PROCESS"]["KEYWORDS"][index]
        golden_label = config["DATA_PROCESS"]["GOLDEN_LABELS"][index]
        logger.info("processing keyword: {key}, with golden label: {golden_label}".format(key=key, golden_label=golden_label))
        prompt_num = 0
        df["{key}_result".format(key=key)] = 0
        for i in range(config["DATA_PROCESS"]["PROMPT_NUM"]):
            if "{key}_prompt{id}_meta".format(key=key, id=i) in column_names:
                prompt_num += 1
                df["{key}_prompt{id}".format(key=key, id=i)] = df["{key}_prompt{id}_meta".format(key=key, id=i)].apply(
                    lambda x: 
                        1 if x.find("My Answer is Yes") != -1 
                        else 0
                )
                df["{key}_result".format(key=key)] += df["{key}_prompt{id}".format(key=key, id=i)]
                true_positive = len(df[(df['{key}_prompt{id}'.format(key=key,id=i)] == 1) & (df['label'] == golden_label)])
                false_positive = len(df[(df['{key}_prompt{id}'.format(key=key,id=i)] == 1) & (df['label'] != golden_label)])
                true_negative = len(df[(df['{key}_prompt{id}'.format(key=key,id=i)] == 0) & (df['label'] != golden_label)])
                false_negative = len(df[(df['{key}_prompt{id}'.format(key=key,id=i)] == 0) & (df['label'] == golden_label)])
                # logger.info("{index}: true_positive={true_positive}, false_positive={false_positive}, true_negative={true_negative}, false_negative={false_negative}".format(
                #     index=i,
                #     true_positive=true_positive,
                #     false_positive=false_positive,
                #     true_negative=true_negative,
                #     false_negative=false_negative
                # ))
                logger.info("{index}: precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, accuracy={accuracy:.4f}".format(
                    index=i,
                    precision=true_positive / (true_positive + false_positive),
                    recall=true_positive / (true_positive + false_negative),
                    f1=2 * true_positive / (2 * true_positive + false_positive + false_negative),
                    accuracy=(true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
                ))
            else:
                break
        output = [[len(df), prompt_num, len(df) * prompt_num]]
        i = 0
        for index, row in df.iterrows():
            for prompt_id in range(prompt_num):
                output.append([i, prompt_id, row["{key}_prompt{num}".format(key=key, num=prompt_id)]])
            i += 1
        with open(config["DATA_PROCESS"]["TMP_PATH"], 'w') as f:
            for line in output:
                f.write(" ".join([str(i) for i in line]) + "\n")
        # majority voting
        df["{key}_result".format(key=key)] = df["{key}_result".format(key=key)].apply(
            lambda x: 
                1 if x > prompt_num / 2 
                else 0
        )
        beta = len(df[(df['{key}_result'.format(key=key)] == 1)]) / len(df)
        logger.info("beta = {beta}".format(beta=beta))
        model = Binary1dSignalModel()
        model.set_model_param(prm={'beta': beta})
        model.load_data(config["DATA_PROCESS"]["TMP_PATH"])
        model.optimize_param()
        eprm = {
            'wkr' : model.get_worker_param(), 
            'img' : model.get_image_param()
        }
        for index in range(len(eprm['wkr'])):
            logger.info("{index}, tau/w={num:.4f}, sigma={sigma:.4f}".format(index=index, num=eprm['wkr'][index][1] / eprm['wkr'][index][0], sigma=1 / eprm['wkr'][index][0]))