from utils.file import read_csv_file, read_txt_file
from utils.load_config import read_yaml_config
from utils.log import get_logger
from cubam_new import Cubam
import torch

logger = get_logger("INFO", "prompt analysis")

def prompt_analysis(args):
    # read config
    config = read_yaml_config("./config.yaml")
    logger.info(args)
    logger.info(config)

    # read data
    df = read_csv_file(config["DATA_PROCESS"]["RESULT_PATH"])
    logger.info(df.info())
    column_names = df.columns.tolist()
    
    # read keywords
    keywords = read_txt_file(config["DATA_PROCESS"]["KEYWORDS_PATH"])
    valid_keywords = []
    index = -1

    # prompt analysis
    for _i, key in enumerate(keywords):
        logger.info("processing keyword: {key}".format(key=key))

        # check if the keyword is valid
        prompt_num = 0
        if "{key}_prompt0_meta".format(key=key) not in column_names:
            continue
        index += 1
        valid_keywords.append(key)

        # read prompt result
        prompt_df = read_csv_file(config["DATA_PROCESS"]["PROMPT_PATH"] + "prompt_result_" + str(index) + ".csv")
        logger.info(config["DATA_PROCESS"]["PROMPT_PATH"] + "prompt_result_" + str(index) + ".csv")
        logger.info(prompt_df.info())

        # calculate prompt result
        df["{key}_result".format(key=key)] = 0
        for i in range(config["DATA_PROCESS"]["PROMPT_NUM"]):
            if "{key}_prompt{id}_meta".format(key=key, id=i) in column_names:
                prompt_num += 1
                df["{key}_result".format(key=key)] += df["{key}_prompt{id}".format(key=key, id=i)]
            else:
                break
        logger.info(prompt_num)

        # genearte input to network
        L = []
        for idx, row in df.iterrows():
            L.append([])
            for prompt_id in range(prompt_num):
                L[-1].append(1 if row["{key}_prompt{num}".format(key=key, num=prompt_id)] == 1 else 0)

        # train network
        cubam = Cubam(len(df), prompt_num)
        cubam.to(0)
        cubam.train()
        optimizer = torch.optim.Adam(cubam.parameters(), lr=0.1)
        L = torch.tensor(L, dtype=torch.float32, device=0)
        for i in range(1600):
            loss = cubam(L)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 100 == 99:
                logger.info("epoch {epoch}, loss {loss}".format(epoch=i + 1, loss=loss))
        
        # save result
        prompt_df["tau"] = cubam.tau.data.tolist()
        prompt_df["sigma"] = [item + 0.05 for item in cubam.sigma.data.tolist()]
        prompt_df.to_csv(config["DATA_PROCESS"]["PROMPT_PATH"] + "prompt_final_result_" + str(index) + ".csv", index=False)

        # print result
        portions = []
        for idx in range(prompt_num):
            portions.append(len(df[(df["{key}_prompt{id}".format(key=key, id=idx)] == 1)]) / len(df))
        for idx, portion, tau, sigma in zip(range(len(portions)), portions, prompt_df["tau"], prompt_df["sigma"]):
            logger.info("portion{portion:.4f}, tau{tau:.4f}, sigma{sigma:.4f}, prompt {prompt} ".format(prompt=prompt_df.at[idx, '{key}_prompt'.format(key=key)], portion=portion, tau=tau, sigma=sigma))
    
    # save to file
    logger.info("valid keywords: {valid_keywords}".format(valid_keywords=valid_keywords))
    with open(config["DATA_PROCESS"]["VALID_KEYWORDS_PATH"], 'w') as f:
        for key in valid_keywords:
            f.write(key + "\n")