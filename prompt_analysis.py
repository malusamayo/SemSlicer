from utils.file import read_csv_file, read_txt_file
from utils.load_config import read_yaml_config
from utils.log import get_logger
# from cubam_py.Binary1dSignalModel import Binary1dSignalModel
# import numpy as np
from cubam_new import Cubam
import torch

logger = get_logger("INFO", "prompt analysis")

def prompt_analysis(args):
    config = read_yaml_config("./config.yaml")
    logger.info(args)
    logger.info(config)
    df = read_csv_file(config["DATA_PROCESS"]["RESULT_PATH"])
    logger.info(df.info())
    column_names = df.columns.tolist()
    
    keywords = read_txt_file(config["DATA_PROCESS"]["KEYWORDS_PATH"])
    valid_keywords = []
    index = -1
    for _i, key in enumerate(keywords):
        logger.info("processing keyword: {key}".format(key=key))
        prompt_num = 0
        if "{key}_prompt0_meta".format(key=key) not in column_names:
            continue
        index += 1
        valid_keywords.append(key)
        prompt_df = read_csv_file(config["DATA_PROCESS"]["PROMPT_PATH"] + "prompt_result_" + str(index) + ".csv")
        logger.info(config["DATA_PROCESS"]["PROMPT_PATH"] + "prompt_result_" + str(index) + ".csv")
        logger.info(prompt_df.info())
        df["{key}_result".format(key=key)] = 0
        for i in range(config["DATA_PROCESS"]["PROMPT_NUM"]):
            # logger.info("{key}_prompt{id}_meta".format(key=key, id=i))
            # logger.info("{key}_prompt{id}_meta".format(key=key, id=i) in column_names)

            if "{key}_prompt{id}_meta".format(key=key, id=i) in column_names:
                # logger.info(df["{key}_prompt{id}_meta".format(key=key, id=i)])
                prompt_num += 1
                # for x in df["{key}_prompt{id}_meta".format(key=key, id=i)]:
                #     if type(x) != str:
                #         logger.info(x)
                #         exit(-1)
                # df["{key}_prompt{id}".format(key=key, id=i)] = df["{key}_prompt{id}_meta".format(key=key, id=i)].apply(
                #     lambda x: 
                #         1 if x.find("My Answer is Yes") != -1 
                #         else 0
                # )
                df["{key}_result".format(key=key)] += df["{key}_prompt{id}".format(key=key, id=i)]
            else:
                break
        logger.info(prompt_num)
        # output = [[len(df), prompt_num, len(df) * prompt_num]]
        i = 0
        L = []
        for idx, row in df.iterrows():
            L.append([])
            for prompt_id in range(prompt_num):
                L[-1].append(1 if row["{key}_prompt{num}".format(key=key, num=prompt_id)] == 1 else 0)
                # output.append([i, prompt_id, row["{key}_prompt{num}".format(key=key, num=prompt_id)]])
            # i += 1
        cubam = Cubam(len(df), prompt_num)
        cubam.to(0)
        # for p in cubam.parameters():
        #     logger.info(p)
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
        # for p in cubam.parameters():
        #     logger.info(p)
        # with open(config["DATA_PROCESS"]["TMP_PATH"], 'w') as f:
        #     for line in output:
        #         f.write(" ".join([str(i) for i in line]) + "\n")
        # majority voting
        # df["{key}_result".format(key=key)] = df["{key}_result".format(key=key)].apply(
        #     lambda x: 
        #         1 if x > prompt_num / 2 
        #         else 0
        # )
        # beta = max(len(df[(df['{key}_result'.format(key=key)] == 1)]), 1) / len(df)
        prompt_df["tau"] = cubam.tau.data.tolist()
        prompt_df["sigma"] = [item + 0.05 for item in cubam.sigma.data.tolist()]
        prompt_df.to_csv(config["DATA_PROCESS"]["PROMPT_PATH"] + "prompt_final_result_" + str(index) + ".csv", index=False)
        portions = []
        for idx in range(prompt_num):
            portions.append(len(df[(df["{key}_prompt{id}".format(key=key, id=idx)] == 1)]) / len(df))
        for idx, portion, tau, sigma in zip(range(len(portions)), portions, prompt_df["tau"], prompt_df["sigma"]):
            logger.info("portion{portion:.4f}, tau{tau:.4f}, sigma{sigma:.4f}, prompt {prompt} ".format(prompt=prompt_df.at[idx, '{key}_prompt'.format(key=key)], portion=portion, tau=tau, sigma=sigma))
            # logger.info("portion{portion:.4f}, prompt {prompt} ".format(prompt=prompt_df.at[idx, '{key}_prompt'.format(key=key)], portion=portion))
        # sigt = np.std(portions) * 2 + 0.1
        # logger.info("beta = {beta}".format(beta=beta))
        # logger.info("sigt = {sigt}".format(sigt=sigt))
        # model = Binary1dSignalModel()
        # model.set_model_param(prm={'beta': beta})
        # model.load_data(config["DATA_PROCESS"]["TMP_PATH"])
        # model.optimize_param()
        # eprm = {
        #     'wkr' : model.get_worker_param(), 
        #     'img' : model.get_image_param()
        # }
        # tau = [eprm['wkr'][idx][1] / eprm['wkr'][idx][0] for idx in range(len(eprm['wkr']))]
        # sigma = [1 / eprm['wkr'][idx][0] for idx in range(len(eprm['wkr']))]
        #     # logger.info(prompt_df.at[idx, '{key}_prompt'.format(key=key)])
        #     logger.info("tau{tau:.4f} sigma{sigma:.4f} portion{portion:.4f}, prompt {prompt} ".format(prompt=prompt_df.at[idx, '{key}_prompt'.format(key=key)], index=idx, tau=tau_i, sigma=sigma_i, portion=portion))
        # # a = input("continue")
        # # continue
        # prompt_df["tau"] = tau
        # prompt_df["sigma"] = sigma
        # logger.info("written to {path}".format(path=config["DATA_PROCESS"]["PROMPT_PATH"] + "prompt_final_result_" + str(index) + ".csv"))
        # input("continue")
    logger.info("valid keywords: {valid_keywords}".format(valid_keywords=valid_keywords))
    with open(config["DATA_PROCESS"]["VALID_KEYWORDS_PATH"], 'w') as f:
        for key in valid_keywords:
            f.write(key + "\n")