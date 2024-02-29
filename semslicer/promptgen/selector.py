from ..utils.file import read_csv_file, read_txt_file
from ..utils.log import get_logger
from ..utils.config import config
from .cubam_new import Cubam
import torch

logger = get_logger("INFO", "prompt analysis")

def select_usp_examples(dialogs, results, probs, nums):
    num_per_class = int(nums / 2)
    _, indices_a = torch.topk(probs[:, 0], num_per_class)
    _, indices_b = torch.topk(probs[:, 1], num_per_class)

    # selected_idx = indices_a.tolist() + indices_b.tolist()
    selected_idx = [v for p in zip(indices_a.tolist(), indices_b.tolist()) for v in p] #interleave two lists
    return [dialogs[idx] for idx in selected_idx], [results[idx] for idx in selected_idx]

def select_boundary_examples(dialogs, probs, nums):

    max_probs = torch.maximum(probs[:, 0], probs[:, 1])
    _, selected_idx = torch.topk(-max_probs, nums)

    # logger.info(max_probs[selected_idx])
    return [dialogs[idx] for idx in selected_idx]

def select_random_examples(dialogs, nums):
    torch.manual_seed(42)
    selected_idx = torch.randperm(len(dialogs))[:nums]
    return [dialogs[idx] for idx in selected_idx]

class PromptSelector:

    def __init__(self, ):
        self.noise_estimate_flag = config["PROMPT_ANALYSIS"]["NOISE_ESTIMATE"]

    def noise_estimate(self, df, key, prompt_num):
        # genearte input to network
        L = []
        for idx, row in df.iterrows():
            L.append([])
            for prompt_id in range(prompt_num):
                L[-1].append(1 if row["{key}_prompt{num}".format(key=key, num=prompt_id)] == 1 else 0)
        # L = df[[f"{key}_prompt{prompt_id}"]].values

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

        biases = cubam.tau.data.tolist()
        noises = [item + 0.05 for item in cubam.sigma.data.tolist()]
        return biases, noises

    def accuracy_estimate(self, df, key, prompt_num):
        df["majority_vote_{key}".format(key=key)] = ((df["{key}_result".format(key=key)] / prompt_num) > 0.5).astype(int)
        accs = [(df["{key}_prompt{id}".format(key=key, id=i)] == df["majority_vote_{key}".format(key=key)]).mean() for i in range(prompt_num)]
        return accs

    def select_prompt(self, prompt_df, keyword, criteria):
        """
        Select prompts based on predictions

        Parameters:
            prompt_df: prompt metadata analyzed from predictions
            keyword: keyword
            criteria: criteria for prompt selection
        """
        if criteria == 'min_noise':
            noise = prompt_df[f'{keyword}_sigma'].apply(lambda x: x if x > 0 else -x)
            min_noise_idx = noise.idxmin()
            prompt = prompt_df.at[min_noise_idx, '{keyword}_prompt'.format(keyword=keyword)]
        elif criteria == 'default':
            prompt = prompt_df.at[0, '{keyword}_prompt'.format(keyword=keyword)]
        elif criteria == 'maj_vote':
            max_acc_idx = prompt_df[f'{keyword}_pseudo_acc'].idxmax()
            prompt = prompt_df.at[max_acc_idx, '{keyword}_prompt'.format(keyword=keyword)]
        return prompt

    def analyze(self, keywords):
        # read data
        df = read_csv_file(config["EXPERIMENT"]["SLICE_RESULT_PATH"])
        logger.info(df.info())
        column_names = df.columns.tolist()

        valid_keywords = []
        index = -1

        # read prompt result
        prompt_df = read_csv_file(config["EXPERIMENT"]["PROMPT_PATH"].format(key_idx=index))
        logger.info(config["EXPERIMENT"]["PROMPT_PATH"].format(key_idx=index))
        logger.info(prompt_df.info())

        # prompt analysis
        for _i, key in enumerate(keywords):
            logger.info("processing keyword: {key}".format(key=key))

            # check if the keyword is valid
            prompt_num = 0
            if "{key}_prompt0_meta".format(key=key) not in column_names:
                continue
            index += 1
            valid_keywords.append(key)

            # calculate prompt result
            df["{key}_result".format(key=key)] = 0
            for i in range(len(prompt_df)):
                if "{key}_prompt{id}_meta".format(key=key, id=i) in column_names:
                    prompt_num += 1
                    df["{key}_result".format(key=key)] += df["{key}_prompt{id}".format(key=key, id=i)]
                else:
                    break
            
            # calculate pseudo accuracy
            prompt_df["{key}_pseudo_acc"] = self.accuracy_estimate(df, key, prompt_num)

            if self.noise_estimate_flag:
                prompt_df["{key}_tau"], prompt_df["{key}_sigma"] = self.noise_estimate(df, key, prompt_num)
            

            # # estimated labels
            # df[f"estimated_label_{key}"] = cubam.x.data.tolist()
            # df.to_csv(config["PROMPT_ANALYSIS"]["RESULT_PATH"] + '.cp.csv', index=False)

            # # print result
            # portions = []
            # for idx in range(prompt_num):
            #     portions.append(len(df[(df["{key}_prompt{id}".format(key=key, id=idx)] == 1)]) / len(df))
            # for idx, portion, tau, sigma in zip(range(len(portions)), portions, prompt_df["tau"], prompt_df["sigma"]):
            #     logger.info("portion{portion:.4f}, tau{tau:.4f}, sigma{sigma:.4f}, prompt {prompt} ".format(prompt=prompt_df.at[idx, '{key}_prompt'.format(key=key)], portion=portion, tau=tau, sigma=sigma))

    
        logger.info("valid keywords: {valid_keywords}".format(valid_keywords=valid_keywords))
        # save result
        prompt_df.to_csv(config["EXPERIMENT"]["FINAL_PROMPT_PATH"], index=False)

if __name__ == "__main__":
    keywords = read_txt_file(config["EXPERIMENT"]["KEYWORDS_PATH"])
    
    selector = PromptSelector()
    selector.analyze(keywords)

    for index, keyword in enumerate(keywords):
        logger.info("processing keyword: {key}".format(key=keyword))
        # read prompt
        prompt_df = read_csv_file(config["EXPERIMENT"]["FINAL_PROMPT_PATH"].format(key_idx=index))

        # select prompt
        prompt = selector.select_prompt(prompt_df, keyword, criteria='maj_vote')
        logger.info("prompt = {prompt}".format(prompt=prompt.split('\n')[0]))