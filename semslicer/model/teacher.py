from openai import OpenAI
from .query_utils import query_batch
import tqdm

class OpenAIModel:
    def __init__(self, model_name="gpt-4-turbo-preview", model_size=""):
        self.model = OpenAI().chat.completions
        self.model_name = model_name

    def _send_request(
        self,
        dialogs,
        batch_size=1,
        max_gen_len=1024,
        temperature=1,
        top_p=0.9,
        return_probs=False,
        labels=None,
        mimic_starting_response=''
    ):
        results = []
        # system_prompt = dialogs[0][0]["content"]
        # prompts = [dialog[1]["content"] for dialog in dialogs]
        # results = query_batch(prompts, 
        #     batch_size=batch_size,
        #     model_name=self.model_name, 
        #     system_msg=system_prompt, 
        # temperature=temperature)
        if mimic_starting_response != '':
            dialogs = [dialog + [{"role": "assistant", "content": mimic_starting_response}] for dialog in dialogs]
        for dialog in tqdm.tqdm(dialogs):
            response = self.model.create(
                model=self.model_name,
                messages=dialog,
                temperature=temperature,
            )
            results.append(response.choices[0].message.content)
        return results

if __name__ == "__main__":
    teacher = OpenAIModel()
    dialogs = [[
        {"role": "system", "content": "Is the text relavnt to mental health? Answer ONLY yes or no. Do NOT explain your answer."},
        {"role": "user", "content": "Text: I am feeling anxious. I am not sure what to do. I feel like I am going to have a panic attack.\n Answer:"}
    ], [
        {"role": "system", "content": "Is the text relavnt to mental health? Answer ONLY yes or no. Do NOT explain your answer."},
        {"role": "user", "content": "Text: I feel perfect today.\n Answer:"}
    ]]
    results = teacher._send_request(dialogs, temperature=0, batch_size=5)
    print(results)