from openai import OpenAI

class TeacherModel:
    def __init__(self, model_name="gpt-4-turbo-preview"):
        self.model = OpenAI().chat.completions
        self.model_name = model_name

    def label(self, dialogs):
        results = []
        for dialog in dialogs:
            response = self.model.create(
                model=self.model_name,
                messages=dialog
            )
            results.append(response.choices[0].message.content)
        return results

if __name__ == "__main__":
    teacher = TeacherModel()
    dialogs = [[
        {"role": "system", "content": "Is the text relavnt to mental health? Answer ONLY yes or no. Do NOT explain your answer."},
        {"role": "user", "content": "Text: I am feeling anxious. I am not sure what to do. I feel like I am going to have a panic attack.\n Answer:"}
    ], [
        {"role": "system", "content": "Is the text relavnt to mental health? Answer ONLY yes or no. Do NOT explain your answer."},
        {"role": "user", "content": "Text: I feel perfect today.\n Answer:"}
    ]]
    results = teacher.label(dialogs)
    print(results)