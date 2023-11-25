import enum
import warnings
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from transformers.pipelines.text2text_generation import Text2TextGenerationPipeline
import torch
import torch.nn.functional as F


class ReturnType(enum.Enum):
    TENSORS = 0
    TEXT = 1

class Text2TextGenerationPipelineWithProbs(Text2TextGenerationPipeline):
    # Used in the return key of the pipeline.
    return_name = "generated"

    def __init__(self, label_space, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_ids = self.tokenizer(label_space, return_tensors="pt")["input_ids"].t()[0]

    def _forward(self, model_inputs, **generate_kwargs):
        if self.framework == "pt":
            in_b, input_length = model_inputs["input_ids"].shape
        elif self.framework == "tf":
            in_b, input_length = tf.shape(model_inputs["input_ids"]).numpy()

        self.check_inputs(
            input_length,
            generate_kwargs.get("min_length", self.model.config.min_length),
            generate_kwargs.get("max_length", self.model.config.max_length),
        )

        outputs = self.model.generate(**model_inputs, **generate_kwargs, output_scores=True, return_dict_in_generate=True)

        # take logits for the first token (i.e., label)
        # and convert to probabilities
        logits = outputs["scores"][0] 
        selected_logits = logits[:, self.label_ids].to(torch.float32)
        probs = F.softmax(selected_logits, dim=1)

        output_ids = outputs["sequences"]
        out_b = output_ids.shape[0]
        if self.framework == "pt":
            output_ids = output_ids.reshape(in_b, out_b // in_b, *output_ids.shape[1:])
        elif self.framework == "tf":
            output_ids = tf.reshape(output_ids, (in_b, out_b // in_b, *output_ids.shape[1:]))
        return {"output_ids": output_ids, "probs": probs}

    def postprocess(self, model_outputs, return_type=ReturnType.TEXT, clean_up_tokenization_spaces=False):
        records = []
        for output_ids, probs in zip(model_outputs["output_ids"][0], model_outputs["probs"]):
            if return_type == ReturnType.TENSORS:
                record = {f"{self.return_name}_token_ids": output_ids}
            elif return_type == ReturnType.TEXT:
                record = {
                    f"{self.return_name}_text": self.tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    ),
                    "probs": probs
                }
            records.append(record)
        return records

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto", load_in_8bit=True)
    generator = Text2TextGenerationPipelineWithProbs(label_space = ['yes', 'no'], model=model,
                    tokenizer=tokenizer, batch_size=3)

    input_template = """# Text
{text}

# Question
{question} Your answer is yes or no.

# Answer
My answer is """

    questions = ['Does the text mention anything about age?',
    'Does the text have any information about age?',
    'Does the text discuss age in any way?',
    'Does the text address age in any context?',
    'Does the text include any references to age?',
    'Does the text make any mention of age-related topics?',
    'Does the text touch on age in any way?',
    'Does the text have any content related to age?',
    'Does the text have any content related to crime?']

    input_examples = ['A young man and his parents had a meeting scheduled together for the next morning.',
        'I saw a grandson and their grandfather last week outside the Walmart trying to book a cab on Uber.',
        # "There's two professors for the course, a Gen-Xer and a Millennial, which is good because you can pick whom to talk to depending on if you want to actually understand the subject matter or if you just want an easy grade.",
        # "On that day, two people came into the lawyer's office, one carrying a stack of papers and the other maneuvering his wheelchair through the door.",
        "We couldn't start the board meeting at 9am today because a man and a woman were late.",
        # "The police stopped a Hispanic man and a Native American man the other day just outside the arena."
    ]
    input_text = [input_template.format(text=text, question=questions[0]) for text in input_examples]
    print(generator(input_text))