# Cost Model

For each slice, we can break down the cost into two parts: (1) a
one-time prompt construction cost and (2) a labeling cost for each
example.

Next, we discuss how we estimate cost of LLMs and humans:

#### Student model

Based on the pricing from cloud provider[^1], a model (`Mixtral 8*7B`)
with similar capability to our student model (`flan-t5-xxl`) costs
\$0.00045 per 1K input token, and \$0.0007 per 1K output token. This
translates into \$0.000043 for every zero-shot annotation, and \$0.00028
for every few-shot annotation.

#### Teacher model

Our teacher/creator model (`gpt-4-turbo-preview`) costs \$10 per 1M
input token, and \$30 per 1M output token. This translates into
\$0.00096 for every zero-shot annotation, \$0.011 for every synthetic
example generation, \$0.0013 for every instruction generation, and
\$0.0037 for every instruction refinement.

#### Human labeling

Crowdworkers can label 180 annotations per hour and are paid around \$13
per hour.[^2] This translates into \$0.072 for every annotation.

#### Instruction refinement with human

On average, it takes around 8 minutes per slice to refine the
instruction. Based on average data scientist salary (\$59 per hour),[^3]
this translates into \$7.87 per slice,

From the cost estimate, we can calculate the cost for each component of
(see below) and
hence the cost of all experiment configurations
(Table 2 in our paper).

---

```
             **Component** **Avg. \#token**            **Cost**     
                           input              output   cost/anno.   cost/slice
         **Label/student** 93.17              1        \$0.000043   \$0.258
**Label/student-few-shot** 523.79             1        \$0.00028    \$1.68
           **Label/human** \-                 \-       \$0.072      \$432
**Few-shot-label/teacher** 93.17              1        \$0.00096    \$0.00768
**Few-shot-input/teacher** 621                156.25   \-           \$0.011
 **Instruction/model-gen** 107                8.88     \-           \$0.0013
  **Instruction/model-rf** 58                 105      \-           \$0.0037
  **Instruction/human-rf** \-                 \-       \-           \$7.87
```

---

# Models for RQ3

We used `cardiffnlp/twitter-roberta-base-hate-latest` for HateCheck,
`DataMonke/bert-base-uncased-finetuned-review-sentiment-analysis` for
Amazon, and `unitary/toxic-bert` for CivilComments.

[^1]: <https://aws.amazon.com/bedrock/pricing/>

[^2]: <https://monkeylearn.com/blog/mechanical-turk-101-use-mturk-tagging-training-data/>

[^3]: <https://www.ziprecruiter.com/Salaries/DATA-Scientist-Salary-per-Hour>
