# SemSlicer

Slice any datasets on any semantic criteria with LLMs.


### Experiments Reproduction
To reproduce our experiments (automated setups), please run `sbatch run.sh {dataset}` on a cluster, using the according datasets.


### Notebook Usage
```python
%env OPENAI_API_KEY={put your key here}
from semslicer.slicer import InteractiveSlicer
import pandas as pd
data = pd.read_csv("data/data/civil_comments_sampled.csv").sample(20)
concept = "Muslim"

slicer = InteractiveSlicer(concept, data,
    {
        'few-shot': True,
        'few-shot-size': 8,
        'instruction-source': 'template',
        'student-model': 'gpt-3.5-turbo',
        'teacher-model': 'gpt-4-turbo-preview'
    }
)

slicer.show_prompt()
```
```python
llm_slicing = slicer.gen_slicing_func()
m_slice = data[data['context'].map(llm_slicing)]
m_slice['context'].sample(2)
```