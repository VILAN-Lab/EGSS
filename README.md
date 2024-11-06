# EGSS
<!--EGSS paper code-->
## Prerequisites

- Python 3.6
- PyTorch == 1.4.0

## Quick start:

- Dataset

All dataset under the directory of `squad-processed`, include [SQuAD corpus](`bio`, `linguistic features`,`src`, `tgt`).
You need to run `answer_to_entity.py`,`ans-entity_to_qtype.py`,`other_covert.py`, and `style_question.py` in order to get the question type.
Dependency parsing relations need to be obtained yourself using the Stanford dependency tools and converted to adj.txt via `adj_json.py`
We utilize the glove embedding, please download the *glove.840b.300d.txt* and put it in the `data`.

- Data preprocess

Run following command:

1. `python preprocess_data.py`

Then we will get two files

1. `data/embedding.pkl`
2. `data/word2idx.pkl`

- Training

Run command:

`python main.py`

- Inference

Run command:

`python main.py -train False`

Then the output file will be save in directory of `result/`
