# QRRetriever: A General-Purpose Retriever Built on Top of QRHead

<p align="center">
  <img width="80%" alt="image" src="assets/qrheadlogo.png">
</p>


[[Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking (EMNLP 2025)](https://arxiv.org/pdf/2506.09944)]

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-yellow.svg?logo=huggingface)](https://huggingface.co/datasets/PrincetonPLI/QRHead)

**QRRetriever** is a general-purpose retriever that uses the attention scores of QRHead (Query-Focused Retrieval Heads) of language models for retrieval from long context.

## TODO
[☑️] QRHead detection code

## Installation

Please first install the following packages:
* `torch`
* `transformers` (tested with versions `4.44.1` to `4.48.3`)
* `flash_attn`

Next, install `qrretriever` by running:
```bash
pip install -e .
```

## Usage
Using `QRRetriever` is simple. We provide a minimal example in `examples/qrretriever_example.py`.

```python
from qrretriever.attn_retriever import QRRetriever
retriever = QRRetriever(model_name_or_path="meta-llama/Llama-3.1-8B-Instruct")

query = "Which town in Nizhnyaya has the largest population?"
docs = [
    {"idx": "test0", "title": "Kushva", "paragraph_text": "Kushva is the largest town in Nizhnyaya. It has a population of 1,000."},
    {"idx": "test1", "title": "Levikha", "paragraph_text": "Levikha is a bustling town in Nizhnyaya. It has a population of 200,000."},
]

scores = retriever.score_docs(query, docs)
print(scores)
# expected output: {'test0': 0.63, 'test1': 1.17}
```

**Supported models**:
* `Llama-3.2-1B-Instruct`
* `Llama-3.2-3B-Instruct`
* `Llama-3.1-8B-Instruct`
* `Llama-3.1-70B-Instruct`
* `Qwen2.5-7B-Instruct`

## Reproducing Experiments on Long-Context Reasoning and BEIR Re-ranking
Please refer to the [README](https://github.com/princeton-pli/QRHead/tree/main/exp_scripts) in `exp_scripts` for
* QRHead detection
* Running and evaluating retrieval
* Running and evaluating generation

## Citation

```bibtex
@inproceedings{zhang25qrhead,
    title={Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking},
    author={Wuwei Zhang and Fangcong Yin and Howard Yen and Danqi Chen and Xi Ye},
    booktitle={Proceedings of EMNLP},
    year={2025}
}
```

## Credits

Part of the code is adapted from [In-Context-Reranking](https://github.com/OSU-NLP-Group/In-Context-Reranking).
