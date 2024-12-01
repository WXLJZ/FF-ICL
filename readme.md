# Metaphor Interpretation with Feature-driven In-context Learning Integrating Prior Prediction Feedback

---

**Authors:** Anonymous Authors


## 📋 Table of Contents
- [Introduction](#anchor-introduction)
- [Code Structure](#anchor-code-structure)
- [Environment and Requirements](#anchor-environment-and-requirements)
- [Running](#anchor-running)
- [Data Description](#anchor-data-description)

---

<a id="anchor-introduction"></a>
## 📌 Introduction

Metaphors, as a common linguistic expression, help people intuitively understand complex concepts in communication, writing, and cognition. Metaphor interpretation is a central task in the computational metaphor processing community. However, the critical modeling element in the tasks—metaphor components—has rarely been studied as a direct research focus. This paper centers on metaphor components and proposes a metaphor interpretation framework based on large language models. Specifically, we integrate in-context learning and feedback mechanisms inspired by human learning processes. First, we design a machine feedback mechanism that simulates human feedback to perform prior predictions on training samples, generating a candidate demonstration pool enriched with prediction results and feedback information. Next, a multi-head graph attention network is introduced to capture the linguistic and structural information embedded in metaphorical expressions, producing feature-rich representations and constructing a vector repository. Based on this repository, the framework retrieves demonstrations most relevant to the input query across different feature dimensions, incorporating in-context prompts to efficiently fine-tune the LLM. Experiments and analyses on public datasets demonstrate the superiority of our approach. Furthermore, additional metaphor concept mapping experiments validate the crucial role of metaphor components in downstream computational metaphor tasks. 

---

<a id="anchor-code-structure"></a>
## 📂 Code Structure
```angular2html
├── 📁 data——The reconstruct data from CSR and CMRE.
└── 📁 src——the code of method
    ├── 📁 gnnencoder——GAT encoder
    └── 📁 Icl——In-context learning
```

---

<a id="anchor-environment-and-requirements"></a>
## 🛠 Environment and Requirements
- **Python version:** 3.8 or above
- **GPU** One NVIDIA GeForce RTX 3090 24G
- **Dependencies** Refer to `requirements.txt` for the complete list. It is recommended to install directly with the following command:
```shell
pip install -r requirements.txt
```
---

<a id="anchor-running"></a>
## 🚀 Running
><span style="color:red;font-weight:800;">Note</span>: Different devices and hardware may have different parameter configurations, and the parameter settings can be adjusted as needed.

### Step1 Feedback Acquisition
```shell
CUDA_VISIBLE_DEVICES=0 bash run_ficl.sh
```
```shell
# Important parameter description
--dataset_name # The dataset name to be processed. (CSR, CMRE)
--dataset_type # The type of dataset to be processed. (train)
--model_name_or_path # The location of the LLM.
```

### Step2 Train Multi-GAT Encoder
```shell
CUDA_VISIBLE_DEVICES=0 bash run_gnn.sh
```
```shell
# Important parameter description
--dataset_name # The dataset name to be processed. (CSR, CMRE)
--bert_model_path # The location of the bert-base-chinese.
```


### Step3 Task Fine-tuning
```shell
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

```shell
# Important parameter description
--method # The in-context demonstration retrieval methods. (gnn, fixed, random, sbert, KATE, kmeans, bm25, mmr)
--selected_k # Number of demonstration selected.
--bert_path # The location of the bert-base-chinese.
--sbert_path # The location of the sentence-transformers. (all-MiniLM-L6-v2)
--gnn_path # The location of the GAT encoder. (The model saved in step2)
--dataset_dir # The location of the dataset.
--dataset_name # The dataset name to be processed. (CSR, CMRE)
--model_name_or_path # The location of the LLM.
```

## 📄 Data Description
>The CSR dataset is sourced from Liu et al.
```json
@inproceedings{liu-etal-2018-neural,
    title = "Neural Multitask Learning for Simile Recognition",
    author = "Liu, Lizhen  and Hu, Xiao  and Song, Wei and Fu, Ruiji and Liu, Ting  and Hu, Guoping",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = "oct - nov",
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1183",
    doi = "10.18653/v1/D18-1183",
    pages = "1543--1553"
}
```
>The CMRE dataset is sourced from Chen et al.
```json
@inproceedings{chen-etal-2023-chinese,
    title = "{C}hinese Metaphorical Relation Extraction: Dataset and Models",
    author = "Chen, Guihua  and Wu, Tiantian and Cheng, MiaoMiao and Han, Xu and Gong, Jiefu  and Wang, Shijin and Song, Wei",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = "dec",
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.609",
    doi = "10.18653/v1/2023.findings-emnlp.609",
    pages = "9085--9095"
}
```

