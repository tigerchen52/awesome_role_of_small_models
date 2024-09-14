# The Role of Small Models
[![PDF](https://img.shields.io/badge/PDF-2409.06857-green)](https://arxiv.org/abs/2409.06857)
![GitHub License](https://img.shields.io/github/license/tigerchen52/role_of_small_models)
![](https://img.shields.io/badge/PRs-Welcome-red) 

This work is ongoing, and we welcome any comments or suggestions. 

Please feel free to reach out if you find we have overlooked any relevant papers.

<!-- Big font size -->
<h2 align="center">
What is the Role of Small Models in the LLM Era: A Survey
</h2> 


<p align="center">
    Lihu Chen<sup>1</sup>&nbsp&nbsp
    Gaël Varoquaux<sup>2</sup>&nbsp&nbsp
</p>  


<p align="center">
<sup>1</sup> Imperial College London, UK &nbsp&nbsp
<sup>2</sup>  Soda, Inria Saclay, France &nbsp&nbsp
</p>
<div align="center">
  <img src="imgs/collaboration.png" width="500"><br>
</div>
<br>


## Content List
- [Collaboration](#collaboration)
  - [SMs Enhance LLMs](#sms-enhance-llms)
    - [Data Curation](#data-curation)
      - [Curating pre-training data](#curating-pre-training-data)
      - [Curating Instruction-tuning Data](#curating-instruction-tuning-data)
    - [Weak-to-Strong Paradigm](#weak-to-strong-paradigm)
    - [Efficient Inference](#efficient-inference)
      - [Ensembling different-size models to reduce inference costs](#ensembling-different-size-models-to-reduce-inference-costs)
      - [Speculative Decoding](#speculative-decoding)
    - [Evaluating LLMs](#evaluating-llms)
    - [Domain Adaptation](#domain-adaptation)
      - [Using domain-specific SMs to generate knowledge for LLMs at reasoning time](#using-domain-specific-sms-to-generate-knowledge-for-llms-at-reasoning-time)
      - [Using domain-specific SMs to adjust token probability of LLMs at decoding time](#using-domain-specific-sms-to-adjust-token-probability-of-llms-at-decoding-time)
    - [Retrieval Augmented Generation](#retrieval-augmented-generation)
    - [Prompt-based Reasoning](#prompt-based-reasoning)
    - [Deficiency Repair](#deficiency-repair)
      - [Developing SM plugins to repair deficiencies](#developing-sm-plugins-to-repair-deficiencies)
      - [Contrasting LLMs and SMs for better generations](#contrasting-llms-and-sms-for-better-generations)
  - [LLMs Enhance SMs](#llms-enhance-sms)
    - [Knowledge Distillation](#knowledge-distillation)
      - [Black-box Distillation](#black-box-distillation)
      - [White-box distillation](#white-box-distillation)
    - [Data Synthesis](#data-synthesis)
      - [Data Augmentation](#data-augmentation)
      - [Training Data Generation](#training-data-generation)


# Collaboration <a name="collaboration"></a>

## SMs Enhance LLMs <a name="sms-enhance-llms"></a>

### Data Curation <a name="data-curation"></a>

#### Curating pre-training data: <a name="curating-pre-training-data"></a>
- Data Selection [\[36\]](#36)
- Data Reweighting [\[37\]](#37)

#### Curating Instruction-tuning Data <a name="curating-instruction-tuning-data"></a>
- [\[35\]](#35)

### Weak-to-Strong Paradigm <a name="weak-to-strong-paradigm"></a>

#### Using weaker (smaller) models to align stronger (larger) models <a name="using-weaker-smaller-models-to-align-stronger-larger-models"></a>
- [\[34\]](#34)

### Efficient Inference <a name="efficient-inference"></a>

#### Ensembling different-size models to reduce inference costs: <a name="ensembling-different-size-models-to-reduce-inference-costs"></a>
- Model Cascading [\[32\]](#32)
- Model Routing [\[33\]](#33)

#### Speculative Decoding <a name="speculative-decoding"></a>
- [\[31\]](#31)

### Evaluating LLMs <a name="evaluating-llms"></a>

#### Using SMs to evaluate LLM's generations: <a name="using-sms-to-evaluate-llms-generations"></a>
- General Evaluation [\[28\]](#28)
- Uncertainty [\[29\]](#29)
- Performance Prediction [\[30\]](#30)

### Domain Adaptation <a name="domain-adaptation"></a>

#### Using domain-specific SMs to adjust token probability of LLMs at decoding time <a name="using-domain-specific-sms-to-adjust-token-probability-of-llms-at-decoding-time"></a>
- [\[26\]](#26)

#### Using domain-specific SMs to generate knowledge for LLMs at reasoning time <a name="using-domain-specific-sms-to-generate-knowledge-for-llms-at-reasoning-time"></a>
- [\[27\]](#27)

### Retrieval Augmented Generation <a name="retrieval-augmented-generation"></a>

#### Using SMs to retrieve knowledge for enhancing generations: <a name="using-sms-to-retrieve-knowledge-for-enhancing-generations"></a>
- Documents [\[20\]](#20)
- Knowledge Bases [\[21\]](#21)
- Tables [\[22\]](#22)
- Codes [\[23\]](#23)
- Tools [\[24\]](#24)
- Images [\[25\]](#25)

### Prompt-based Reasoning <a name="prompt-based-reasoning"></a>

#### Using SMs to augment prompts for LLMs: <a name="using-sms-to-augment-prompts-for-llms"></a>
- Retrieving Prompts [\[16\]](#16)
- Decomposing Complex Problems [\[17\]](#17)
- Generating Pseudo Labels [\[18\]](#18)
- Feedback [\[19\]](#19)

### Deficiency Repair <a name="deficiency-repair"></a>

#### Developing SM plugins to repair deficiencies: <a name="developing-sm-plugins-to-repair-deficiencies"></a>
- Hallucinations [\[14\]](#14)
- Out-Of-Vocabulary Words [\[15\]](#15)

#### Contrasting LLMs and SMs for better generations: <a name="contrasting-llms-and-sms-for-better-generations"></a>
- Reducing Repeated Texts [\[10\]](#10)
- Mitigating Hallucinations [\[11\]](#11)
- Augmenting Reasoning Capabilities [\[12\]](#12)
- Safeguarding Privacy [\[13\]](#13)

## LLMs Enhance SMs <a name="llms-enhance-sms"></a>

### Knowledge Distillation <a name="knowledge-distillation"></a>

#### Black-box Distillation: <a name="black-box-distillation"></a>
- Chain-Of-Thought Distillation [\[8\]](#8)
- Instruction Following Distillation [\[9\]](#9)

#### White-box Distillation: <a name="white-box-distillation"></a>
- Logits [\[6\]](#6)
- Intermediate Features [\[7\]](#7)

### Data Synthesis <a name="data-synthesis"></a>

#### Data Augmentation: <a name="data-augmentation"></a>
- Text Paraphrase [\[3\]](#3)
- Query Rewriting [\[4\]](#4)
- Specific Tasks [\[5\]](#5)

#### Training Data Generation: <a name="training-data-generation"></a>
- Label Annotation [\[1\]](#1)
- Dataset Generation [\[2\]](#2)




## Citation


```
@misc{chen2024rolesmallmodelsllm,
      title={What is the Role of Small Models in the LLM Era: A Survey}, 
      author={Lihu Chen and Gaël Varoquaux},
      year={2024},
      eprint={2409.06857},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.06857}, 
}
``````
