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


# Collaboration

## SMs Enhance LLMs

### Data Curation
#### Curating pre-training data
#### Curating Instruction-tuning Data

### Weak-to-Strong Paradigm

### Efficient Inference
#### Ensembling different-size models to reduce inference costs
#### Speculative Decoding

### Evaluating LLMs

### Domain Adaptation
#### Using domain-specific SMs to generate knowledge for LLMs at reasoning time
#### Using domain-specific SMs to adjust token probability of LLMs at decoding time

### Retrieval Augmented Generation

### Prompt-based Reasoning

### Deficiency Repair
#### Developing SM plugins to repair deficiencies
#### Contrasting LLMs and SMs for better generations

## LLMs Enhance SMs

### Knowledge Distillation
#### Black-box Distillation
#### White-box distillation

### Data Synthesis
#### Data Augmentation
#### Training Data Generation


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
