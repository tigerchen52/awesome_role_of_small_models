# The Role of Small Models
[![Awesome](https://awesome.re/badge.svg)]() 
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
- [Competition](#competition)
  - [Computation-constrained Environment](#computation-constrained-environment)
  - [Task-specific Environment](#task-specific-environment)
  - [Interpretability-required Environment](#interpretability-required-environment)


# Collaboration <a name="collaboration"></a>

    
## SMs Enhance LLMs <a name="sms-enhance-llms"></a>

### Data Curation <a name="data-curation"></a>

#### Curating pre-training data <a name="curating-pre-training-data"></a>
<table>
  <tr>
    <th>Title</th>
    <th>Topic</th>
    <th>Venue</th>
    <th>Code</th>
  </tr>
  <tr>
    <td>Data selection for language models via importance resampling</td>
    <td>Data Selection</td>
    <td><a href="https://arxiv.org/abs/2302.03169"> 
      <img src="https://img.shields.io/badge/PDF-NeurIPS 2023-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/p-lambda/dsir">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
  </tr>
  <tr>
    <td>When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale</td>
    <td>Data Selection</td>
    <td><a href="https://arxiv.org/abs/2309.04564"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2023-10868" alt="PDF Badge">
      </a></td>
    <td> </td>
  </tr>
 <tr>
    <td>CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data</td>
    <td>Data Selection</td>
    <td><a href="https://aclanthology.org/2020.lrec-1.494/"> 
      <img src="https://img.shields.io/badge/PDF-LREC 2020-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/facebookresearch/cc_net">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"> 
   </a> </td>
  </tr>
 <tr>
    <td>QuRating: Selecting High-Quality Data for Training Language Models</td>
    <td>Data Selection</td>
    <td><a href="https://arxiv.org/abs/2402.09739"> 
      <img src="https://img.shields.io/badge/PDF-ICML 2024-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/princeton-nlp/QuRating">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"> 
   </a> </td>
  </tr>
  <tr>
    <td>DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining</td>
    <td>Data Reweighting</td>
    <td><a href="https://arxiv.org/abs/2305.10429"> 
      <img src="https://img.shields.io/badge/PDF-NeurIPS 2023-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/sangmichaelxie/doremi">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"> 
   </a> </td>
  </tr>
</table>


#### Curating Instruction-tuning Data <a name="curating-instruction-tuning-data"></a>
<table>
  <tr>
    <th>Title</th>
    <th>Topic</th>
    <th>Venue</th>
    <th>Code</th>
  </tr>
  <tr>
    <td>MoDS: Model-oriented Data Selection for Instruction Tuning</td>
    <td>Data Selection</td>
    <td><a href="https://arxiv.org/abs/2311.15653"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2023-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/CASIA-LM/MoDS">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
  </tr>
  <tr>
    <td>LESS: Selecting Influential Data for Targeted Instruction Tuning</td>
    <td>Data Selection</td>
    <td><a href="https://arxiv.org/abs/2402.04333"> 
      <img src="https://img.shields.io/badge/PDF-ICML 2024-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/princeton-nlp/LESS">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
<tr>
    <td>What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning</td>
    <td>Data Selection</td>
    <td><a href="https://openreview.net/forum?id=BTKAeLqLMw"> 
      <img src="https://img.shields.io/badge/PDF-ICLR 2024-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/hkust-nlp/deita">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
</table>

### Weak-to-Strong Paradigm <a name="weak-to-strong-paradigm"></a>

#### Using weaker (smaller) models to align stronger (larger) models <a name="using-weaker-smaller-models-to-align-stronger-larger-models"></a>
<table>
  <tr>
    <th>Title</th>
    <th>Topic</th>
    <th>Venue</th>
    <th>Code</th>
  </tr>
  <tr>
    <td>Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision</td>
    <td>Weak-to-Strong</td>
    <td><a href="https://arxiv.org/abs/2312.09390"> 
      <img src="https://img.shields.io/badge/PDF-ICML 2024-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/openai/weak-to-strong">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
  </tr>
<tr>
    <td>Weak-to-Strong Search: Align Large Language Models via Searching over Small Language Models</td>
    <td>Weak-to-Strong</td>
    <td><a href="https://arxiv.org/abs/2405.19262"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2024-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/ZHZisZZ/weak-to-strong-search">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
  </tr>
<tr>
    <td>Co-Supervised Learning: Improving Weak-to-Strong Generalization with Hierarchical Mixture of Experts</td>
    <td>Weak-to-Strong</td>
    <td><a href="https://arxiv.org/abs/2402.15505"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2024-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/yuejiangliu/csl">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
</tr>
<tr>
    <td>Improving Weak-to-Strong Generalization with Reliability-Aware Alignment</td>
    <td>Weak-to-Strong</td>
    <td><a href="https://arxiv.org/abs/2406.19032"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2024-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/Irenehere/ReliableAlignment">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
</tr>
<tr>
    <td>Aligner: Efficient Alignment by Learning to Correct</td>
    <td>Weak-to-Strong</td>
    <td><a href="https://arxiv.org/abs/2402.02416"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2024-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/Aligner2024/aligner">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
</tr>
<tr>
    <td>Vision Superalignment: Weak-to-Strong Generalization for Vision Foundation Models
</td>
    <td>Weak-to-Strong</td>
    <td><a href="https://arxiv.org/abs/2402.03749"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2024-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/ggjy/vision_weak_to_strong">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
</tr>
</table>


### Efficient Inference <a name="efficient-inference"></a>

#### Ensembling different-size models to reduce inference costs <a name="ensembling-different-size-models-to-reduce-inference-costs"></a>
<table>
  <tr>
    <th>Title</th>
    <th>Topic</th>
    <th>Venue</th>
    <th>Code</th>
  </tr>
  <tr>
    <td>Efficient Edge Inference by Selective Query</td>
    <td>Model Cascading</td>
    <td><a href="https://openreview.net/forum?id=jpR98ZdIm2q"> 
      <img src="https://img.shields.io/badge/PDF-ICLR 2023-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/anilkagak2/Hybrid_Models">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
  </tr>
  <tr>
    <td>FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance</td>
    <td>Model Cascading</td>
    <td><a href="https://arxiv.org/abs/2305.05176"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2023-10868" alt="PDF Badge">
      </a></td>
    <td> </td>
  </tr>
  <tr>
    <td>Data Shunt: Collaboration of Small and Large Models for Lower Costs and Better Performance</td>
    <td>Model Cascading</td>
    <td><a href="https://ojs.aaai.org/index.php/AAAI/article/view/29003"> 
      <img src="https://img.shields.io/badge/PDF-AAAI 2024-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/Anfeather/Data-Shunt">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
</tr>
<tr>
<td>AutoMix: Automatically Mixing Language Models</td>
    <td>Model Cascading</td>
    <td><a href="https://arxiv.org/abs/2310.12963"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2023-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/automix-llm/automix">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
</tr>
<tr>
<td>Routing to the Expert: Efficient Reward-guided Ensemble of Large Language Models</td>
    <td>Model Routing</td>
    <td><a href="https://aclanthology.org/2024.naacl-long.109/"> 
      <img src="https://img.shields.io/badge/PDF-NAACL 2024-10868" alt="PDF Badge">
      </a></td>
    <td></td>
</tr>
<tr>
<td>Tryage: Real-time, intelligent Routing of User Prompts to Large Language Models</td>
    <td>Model Routing</td>
    <td><a href="https://arxiv.org/abs/2308.11601"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2023-10868" alt="PDF Badge">
      </a></td>
    <td></td>
</tr>
<tr>
<td>OrchestraLLM: Efficient Orchestration of Language Models for Dialogue State Tracking</td>
    <td>Model Routing</td>
    <td><a href="https://aclanthology.org/2024.naacl-long.79/"> 
      <img src="https://img.shields.io/badge/PDF-NAACL 2024-10868" alt="PDF Badge">
      </a></td>
    <td></td>
</tr>
<tr>
    <td>RouteLLM: Learning to Route LLMs with Preference Data</td>
    <td>Model Routing</td>
    <td><a href="https://arxiv.org/abs/2406.18665"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2024-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/lm-sys/RouteLLM">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
</tr>
<tr>
    <td>Fly-Swat or Cannon? Cost-Effective Language Model Choice via Meta-Modeling</td>
    <td>Model Routing</td>
    <td><a href="https://dl.acm.org/doi/10.1145/3616855.3635825"> 
      <img src="https://img.shields.io/badge/PDF-WSDM 2024-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/epfl-dlab/forc">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
</tr>
</table>

#### Speculative Decoding <a name="speculative-decoding"></a>
<table>
  <tr>
    <th>Title</th>
    <th>Topic</th>
    <th>Venue</th>
    <th>Code</th>
  </tr>
  <tr>
    <td>Fast Inference from Transformers via Speculative Decoding</td>
    <td>Speculative Decoding</td>
    <td><a href="https://arxiv.org/abs/2211.17192"> 
      <img src="https://img.shields.io/badge/PDF-ICML 2023-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/feifeibear/LLMSpeculativeSampling">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
  </tr>
  <tr>
    <td>Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding</td>
    <td>Speculative Decoding</td>
    <td><a href="https://arxiv.org/abs/2401.07851"> 
      <img src="https://img.shields.io/badge/PDF-ACL 2024-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/hemingkx/SpeculativeDecodingPapers">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
<tr>
    <td>Accelerating Large Language Model Decoding with Speculative Sampling</td>
    <td>Speculative Decoding</td>
    <td><a href="https://arxiv.org/abs/2302.01318"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2024-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/hemingkx/SpeculativeDecodingPapers">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
</table>

### Evaluating LLMs <a name="evaluating-llms"></a>

#### Using SMs to evaluate LLM's generations <a name="using-sms-to-evaluate-llms-generations"></a>
<table>
  <tr>
    <th>Title</th>
    <th>Topic</th>
    <th>Venue</th>
    <th>Code</th>
  </tr>
  <tr>
    <td>BERTScore: Evaluating Text Generation with BERT</td>
    <td>General Evaluation</td>
    <td><a href="https://openreview.net/forum?id=SkeHuCVFDr"> 
      <img src="https://img.shields.io/badge/PDF-ICLR 2020-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/Tiiiger/bert_score">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
  </tr>
  <tr>
    <td>BARTScore: Evaluating Generated Text as Text Generation</td>
    <td>General Evaluation</td>
    <td><a href="https://proceedings.neurips.cc/paper/2021/hash/e4d2b6e6fdeca3e60e0f1a62fee3d9dd-Abstract.html"> 
      <img src="https://img.shields.io/badge/PDF-NeurIPS 2021-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/neulab/BARTScore">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
<tr>
    <td>Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation</td>
    <td>Uncertainty</td>
    <td><a href="https://arxiv.org/abs/2302.09664"> 
      <img src="https://img.shields.io/badge/PDF-ICLR 2023-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/lorenzkuhn/semantic_uncertainty">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
<tr>
    <td>Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models</td>
    <td>Uncertainty</td>
    <td><a href="https://arxiv.org/abs/2303.08896"> 
      <img src="https://img.shields.io/badge/PDF-EMNLP 2023-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/potsawee/selfcheckgpt">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
<tr>
    <td>ProxyLM: Predicting Language Model Performance on Multilingual Tasks via Proxy Models</td>
    <td>Performance Prediction</td>
    <td><a href="https://arxiv.org/abs/2406.09334"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2024-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/davidanugraha/proxylm">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
</table>

### Domain Adaptation <a name="domain-adaptation"></a>

#### Using domain-specific SMs to adjust token probability of LLMs at decoding time <a name="using-domain-specific-sms-to-adjust-token-probability-of-llms-at-decoding-time"></a>
<table>
  <tr>
    <th>Title</th>
    <th>Topic</th>
    <th>Venue</th>
    <th>Code</th>
  </tr>
  <tr>
    <td>CombLM: Adapting Black-Box Language Models through Small Fine-Tuned Models</td>
    <td>White-box Domain Adaptation</td>
    <td><a href="https://aclanthology.org/2023.emnlp-main.180"> 
      <img src="https://img.shields.io/badge/PDF-EMNLP 2023-10868" alt="PDF Badge">
      </a></td>
    <td></td>
  </tr>
  <tr>
    <td>Inference-Time Policy Adapters (IPA): Tailoring Extreme-Scale LMs without Fine-tuning</td>
    <td>White-box Domain Adaptation</td>
    <td><a href="https://aclanthology.org/2023.emnlp-main.424/"> 
      <img src="https://img.shields.io/badge/PDF-EMNLP 2023-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/GXimingLu/IPA">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
  <tr>
    <td>Tuning Language Models by Proxy</td>
    <td>White-box Domain Adaptation</td>
    <td><a href="https://arxiv.org/abs/2401.08565"> 
      <img src="https://img.shields.io/badge/PDF-COLM 2024-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/alisawuffles/proxy-tuning">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
</table>

#### Using domain-specific SMs to generate knowledge for LLMs at reasoning time <a name="using-domain-specific-sms-to-generate-knowledge-for-llms-at-reasoning-time"></a>
<table>
  <tr>
    <th>Title</th>
    <th>Topic</th>
    <th>Venue</th>
    <th>Code</th>
  </tr>
  <tr>
    <td>Knowledge Card: Filling LLMs' Knowledge Gaps with Plug-in Specialized Language Models</td>
    <td>Black-box Domain Adaptation</td>
    <td><a href="https://arxiv.org/abs/2305.09955"> 
      <img src="https://img.shields.io/badge/PDF-ICLR 2024-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/BunsenFeng/Knowledge_Card">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
  </tr>
  <tr>
    <td>BLADE: Enhancing Black-box Large Language Models with Small Domain-Specific Models</td>
    <td>Black-box Domain Adaptation</td>
    <td><a href="https://arxiv.org/abs/2403.18365"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2024-10868" alt="PDF Badge">
      </a></td>
    <td> </td>
  </tr>
</table>

### Retrieval Augmented Generation <a name="retrieval-augmented-generation"></a>

#### Using SMs to retrieve knowledge for enhancing generations: <a name="using-sms-to-retrieve-knowledge-for-enhancing-generations"></a>
<table>
  <tr>
    <th>Title</th>
    <th>Topic</th>
    <th>Venue</th>
    <th>Code</th>
  </tr>
  <tr>
    <td>Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</td>
    <td>Documents</td>
    <td><a href="https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html"> 
      <img src="https://img.shields.io/badge/PDF-NeurIPS 2020-10868" alt="PDF Badge">
      </a></td>
    <td> </td>
  </tr>
  <tr>
    <td>KnowledGPT: Enhancing Large Language Models with Retrieval and Storage Access on Knowledge Bases</td>
    <td>Knowledge Bases</td>
    <td><a href="https://arxiv.org/abs/2308.11761"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2023-10868" alt="PDF Badge">
      </a></td>
    <td> </td>
  </tr>
  <tr>
    <td>End-to-End Table Question Answering via Retrieval-Augmented Generation</td>
    <td>Tables</td>
    <td><a href="https://arxiv.org/abs/2203.16714"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2022-10868" alt="PDF Badge">
      </a></td>
    <td> </td>
  </tr>
  <tr>
    <td>DocPrompting: Generating Code by Retrieving the Docs </td>
    <td>Codes</td>
    <td><a href="https://openreview.net/forum?id=ZTCxT2t2Ru"> 
      <img src="https://img.shields.io/badge/PDF-ICLR 2023-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/shuyanzhou/docprompting">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
  </tr>
  <tr>
    <td>Toolformer: Language Models Can Teach Themselves to Use Tools </td>
    <td>Tools</td>
    <td><a href="https://openreview.net/forum?id=Yacmpz84TH"> 
      <img src="https://img.shields.io/badge/PDF-NeurIPS 2023-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/conceptofmind/toolformer">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
  </tr>
  <tr>
    <td>Retrieval-Augmented Multimodal Language Modeling</td>
    <td>Images</td>
    <td><a href="https://arxiv.org/abs/2211.12561"> 
      <img src="https://img.shields.io/badge/PDF-ICML 2023-10868" alt="PDF Badge">
      </a></td>
    <td></td>
  </tr>
</table>

### Prompt-based Reasoning <a name="prompt-based-reasoning"></a>

#### Using SMs to augment prompts for LLMs <a name="using-sms-to-augment-prompts-for-llms"></a>
<table>
  <tr>
    <th>Title</th>
    <th>Topic</th>
    <th>Venue</th>
    <th>Code</th>
  </tr>
  <tr>
    <td>UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation</td>
    <td>Retrieving Prompts</td>
    <td><a href="https://aclanthology.org/2023.emnlp-main.758/"> 
      <img src="https://img.shields.io/badge/PDF-EMNLP 2023-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/microsoft/LMOps">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
  </tr>
  <tr>
    <td>Small Language Models Fine-tuned to Coordinate Larger Language Models improve Complex Reasoning</td>
    <td>Decomposing Complex Problems</td>
    <td><a href="https://aclanthology.org/2023.emnlp-main.225/"> 
      <img src="https://img.shields.io/badge/PDF-EMNLP 2023-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/LCS2-IIITD/DaSLaM">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
  <tr>
    <td>Small Models are Valuable Plug-ins for Large Language Models</td>
    <td>Generating Pseudo Labels</td>
    <td><a href="https://arxiv.org/abs/2305.08848"> 
      <img src="https://img.shields.io/badge/PDF-ACL 2024-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/JetRunner/SuperICL">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
  <tr>
    <td>Can Small Language Models Help Large Language Models Reason Better?: LM-Guided Chain-of-Thought</td>
    <td>Generating Pseudo Labels</td>
    <td><a href="https://aclanthology.org/2024.lrec-main.252/"> 
      <img src="https://img.shields.io/badge/PDF-COLING 2024-10868" alt="PDF Badge">
      </a></td>
    <td> </td>
  </tr>
  <tr>
    <td>CaLM: Contrasting Large and Small Language Models to Verify Grounded Generation
</td>
    <td>Generating Feedback</td>
    <td><a href="https://arxiv.org/abs/2406.05365"> 
      <img src="https://img.shields.io/badge/PDF-ACL 2024-10868" alt="PDF Badge">
      </a></td>
    <td> </td>
  </tr>
  <tr>
    <td>Small Language Models Improve Giants by Rewriting Their Outputs</td>
    <td>Generating Feedback</td>
    <td><a href="https://aclanthology.org/2024.eacl-long.165/"> 
      <img src="https://img.shields.io/badge/PDF-EACL 2024-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/GeorgeVern/lmcor">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
  </tr>
</table>

### Deficiency Repair <a name="deficiency-repair"></a>

#### Developing SM plugins to repair deficiencies: <a name="developing-sm-plugins-to-repair-deficiencies"></a>
<table>
  <tr>
    <th>Title</th>
    <th>Topic</th>
    <th>Venue</th>
    <th>Code</th>
  </tr>
  <tr>
    <td>Small Agent Can Also Rock! Empowering Small Language Models as Hallucination Detector</td>
    <td>Hallucinations</td>
    <td><a href="https://arxiv.org/abs/2406.11277/"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2024-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/RUCAIBox/HaluAgent">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
  </tr>
  <tr>
    <td>Reconfidencing LLMs from the Grouping Loss Perspective</td>
    <td>Hallucinations</td>
    <td><a href="https://arxiv.org/abs/2402.04957"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2024-10868" alt="PDF Badge">
      </a></td>
    <td></td>
  </tr>
  <tr>
    <td>Imputing Out-of-Vocabulary Embeddings with LOVE Makes LanguageModels Robust with Little Cost</td>
    <td>Out-Of-Vocabulary Words</td>
    <td><a href="https://arxiv.org/abs/2305.08848"> 
      <img src="https://img.shields.io/badge/PDF-ACL 2022-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/tigerchen52/LOVE">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
</table>

#### Contrasting LLMs and SMs for better generations: <a name="contrasting-llms-and-sms-for-better-generations"></a>
<table>
  <tr>
    <th>Title</th>
    <th>Topic</th>
    <th>Venue</th>
    <th>Code</th>
  </tr>
  <tr>
    <td>Contrastive Decoding: Open-ended Text Generation as Optimization</td>
    <td>Reducing Repeated Texts</td>
    <td><a href="https://aclanthology.org/2023.acl-long.687/"> 
      <img src="https://img.shields.io/badge/PDF-ACL 2023-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/XiangLi1999/ContrastiveDecoding">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
  </tr>
  <tr>
    <td>Alleviating Hallucinations of Large Language Models through Induced Hallucinations</td>
    <td>Mitigating Hallucinations</td>
    <td><a href="https://arxiv.org/abs/2312.15710"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2023-10868" alt="PDF Badge">
      </a></td>
    <td></td>
  </tr>
  <tr>
    <td>Contrastive Decoding Improves Reasoning in Large Language Models</td>
    <td>Augmenting Reasoning Capabilities</td>
    <td><a href="https://arxiv.org/abs/2309.09117"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2023-10868" alt="PDF Badge">
      </a></td>
    <td> </td>
  </tr>
  <tr>
    <td>CoGenesis: A Framework Collaborating Large and Small Language Models for Secure Context-Aware Instruction Following</td>
    <td>Safeguarding Privacy</td>
    <td><a href="https://arxiv.org/abs/2403.03129"> 
      <img src="https://img.shields.io/badge/PDF-ACL 2024-10868" alt="PDF Badge">
      </a></td>
    <td> </td>
  </tr>
</table>

## LLMs Enhance SMs <a name="llms-enhance-sms"></a>

### Knowledge Distillation <a name="knowledge-distillation"></a>

#### Black-box Distillation: <a name="black-box-distillation"></a>
<table>
  <tr>
    <th>Title</th>
    <th>Topic</th>
    <th>Venue</th>
    <th>Code</th>
  </tr>
  <tr>
    <td>Explanations from Large Language Models Make Small Reasoners Better</td>
    <td>Chain-Of-Thought Distillation</td>
    <td><a href="https://arxiv.org/abs/2210.06726"> 
      <img src="https://img.shields.io/badge/PDF-ACL 2023-10868" alt="PDF Badge">
      </a></td>
    <td> </td>
  </tr>
  <tr>
    <td>Distilling Step-by-Step! Outperforming Larger Language Models
with Less Training Data and Smaller Model Sizes</td>
    <td>Chain-Of-Thought Distillation</td>
    <td><a href="https://arxiv.org/abs/2305.02301"> 
      <img src="https://img.shields.io/badge/PDF-ACL 2023-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/google-research/distilling-step-by-step">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
  <tr>
    <td>Distilling Reasoning Capabilities into Smaller Language Models</td>
    <td>Chain-Of-Thought Distillation</td>
    <td><a href="https://aclanthology.org/2023.findings-acl.441/"> 
      <img src="https://img.shields.io/badge/PDF-ACL 2023-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/kumar-shridhar/Distiiling-LM">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
  </tr>
  <tr>
    <td>Teaching Small Language Models to Reason</td>
    <td>Chain-Of-Thought Distillation</td>
    <td><a href="https://aclanthology.org/2023.acl-short.151/"> 
      <img src="https://img.shields.io/badge/PDF-ACL 2023-10868" alt="PDF Badge">
      </a></td>
    <td> </td>
  </tr>
  <tr>
    <td>Symbolic Chain-of-Thought Distillation: Small Models Can Also "Think" Step-by-Step
</td>
    <td>Chain-Of-Thought Distillation</td>
    <td><a href="https://arxiv.org/abs/2306.14050"> 
      <img src="https://img.shields.io/badge/PDF-ACL 2023-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/liunian-harold-li/scotd">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a>  </td>
  </tr>
  <tr>
    <td>Specializing Smaller Language Models towards Multi-Step Reasoning</td>
    <td>Chain-Of-Thought Distillation</td>
    <td><a href="https://arxiv.org/abs/2301.12726"> 
      <img src="https://img.shields.io/badge/PDF-ICML 2023-10868" alt="PDF Badge">
      </a></td>
    <td>  </td>
  </tr>
  <tr>
    <td>TinyLLM: Learning a Small Student from Multiple Large Language Models
</td>
    <td>Chain-Of-Thought Distillation</td>
    <td><a href="https://arxiv.org/abs/2402.04616"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2024-10868" alt="PDF Badge">
      </a></td>
    <td> </td>
  </tr>
  <tr>
    <td>Lion: Adversarial Distillation of Proprietary Large Language Models</td>
    <td>Instruction Following Distillation</td>
    <td><a href="https://arxiv.org/abs/2305.12870"> 
      <img src="https://img.shields.io/badge/PDF-EMNLP 2023-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/YJiangcm/Lion">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
      <tr>
    <td>Selective Reflection-Tuning: Student-Selected Data Recycling for LLM Instruction-Tuning</td>
    <td>Instruction Following Distillation</td>
    <td><a href="https://arxiv.org/abs/2402.10110"> 
      <img src="https://img.shields.io/badge/PDF-ACL 2024-10868" alt="PDF Badge">
      </a></td>
    <td> <a href="https://github.com/tianyi-lab/Reflection_Tuning">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a> </td>
  </tr>
</table>

#### White-box Distillation: <a name="white-box-distillation"></a>
<table>
  <tr>
    <th>Title</th>
    <th>Topic</th>
    <th>Venue</th>
    <th>Code</th>
  </tr>
  <tr>
    <td>DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter</td>
    <td>Logits</td>
    <td><a href="https://arxiv.org/abs/1910.01108"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2019-10868" alt="PDF Badge">
      </a></td>
    <td> </td>
  </tr>
  <tr>
    <td>ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers</td>
    <td>Intermediate Features</td>
    <td><a href="https://arxiv.org/abs/2206.01861"> 
      <img src="https://img.shields.io/badge/PDF-NeurIPS 2022-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/microsoft/DeepSpeed">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
  <tr>
    <td>Less is More: Task-aware Layer-wise Distillation for Language Model Compression</td>
    <td>Intermediate Features</td>
    <td><a href="https://arxiv.org/abs/2210.01351"> 
      <img src="https://img.shields.io/badge/PDF-ICML 2023-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/cliang1453/task-aware-distillation">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
  <tr>
    <td>MiniLLM: Knowledge Distillation of Large Language Models</td>
    <td>Intermediate Features</td>
    <td><a href="https://arxiv.org/abs/2306.08543"> 
      <img src="https://img.shields.io/badge/PDF-ICLR 2024-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/microsoft/LMOps/tree/main/minillm">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
  <tr>
    <td>LLM-QAT: Data-Free Quantization Aware Training for Large Language Models</td>
    <td>Intermediate Features</td>
    <td><a href="https://arxiv.org/abs/2305.17888"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2023-10868" alt="PDF Badge">
      </a></td>
    <td></td>
  </tr>
</table>

### Data Synthesis <a name="data-synthesis"></a>

#### Data Augmentation: <a name="data-augmentation"></a>
<table>
  <tr>
    <th>Title</th>
    <th>Topic</th>
    <th>Venue</th>
    <th>Code</th>
  </tr>
  <tr>
    <td>Improving data augmentation for low resource speech-to-text translation with diverse paraphrasing</td>
    <td>Text Paraphrase</td>
    <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S0893608022000260"> 
      <img src="https://img.shields.io/badge/PDF-NN 2022-10868" alt="PDF Badge">
      </a></td>
    <td> </td>
  </tr>
  <tr>
    <td>Paraphrasing with Large Language Models</td>
    <td>Text Paraphrase</td>
    <td><a href="https://aclanthology.org/D19-5623/"> 
      <img src="https://img.shields.io/badge/PDF-NGT 2019-10868" alt="PDF Badge">
      </a></td>
    <td></td>
  </tr>
  <tr>
    <td>Query Rewriting for Retrieval-Augmented Large Language Models</td>
    <td>Query Rewriting</td>
    <td><a href="https://arxiv.org/abs/2305.14283"> 
      <img src="https://img.shields.io/badge/PDF-EMNLP 2023-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/xbmxb/RAG-query-rewriting">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
  <tr>
    <td>LLMvsSmall Model? Large Language Model Based Text Augmentation Enhanced Personality Detection Model</td>
    <td>Specific Tasks</td>
    <td><a href="https://arxiv.org/abs/2403.07581"> 
      <img src="https://img.shields.io/badge/PDF-NLP4ConvAI 2022-10868" alt="PDF Badge">
      </a></td>
    <td></td>
  </tr>
  <tr>
    <td>Data Augmentation for Intent Classification with Off-the-shelf Large Language Models</td>
    <td>Specific Tasks</td>
    <td><a href="https://aclanthology.org/2022.nlp4convai-1.5/"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2023-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/ServiceNow/data-augmentation-with-llms">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
  <tr>
    <td>Weakly Supervised Data Augmentation Through Prompting for Dialogue Understanding</td>
    <td>Specific Tasks</td>
    <td><a href="https://arxiv.org/abs/2210.14169"> 
      <img src="https://img.shields.io/badge/PDF-SyntheticData4ML 2023-10868" alt="PDF Badge">
      </a></td>
    <td></td>
  </tr>
</table>


#### Training Data Generation: <a name="training-data-generation"></a>
<table>
  <tr>
    <th>Title</th>
    <th>Topic</th>
    <th>Venue</th>
    <th>Code</th>
  </tr>
  <tr>
    <td>Want To Reduce Labeling Cost? GPT-3 Can Help</td>
    <td>Label Annotation</td>
    <td><a href="https://aclanthology.org/2021.findings-emnlp.354/"> 
      <img src="https://img.shields.io/badge/PDF-EMNLP 2021-10868" alt="PDF Badge">
      </a></td>
    <td> </td>
  </tr>
  <tr>
    <td>Self-Guided Noise-Free Data Generation for Efficient Zero-Shot Learning</td>
    <td>Label Annotation</td>
    <td><a href="https://arxiv.org/abs/2205.12679"> 
      <img src="https://img.shields.io/badge/PDF-ICLR 2023-10868" alt="PDF Badge">
      </a></td>
    <td></td>
  </tr>
  <tr>
    <td>ZeroGen: Efficient Zero-shot Learning via Dataset Generation</td>
    <td>Dataset Generation</td>
    <td><a href="https://aclanthology.org/2022.emnlp-main.801/"> 
      <img src="https://img.shields.io/badge/PDF-EMNLP 2022-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/HKUNLP/ZeroGen">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
  <tr>
    <td>Generating Training Data with Language Models: Towards Zero-Shot Language Understanding</td>
    <td>Dataset Generation</td>
    <td><a href="https://arxiv.org/abs/2202.04538"> 
      <img src="https://img.shields.io/badge/PDF-NeurIPS 2022-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/yumeng5/SuperGen">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
  <tr>
    <td>Increasing Diversity While Maintaining Accuracy: Text Data Generation with Large Language Models and Human Interventions</td>
    <td>Dataset Generation</td>
    <td><a href="https://arxiv.org/abs/2306.04140"> 
      <img src="https://img.shields.io/badge/PDF-ACL 2023-10868" alt="PDF Badge">
      </a></td>
    <td></td>
  </tr>
  <tr>
    <td>Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations</td>
    <td>Dataset Generation</td>
    <td><a href="https://arxiv.org/abs/2310.07849"> 
      <img src="https://img.shields.io/badge/PDF-EMNLP 2023-10868" alt="PDF Badge">
      </a></td>
    <td></td>
  </tr>
  <tr>
    <td>Does Synthetic Data Generation of LLMs Help Clinical Text Mining?</td>
    <td>Dataset Generation</td>
    <td><a href="https://arxiv.org/abs/2303.04360"> 
      <img src="https://img.shields.io/badge/PDF-arXiv 2023-10868" alt="PDF Badge">
      </a></td>
    <td></td>
  </tr>
  <tr>
    <td>Exploiting Asymmetry for Synthetic Training Data Generation: SynthIE and the Case of Information Extraction</td>
    <td>Dataset Generation</td>
    <td><a href="https://arxiv.org/abs/2303.04132"> 
      <img src="https://img.shields.io/badge/PDF-EMNLP 2023-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/epfl-dlab/SynthIE">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
  <tr>
    <td>ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection</td>
    <td>Dataset Generation</td>
    <td><a href="https://aclanthology.org/2022.acl-long.234/"> 
      <img src="https://img.shields.io/badge/PDF-ACL 2022-10868" alt="PDF Badge">
      </a></td>
    <td><a href="https://github.com/microsoft/toxigen">
  <img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="PDF Badge"></a></td>
  </tr>
</table>


# Competition <a name="competition"></a>

<details>

## Computation-constrained Environment <a name="computation-constrained-environment"></a>
## Task-specific Environment <a name="task-specific-environment"></a>
## Interpretability-required Environment <a name="interpretability-required-environment"></a>

</details>

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
