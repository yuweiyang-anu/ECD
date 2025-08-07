<h1 align="center">
  Effective Training Data Synthesis for Improving MLLM Chart Understanding
</h1>
<h3 align="center" style="color: blue; font-weight: normal;">
  ICCV 2025
</h3>

<p align="center">
  Yuwei Yang<sup>1</sup>, Zeyu Zhang<sup>1</sup>, Yunzhong Hou<sup>1</sup>, Zhuowan Li<sup>4</sup>, Gaowen Liu<sup>3</sup>, Ali Payani<sup>3</sup>, Yuan-Sen Ting<sup>2</sup>, Liang Zheng<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Australian National University &emsp;
  <sup>2</sup>Ohio State University &emsp;
  <sup>3</sup>Cisco &emsp;
  <sup>4</sup>Johns Hopkins University
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/ChartFoundation/ECD-10k-Images">🤗 Dataset </a> &ensp;
  <a href="https://huggingface.co/datasets/ChartFoundation/ECDBench">🥇 Benchmark </a> &ensp;
  <a href="https://huggingface.co/ChartFoundation/ECD_Finetuned_MLLMs"> 🧠 Models </a> &ensp;
  <br><br>
</p>

![teaser](overview.png)

## ✨ Abstract
Being able to effectively read scientific plots, or chart understanding, is a central part toward building effective agents for science. However, existing multimodal large language models (MLLMs), especially open-source ones, are still falling behind with a typical success rate of 30%-50% on challenging benchmarks. Previous studies on fine-tuning MLLMs with synthetic charts are often restricted by their inadequate similarity to the real charts, which could compromise model training and performance on complex real-world charts. In this study, we show that modularizing chart generation and diversifying visual details improves chart understanding capabilities. In particular, we design a five-step data synthesis pipeline, where we separate data and function creation for single plot generation, condition the generation of later subplots on earlier ones for multi-subplot figures, visually diversify the generated figures, filter out low quality data, and finally generate the question-answer (QA) pairs with GPT-4o. 
This approach allows us to streamline the generation of fine-tuning datasets and introduce the effective chart dataset (ECD), which contains 10k+ chart images and 300k+ QA pairs, covering 25 topics and featuring 250+ chart type combinations with high visual complexity. We show that ECD consistently improves the performance of various MLLMs on a range of real-world and synthetic test sets.

## ♻️ Data Generation
The overall data generation pipeline used in our work is open-sourced and available for customization. It includes all necessary components for generating ECD synthetic data tailored to MLLM training and evaluation. For a detailed implementation of the pipeline, please refer to the [README](https://github.com/yuweiyang-anu/ECD/tree/main/data_generation_pipeline/README.md) file in the `data_generation_pipeline` directory. The complete ECD dataset (including chart images, codes and QA pairs) generated using this pipeline is publicly available on [🤗Hugging Face](https://huggingface.co/datasets/ChartFoundation/ECD-10k-Images).

## 🛠️ SFT on MLLMs

## 🏆 Evaluation

## 📅 Citation
If ECD proves beneficial to your research, please consider citing our paper in your work:

```
@inproceedings{yang2025effective,
     title={Effective Training Data Synthesis for Improving MLLM Chart Understanding},
     author={Yang, Yuwei and Zhang, Zeyu and Hou, Yunzhong and Li, Zhuowan and Liu, Gaowen and Payani, Ali and Ting, Yuan-Sen and Zheng, Liang},
     booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
     year={2025}
 }
```


