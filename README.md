# <img src="./static/images/fakereasoning.png" alt="fakereasoning" style="width: 5%;" /> FakeReasoning

[![](https://img.shields.io/badge/arXiv-2503.21210-b31b1b)](https://arxiv.org/abs/2503.21210)
[![](https://img.shields.io/badge/Project-Page-green)](https://pris-cv.github.io/FakeReasoning/)
[![](https://img.shields.io/badge/Dataset-Huggingface-orange)](https://huggingface.co/datasets/AnnaGao/MMFR-Dataset)
[![](https://img.shields.io/badge/Model-Huggingface-orange)](https://huggingface.co/AnnaGao/FakeReasoning)

This is the source code for *Towards Generalizable Forgery Detection and Reasoning*. In this paper,

* We formulate detection and explanation as a unified Forgery Detection and Reasoning task (FDR-Task), leveraging Multi-Modal Large Language Models (MLLMs) to provide accurate detection through reliable reasoning over forgery attributes. 
* We introduce the Multi-Modal Forgery Reasoning dataset (MMFR-Dataset), a large-scale dataset containing 120K images across 10 generative models, with 378K reasoning annotations on forgery attributes, enabling comprehensive evaluation of the FDR-Task. 
* We propose FakeReasoning, a forgery detection and reasoning framework with three key components: 1) a dual-branch visual encoder that integrates CLIP and DINO to capture both high-level semantics and low-level artifacts; 2) a Forgery-Aware Feature Fusion Module that leverages DINO's attention maps and cross-attention mechanisms to guide MLLMs toward forgery-related clues; 3) a Classification Probability Mapper that couples language modeling and forgery detection, enhancing overall performance. 

## News

* **Aug 27 2025**: The [pretrained model](https://huggingface.co/AnnaGao/FakeReasoning) and source code are released. If you have followed our earlier work, please note that both the **dataset** and **method** have been updated. Check details on [arXiv](https://arxiv.org/abs/2503.21210).
* **Jun 11 2025**: The [MMFR-Dataset](https://huggingface.co/datasets/AnnaGao/MMFR-Dataset) is released! Also we provide codes to follow our dataset construction pipeline. 
* **Apr 15 2025**:  The [Project Page](https://pris-cv.github.io/FakeReasoning/) of our paper has been published! Click to find more about performance of FakeReasoning and samples in MMFR-Dataset.
* **Mar 27 2025**:  [Our Paper](https://arxiv.org/abs/2503.21210) is released on arXiv.

## Dataset

The training set of MMFR-Dataset contains 50K fake images with 129K reasoning annotations and 50K real images with 183K reasoning annotations. The evaluation sets of MMFR-Dataset contains 20K images with 66K reasoning annotations across 10 generative models. 

### Download

MMFR-Dataset is available on [huggingface](https://huggingface.co/datasets/AnnaGao/MMFR-Dataset). Download all split `.tar` files, concatenate them into a single archive, and then extract the dataset.

### Structure

```markdown
./
├── diffusiondb
│   ├── part-000001
│   │   ├── 0a3c75bb-4bd0-47c8-a2ba-e2aee92ad43f.png
│   │   └── [...]
│   ├── [...]
│   ├── part-000051
│   └── diffusiondb_reasoning.json
├── laion
│   ├── 00000
│   │   ├── 000000000.jpg
│   │   └── [...]
│   ├── [...]
│   ├── 00047
│   └── laion_reasoning.json
├── evaluation_sets
│   ├── stablediffusion
│   │   ├── 0_real
│   │   ├── 1_fake
│   │   └── stablediffusion_reasoning.json
│   ├── [...]
│   └── gigagan
└── forgery_reasoning_cot.json
```

`forgery_reasoning_cot.json` contains instruction-CoT annotations for the training set. We also provide original reasoning annotations in `diffusiondb_reasoning.json` and `laion_reasoning.json` (for the training set). Reasoning annotations for evaluation sets, such as `stablediffusion_reasoning.json`, can be found within their respective subfolders.

### Generation

Codes are included in `./mmfr_generation/`. We use batch API of GPT-4o for dataset generation. To follow our construction pipeline:

1. Generate jsonl files with `get_jsonl.py` for batch requests.
2. Upload your jsonl files and get output from GPT-4o with `batch_api_generation.ipynb`.
3. Organize original output from GPT-4o to structured reasoning annotation with `output_to_reasoning.py`.

## Install

The implementation is based on **torch==2.1.2+cu121**.

1. Clone this repository and navigate to the LLaVA folder

```bash
git clone https://github.com/PRIS-CV/FakeReasoning.git
cd LLaVA
```

2. Install required packages

```bash
conda create -n fakereasoning python=3.10
conda activate fakereasoning
pip install -e .
```

3. Install additional dependencies for training

```bash
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

If the installation of flash-attn fails, please visit the [official GitHub release page](https://github.com/Dao-AILab/flash-attention/releases) and install the corresponding `.whl` package.

4. Install additional dependencies for evaluation

```bash
pip install nltk
pip install rouge-score
```

5. Download base models

FakeReasoning is built upon the following models:

- [CLIP](https://huggingface.co/openai/clip-vit-large-patch14)
- [DINO](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth)
- [MoF-Models](https://huggingface.co/MMVP/MoF_Models)

Please download the corresponding pretrained weights before running the framework.

## Inference

The pretrained model of FakeReasoning is available on [Hugging Face](https://huggingface.co/AnnaGao/FakeReasoning). To use your local weight of openai/clip-vit-large-patch14-336, modify the `"mm_vision_tower"` in `config.json` to `path_to_clip-vit-large-patch14-336`.

#### Inference with a single image

```bash
cd LLaVA/forgery_eval
export DINO_PATH='path_to_dinov2-main'
export DINO_WEIGHT='path_to_dinov2_vitl14_pretrain.pth'
```

```bash
python inference.py \
--model-path path_to_FakeReasoning_weights \
--img_path commonFake_COCO_if_stage_III_189.png
```

#### Evaluation on the MMFR-Dataset

```bash
python eval.py \
--model-path path_to_FakeReasoning_weights \
--dataset_path path_to_MMFR-Dataset \
--result_folder ./results \
--clip_path path_to_clip-vit-large-patch14-336
```

⚠️ **Note:** Multi-GPU inference is currently not supported. Please ensure that you have at least **30 GB of GPU memory** available on a single GPU to run inference and evaluation. 

## Training

FakeReasoning is trained on 8× A800 GPUs (40GB) for 3 epochs, with the entire training completed in about 7 hours.

```bash
cd LLaVA
export DINO_PATH='path_to_dinov2-main'
export DINO_WEIGHT='path_to_dinov2_vitl14_pretrain.pth'
```

```bash
bash finetune_task_lora.sh \
--data_path path_to_forgery_reasoning_cot.json \
--model_name_or_path path_to_MoF_Models \
--image_folder path_to_MMFR-Dataset \
--vision_tower path_to_clip-vit-large-patch14-336
```

⚠️ **Note:** If you change the number of training devices, always ensure:

per_device_train_batch_size × gradient_accumulation_steps × num_gpus = 128

## Citation

If you find this work useful for your research, please kindly cite our paper:

```latex
@article{gao2025fakereasoning,
  title={FakeReasoning: Towards Generalizable Forgery Detection and Reasoning},
  author={Gao, Yueying and Chang, Dongliang and Yu, Bingyao and Qin, Haotian and Chen, Lei and Liang, Kongming and Ma, Zhanyu},
  journal={arXiv preprint arXiv:2503.21210},
  year={2025},
  url={https://arxiv.org/abs/2503.21210}
}
```

## Acknowledgement

We are thankful to [LLaVA](https://github.com/haotian-liu/LLaVA), [MMVP](https://github.com/tsb0601/MMVP), [DINOv2](https://github.com/facebookresearch/dinov2), [UniFD](https://github.com/WisconsinAIVision/UniversalFakeDetect), and [MCAN](https://github.com/MILVLG/mcan-vqa) for releasing their models and code as open-source contributions.
