# <img src="./static/images/fakereasoning.png" alt="fakereasoning" style="width: 5%;" /> FakeReasoning

[<img src="https://img.shields.io/badge/arXiv-2503.21210-b31b1b.svg" alt="arXiv" style="zoom:150%;" />](https://arxiv.org/abs/2503.21210)[<img src="https://img.shields.io/badge/Project-Page-green.svg" alt="Project Page" style="zoom:150%;" />](https://pris-cv.github.io/FakeReasoning/)[<img src="https://img.shields.io/badge/Dataset-Page-orange.svg" alt="Dataset" style="zoom:150%;" />](https://huggingface.co/datasets/AnnaGao/MMFR-Dataset)

This is the source code for *FakeReasoning: Towards Generalizable Forgery Detection and Reasoning*. 

**Abstract:** Accurate and interpretable detection of AI-generated images is essential for mitigating risks associated with AI misuse. However, the substantial domain gap among generative models makes it challenging to develop a generalizable forgery detection model. Moreover, since every pixel in an AI-generated image is synthesized, traditional saliency-based forgery explanation methods are not well suited for this task. To address these challenges, we propose modeling AI-generated image detection and explanation as a Forgery Detection and Reasoning task (FDR-Task), leveraging vision-language models (VLMs) to provide accurate detection through structured and reliable reasoning over forgery attributes. To facilitate this task, we introduce the Multi-Modal Forgery Reasoning dataset (MMFR-Dataset), a large-scale dataset containing 100K images across 10 generative models, with 10 types of forgery reasoning annotations, enabling comprehensive evaluation of FDR-Task. Additionally, we propose FakeReasoning, a forgery detection and reasoning framework with two key components. First, Forgery-Aligned Contrastive Learning enhances VLMs' understanding of forgery-related semantics through both cross-modal and intra-modal contrastive learning between images and forgery attribute reasoning. Second, a Classification Probability Mapper bridges the optimization gap between forgery detection and language modeling by mapping the output logits of VLMs to calibrated binary classification probabilities. 

## News

* **Jun 11 2025**: The [MMFR-Dataset](https://huggingface.co/datasets/AnnaGao/MMFR-Dataset) is released! Also we provide codes to follow our dataset construction pipeline. 

* **Apr 15 2025**:  The [Project Page](https://pris-cv.github.io/FakeReasoning/) of our paper has been published! Click to find more about performance of FakeReasoning and samples in MMFR-Dataset.
* **Mar 27 2025**:  [Our Paper](https://arxiv.org/abs/2503.21210) is released on arXiv.

## TODO

- [x] Release paper
- [x] Release project page
- [x] Release MMFR-Dataset
- [ ] Release checkpoints and inference code
- [ ] Release Training code

## Dataset

The training set of MMFR-Dataset contains 50K fake images with 129K reasoning annotations and 50K real images with 183K reasoning annotations. The evaluation sets of MMFR-Dataset contains 20K images with 66K reasoning annotations across 2 classes and 10 generative models. 

### Download

MMFR-Dataset is available on [huggingface](https://huggingface.co/datasets/AnnaGao/MMFR-Dataset).

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
│   ├── biggan
│   │   ├── 0_real
│   │   ├── 1_fake
│   │   └── biggan_reasoning.json
│   ├── [...]
│   └── styleganxl
└── forgery_reasoning_cot.json
```

`forgery_reasoning_cot.json` contains instruction-CoT annotations for the training set. We also provide original reasoning annotations in `diffusiondb_reasoning.json` and `laion_reasoning.json` (for the training set). Reasoning annotations for evaluation sets, such as `biggan_reasoning.json`, can be found within their respective subfolders.

### Generation

Codes are included in `./mmfr_generation/`. We use batch API of GPT-4o for dataset generation. To follow our construction pipeline:

1. Generate jsonl files with `get_jsonl.py` for batch requests.
2. Upload your jsonl files and get output from GPT-4o with `batch_api_generation.ipynb`.
3. Organize original output from GPT-4o to structured reasoning annotation with `output_to_reasoning.py`.

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

