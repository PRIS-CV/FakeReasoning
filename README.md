# <img src="./static/images/fakereasoning.png" alt="fakereasoning" style="zoom:10%;" /> FakeReasoning

[![arXiv](https://img.shields.io/badge/arXiv-2503.21210-b31b1b.svg)](https://arxiv.org/abs/2503.21210)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://pris-cv.github.io/FakeReasoning/)

This is the source code for *FakeReasoning: Towards Generalizable Forgery Detection and Reasoning*. 

**Abstract:** Accurate and interpretable detection of AI-generated images is essential for mitigating risks associated with AI misuse. However, the substantial domain gap among generative models makes it challenging to develop a generalizable forgery detection model. Moreover, since every pixel in an AI-generated image is synthesized, traditional saliency-based forgery explanation methods are not well suited for this task. To address these challenges, we propose modeling AI-generated image detection and explanation as a Forgery Detection and Reasoning task (FDR-Task), leveraging vision-language models (VLMs) to provide accurate detection through structured and reliable reasoning over forgery attributes. To facilitate this task, we introduce the Multi-Modal Forgery Reasoning dataset (MMFR-Dataset), a large-scale dataset containing 100K images across 10 generative models, with 10 types of forgery reasoning annotations, enabling comprehensive evaluation of FDR-Task. Additionally, we propose FakeReasoning, a forgery detection and reasoning framework with two key components. First, Forgery-Aligned Contrastive Learning enhances VLMs' understanding of forgery-related semantics through both cross-modal and intra-modal contrastive learning between images and forgery attribute reasoning. Second, a Classification Probability Mapper bridges the optimization gap between forgery detection and language modeling by mapping the output logits of VLMs to calibrated binary classification probabilities. 

## News

* **Apr 15 2025**:  The [Project Page](https://pris-cv.github.io/FakeReasoning/) of our paper has been published! Click to find more about performance of FakeReasoning and samples in MMFR-Dataset.
* **Mar 27 2025**:  [Our Paper](https://arxiv.org/abs/2503.21210) is released on arXiv.

## TODO

- [x] Release paper
- [x] Release project page
- [ ] Write environment setting
- [ ] Release checkpoints and inference code

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

