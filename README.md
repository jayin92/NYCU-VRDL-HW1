# NYCU-VRDL-HW1

## Introduction

This homework addresses the challenging task of fine-grained image classification across 100 diverse classes, encompassing insects, plants, and birds. The dataset presents several key challenges: (1) complex scenes with mixed foreground and background elements, (2) images containing multiple objects of interest, and (3) significant intra-class variation, such as classes containing both butterflies and their caterpillar forms. The dataset comprises 21,024 images for training and validation, with an additional 2,344 test images for evaluation.

To tackle these challenges, I implement ResNeSt (Zhang et al., 2022) as the backbone architecture, which enhances the original ResNet (He et al., 2016) design with Split-Attention blocks for improved feature representation. For robust training, I employ a comprehensive augmentation strategy combining **TrivialAugmentWide** (Müller & Hutter, 2021), **Mixup** (Zhang et al., 2018), and **CutMix** (Yun et al., 2019) techniques. This multi-faceted approach to data augmentation addresses the various object scales, positions, and contextual variations present in the dataset.

Additionally, I implement test-time augmentation to further improve prediction accuracy by averaging predictions across multiple transformed versions of each test image. This comprehensive approach yields excellent results, achieving 93% validation accuracy and 97% testing accuracy. The complete implementation, including training configurations and evaluation scripts, is available in the project repository at [https://github.com/jayin92/NYCU-VRDL-HW1](https://github.com/jayin92/NYCU-VRDL-HW1).

## Training

```bash
python train_cutmix.py --epochs 80 --use_wandb --model_id resnest200 --use_cutmix --data_dir ./data --image_size 320
```

## Create submission
```bash
python test.py --data_dir ./data/ --test_dir data/test/ --checkpoint <path to checkpoint> --model_id resnest200 --image_size 320 --unlabeled_test
```
