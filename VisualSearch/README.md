# Visual Search Model

## Overview
This model performs image-based searches using Janus Pro as the LLM, OWViT as the detection model, and SegmentAnything for segmentation.

## Fine-tuning JanusPro-7B
To fine-tune JanusPro-7B with DeepSpeed and LoRA, run `finetune/finetune.sh`. See for an explanation of the [DeepSpeed configuration](finetune/zero3.md`).  
**Dataset for Fine-tuning with DeepSpeed:** [seal_vqa_data](https://huggingface.co/datasets/craigwu/seal_vqa_data/tree/main)

## VSM Model
For details about the VSM model, refer to [VSM Model](VSM_Janus_Explanation.md). [Development notes](dev_vsm_doc.md) are available.

## Fine-tuning the VSM Model
The VSM model is fine-tuned separately from JanusPro. Current work involves fine-tuning the VSM model, which integrates three components: JanusPro-7B, OWViT, and SegmentAnything.

## Training Data for the VSM Model

The training data consists of the following datasets:

1. **General Segmentation and Detection Datasets**  
    - [COCO-2017](https://cocodataset.org/#download)  
    - [COCO-Stuff](https://github.com/nightrome/cocostuff)  
    - [PACO-LVIS](https://github.com/facebookresearch/paco/tree/main#dataset-setup)  
    - [Objects364-V2](https://www.objects365.org/overview.html)

2. **Referring Segmentation Datasets**  
    - [refCOCO](https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip)  
    - [refCOCO+](https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip)  
    - [refCOCOg](https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip)  
    - [refCLEF](https://web.archive.org/web/20220413011817/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip) ([saiapr_tc-12](https://web.archive.org/web/20220515000000/http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip))

3. **Mixed Grounding Datasets**  
    Follow the instructions [here](https://github.com/ashkamath/mdetr/blob/main/.github/pretrain.md) to prepare the GQA images, Flickr30K images, and pre-processed annotations.

4. **Visual Question Answering Dataset**  
    - [LLaVA-80K Instruction Data](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_80k.json)  
    - [Possible Location QA Data](https://huggingface.co/datasets/craigwu/vsm_vqa_data)  
    - **Image Data:** COCO-2017

### Data Organization
After downloading the datasets, organize them as follows:
```
├── dataset
│   ├── coco2017
│   │   ├── annotations
│   │   └── train2017
│   ├── cocostuff
│   │   ├── annotations
│   │   └── train2017
│   ├── vsm_vqa_data
│   │   ├── llava_instruct_80k.json
│   │   └── possible_locations_conv_86k.json
│   ├── refer_seg
│   │   ├── images
│   │   │   ├── saiapr_tc-12
│   │   │   └── mscoco
│   │   │       └── images
│   │   │           └── train2014
│   │   ├── refclef
│   │   ├── refcoco
│   │   ├── refcoco+
│   │   └── refcocog
│   ├── MixedGrounding
│   │   ├── flickr30k-images
│   │   ├── GQA
│   │   │   └── images
│   │   ├── final_flickr_separateGT_train.json
│   │   └── final_mixed_train.json
│   ├── object365
│   │   ├── images
│   │   └── zhiyuan_objv2_train.json
│   └── vlpart
│       └── paco
│           └── annotations
```

### Data Preprocessing
After organizing the data, navigate to the `VisualSearch` folder and run:
```bash
python preprocess_data.py --data_dir DATASET_FOLDER
```