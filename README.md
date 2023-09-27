## You Only Look at Screens: Multimodal Chain-of-Action Agents

*"In every systematic inquiry (methodos) where there are first principles, or causes, or elements, knowledge and science result from acquiring knowledge of these; for we think we know something just in case we acquire knowledge of the primary causes, the primary first principles, all the way to the elements."*

<p align="right">-- Aristotle (384 BC - 322 BC)</p>

![](overview.jpg)

Auto-UI is a multimodal solution that directly interacts with the interface, bypassing the need for environment parsing or reliance on application-dependent APIs. To improve the agent's action prediction capability, we propose a novel chain-of-action technique, where a chain of action is a series of intermediate previous action histories and future action plans that lead to action prediction.

## Requirements

Install all required python dependencies:

```
pip install -r requirements.txt
```

## Datasets

Download the processed dataset from the following repository: https://huggingface.co/cooelf/Auto-UI/tree/main.

## Extract Features (optional)

The following script will download and processed the AITW dataset. 

```
CUDA_VISIBLE_DEVICES=0 python fetch_features.py --dataset general --split_file dataset/general_texts_splits.json --output_dir dataset/blip/general_blip
CUDA_VISIBLE_DEVICES=1 python fetch_features.py --dataset install --split_file dataset/install_texts_splits.json --output_dir dataset/blip/install_blip
CUDA_VISIBLE_DEVICES=2 python fetch_features.py --dataset google_apps --split_file dataset/google_apps_texts_splits.json --output_dir dataset/blip/google_apps_blip
CUDA_VISIBLE_DEVICES=3 python fetch_features.py --dataset single --split_file dataset/single_texts_splits.json --output_dir dataset/blip/single_blip
CUDA_VISIBLE_DEVICES=4 python fetch_features.py --dataset web_shopping --split_file dataset/web_shopping_texts_splits.json --output_dir dataset/blip/web_shopping_blip
```
The structure of the dataset folder should be:

```
dataset
├── general_texts_splits.json
├── install_texts_splits.json
├── google_apps_texts_splits.json
├── single_texts_splits.json
├── web_shopping_texts_splits.json
├── blip
│   └── general_blip_train.obj
│   └── general_blip_val.obj
│   └── general_blip_test.obj
│   └── ...
```

## Instructions

### Training 

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --data_root blip \
    --model declare-lab/flan-alpaca-base \
    --epoch 10 --lr 1e-4 \
    --user_msg seq_future_blip_axis_all0.1_hist8_future4 --img_type blip --img_dim 1408 \
    --bs 4 --eval_bs 16 --input_len 512 --output_len 128 --eval_acc 40 \
    --transform_axis --warmup_ratio 0.05 \
    --all_data 0.1 \
    --use_history 8 \
    --use_future 4 \
    --eval_subset dataset/blip/general_blip \
    --output_dir experiments
```

### Inference 

Our trained models are available at https://huggingface.co/cooelf/Auto-UI/tree/main.

```
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --data_root dataset/blip/general_blip \
    --model declare-lab/flan-alpaca-base \
    --epoch 10 --lr 1e-4 \
    --user_msg seq_future_blip_axis_all0.1_hist8_future4 --img_type blip --img_dim 1408 \
    --bs 4 --eval_bs 16 --input_len 512 --output_len 128 --eval_acc 40 \
    --transform_axis --warmup_ratio 0.05 \
    --use_history 8 \
    --use_future 4 \
    --eval_name general \
    --evaluate_dir Auto-UI-Base
```

## Citing Auto-UI

```
@article{zhan2023autoui,
  title={You Only Look at Screens: Multimodal Chain-of-Action Agents},
  author={Zhan, Zhuosheng and Zhang, Aston},
  journal={arXiv preprint arXiv:2309.11436},
  year={2023}
}
```

## License

This project is licensed under the Apache-2.0 License.