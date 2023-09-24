import action_type, action_matching
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import json
import jax.numpy as jnp
import argparse
import pickle
import torch
import tensorflow as tf
from PIL import Image
from transformers import AutoProcessor, Blip2Model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
model.to(device)
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

dataset_directories = {
    'general': 'gs://gresearch/android-in-the-wild/general/*',
    'google_apps': 'gs://gresearch/android-in-the-wild/google_apps/*',
    'install': 'gs://gresearch/android-in-the-wild/install/*',
    'single': 'gs://gresearch/android-in-the-wild/single/*',
    'web_shopping': 'gs://gresearch/android-in-the-wild/web_shopping/*',
}

def _decode_image(
    example,
    image_height,
    image_width,
    image_channels,
):
    """Decodes image from example and reshapes.

    Args:
        example: Example which contains encoded image.
        image_height: The height of the raw image.
        image_width: The width of the raw image.
        image_channels: The number of channels in the raw image.

    Returns:
        Decoded and reshaped image tensor.
    """
    image = tf.io.decode_raw(
        example.features.feature['image/encoded'].bytes_list.value[0],
        out_type=tf.uint8,
    )

    height = tf.cast(image_height, tf.int32)
    width = tf.cast(image_width, tf.int32)
    n_channels = tf.cast(image_channels, tf.int32)

    return tf.reshape(image, (height, width, n_channels))

def parse_episode(
    episode,
    get_images = False,
    get_annotations = False,
    get_actions = False,
):
    parsed_episode = []
    for i, ex in enumerate(episode):
        goal = ex.features.feature['goal_info'].bytes_list.value[0].decode('utf-8')
        step_id = ex.features.feature['step_id'].int64_list.value[0]
        # episode_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
        output_ep = {
            "goal": goal,
            "step_id": step_id
        }

        image_height = ex.features.feature['image/height'].int64_list.value[0]
        image_width = ex.features.feature['image/width'].int64_list.value[0]
        image_channels = ex.features.feature['image/channels'].int64_list.value[0]
        if get_images:
            image = _decode_image(ex, image_height, image_width, image_channels)
            image = image.numpy()
            image = Image.fromarray(image).convert('RGB')

            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
                image_features = model.get_image_features(**inputs).pooler_output[0]
                image_features = image_features.detach().cpu()
            output_ep["image"] = image_features

        if get_annotations:
            flattened_positions = np.array(
            ex.features.feature['image/ui_annotations_positions'].float_list.value
            )
            ui_text = ex.features.feature['image/ui_annotations_text'].bytes_list.value
            ui_text = [value.decode('utf-8') for value in ui_text]
            ui_type = ex.features.feature['image/ui_annotations_ui_types'].bytes_list.value
            ui_type = [value.decode('utf-8') for value in ui_type]

            positions = np.reshape(flattened_positions, (-1, 4)) #(y, x, height, width)
            output_ep["ui_positions"] = positions
            output_ep["ui_text"] = ui_text
            output_ep["ui_type"] = ui_type
        
        if get_actions:
            touch_y, touch_x = ex.features.feature['results/yx_touch'].float_list.value
            lift_y, lift_x = ex.features.feature['results/yx_lift'].float_list.value
            ex_action_type = ex.features.feature['results/action_type'].int64_list.value[0]

            ex_action_type = action_type.ActionType(ex_action_type).name

            type_text = (ex.features.feature['results/type_action'].bytes_list.value[0].decode('utf-8'))
            
            output_ep["result_touch_yx"] = [touch_y, touch_x]
            output_ep["result_lift_yx"] = [lift_y, lift_x]
            output_ep["result_action"] = [ex_action_type, type_text]

        parsed_episode.append(output_ep)
    return parsed_episode

def fetch_episode(dataset_name, data_split, get_images, get_annotations, get_actions):
    filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
    dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()

    with open (data_split, "r") as rp:
        split_data = json.load(rp)
        train_data = split_data["train"]
        val_data = split_data["val"]
        test_data = split_data["test"]
        print(f"train_data size: {len(train_data)}, val_data size: {len(val_data)}, test_data size: {len(test_data)}")

    all_parsed_episode = {
        "train": [],
        "val": [],
        "test": [],
    }
    total_screens = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    episode = []
    episode_id = None
    
    for d in tqdm(dataset):
        ex = tf.train.Example()
        ex.ParseFromString(d)
        ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
        # if (ep_id not in train_data) & (ep_id not in test_data):
        #     continue
        if episode_id is None:
            episode_id = ep_id
            episode.append(ex)
        elif ep_id == episode_id:
            episode.append(ex)
        else:
            # save data
            try:
                output = parse_episode(episode, get_images=get_images, get_annotations=get_annotations, get_actions=get_actions)
            except Exception as exc:
                print(exc)
                #  bad data point; init a new episode
                episode_id = ep_id
                episode = [ex]

            if episode_id in train_data:
                curr_split = "train"
            elif episode_id in val_data:
                curr_split = "val"
            elif episode_id in test_data:
                curr_split = "test"
            else:
                assert "error episode"
            
            all_parsed_episode[curr_split].append({"episode_id":episode_id, "data":output})
            total_screens[curr_split] += len(episode)
            # init a new episode
            episode_id = ep_id
            episode = [ex]
    # last episode
    if len(episode) > 0:
        # save data
        output = parse_episode(episode, get_images=get_images, get_annotations=get_annotations, get_actions=get_actions)
        if episode_id in train_data:
            curr_split = "train"
        elif episode_id in val_data:
            curr_split = "val"
        elif episode_id in test_data:
            curr_split = "test"
        else:
            assert "error episode"
        
        all_parsed_episode[curr_split].append({"episode_id":episode_id, "data":output})
        total_screens[curr_split] += len(episode)

    print(len(all_parsed_episode["train"]), total_screens["train"], len(all_parsed_episode["val"]), total_screens["val"], len(all_parsed_episode["test"]), total_screens["test"])
    return all_parsed_episode

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='general')
    parser.add_argument("--split_file", type=str, default="dataset/general_texts_splits.json")
    parser.add_argument('--output_dir', type=str, default='dataset/t5/general_parsed_episode_t5_clip')
    parser.add_argument('--get_images', default=True, action='store_true')
    parser.add_argument('--get_annotations', default=True, action='store_true')
    parser.add_argument('--get_actions', default=True, action='store_true')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    all_parsed_episode = fetch_episode(args.dataset, args.split_file, args.get_images, args.get_annotations, args.get_actions)
    
    with open(f"{args.output_dir}_train.obj", "wb") as wp:
        pickle.dump(all_parsed_episode["train"],wp)
    with open(f"{args.output_dir}_val.obj", "wb") as wp:
        pickle.dump(all_parsed_episode["val"],wp)
    with open(f"{args.output_dir}_test.obj", "wb") as wp:
        pickle.dump(all_parsed_episode["test"],wp)
