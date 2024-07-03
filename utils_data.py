from torch.utils.data import Dataset
import torch
import pickle
from tqdm import tqdm
import action_matching, action_type
import numpy as np
import jax.numpy as jnp
import random
import re
img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
    "vit": (577, 768),
    "vit-large": (145, 1024),
    "vit-global": (1, 768),
    "vit-merge": (578, 768),
}


def load_data(args, split):
    target_text = []
    source_text = []
    source_image = []
    anno_positions = []

    if args.all_data:
        if split == "train":
            data = []
            for subdir in ["general", "google_apps", "install", "single", "web_shopping"]:
                print(f"loading {subdir}", len(data))
                with open(f"dataset/blip/{subdir}_{args.data_root}_{split}.obj", "rb") as rp:
                    sub_data = pickle.load(rp)
                if subdir == "google_apps":
                    sub_data = random.sample(sub_data, int(len(sub_data) * args.all_data))
                data.extend(sub_data)
        else:
            # we use general subset for dev/test
            with open(f"{args.eval_subset}_{split}.obj", "rb") as rp:
                    data = pickle.load(rp)
    else:
        with open(f"{args.data_root}_{split}.obj", "rb") as rp:
            data = pickle.load(rp)
            if args.data_ratio:
                data = random.sample(data, int(len(data) * args.data_ratio))

    for qid, episode in enumerate(tqdm(data)):
        episode_id = episode["episode_id"]
        episode_data = episode["data"]
        if args.use_history:
            history_action = []
            if args.use_img_history:
                history_image = [torch.zeros(args.img_dim)] * args.use_history

        for step_idx, step_data in enumerate(episode_data):
            question = step_data["goal"]
            question = f"Goal: {question}"

            image = step_data["image"]

            ui_positions = step_data["ui_positions"]
            ui_text = step_data["ui_text"]
            ui_type = step_data["ui_type"]

            if args.use_layout:
                icon_string = ""
                for ui_idx, ui_type_i in enumerate(ui_type):
                    ui_axis = ui_positions[ui_idx]
                    top, left, height, width = ui_axis
                    # The y-axis is inverted for AndroidEnv, so bottom = top + height.
                    bottom, right = top + height, left + width
                    ui_axis = [top, left, bottom, right]
                    ui_axis = ["{:.4f}".format(axis) for axis in ui_axis]
                    ui_axis = f"({ui_axis[0]}, {ui_axis[1]}, {ui_axis[2]}, {ui_axis[3]})"
                    if ui_type_i == "TEXT":
                        icon_string += f'<p id={ui_idx} class="text" alt="{ui_axis}">{ui_text[ui_idx]}</p>\n'
                    elif "ICON" in ui_type_i:
                        icon_string += f'<img id={ui_idx} class={ui_type_i} alt="{ui_axis}">{ui_text[ui_idx]}</p>\n'
                    else:
                        print(icon_string)
                        assert "parsing ui failed!!!"
                
                question = f"{question}\nScreen: {icon_string}"
                # print(question)
            result_touch_yx = step_data["result_touch_yx"]
            result_lift_yx = step_data["result_lift_yx"]
            result_action = step_data["result_action"][0]
            result_text = step_data["result_action"][1]

            result_text = result_text.replace("\\", "").replace('"','').replace("'","")

            if args.transform_axis:
                scroll_map = {
                    "up": [[0.8000, 0.5000], [0.2000, 0.5000]],
                    "down": [[0.2000, 0.5000], [0.8000, 0.5000]],
                    "left": [[0.5000, 0.8000], [0.5000, 0.2000]],
                    "right": [[0.5000, 0.2000], [0.5000, 0.8000]]
                }
                action_touch_yx = jnp.asarray(result_touch_yx)
                action_lift_yx = jnp.asarray(result_lift_yx)
                if result_action == "DUAL_POINT":
                    if is_tap_action(action_touch_yx, action_lift_yx):
                        result_touch_yx = [round(axis, 4) for axis in result_touch_yx]
                        # if touching, the lift can be the same as touch
                        result_lift_yx = result_touch_yx
                    else:
                        drags_match = _check_drag_actions_match(
                            action_touch_yx, action_lift_yx
                        )
                        result_touch_yx, result_lift_yx = scroll_map[drags_match]

            target_action = f'"action_type": "{result_action}", "touch_point": "{result_touch_yx}", "lift_point": "{result_lift_yx}", "typed_text": "{result_text}"'
            
            if args.use_history:
                prev_actions = "\n".join(history_action)
                question = f"Previous Actions: {prev_actions}\n{question}"
                if args.use_img_history:
                    image = history_image + [image]
                    image = torch.stack(image)

            if args.use_future:
                future_actions = episode_data[step_idx:]
                if len(future_actions) > args.use_future:
                    future_actions = future_actions[:args.use_future]
                future_actions = "[" + ",".join([action_t["result_action"][0] for action_t in future_actions]) + "]\n"
                target_action_label = "Action Plan: " + future_actions + "; Action Decision: " + target_action

            source_text.append(question)
            source_image.append(image)
            target_text.append(target_action_label)
            anno_positions.append(ui_positions)

            if args.use_history:
                history_action.append(target_action)
                if args.use_img_history:
                    history_image.append(image[-1])
                    history_image.pop(0)
                if len(history_action) > args.use_history:
                    history_action.pop(0)
                        

        if args.debug_num:
            if int(qid) > args.debug_num:
                break
            
    return source_text, source_image, target_text, anno_positions

_SWIPE_DISTANCE_THRESHOLD = 0.04
def is_tap_action(normalized_start_yx, normalized_end_yx):
    distance = jnp.linalg.norm(
        jnp.array(normalized_start_yx) - jnp.array(normalized_end_yx))
    return distance <= _SWIPE_DISTANCE_THRESHOLD

def _check_drag_actions_match(
    drag_touch_yx,
    drag_lift_yx,
):
    """Determines if two drag actions are the same."""
    # Store drag deltas (the change in the y and x coordinates from touch to
    # lift), magnitudes, and the index of the main axis, which is the axis with
    # the greatest change in coordinate value (e.g. a drag starting at (0, 0) and
    # ending at (0.3, 0.5) has a main axis index of 1).
    drag_1_deltas = drag_lift_yx - drag_touch_yx
    drag_1_magnitudes = jnp.abs(drag_1_deltas)
    drag_1_main_axis = np.argmax(drag_1_magnitudes)

    # y axis
    if drag_1_main_axis == 0:
        if drag_1_deltas[0] < 0:
            scroll = "up"
        else:
            scroll = "down"
    elif drag_1_main_axis == 1:
        if drag_1_deltas[1] < 0:
            scroll = "left"
        else:
            scroll = "right"
            
    return scroll

class AITWDatasetImg(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, data, tokenizer, source_len, target_len
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = target_len
        self.source_text = data[0]
        self.source_image = data[1]
        self.target_text = data[2]
        self.anno_positions = data[3]
            
    def __len__(self):
        """returns the length of dataframe"""
        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        source_image = self.source_image[index]
        target_text_org = str(self.target_text[index])


        # abc = self.tokenizer.tokenize(target_text)
        # print(len(abc))

        pattern = r'(?<=Action Decision:\s).*'
        result = re.search(pattern, target_text_org)
        target_text = result.group(0)
        target_text = target_text.strip()
        
        target_dict = eval("{" + target_text + "}")
        action = action_type.ActionType[target_dict["action_type"]].value

        touch_point = eval(target_dict["touch_point"])
        lift_point = eval(target_dict["lift_point"])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text_org = " ".join(target_text_org.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text_org],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        
        image_ids = torch.tensor(source_image).squeeze()
        vis_attention_mask = torch.tensor([1]).squeeze()

        act_ids = torch.tensor(action).squeeze()
        touch_point = torch.tensor(touch_point).squeeze()
        lift_point = torch.tensor(lift_point).squeeze()
        
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids,
            "target_act": act_ids,
            "target_touch": touch_point,
            "target_lift": lift_point
        }
