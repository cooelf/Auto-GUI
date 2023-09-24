import os
import numpy as np
import torch
import os
import re
import json
import argparse
import random
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from model import T5ForMultimodalGeneration
from utils_data import AITWDatasetImg, load_data
from rich.table import Column, Table
from rich import box
from rich.console import Console
console = Console(record=True)
import action_matching, action_type
import evaluate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='dataset/blip/general_blip')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default='declare-lab/flan-alpaca-base')
    parser.add_argument('--data_ratio', type=float, default=None)
    parser.add_argument('--eval_name', type=str, default=None, help='the saved subset name used for evaluation')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--debug_num', type=int, default=None)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=256)
    parser.add_argument('--img_dim', type=int, default=1408)
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--all_data', type=float, default=None, help='whether using all the data for training. Set the ratio for google apps to save computation')
    parser.add_argument('--eval_subset', type=str, default=None, help='use which subset for evaluation/test when training with all data')
    parser.add_argument('--use_history', type=int, default=8, help='use textual action history')
    parser.add_argument('--use_img_history', action='store_true', help='use screen history')
    parser.add_argument('--use_future', type=int, default=16, help='planning the future actions before giving the current action')
    parser.add_argument('--use_layout', action='store_true', help='use annotated layout information')
    parser.add_argument('--transform_axis', default=True, action='store_true', help='use coordinate normalization')
    parser.add_argument('--use_generate', default=True, action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="debug", help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default="blip", help='type of image features')
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    return args
        
if __name__ == '__main__':

    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )
    
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")
    
    if args.evaluate_dir is not None:
        save_dir = args.evaluate_dir
    else:
        model_name = args.model.replace("/","-")
        gpu_count = torch.cuda.device_count()
        save_dir = f"{args.output_dir}/{args.user_msg}_{model_name}_{args.img_type}_lr{args.lr}_bs{args.bs * gpu_count}_ip{args.input_len}_op{args.output_len}_ep{args.epoch}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    print(save_dir)
    
    model = T5ForMultimodalGeneration.from_pretrained(args.model, args.img_dim) 
    
    
    if args.evaluate_dir is not None:
        train_set = None
    else:
        training_data = load_data(args, "train")
        train_set = AITWDatasetImg(
            training_data,
            tokenizer,
            args.input_len,
            args.output_len
            )
    eval_data = load_data(args, "val")
    eval_set = AITWDatasetImg(
        eval_data,
        tokenizer,
        args.input_len,
        args.output_len
    )
    test_data = load_data(args, "test")
    test_set = AITWDatasetImg(
        test_data,
        tokenizer,
        args.input_len,
        args.output_len
    )

    datacollator = DataCollatorForSeq2Seq(tokenizer)
    print("model parameters: ", model.num_parameters())

    # rougel for rationale generation
    metric = evaluate.load("rouge")
    def compute_metrics_rouge(eval_preds):
        preds, targets = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds= np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        result = metric.compute(predictions=preds, references=targets)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # only use the last model for evaluation to save time
    if args.final_eval:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=False,
            warmup_ratio=args.warmup_ratio,
            evaluation_strategy="no",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit = 2,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            report_to="none",
            local_rank=args.local_rank
        )
    # evaluate at each epoch
    else:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=True,
            warmup_ratio=args.warmup_ratio,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit = 2,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            metric_for_best_model="rougeL",
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            load_best_model_at_end=True,
            report_to="none",
            local_rank=args.local_rank
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics = compute_metrics_rouge
    )

    if args.evaluate_dir is None:
        trainer.train()
        trainer.save_model(save_dir)
        
    # metrics = trainer.evaluate(eval_dataset = test_set, max_length=args.output_len)
    # trainer.log_metrics("test", metrics)
    # trainer.save_metrics("test", metrics)
    metrics = {}

    predict_results = trainer.predict(test_dataset=test_set, max_length=args.output_len) 
    if trainer.is_world_process_zero():
        preds, targets = predict_results.predictions, predict_results.label_ids
        preds= np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        action_correct = 0
        text_correct = 0
        type_correct = 0
        
        reference_test_positions = test_set.anno_positions

        output_data = []

        pattern = r'(?<=Action Decision:\s).*'

        assert len(preds) == len(targets)  == len(reference_test_positions)
        for idx, pred in enumerate(preds):
            try:
                result = re.search(pattern, targets[idx])
                target_text = result.group(0)
                target_text = target_text.strip()

                reference = eval("{" + target_text + "}")
            except:
                print("reference error")
                continue

            try:
                result = re.search(pattern, preds[idx])
                pred_text = result.group(0)
                pred_text = pred_text.strip()

                pred = eval("{" + pred_text + "}")
                action_1_touch_yx = eval(pred["touch_point"])
                action_1_lift_yx = eval(pred["lift_point"])
                action_1_action_type = action_type.ActionType[pred["action_type"]].value
                action_1_typed_text = pred["typed_text"].lower()
                action_1_typed_text = action_1_typed_text.strip()

                action_1_wrap = f'"action_type": "{action_1_action_type}", "touch_point": "{action_1_touch_yx}", "lift_point": "{action_1_lift_yx}", "typed_text": "{action_1_typed_text}"'
                action_1_wrap = action_1_wrap.replace('"', "'")
            except:
                pred = '{ "action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": "Invalid"}'
            
            action_2_touch_yx = eval(reference["touch_point"])
            action_2_lift_yx = eval(reference["lift_point"])
            action_2_action_type = action_type.ActionType[reference["action_type"]].value
            action_2_typed_text = reference["typed_text"].lower()
            
            action_2_wrap = f'"action_type": "{action_2_action_type}", "touch_point": "{action_2_touch_yx}", "lift_point": "{action_2_lift_yx}", "typed_text": "{action_2_typed_text}"'
            action_2_wrap = action_2_wrap.replace('"', "'")

            annotation_positions = reference_test_positions[idx]

            try:
                check_match = action_matching.check_actions_match(
                    action_1_touch_yx,
                    action_1_lift_yx,
                    action_1_action_type,
                    action_2_touch_yx,
                    action_2_lift_yx,
                    action_2_action_type,
                    annotation_positions
                )

            except Exception as exc:
                print(idx, action_1_touch_yx, action_1_lift_yx)
                check_match = False
                match_label = "invalid"

            if check_match:
                action_correct += 1
                match_label = 1
            else:
                match_label = 0
            if check_match and (action_1_typed_text in action_2_typed_text or action_2_typed_text in action_1_typed_text):
                text_correct += 1
            if action_1_action_type == action_2_action_type:
                type_correct += 1

            action_data = {"pred": action_1_wrap, "target": action_2_wrap, "match_label": match_label}
            output_data.append(action_data)

        metrics["accuracy"] = "{:.2f}".format(action_correct/len(targets) * 100)
        metrics["text_acc"] = "{:.2f}".format(text_correct/len(targets) * 100)
        metrics["type_acc"] = "{:.2f}".format(type_correct/len(targets) * 100)
        metrics["action_correct"] = action_correct
        metrics["text_correct"] = text_correct
        metrics["type_correct"] = type_correct
        metrics["total_num"] = len(targets)
        print(metrics)
        output_data = {
            "metrics": metrics,
            "data": output_data
        }
        print(save_dir)
        if args.eval_name:
            output_prediction_file = os.path.join(save_dir,f"predictions_ans_test_{args.eval_name}.json")
        else:
            output_prediction_file = os.path.join(save_dir,"predictions_ans_test.json")
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(output_data, indent=4))

