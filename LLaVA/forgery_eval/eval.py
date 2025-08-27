import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import json
import random
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPModel

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc  


def similarity(args, outputs, gt, device):
    clip_model = CLIPModel.from_pretrained(args.clip_path)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_path)
    clip_model.to(device)
    with torch.no_grad():
        outputs = clip_processor(text=[outputs], return_tensors="pt", padding=True, truncation=True).to(device)
        gt = clip_processor(text=[gt], return_tensors="pt", padding=True, truncation=True).to(device)
        
        outputs_features = clip_model.get_text_features(**outputs)
        gt_feature = clip_model.get_text_features(**gt)

        outputs_features = outputs_features / outputs_features.norm(dim=-1, keepdim=True)
        gt_feature = gt_feature / gt_feature.norm(dim=-1, keepdim=True)
    
    similarity = (outputs_features @ gt_feature.T).item()
    return similarity


def bleu(outputs, gt):
    outputs_tokens = outputs.split()
    gt_tokens = gt.split()
    bleu1 = sentence_bleu([gt_tokens], outputs_tokens, weights=(1, 0, 0, 0))
    return bleu1


def rouge_l(outputs, gt):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(gt, outputs)
    rouge_l_f1 = scores['rougeL'].fmeasure
    return rouge_l_f1


class ReasoningDataset(Dataset):
    def __init__(self,  img_folder, 
                        anno_path,
                        max_sample=None
                        ):

        with open(anno_path, 'r') as file:
            self.json = json.load(file)

        self.real_list = []
        self.fake_list = []
        root_path = os.path.dirname(img_folder)
        for image_name, attributes in self.json.items():
            if "0_real" in image_name:
                img_path = os.path.join(root_path, image_name)
                reasoning = " ".join(attributes[key] for key in sorted(attributes) if attributes[key] is not None)
                self.real_list.append({'img_path': img_path, 'reasoning': reasoning, 'label': 0})
            elif '1_fake' in image_name:
                img_path = os.path.join(root_path, image_name)
                reasoning = " ".join(attributes[key] for key in sorted(attributes) if attributes[key] is not None)
                self.fake_list.append({'img_path': img_path, 'reasoning': reasoning, 'label': 1})
            else:
                raise ValueError(f"not a standard image name {image_name}")

        if max_sample is not None:
            self.real_list = self.real_list[:min(max_sample, len(self.real_list))]
            self.fake_list = self.fake_list[:min(max_sample, len(self.fake_list))]
        assert len(self.real_list) == len(self.fake_list)

        self.total_list = self.real_list + self.fake_list
        random.shuffle(self.total_list)


    def __len__(self):
        return len(self.total_list)
    

    def __getitem__(self, idx):
        item = self.total_list[idx]
        img_path, gt, label = item['img_path'], item['reasoning'], item['label']
        image = Image.open(img_path).convert("RGB")
        return image, label, gt


def validate(args, model, input_ids, loader):

    y_true, y_pred = [], []
    pre_failure_count = 0

    metrics = {
        "bleu1_pre": [],
        "rougel_pre": [],
        "css_pred": [],
    }

    print ("Length of dataset: %d" %(len(loader)))
    for image, label, gt in tqdm(loader):

        image_tensor = process_images(
            [image],
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        conclusion_content = re.search(r'<CONCLUSION>(.*?)</CONCLUSION>', outputs, re.DOTALL)
        if conclusion_content:
            conclusion_content = conclusion_content.group(1).strip()
        else:
            print(f"failure: {outputs}")
            pre_failure_count = pre_failure_count + 1
            continue

        if 'This image is fake.' in conclusion_content:
            pred_label = 1
        elif 'This image is real.' in conclusion_content:
            pred_label = 0
        else:
            print(f"failure: {outputs}")
            pre_failure_count = pre_failure_count + 1
            continue
        
        reasoning_content = re.search(r'<REASONING>(.*?)</REASONING>', outputs, re.DOTALL)
        if reasoning_content:
            reasoning_content = reasoning_content.group(1).strip()
        else:
            print(f"failure: {outputs}")
            pre_failure_count = pre_failure_count + 1
            continue

        # detection metrics
        y_pred.append(pred_label)
        y_true.append(label)

        # reasoning metrics
        metrics["css_pred"].append(similarity(args, reasoning_content, gt, model.device))
        metrics["bleu1_pre"].append(bleu(reasoning_content, gt))
        metrics["rougel_pre"].append(rouge_l(reasoning_content, gt))

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    pre_failure = pre_failure_count/len(loader)

    css = np.mean(metrics["css_pred"])
    bleu1 = np.mean(metrics["bleu1_pre"])
    rougel = np.mean(metrics["rougel_pre"])

    return r_acc0, f_acc0, acc0, bleu1, rougel, css, pre_failure
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="path_to_FakeReasoning_weights")
    parser.add_argument("--model-base", type=str, default=None)

    parser.add_argument('--dataset_path', type=str, default='path_to_MMFR-Dataset')
    parser.add_argument('--result_folder', type=str, default='./results')
    parser.add_argument("--clip_path", type=str, default="path_to_clip-vit-large-patch14-336")
    parser.add_argument('--max_sample', type=int, default=10)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")  
    args = parser.parse_args()
    
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)
    
    set_seed()
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device=args.device)
    
    qs = "Is this image real or fake? Please describe the image, reasoning step-by-step and conclude with 'this image is real' or 'this image is fake'."
        
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    vals = ['stablediffusion', 'dalle3', 'if', 'midjourney', 'kandinsky',
           'pixart', 'flux', 'gpt4o', 'styleganxl', 'gigagan']
    
    for val in vals:
        print(f"Evaluating: {val}")
        img_folder = os.path.join(args.dataset_path, 'evaluation_sets', val)
        anno_path = os.path.join(img_folder, f"{val}_reasoning.json")
        dataset = ReasoningDataset(img_folder, anno_path, args.max_sample)
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0) 
        
        r_acc0, f_acc0, acc0, bleu1, rougel, css, pre_failure = validate(args, model, input_ids, loader)
        with open(os.path.join(args.result_folder, 'metric.txt'), 'a') as f:
            f.write(f"{val} {round(r_acc0*100,2)} {round(f_acc0*100,2)} {round(acc0*100,2)} {round(bleu1,2)} {round(rougel,2)} {round(css,2)}\n")


