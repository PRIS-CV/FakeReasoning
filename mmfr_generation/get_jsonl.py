"""
annotation: generate jsonl requests for batch api
date:
"""

import json
import os
import base64
import random
from tqdm import tqdm
from prompt import qs_real

folder_path = ""
target_root:str = ""
save_format:str = 'batchinput-{:0>5d}.jsonl' # batchinput-00000.jsonl

tol = 3000 
images_dict = {}

files = os.listdir(folder_path)[:tol]
random.shuffle(files)
for img_name in tqdm(files):
    img_path = os.path.join(folder_path,img_name)
    with open(img_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        images_dict[img_path] = encoded_string

if not os.path.exists(target_root):
    os.mkdir(target_root)

# since openai has the upload limit of 100M, organize N images in one jsonl, depending on image resolution
N:int = 50
begin_id:int = 0
total_jsonl = []
new_dicts = []

counter:int = 0
for image_name, base64_image in tqdm(images_dict.items()):
    filename = os.path.basename(image_name)
    custom_id = f"stylegan2/0_real/{filename}"

    new_dict = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-2024-08-06",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": qs_real},  
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            "max_tokens": 300
        }
    }
    new_dicts.append(new_dict)
    counter += 1
    if counter == N:
        total_jsonl.append(new_dicts)
        counter = 0
        new_dicts = []

if len(new_dicts) != 0:
    total_jsonl.append(new_dicts)

for idx, data in tqdm(enumerate(total_jsonl)):
    with open(os.path.join(target_root, save_format.format(idx+begin_id)), 'w') as jsonl_file:
        for item in data:
            jsonl_file.write(json.dumps(item) + '\n')

