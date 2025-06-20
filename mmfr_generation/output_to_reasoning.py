"""
annotation: extract output files to reasoning 
date:
"""

import json
import os
from output_control import fake_output_control, real_output_control
from output_control import authentic_attributes, forgery_attributes

jsonl_folder = ""
output_file = ""

# real images reasoning
json_failure_count = 0
dataset_json = {}
count = 0
for jsonl_name in os.listdir(jsonl_folder):
    file_path = os.path.join(jsonl_folder, jsonl_name)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            count += 1
            data_dict = json.loads(line)
            outputs = data_dict["response"]["body"]['choices'][0]['message']['content']
            
            # retune {} if output control failed
            answer_json = real_output_control(outputs)
            if not answer_json:
                #json_failure_case.append(outputs)
                json_failure_count = json_failure_count + 1
                continue

            forgery_reasoning = answer_json["Attributes"]
            for attribute in authentic_attributes:
                if attribute in forgery_reasoning.keys():
                    continue
                else:
                    forgery_reasoning[attribute] = None
            dataset_json[data_dict["custom_id"]] = forgery_reasoning

print(f"total: {count}")
print(f"failure: {json_failure_count}")
print(f"success: {len(dataset_json)}")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset_json, f, ensure_ascii=False, indent=4)


# fake images reasoning
json_failure_count = 0
dataset_json = {}
count = 0
for jsonl_name in os.listdir(jsonl_folder):
    file_path = os.path.join(jsonl_folder, jsonl_name)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            count += 1
            data_dict = json.loads(line)
            outputs = data_dict["response"]["body"]['choices'][0]['message']['content']
            
            # retune {} if output control failed
            answer_json = fake_output_control(outputs)
            if not answer_json:
                #json_failure_case.append(outputs)
                json_failure_count = json_failure_count + 1
                continue

            forgery_reasoning = answer_json["Forgery Attributes"]
            dataset_json[data_dict["custom_id"]] = forgery_reasoning

print(f"total: {count}")
print(f"failure: {json_failure_count}")
print(f"success: {len(dataset_json)}")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset_json, f, ensure_ascii=False, indent=4)

