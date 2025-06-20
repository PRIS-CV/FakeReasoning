"""
annotation: output control
date:
"""

forgery_attributes = {
    "Inconsistent Lighting and Shadows",
    "Irregular Image Edges",
    "Anomalous Image Texture",
    "Low Image Quality",
    "Semantic Irrationality"
}

authentic_attributes = {
    "Consistent Lighting and Shadows",
    "Smooth and Natural Image Edges",
    "Natural and Consistent Image Texture",
    "High Image Quality",
    "Semantic Rationality"
}


def fake_output_control(outputs):
    try:
        start_index = outputs.find('{')
    except Exception as e:
        return {}

    if start_index == -1:
        return {}

    endmark_index = outputs[::-1].find('}')
    end_index = len(outputs) - endmark_index
    standard_outputs = outputs[start_index:end_index]

    try:
        answer_json = eval(standard_outputs)
    except Exception as e:
        return {}

    if "Answer" not in answer_json.keys() or "Forgery Attributes" not in answer_json.keys():
        return {}
    
    if 'fake' not in answer_json['Answer']:
        return {}
    
    if not isinstance(answer_json.get("Forgery Attributes"), dict):
        return {}
    
    forgery_reasoning = {
        key: value
        for key, value in answer_json["Forgery Attributes"].items()
        if key in forgery_attributes
    }
    answer_json["Forgery Attributes"] = forgery_reasoning

    exist_attributes = answer_json["Forgery Attributes"].keys()
    for attribute in forgery_attributes:
        if attribute not in exist_attributes:
            answer_json["Forgery Attributes"][attribute] = None

    return answer_json


def real_output_control(outputs):
    try:
        start_index = outputs.find('{')
    except Exception as e:
        return {}

    if start_index == -1:
        return {}

    endmark_index = outputs[::-1].find('}')
    end_index = len(outputs) - endmark_index
    standard_outputs = outputs[start_index:end_index]

    try:
        answer_json = eval(standard_outputs)
    except Exception as e:
        return {}

    if "Answer" not in answer_json.keys() or "Attributes" not in answer_json.keys():
        return {}
    
    if 'real' not in answer_json['Answer']:
        #print(answer_json['Answer'])
        return {}
    
    # # 不存在的伪造属性设置为None
    # exist_attributes = answer_json["Forgery Attributes"].keys()
    # for attribute in forgery_attributes:
    #     if attribute not in exist_attributes:
    #         answer_json["Forgery Attributes"][attribute] = None

    return answer_json