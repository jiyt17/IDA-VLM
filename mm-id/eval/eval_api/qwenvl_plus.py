# export DASHSCOPE_API_KEY=xxxxxxxx

from http import HTTPStatus
import dashscope
from tqdm import tqdm
import re
import time
import json

res = []
new_test_set = json.load(open('/mnt/bn/automl-aigc/yatai/data/benchmark/mm-id/benchmark_new_matchoi.json', 'r'))

for id in tqdm(range(len(new_test_set))):
    # if id < res_len:
    #     continue
    test_sample = new_test_set[id]
    question = test_sample['question']
    question = re.split(r'<img>|</img>\n', question)

    if test_sample['type'] != 'matching':
        continue

    for i in range(len(question)):
        if question[i] in test_sample:
            img_path = test_sample[question[i]]
            question[i] = {"image": "file://"+img_path}
        else:
            question[i] = {'text': question[i]}
    
    # if test_sample['type'] == 'location':
    #     question[-1] = {"text": f"Ground {test_sample['ground_target']} with coordinates."}
    error_time = 0
    if question[-1]['text'] == '.':
        question = question[:-1]
    print(question)
    while True:
        try:
            response = dashscope.MultiModalConversation.call(
                model='qwen-vl-max', 
                messages=[ 
                    { "role": "system", 
                        "content": [
                            # {"text":"You need to give coordinates of bounding box of one given character or object. The answer form should only be '<ref>xxx</ref><box>(x1,y1),(x2,y2)</box>.', where x1 is left side of bounding box, y1 is upper side, x2 is right side, y2 is bottom side, they are all integers."}
                            {"text":"You are a helpful and precise assistant for providing a answer to the question. You need recognize instance identity to answer questions. Your answer must be 'A' or 'B' or 'C' or 'D'. "}
                            # {"text":"You are a helpful and precise assistant for providing a answer to the question. You need recognize instance identity to answer questions about reference characters or give a caption with character names for continuous images."}
                            ]
                    },
                    { "role": "user", 
                        "content": question
                    } 
                ] 
            ) 
            qwen_result = response.output.choices[0].message.content
            error_time = 0
            break
        except Exception as e:

            error_time += 1
            print(e)
            time.sleep(1)
            if error_time > 20:
                qwen_result = ""
                print(id)
                break
            continue

    res.append({'id': test_sample['id'], 'question': test_sample['question'], 'prediction': qwen_result})
    with open('qwenvl_max_pred_match.json', 'w') as f:
        json.dump(res, f, indent=4)

