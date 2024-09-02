import cv2
import base64 
import openai
import json
import time
import re
import os
from tqdm import tqdm

res = []
# res = json.load(open('/mnt/bn/automl-aigc/yatai/data/benchmark/mm-id/eval_models/gpt4v_pred_mi.json', 'r'))
new_test_set = json.load(open('/mnt/bn/automl-aigc/yatai/data/benchmark/mm-id/benchmark_mi.json', 'r'))
# new_test_set = new_test_set[118:]

model = "gptv"
client = openai.AzureOpenAI( 
    azure_endpoint="https:xxxxxxxxx", 
    api_version="gpt-4-vision", 
    api_key="xxxxxxxx") 


for id in tqdm(range(len(new_test_set))):
    test_sample = new_test_set[id]
    if test_sample['type'] == 'location':
        continue
    question = test_sample['question']
    question = re.split(r'<img>|</img>\n', question)

    for i in range(len(question)):
        if question[i] in test_sample:
            img_path = test_sample[question[i]]
            img = cv2.imread(img_path)
            success, encoded_image = cv2.imencode('.jpg', img)
            content = encoded_image.tobytes() 
            base64_image = base64.b64encode(content) 
            base64_string = base64_image.decode('utf-8')
            question[i] = {"image": base64_string}
    # img_size = img.shape[:2]
    # question[-1] = f"(the height of the image is {img_size[0]}, the width is {img_size[1]}) Ground {test_sample['ground_target']}"
    # print(question)
    while True:
        try:
            response = client.chat.completions.create(
                model=model, 
                messages=[ 
                    { "role": "system", 
                        "content": [
                            # "You need to give coordinates of bounding box of some given characters or objects. The answer form should be 'bbox: [x1, y1, x2, y2].', where x1 is left side of bounding box, y1 is upper side, x2 is right side, y2 is bottom side, they are all integers."
                            # "You are a helpful and precise assistant for providing a answer to the question. You need recognize instance identity to answer questions about reference characters or give a caption with character names. You must provide an exact answer."
                            "You are a helpful and precise assistant for providing a answer to the question. You need recognize instance identity to answer questions about reference characters or give a caption with character names for multiple continuous images. You must provide an exact answer."
                            ]
                    },
                    { "role": "user", 
                        "content": question
                    } 
                ] 
            ) 
            gpt_result = response.choices[0].message.content
            break
        except Exception as e:
            print(e)
            if 'Error code: 429' in str(e):
                time.sleep(5)
                continue
            else:
                print(test_sample['id'])
                break

    res.append({'id': test_sample['id'], 'question': test_sample['question'], 'prediction': gpt_result})
    with open('gpt4v_pred_mi_new.json', 'w') as f:
        json.dump(res, f, indent=4)
    
    time.sleep(80)
