      
import cv2
import base64 
import openai
import json
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import os

class GPT4:
    def __init__(self, api_version, api_key, model): 
        super().__init__ () 
        self.model = model 
        self.client = openai.AzureOpenAI( 
            azure_endpoint="xxxxxxxxx", 
            api_version=api_version, 
            api_key=api_key) 
    def message(self, content): 
        completion = self.client.chat.completions.create(
            model=self.model, 
            messages=[{
                'role': 'system',
                'content': "You are a helpful and precise assistant for evaluating answers. We would like to request your feedback on the quality of an AI assistant's answer according to the given question and ground truth. The question, answer of AI assistant and ground truth will be signed by 'question', 'prediction' and 'GT'. You need to judge whether the overall meanings of prediction and ground truth answer are consistent or not. Please pay more attention to the correspondence of character names and their states or actions. \nPlease rate the helpfulness, relevance, accuracy of the responses. You should give an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only one value indicating the score for Assistant answer.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
            }, {
                'role': 'user',
                'content': content,
            }],
        ) 
        return completion.choices[0].message.content

def calculate_iou(box1, box2):
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    x1_intersection = max(x1_box1, x1_box2)
    y1_intersection = max(y1_box1, y1_box2)
    x2_intersection = min(x2_box1, x2_box2)
    y2_intersection = min(y2_box1, y2_box2)

    intersection_width = max(0, x2_intersection - x1_intersection)
    intersection_height = max(0, y2_intersection - y1_intersection)
    intersection_area = intersection_width * intersection_height

    box1_area = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    box2_area = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou

if __name__ == '__main__':

    gpt_handler = GPT4(api_version="gpt-4-1106-preview", 
        api_key= 'xxxxxxxxx', 
        model="gpt-4-1106-preview")

    pre_list = [
        "/mnt/bn/automl-aigc/yatai/Qwen-VL/result/qwen_beta_idadapter_full_llavashare_mini_2/checkpoint-3000/prediction.json",
    ]
    save_list = [
        './review_results/review_beta_idad_llavashare_mix_mini2_3000s.json',
    ]
    gt_list = [
        './benchmark/benchmark_mmid.json',
    ]
    for index in range(len(pre_list)):
        gt = json.load(open(gt_list[index], 'r'))
        pre = json.load(open(pre_list[index], 'r'))
        pre_new = {}
        for p in pre:
            pre_new[p['id']] = p['prediction']
        prediction = {}
        for i in range(len(gt)):
            if gt[i]['type'] == 'caption':
                prediction[gt[i]['id']] = ['Give an image description.', pre_new[gt[i]['id']], gt[i]['answer']]
            elif gt[i]['type'] == 'QA':
                ques = gt[i]['question'].split('\n')[-1]
                prediction[gt[i]['id']] = [ques, pre_new[gt[i]['id']], gt[i]['answer']]
        
        res = []
        for test_id in tqdm(prediction.keys()):
            content = f"Question: {prediction[test_id][0]} Prediction: {prediction[test_id][1]} GT: {prediction[test_id][2]}"

            while True:
                try:
                    review = gpt_handler.message(content=content) 
                    break
                except Exception as e:
                    print(e)
                    time.sleep(5)
                    continue
            
            res.append({'id':test_id, 'prediction':prediction[test_id][1], 'GT':prediction[test_id][2], 'score':review.strip().split('\n')[0], 'review':review})
            
            with open(save_list[index], 'w') as f:
                json.dump(res, f, indent=4)

    # calculate review scores
    review = json.load(open('./review_results/review_beta_idad_llavashare_mix_mini2_3000s.json', 'r'))
    gt = json.load(open('./benchmark/benchmark_mmid.json', 'r'))
    caption_scores = []
    qa_scores = []
    for sample in review:
        # print(sample['id'])
        t = gt[sample['id']-1]['type']
        s = sample['score']
        if '/' in s:
            assert s[-2:] == '10'
            s = s[0]
        if t == 'caption':
            caption_scores.append(int(s))
        elif t == 'QA':
            qa_scores.append(int(s))
    print('caption:', len(caption_scores), np.mean(caption_scores))
    print('qa:', len(qa_scores), np.mean(qa_scores))
        
    # location
    gt = json.load(open('./benchmark/benchmark_mmid.json', 'r'))
    pre = json.load(open('/mnt/bn/automl-aigc/yatai/Qwen-VL/result/qwen_beta_idadapter_full_llavashare_mini_2/checkpoint-3000/prediction.json', 'r'))
    pre_new = {}
    for p in pre:
        pre_new[p['id']] = p['prediction']
    location_scores = []
    for i in range(426):
        if gt[i]['type'] == 'location':
            gt_loc = gt[i]['answer'][1:-1].split(',')
            gt_loc = [int(j) for j in gt_loc]

            # ours
            pre_loc = pre_new[gt[i]['id']].split('box')[1][2:-3]
            pre_loc = pre_loc.replace('(','').replace(')','').split(',')
            pre_loc = [int(j) for j in pre_loc]
            img_num = len(gt[i]) - 5
            img_path = gt[i][f'image_{img_num}']
            image = Image.open(img_path)
            h = image.size[1]
            w = image.size[0]
            pre_loc = [int(w * pre_loc[0]/1000), int(h * pre_loc[1]/1000), int(w * pre_loc[2]/1000), int(h * pre_loc[3]/1000)]
            # gpt4v
            # try:
            #     pre_loc = pre_new[gt[i]['id']]
            #     left_s = pre_loc.find('[')
            #     right_s = pre_loc.find(']')
            #     pre_loc = pre_loc[left_s+1:right_s].split(',')
            #     pre_loc = [int(loc) for loc in pre_loc]
            #     print(pre_loc)
            # except:
            #     print(gt[i]['id'])
            #     location_scores.append(0)
            #     continue

            location_scores.append(calculate_iou(gt_loc, pre_loc))
    acc_num = 0
    for score in location_scores:
        if score > 0.5:
            acc_num += 1
    print(len(location_scores), np.mean(location_scores), acc_num/len(location_scores))

    # matching
    pre_new = {}
    for p in pre:
        pre_new[p['id']] = p['prediction']
    matching_scores = []
    for i in range(426):
        if gt[i]['type'] == 'matching':
            gt_ans = gt[i]['answer'][-1]
            # pre_ans = pre[i]['prediction'][-1]
            # assert pre_ans in ['1', '2', '3', '4']
            prediction = pre_new[gt[i]['id']]

            pre_ans = '0'
            if '1' in prediction or 'first' in prediction:
                pre_ans = '1'
            elif '2' in prediction or 'second' in prediction:
                pre_ans = '2'
            elif '3' in prediction or 'third' in prediction:
                pre_ans = '3'
            elif '4' in prediction or 'fourth' in prediction:
                pre_ans = '4'
            else:
                print(gt[i]['id'])
            # assert prediction in ['A', 'B', 'C', 'D']
            if gt_ans == pre_ans:
            # if gt_ans == prediction:
                matching_scores.append(1)
            else:
                matching_scores.append(0)
    print('matching', len(matching_scores), np.mean(matching_scores))
