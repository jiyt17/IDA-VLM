      
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
            azure_endpoint="https://search.bytedance.net/gpt/openapi/online/v2/crawl", 
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

    # gpt_handler = GPT4(api_version="gpt-4-1106-preview", 
    #     api_key= 'xxxxxxx', 
    #     model="gpt-4-1106-preview")

    # gt = json.load(open('/mnt/bn/automl-aigc/yatai/data/benchmark/mm-id/benchmark_mi.json', 'r'))
    # pre = json.load(open('/mnt/bn/automl-aigc/yatai/data/benchmark/mm-id/eval_models/qwenvl_max_pred_mi.json', 'r'))
    # pre_new = {}
    # for p in pre:
    #     pre_new[p['id']] = p['prediction']
    # prediction = {}
    # for i in range(len(gt)):
    #     if gt[i]['type'] == 'caption':
    #         caption = pre_new[gt[i]['id']]
    #         caption_new = ""
    #         for cap in caption:
    #             for k,v in cap.items():
    #                 if k == 'box':
    #                     caption_new += v[v.find('<ref>')+5: v.find('</ref>')]
    #                 elif k == 'text':
    #                     caption_new += v
    #         prediction[gt[i]['id']] = ['Give an image description.', caption_new, gt[i]['answer']]
    #     elif gt[i]['type'] == 'QA':
    #         ques = gt[i]['question'].split('\n')[-1]
    #         caption = pre_new[gt[i]['id']]
    #         caption_new = ""
    #         for cap in caption:
    #             for k,v in cap.items():
    #                 if k == 'box':
    #                     caption_new += v[v.find('<ref>')+5: v.find('</ref>')]
    #                 elif k == 'text':
    #                     caption_new += v
    #         prediction[gt[i]['id']] = [ques, caption_new, gt[i]['answer']]
    #     # elif gt[i]['type'] == 'matching':
    #     #     ques = f"Which is {gt[i]['reference_target']}"
    #     #     prediction[gt[i]['id']] = [ques, pre_new[gt[i]['id']], gt[i]['answer']]
    
    # res = []
    # for test_id in tqdm(prediction.keys()):
    #     # test_id = 314
    #     content = f"Question: {prediction[test_id][0]} Prediction: {prediction[test_id][1]} GT: {prediction[test_id][2]}"
    #     # content = "Prediction: Barbara is looking down at food. GT: She is looking down and cutting food with knife or something."

    #     while True:
    #         try:
    #             review = gpt_handler.message(content=content) 
    #             break
    #         except Exception as e:
    #             print(e)
    #             time.sleep(5)
    #             continue
        
    #     res.append({'id':test_id, 'prediction':prediction[test_id][1], 'GT':prediction[test_id][2], 'score':review.split('\n')[0], 'review':review})
        
    #     with open('./review_results/review_qwenvl_max_mi.json', 'w') as f:
    #         json.dump(res, f, indent=4)

    # calculate review scores
    # review = json.load(open('./review_results/review_qwenvl_max_mi.json', 'r'))
    # gt = json.load(open('/mnt/bn/automl-aigc/yatai/data/benchmark/mm-id/benchmark_mi.json', 'r'))
    # caption_scores = []
    # qa_scores = []
    # for sample in review:
    #     t = gt[sample['id']-1]['type']
    #     s = sample['score']
    #     if '/' in s:
    #         assert s[-2:] == '10'
    #         s = s[0]
    #     if t == 'caption':
    #         caption_scores.append(int(s))
    #     elif t == 'QA':
    #         qa_scores.append(int(s))
    # print('caption:', len(caption_scores), np.mean(caption_scores))
    # print('qa:', len(qa_scores), np.mean(qa_scores))
        
    # location
    # wrong_sample = 0
    # gt = json.load(open('/mnt/bn/automl-aigc/yatai/data/benchmark/mm-id/benchmark_new.json', 'r'))
    # pre = json.load(open('/mnt/bn/automl-aigc/yatai/data/benchmark/mm-id/eval_models/qwenvl_plus_pred_loc.json', 'r'))
    # pre_new = {}
    # for p in pre:
    #     pre_new[p['id']] = p['prediction']
    # location_scores = []
    # for i in range(426):
    #     if gt[i]['type'] == 'location':
    #         gt_loc = gt[i]['answer'][1:-1].split(',')
    #         gt_loc = [int(j) for j in gt_loc]
    #         target = gt[i]['ground_target']

    #         pre_loc = 0
    #         # max / plus
    #         for text in pre_new[gt[i]['id']]:
    #             key = list(text.keys())[0]
    #             if key == 'box':
    #                 tar = text['box'][text['box'].find('<ref>')+5: text['box'].find('</ref>')]
    #                 if tar == target:
    #                     pre_loc = text['box'].split('box')[1][2:-3]
    #                     pre_loc = pre_loc.replace('(','').replace(')','').split(',')
    #                     pre_loc = [int(j) for j in pre_loc]
    #                     img_num = len(gt[i]) - 5
    #                     img_path = gt[i][f'image_{img_num}']
    #                     image = Image.open(img_path)
    #                     h = image.size[1]
    #                     w = image.size[0]
    #                     pre_loc = [int(w * pre_loc[0]/1000), int(h * pre_loc[1]/1000), int(w * pre_loc[2]/1000), int(h * pre_loc[3]/1000)]
    #                     break
    #         if pre_loc == 0:
    #             for text in pre_new[gt[i]['id']]:
    #                 key = list(text.keys())[0]
    #                 if key == 'box':
    #                     tar = text['box'][text['box'].find('<ref>')+5: text['box'].find('</ref>')]
    #                     if target in tar:
    #                         pre_loc = text['box'].split('box')[1][2:-3]
    #                         pre_loc = pre_loc.replace('(','').replace(')','').split(',')
    #                         pre_loc = [int(j) for j in pre_loc]
    #                         img_num = len(gt[i]) - 5
    #                         img_path = gt[i][f'image_{img_num}']
    #                         image = Image.open(img_path)
    #                         h = image.size[1]
    #                         w = image.size[0]
    #                         pre_loc = [int(w * pre_loc[0]/1000), int(h * pre_loc[1]/1000), int(w * pre_loc[2]/1000), int(h * pre_loc[3]/1000)]
    #                         break
    #         # chat
    #         # if pre_new[gt[i]['id']].count('<box>') >= 1:
    #         #     pre_loc = pre_new[gt[i]['id']].split('box')[1][2:-3]
    #         #     pre_loc = pre_loc.replace('(','').replace(')','').split(',')
    #         #     pre_loc = [int(j) for j in pre_loc]
    #         #     img_num = len(gt[i]) - 5
    #         #     img_path = gt[i][f'image_{img_num}']
    #         #     image = Image.open(img_path)
    #         #     h = image.size[1]
    #         #     w = image.size[0]
    #         #     pre_loc = [int(w * pre_loc[0]/1000), int(h * pre_loc[1]/1000), int(w * pre_loc[2]/1000), int(h * pre_loc[3]/1000)]
    #         if pre_loc == 0:
    #             location_scores.append(0)
    #             wrong_sample += 1
    #             print(gt[i]['id'])
    #         else:
    #             location_scores.append(calculate_iou(gt_loc, pre_loc))
    # acc_num = 0
    # for score in location_scores:
    #     if score > 0.5:
    #         acc_num += 1
    # print(len(location_scores), wrong_sample, np.mean(location_scores), acc_num/len(location_scores))

    # matching
    gt = json.load(open('/mnt/bn/automl-aigc/yatai/data/benchmark/mm-id/benchmark_new_matchoi.json', 'r'))
    pre = json.load(open('/mnt/bn/automl-aigc/yatai/data/benchmark/mm-id/eval_models/qwenvl_max_pred_match.json', 'r'))
    pre_new = {}
    for p in pre:
        pre_new[p['id']] = p['prediction']
    matching_scores = []
    for i in range(426):
        if gt[i]['type'] == 'matching':
            gt_ans = gt[i]['answer']
            # pre_ans = pre[i]['prediction'][-1]
            # assert pre_ans in ['1', '2', '3', '4']
            # if gt_ans == pre_ans:
            #     matching_scores.append(1)
            # else:
            #     print(gt[i]['id'])
            #     matching_scores.append(0)
            if len(pre_new[gt[i]['id']]) > 0:
                prediction = pre_new[gt[i]['id']][0]['text']
            else:
                prediction = ''
            # pre_ans = '0'
            # if '1' in prediction or 'first' in prediction:
            #     pre_ans = '1'
            # elif '2' in prediction or 'second' in prediction:
            #     pre_ans = '2'
            # elif '3' in prediction or 'third' in prediction:
            #     pre_ans = '3'
            # elif '4' in prediction or 'fourth' in prediction:
            #     pre_ans = '4'
            # else:
            #     print(gt[i]['id'])
            if gt_ans in prediction:
                matching_scores.append(1)
            else:
                matching_scores.append(0)
    print('matching', len(matching_scores), np.mean(matching_scores))
