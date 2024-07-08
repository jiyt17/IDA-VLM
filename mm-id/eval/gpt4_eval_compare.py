      
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
                'content': "We would like to request your feedback on the performance of two AI assistants in response to the user question according to the given ground truth. The question, ground truth answer and predictions of two AI assistants will be signed by 'question', 'GT', 'prediction 1' and 'prediction 2'.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. Please pay more attention to the correspondence of character names and their states or actions.\nPlease first output a single line containing only two values indicating the scores for Prediction 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
                # 'content': "You are a helpful and precise assistant for evaluating answers. We would like to request your feedback on the quality of an AI assistant's answer according to the given question and ground truth. The question, answer of AI assistant and ground truth will be signed by 'question', 'prediction' and 'GT'. You need to judge whether the overall meanings of prediction and ground truth answer are consistent or not. Please pay more attention to the correspondence of character names and their states or actions. \nPlease rate the helpfulness, relevance, accuracy of the responses. You should give an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only one value indicating the score for Assistant answer.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
            }, {
                'role': 'user',
                'content': content,
            }],
        ) 
        return completion.choices[0].message.content


if __name__ == '__main__':

    gpt_handler = GPT4(api_version="gpt-4-1106-preview", 
        api_key= 'xxxxxxx', 
        model="gpt-4-1106-preview")

    gt = json.load(open('benchmark_mmid.json', 'r'))
    pre1 = json.load(open('/mnt/bn/automl-aigc/yatai/Qwen-VL/result/qwen_beta_match_full_idadapter3_llava_mix_mini/checkpoint-1500/prediction.json', 'r'))
    pre2_list = [
        '/mnt/bn/automl-aigc/yatai/data/benchmark/mm-id/eval_models/qwenvl_max_pred.json',
        '/mnt/bn/automl-aigc/yatai/data/benchmark/mm-id/eval_models/qwenvl_plus_pred.json'
    ]
    save_list = [
        'review_results/review_ours_vs_qwenvlmax.json',
        'review_results/review_ours_vs_qwenvlplus.json'
    ]
    for index in range(len(pre2_list)):
        pre2 = json.load(open(pre2_list[index], 'r'))
        pre_new1 = {}
        for p in pre1:
            pre_new1[p['id']] = p['prediction']
        pre_new2 = {}
        for p in pre2:
            pre_new2[p['id']] = p['prediction']
        prediction = {}
        for i in range(len(gt)):
            if gt[i]['type'] == 'caption':
                caption = pre_new2[gt[i]['id']]
                caption_new = ""
                for cap in caption:
                    for k,v in cap.items():
                        if k == 'box':
                            caption_new += v[v.find('<ref>')+5: v.find('</ref>')]
                        elif k == 'text':
                            caption_new += v
                prediction[gt[i]['id']] = ['Give an image description.', gt[i]['answer'], pre_new1[gt[i]['id']], caption_new]
            elif gt[i]['type'] == 'QA':
                ques = gt[i]['question'].split('\n')[-1]
                caption = pre_new2[gt[i]['id']]
                caption_new = ""
                for cap in caption:
                    for k,v in cap.items():
                        if k == 'box':
                            caption_new += v[v.find('<ref>')+5: v.find('</ref>')]
                        elif k == 'text':
                            caption_new += v
                prediction[gt[i]['id']] = [ques, gt[i]['answer'], pre_new1[gt[i]['id']], caption_new]
            
        
        res = []
        for test_id in tqdm(prediction.keys()):
            content = f"Question: {prediction[test_id][0]}\n GT: {prediction[test_id][1]}\n Prediction 1: {prediction[test_id][2]}\n Prediction 2: {prediction[test_id][3]}"

            while True:
                try:
                    review = gpt_handler.message(content=content) 
                    break
                except Exception as e:
                    print(e)
                    time.sleep(5)
                    continue
            print(review)
            res.append({'id':test_id, 'GT':prediction[test_id][1], 'prediction 1':prediction[test_id][2], 'prediction 2':prediction[test_id][3], 'score':review.strip().split('\n')[0], 'review':review})
            
            with open(save_list[index], 'w') as f:
                json.dump(res, f, indent=4)

    # calculate review scores
    review = json.load(open('review_results/review_ours_vs_qwenvlmax.json', 'r'))
    gt = json.load(open('benchmark_mmid.json', 'r'))
    caption_scores1 = []
    qa_scores1 = []
    caption_scores2 = []
    qa_scores2 = []
    for sample in review:
        # print(sample['id'])
        t = gt[sample['id']-1]['type']
        score = sample['score'].strip().split()
        if t == 'caption':
            caption_scores1.append(int(score[0]))
            caption_scores2.append(int(score[1]))
        elif t == 'QA':
            qa_scores1.append(int(score[0]))
            qa_scores2.append(int(score[1]))
    print('prediction1:')
    print('caption:', len(caption_scores1), np.mean(caption_scores1))
    print('qa:', len(qa_scores1), np.mean(qa_scores1))
    print('prediction2:')
    print('caption:', len(caption_scores2), np.mean(caption_scores2))
    print('qa:', len(qa_scores2), np.mean(qa_scores2))

        
