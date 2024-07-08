from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from peft import PeftModel
import json
torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
# tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/automl-aigc/yatai/Qwen-VL/weight/Qwen-VL-Chat", trust_remote_code=True)

model_path = "/mnt/bn/automl-aigc/yatai/Qwen-VL/result/qwen_beta_idadapter_full_llavashare_mini_mini_1/checkpoint-1000"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
# model = AutoModelForCausalLM.from_pretrained("/mnt/bn/automl-aigc/yatai/Qwen-VL/weight/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()
# /mnt/bn/automl-aigc/yatai/Qwen-VL/weight/Qwen-VL-Chat/modeling_qwen.py

# lora
# print('Loading LoRA weights...')
# model = PeftModel.from_pretrained(model, model_path)
# print('Merging LoRA weights...')
# model = model.merge_and_unload()
# print('Model is loaded...')

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
# model.generation_config = GenerationConfig.from_pretrained("/mnt/bn/automl-aigc/yatai/Qwen-VL/weight/Qwen-VL-Chat", trust_remote_code=True)

res = []
new_test_set = json.load(open('benchmark_mmid.json', 'r'))
for id in range(len(new_test_set)):
    test_sample = new_test_set[id]
    
    question = test_sample['question']
    img_num = 0
    for k in test_sample.keys():
        if k.startswith('image'):
            img_num += 1
    if test_sample['id'] < 360:
        for i in range(img_num):
            img_id = f'image_{i+1}'
            question = question.replace(img_id, test_sample[img_id])
    else:
        for i in range(img_num):
            img_id = f'image_{i}'
            question = question.replace(img_id, test_sample[img_id])
    
    response, history = model.chat(tokenizer, query=question, history=None)
    print('user:', question)
    print('assistant: ', response)
    print()
    res.append({'id': test_sample['id'], 'question': test_sample['question'], 'prediction': response})
    if '<box>' in response:
        try:
            image = tokenizer.draw_bbox_on_latest_picture(response, history)
            img_name = response.split('ref')[1][1:-2]
            image.save(f'./grounded_imgs/{img_name}.jpg')
        except:
            print(test_sample['id'])

    
with open(model_path + '/prediction.json', 'w') as f:
    json.dump(res, f, indent=4)
