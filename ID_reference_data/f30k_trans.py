import json
from PIL import Image
import random
import os
import re
from tqdm import tqdm

def crop_and_save_image(input_path, output_path, box):
    # 打开图片
    image = Image.open(input_path)

    # 裁剪图像
    cropped_image = image.crop(box)

    # 增强
    if random.random() > 0.5:
        cropped_image = cropped_image.transpose(Image.FLIP_LEFT_RIGHT)

    # 保存裁剪后的图像
    cropped_image.save(output_path)

train_anno = []
with open('/mnt/bn/automl-aigc/yatai/data/shikra/CWB_flickr30k_train.jsonl', 'r') as f:
    for line in f:
        train_anno.append(json.loads(line))
print(len(train_anno))

template = json.load(open('./flickr30k.json', 'r'))

res = []
id = 0
pattern = re.compile(r"<ph_st>(.*?)<ph_ed>")
key_words = ['people', ' man', ' men', 'woman', 'women', 'boy', 'girl', 'person', 'child', 'lady', 'guy', 'someone', 'kid', 'female', ' male', 'player', 'audience', 'adult', 'biker', 'mom', 'mother']
names = [
    'James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph', 'Charles', 'Thomas',
    'Christopher', 'Daniel', 'Matthew', 'Anthony', 'Mark', 'Donald', 'Steven', 'Paul', 'Andrew', 'Kenneth',
    'George', 'Joshua', 'Brian', 'Edward', 'Kevin', 'Ronald', 'Timothy', 'Jason', 'Jeffrey', 'Ryan',
    'Gary', 'Nicholas', 'Eric', 'Stephen', 'Larry', 'Justin', 'Scott', 'Brandon', 'Benjamin', 'Samuel',
    'Frank', 'Gregory', 'Raymond', 'Alexander', 'Patrick', 'Jack', 'Dennis', 'Jerry', 'Tyler', 'Aaron',
    'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen',
    'Nancy', 'Lisa', 'Betty', 'Margaret', 'Sandra', 'Ashley', 'Kimberly', 'Emily', 'Dorothy', 'Helen',
    'Amanda', 'Melissa', 'Deborah', 'Jessica', 'Laura', 'Cynthia', 'Angela', 'Ruth', 'Sharon', 'Michelle',
    'Anna', 'Carolyn', 'Virginia', 'Samantha', 'Elizabeth', 'Nicole', 'Heather', 'Diane', 'Joyce', 'Sharon',
    'Amber', 'Megan', 'Natalie', 'Grace', 'Tiffany', 'Victoria', 'Tracy', 'Christine', 'Rebecca', 'Rachel'
]

for instance in tqdm(train_anno):
    caption = instance['sentence']
    if len(caption.split()) < 10:
        continue
    matches = pattern.findall(caption)
    people_id = {}
    for i, match in enumerate(matches):
        m = match.lower()
        if any(k in m for k in key_words):
            box_ids = instance['boxes_seq'][i]
            for b_id in box_ids:
                people_id[b_id] = 1
                
    if len(people_id) > 5:
        continue
    people_name = random.sample(names, len(people_id))
    for i,k in enumerate(people_id.keys()):
        people_id[k] = people_name[i]
        
    match_dict = {}
    for i, match in enumerate(matches):
        m = match.lower()
        if any(k in m for k in key_words):
            box_ids = instance['boxes_seq'][i]
            name = ''
            for j,b_id in enumerate(box_ids):
                if len(box_ids) > 1 and j == len(box_ids)-2:
                    name = name + people_id[b_id] + ' and '
                else:
                    name = name + people_id[b_id] + ', '
            name = name[:-2]
            match_dict['<ph_st>'+match+'<ph_ed>'] = name
        else:
            match_dict['<ph_st>'+match+'<ph_ed>'] = match
            
    source_img = '/mnt/bn/automl-aigc/yatai/data/f30k/flickr30k-images/' + instance['image_id'] + '.jpg'
    if len(people_id) == 0:
        question = ''
    else:
        question = 'Given some people characters, '
    flag = 1
    for k,v in people_id.items():
        os.makedirs('/mnt/bn/automl-aigc/yatai/data/shikra/cwb_f30k_imgs/'+str(instance['id']), exist_ok=True)
        people_img_path = '/mnt/bn/automl-aigc/yatai/data/shikra/cwb_f30k_imgs/' + str(instance['id']) + '/' + v + '.jpg'
        question = question + v + ' is <img>' + people_img_path + '</img>\n, ' 
        box = instance['boxes'][k]
        if (box[3] - box[1]) * (box[2]- box[0]) < 3000:
            flag = 0
            break
        crop_and_save_image(source_img, people_img_path, box)
    if flag == 0:
        continue
    
    question = question[:-2] + '. '
    question = question + random.sample(template, 1)[0]
    question = question.replace('<image>', '<img>'+source_img+'</img>\n')

    answer = caption
    for org_m, tar_m in match_dict.items():
        answer = answer.replace(org_m, tar_m)

    conversation = [{"from": "user", "value": question},
                    {"from": "assistant", "value": answer},
                    ]
    
    res.append({"id":id, "conversations": conversation})
    id += 1

with open('qwen_f30k_train.json', 'w') as f:
    json.dump(res, f, indent=4)