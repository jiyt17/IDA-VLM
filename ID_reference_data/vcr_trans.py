import json
from PIL import Image
import random
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
with open('/mnt/bn/automl-aigc/yatai/data/vcr/train.jsonl', 'r') as f:
    for line in f:
        train_anno.append(json.loads(line))
# print(len(train_anno))

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

res = []
id = 0

for ins_id, instance in tqdm(enumerate(train_anno)):
    try:
        ques_org = instance['question_orig']
        ans_org = instance['answer_orig']
        rat_org = instance['rationale_orig']
        objects = instance['objects']
        img_ids = {}
        iid = 1
        pattern = re.compile(r'\b(\d+)\b')
        q_matches = re.finditer(pattern, ques_org)
        a_matches = re.finditer(pattern, ans_org)
        r_matches = re.finditer(pattern, rat_org)
        cur_names = random.sample(names, len(objects))

        flag = 1
        for match in q_matches:
            n = match.group(0)
            if n not in img_ids:
                if int(n) > len(objects):
                    flag = 0
                    break
                elif objects[int(n)-1] == 'person':
                    img_ids[n] = cur_names[int(n)-1]
                else:
                    img_ids[n] = 'image '+ str(iid)
                    iid += 1
        for match in a_matches:
            n = match.group(0)
            if n not in img_ids:
                if int(n) > len(objects):
                    flag = 0
                    break
                elif objects[int(n)-1] == 'person':
                    img_ids[n] = cur_names[int(n)-1]
                else:
                    img_ids[n] = 'image '+ str(iid)
                    iid += 1
        for match in r_matches:
            n = match.group(0)
            if n not in img_ids:
                if int(n) > len(objects):
                    flag = 0
                    break
                elif objects[int(n)-1] == 'person':
                    img_ids[n] = cur_names[int(n)-1]
                else:
                    img_ids[n] = 'image '+ str(iid)
                    iid += 1
        if flag == 0:
            continue
        if len(img_ids) > 5:
            continue

        image_path = '/mnt/bn/automl-aigc/yatai/data/vcr/vcr1images/' + instance['img_fn']
        region_path = image_path[:-3] + 'json'
        boxes = json.load(open(region_path, 'r'))
        size = boxes['height'] * boxes['width']
        flag = 1
        if random.random() > 0.5:
            img_info = f'In the image 0: <img>{image_path}</img>\nthere are some object or person images. '
            for i,box in enumerate(boxes['boxes']):
                if str(i+1) in img_ids:
                    box_coord = box[:4]
                    if (box_coord[3] - box_coord[1]) * (box_coord[2]- box_coord[0]) < size * 0.08:
                        flag = 0
                        break
                    output_image_path = image_path[:-4] + '_' + str(i) + '.jpg'
                    crop_and_save_image(image_path, output_image_path, box_coord)
                    img_info += f'{img_ids[str(i+1)]}: <img>{output_image_path}</img>\n'
        else:
            img_info = ''
            for i,box in enumerate(boxes['boxes']):
                if str(i+1) in img_ids:
                    box_coord = box[:4]
                    if (box_coord[3] - box_coord[1]) * (box_coord[2]- box_coord[0]) < size * 0.08:
                        flag = 0
                        break
                    output_image_path = image_path[:-4] + '_' + str(i) + '.jpg'
                    crop_and_save_image(image_path, output_image_path, box_coord)
                    img_info += f'{img_ids[str(i+1)]} is <img>{output_image_path}</img>\n'
            img_info += f'In the image <img>{image_path}</img>\n'
        if flag == 0:
            continue

        question = ''
        for w in ques_org.split():
            if w[0].isnumeric():
                if len(w) == 1:
                    question = question + img_ids[w] + ' '
                elif w[:2].isnumeric():
                    question = question + img_ids[w[:2]] + w[2:] + ' '
                else:
                    question = question + img_ids[w[0]] + w[1:] + ' '
            else:
                question = question + w + ' '
        question = question[:-1]

        answer = ''
        for w in ans_org.split():
            if w[0].isnumeric():
                if len(w) == 1:
                    answer = answer + img_ids[w] + ' '
                elif w[:2].isnumeric():
                    answer = answer + img_ids[w[:2]] + w[2:] + ' '
                else:
                    answer = answer + img_ids[w[0]] + w[1:] + ' '
            else:
                answer = answer + w + ' '
        answer = answer[:-1]

        rationale = ''
        for w in rat_org.split():
            if w[0].isnumeric():
                if len(w) == 1:
                    rationale = rationale + img_ids[w] + ' '
                elif w[:2].isnumeric():
                    rationale = rationale + img_ids[w[:2]] + w[2:] + ' '
                else:
                    rationale = rationale + img_ids[w[0]] + w[1:] + ' '
            else:
                rationale = rationale + w + ' '
        rationale = rationale[:-1]

        user_input1 = img_info + question
        assistant_output1 = answer
        user_input2 = random.choice(['explain why', 'give a reason for the answer'])
        assistant_output2 = rationale

        conversation = [{"from": "user", "value": user_input1},
                        {"from": "assistant", "value": assistant_output1},
                        {"from": "user", "value": user_input2},
                        {"from": "assistant", "value": assistant_output2},
                        ]
        res.append({"id":id, "conversations": conversation})
        id += 1 
    except:
        print(ins_id)

with open('qwen_train.json', 'w') as f:
    json.dump(res, f, indent=4)