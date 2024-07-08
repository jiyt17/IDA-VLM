import pandas as pd
import json
from PIL import Image
import ast
import random
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

# 读取Parquet文件
anno = ['/mnt/bn/automl-aigc/yatai/data/refcoco/refcocog/data/train-00000-of-00001-4fe3e6340cfb69ed.parquet',
        '/mnt/bn/automl-aigc/yatai/data/refcoco/refcocog/data/validation-00000-of-00001-15168dfe7b5961e5.parquet',
        '/mnt/bn/automl-aigc/yatai/data/refcoco/refcoco/data/train-00000-of-00001-94431d5f4bd5b93f.parquet',
        '/mnt/bn/automl-aigc/yatai/data/refcoco/refcoco/data/validation-00000-of-00001-bfeafdc84ca37aa2.parquet',
        '/mnt/bn/automl-aigc/yatai/data/refcoco/refcocoplus/data/train-00000-of-00001-7294665695c630ee.parquet',
        '/mnt/bn/automl-aigc/yatai/data/refcoco/refcocoplus/data/validation-00000-of-00001-8c57d66282bc60c9.parquet']

res = []
id = 0
org_names = []

for i in range(6):
    df = pd.read_parquet(anno[i])

    title = list(df.columns)

    for index, row in tqdm(df.iterrows()):
        for t in title:
            print(t, row[t])
        image_path = '/mnt/bn/automl-aigc/yatai/data/' + row['image_path']
        file_name = row['file_name']
        if file_name in org_names:
            continue
        org_names.append(file_name)
        dest_path = '/mnt/bn/automl-aigc/yatai/data/refcoco/region_imgs/' + file_name[:-4] + '_crop' + '.jpg'
        bbox = ast.literal_eval(row['raw_anns'])['bbox']
        box_area = bbox[2] * bbox[3]
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        crop_and_save_image(image_path, dest_path, bbox)

        h = ast.literal_eval(row['raw_image_info'])['height']
        w = ast.literal_eval(row['raw_image_info'])['width']
        bbox = [int(1000*bbox[0]/w), int(1000*bbox[1]/h), int(1000*bbox[2]/w), int(1000*bbox[3]/h)]

        user_input1 = f'Picture 1: <img>{dest_path}</img>\nPicture 2:<img>{image_path}</img>\nIdentify the Picture 1 instance in the Picture 2 with grounding:'
        assistant_output1 = f'<ref>Picture 1</ref><box>({bbox[0]},{bbox[1]}),({bbox[2]},{bbox[3]})</box>'
        
        conversation = [{"from": "user", "value": user_input1},
                        {"from": "assistant", "value": assistant_output1},
                        ]
        res.append({"id":id, "conversations": conversation})
        id += 1

with open('qwen_train.json', 'w') as f:
    json.dump(res, f, indent=4)