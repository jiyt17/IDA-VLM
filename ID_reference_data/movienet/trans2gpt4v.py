import json
import os
import shutil
from tqdm import tqdm
import random
from PIL import Image
import re

with open('train_split.txt') as f:
    train_anno = f.readlines()
train_anno = [s.strip() for s in train_anno]
print(len(train_anno))
train_anno = train_anno[::-1]

ins_id = 0
res = []
valid_num = 0
for anno_id in tqdm(train_anno):

    try:

        anno = json.load(open(f'/mnt/bn/automl-aigc/yatai/data/movie/movienet/files/annotation/{anno_id}.json', 'r'))
        meta = json.load(open(f'/mnt/bn/automl-aigc/yatai/data/movie/movienet/files/meta_new/{anno_id}.json', 'r'))

        cast = meta['cast']
        names = {}
        for c in cast:
            names[c['id']] = c['character']

        cast = anno['cast']
        id2img = {}
        img2id = {}
        imgsize = {}
        for c in cast:
            pid = c['pid']
            imgid = str(c['shot_idx']) + '-' + str(c['img_idx'])
            if pid != 'others':
                if pid in id2img:
                    id2img[pid].append(imgid)
                else:
                    id2img[pid] = [imgid]
            if imgid in img2id:
                img2id[imgid].append([pid, c['body']['bbox']])
            else:
                img2id[imgid] = [[pid, c['body']['bbox']]]
            imgsize[imgid] = c['resolution']

        id_idimg = {}
        for id,name in names.items():
            if id in id2img:
                for img in id2img[id]:
                    if len(img2id[img]) == 1:
                        area = imgsize[img][0] * imgsize[img][1]
                        box = img2id[img][0][1]
                        bbox_area = (box[2]-box[0]) * (box[3]-box[1])
                        if bbox_area > area * 0.4:
                            if id in id_idimg:
                                id_idimg[id].append(img)
                            else:
                                id_idimg[id] = [img]

        last_img_shot = -2
        for img, pid in img2id.items():
            if len(pid) < 2 or len(pid) > 5:
                continue
            pid_widimg_num = 0
            if int(img.split('-')[0]) - last_img_shot < 2:
                continue
            for p in pid:
                if p[0] in id_idimg:
                    pid_widimg_num += 1
            if pid_widimg_num < 2 or pid_widimg_num > 4:
                continue
            src_name = 'shot_' + img.split('-')[0].zfill(4)+ '_img_' + img.split('-')[1] + '.jpg'
            src_path = os.path.join('/mnt/bn/automl-aigc/yatai/data/movie/movienet/OpenDataLab___MovieNet/raw/240P', anno_id, src_name)
            name_pid_box = []
            for p in pid:
                if p[0] in id_idimg:
                    name_pid_box.append([p[0], names[p[0]], p[1]])
            img_gpt4v_info = {'idx': ins_id, 'img_path': src_path, 'title': meta['title'], 'movie_id': anno_id, 'img_size': imgsize[img], 'name_box': name_pid_box}
            res.append(img_gpt4v_info)
            valid_num += 1
            ins_id += 1
            last_img_shot = int(img.split('-')[0])
    
    except:
        print(anno_id)
    
print(valid_num)
with open('movie_gpt4v_test.json', 'w') as f:
    json.dump(res, f, indent=4)

anno = json.load(open('movie_gpt4v_test.json', 'r'))
for sample in anno:
    for name in sample['name_box']:
        name.append(str(name[2]))
with open('movie_gpt4v_test_new.json', 'w') as f:
    json.dump(anno, f, indent=4)
