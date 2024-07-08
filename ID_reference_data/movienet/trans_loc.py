import json
import os
import shutil
from tqdm import tqdm
import random

with open('train_split.txt') as f:
    train_anno = f.readlines()
train_anno = [s.strip() for s in train_anno]
print(len(train_anno))

ins_id = 0
res = []
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
                img2id[imgid][pid] = c['body']['bbox']
            else:
                img2id[imgid] = {pid: c['body']['bbox']}
            imgsize[imgid] = c['resolution']

        id_idimg = {}
        for id,name in names.items():
            if id in id2img:
                for img in id2img[id]:
                    if len(img2id[img]) == 1:
                        area = imgsize[img][0] * imgsize[img][1]
                        box = img2id[img][id]
                        bbox_area = (box[2]-box[0]) * (box[3]-box[1])
                        if bbox_area > area * 0.4:
                            if id in id_idimg:
                                id_idimg[id].append(img)
                            else:
                                id_idimg[id] = [img]

        pair_num = 0
        id_imgpair = {}
        for id,idimg in id_idimg.items():
            name = names[id]
            img_pair = []
            for i,img in enumerate(id2img[id]):
                if img == idimg[0]:
                    break
            j = i + 1
            while True:
                if j >= len(id2img[id]):
                    break
                if id2img[id][j] in idimg:
                    i = j
                elif len(img2id[id2img[id][j]]) > 1:
                    area = imgsize[id2img[id][j]][0] * imgsize[id2img[id][j]][1]
                    box = img2id[id2img[id][j]][id]
                    bbox_area = (box[2]-box[0]) * (box[3]-box[1])
                    if bbox_area > area * 0.15:
                        if len(img_pair) > 0:
                            last_pair = img_pair[-1]
                            last_shot = int(last_pair[1].split('-')[0])
                            cur_shot = int(id2img[id][j].split('-')[0])
                            if not (id2img[id][i] == last_pair[0] and cur_shot-last_shot <= 3):
                                pair_num += 1
                                img_pair.append([id2img[id][i], id2img[id][j]])
                        else:
                            pair_num += 1
                            img_pair.append([id2img[id][i], id2img[id][j]])
                j += 1
            id_imgpair[id] = img_pair

        for id, imgpair in id_imgpair.items():
            name = names[id]
            for pair in imgpair:
                box = img2id[pair[1]][id]
                w = imgsize[pair[1]][0]
                h = imgsize[pair[1]][1]
                bbox = [int(1000*box[0]/w), int(1000*box[1]/h), int(1000*box[2]/w), int(1000*box[3]/h)]
                dest_name = 'shot_' + pair[0].split('-')[0].zfill(4)+ '_img_' + pair[0].split('-')[1] + '.jpg'
                dest_path = os.path.join('/mnt/bn/automl-aigc/yatai/data/movie/movienet/OpenDataLab___MovieNet/raw/240P', anno_id, dest_name)
                src_name = 'shot_' + pair[1].split('-')[0].zfill(4)+ '_img_' + pair[1].split('-')[1] + '.jpg'
                src_path = os.path.join('/mnt/bn/automl-aigc/yatai/data/movie/movienet/OpenDataLab___MovieNet/raw/240P', anno_id, src_name)
                user_input1 = f'{name} is <img>{dest_path}</img>\nIn the image: <img>{src_path}</img>\nIdentify {name} with grounding.'
                assistant_output1 = f'<ref>{name}</ref><box>({bbox[0]},{bbox[1]}),({bbox[2]},{bbox[3]})</box>'
                conversation = [{"from": "user", "value": user_input1},
                                {"from": "assistant", "value": assistant_output1},
                                ]
                res.append({"id":ins_id, "conversations": conversation})
                ins_id += 1

    except:
        print(anno_id)

with open('qwen_train.json', 'w') as f:
    json.dump(res, f, indent=4)