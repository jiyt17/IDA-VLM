import json
from tqdm import tqdm
import random
import os

qas = json.load(open('./movie_gpt4v_qa_mi.json', 'r'))
qa_imgs = json.load(open('../movie_gpt4v_train_multiimg_qa.json', 'r'))
print(len(qa_imgs))


ins_id = 0
res = []
anno_id = 0
for sample in tqdm(qas):
    img_info = qa_imgs[sample['img_idx']]
    if anno_id != img_info['movie_id']:
        anno_id = img_info['movie_id']

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
                            idimg_name = 'shot_' + img.split('-')[0].zfill(4)+ '_img_' + img.split('-')[1] + '.jpg'
                            idimg_path = os.path.join('/mnt/bn/automl-aigc/yatai/data/movie/movienet/OpenDataLab___MovieNet/raw/240P', anno_id, idimg_name)
                            if os.path.exists(idimg_path):
                                if id in id_idimg:
                                    id_idimg[id].append(idimg_path)
                                else:
                                    id_idimg[id] = [idimg_path]

    question = ''
    for p in img_info['characters']:
        if p[0] in id_idimg:
            question += p[1]
            idimg = random.choice(id_idimg[p[0]])
            question = question + ' is <img>' + idimg + '</img>\n '
    question = question + "In the following images: "
    src_imgs = ""
    for img in img_info['imgs']:
        src_imgs += f"<img>{img['img_path']}</img>\n"
    question = question + src_imgs + sample['caption'].split('\n')[0]
    answer = sample['caption'].split('\n')[1]

    conversation = [{"from": "user", "value": question},
        {"from": "assistant", "value": answer},
        ]

    res.append({"id":ins_id, "conversations": conversation})
    ins_id += 1


with open('qwen_gpt4v_qa_mi.json', 'w') as f:
    json.dump(res, f, indent=4)
