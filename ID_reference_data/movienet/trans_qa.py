import json
from tqdm import tqdm
import random
import os

# qas = json.load(open('./movie_gpt4v_qa.json', 'r'))

# img_qa = {}
# for qa in qas:
#     if qa['img_path'] in img_qa:
#         img_qa[qa['img_path']].append(qa['qa'])
#     else:
#         img_qa[qa['img_path']] = [qa['qa']]

# with open('/mnt/bn/automl-aigc/yatai/data/movie/movienet/files/train_split.txt') as f:
#     train_anno = f.readlines()
# train_anno = [s.strip() for s in train_anno]
# train_anno = train_anno[::-1]

# ins_id = 0
# res = []
# for anno_id in tqdm(train_anno):
#     try:
#         anno = json.load(open(f'/mnt/bn/automl-aigc/yatai/data/movie/movienet/files/annotation/{anno_id}.json', 'r'))
#         meta = json.load(open(f'/mnt/bn/automl-aigc/yatai/data/movie/movienet/files/meta_new/{anno_id}.json', 'r'))

#         cast = meta['cast']
#         names = {}
#         for c in cast:
#             names[c['id']] = c['character']

#         cast = anno['cast']
#         id2img = {}
#         img2id = {}
#         imgsize = {}
#         for c in cast:
#             pid = c['pid']
#             imgid = str(c['shot_idx']) + '-' + str(c['img_idx'])
#             if pid != 'others':
#                 if pid in id2img:
#                     id2img[pid].append(imgid)
#                 else:
#                     id2img[pid] = [imgid]
#             if imgid in img2id:
#                 img2id[imgid].append([pid, c['body']['bbox']])
#             else:
#                 img2id[imgid] = [[pid, c['body']['bbox']]]
#             imgsize[imgid] = c['resolution']

#         id_idimg = {}
#         for id,name in names.items():
#             if id in id2img:
#                 for img in id2img[id]:
#                     if len(img2id[img]) == 1:
#                         area = imgsize[img][0] * imgsize[img][1]
#                         box = img2id[img][0][1]
#                         bbox_area = (box[2]-box[0]) * (box[3]-box[1])
#                         if bbox_area > area * 0.4:
#                             idimg_name = 'shot_' + img.split('-')[0].zfill(4)+ '_img_' + img.split('-')[1] + '.jpg'
#                             idimg_path = os.path.join('/mnt/bn/automl-aigc/yatai/data/movie/movienet/OpenDataLab___MovieNet/raw/240P', anno_id, idimg_name)
#                             if os.path.exists(idimg_path):
#                                 if id in id_idimg:
#                                     id_idimg[id].append(idimg_path)
#                                 else:
#                                     id_idimg[id] = [idimg_path]

#         for img, pid in img2id.items():
#             src_name = 'shot_' + img.split('-')[0].zfill(4)+ '_img_' + img.split('-')[1] + '.jpg'
#             src_path = os.path.join('/mnt/bn/automl-aigc/yatai/data/movie/movienet/OpenDataLab___MovieNet/raw/240P', anno_id, src_name)
#             if src_path in img_qa:
#                 question = ''
#                 for p in pid:
#                     if p[0] in id_idimg:
#                         question += names[p[0]]
#                         idimg = random.choice(id_idimg[p[0]])
#                         question = question + ' is <img>' + idimg + '</img>\n '
#                 question = question + 'In the image: <img>' + src_path + '</img>\n '
#                 qa = img_qa[src_path]
#                 for qa_ in qa:
#                     question_ = question + qa_.split('\n')[0]
#                     answer_ = qa_.split('\n')[1]

#                     conversation = [{"from": "user", "value": question_},
#                         {"from": "assistant", "value": answer_},
#                         ]

#                     res.append({"id":ins_id, "conversations": conversation})
#                     ins_id += 1

#     except:
#         print(anno_id)

# with open('qwen_gpt4v_qa.json', 'w') as f:
#     json.dump(res, f, indent=4)

# jyt = json.load(open('./movie_gpt4v_qa_jyt.json', 'r'))
# zjc = json.load(open('./movie_gpt4v_qa_zjc.json', 'r'))
# cwf = json.load(open('./movie_gpt4v_qa_cwf.json', 'r'))
# wj = json.load(open('./movie_gpt4v_qa_wj.json', 'r'))

# # jyt_new = []
# # for tmp in jyt:
# #     if 'idx' in tmp and tmp['idx'] > 10000:
# #         continue
# #     jyt_new.append(tmp)

# qa = jyt + zjc + cwf + wj

# print(len(qa))

# res = []
# ind = 1
# for sample in qa:
#     if sample['qa'][:2].lower() == 'no':
#         continue
#     if sample['qa'].count('\n') != 1:
#         continue
#     sample['idx'] = ind
#     res.append(sample)
#     ind += 1

# print(len(res))

# with open('movie_gpt4v_qa.json', 'w') as f:
#     json.dump(res, f, indent=4)

qwen = json.load(open('./qwen_gpt4v_qa.json', 'r'))

res = []
id = 0
for qa in qwen:
    if qa['conversations'][1]['value'].lower()[:2] != 'no':
        qa['id'] = id
        res.append(qa)
        id += 1

with open('qwen_gpt4v_qa.json', 'w') as f:
    json.dump(res, f, indent=4)