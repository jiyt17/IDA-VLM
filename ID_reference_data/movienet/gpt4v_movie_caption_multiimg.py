      
import cv2
import base64 
import openai
import json
import time
import os
from tqdm import tqdm


if __name__ == '__main__':

    model = "gptv"
    client = openai.AzureOpenAI( 
        azure_endpoint="xxxxxx", 
        api_version="gpt-4-vision", 
        api_key="xxxxxxxx") 

    movie_imgs = json.load(open('/mnt/bn/automl-aigc/yatai/data/movie/movienet/files/movie_gpt4v_train_multiimg_new.json', 'r'))

    # resume = json.load(open('movie_gpt4v_caption.json', 'r'))
    # cur_len = resume[-1]['idx']
    cur_len = 2000

    res = []
    for idx, img_info in tqdm(enumerate(movie_imgs)): 
        if idx < cur_len:
            continue
        names = []
        for people in img_info['characters']:
            names.append(people[1])
        names_str = ','.join(names)
        multi_imgs = []
        flag = 1
        for img in img_info['imgs']:
            if not os.path.exists(img['img_path']):
                flag = 0
                break
            image = cv2.imread(img['img_path'])
            success, encoded_image = cv2.imencode('.jpg', image)
            content = encoded_image.tobytes() 
            base64_image = base64.b64encode(content) 
            base64_string = base64_image.decode('utf-8')
            character = 'The characters: '
            for name_box in img['name_box']:
                if name_box[1] in names:
                    character = character + name_box[1] + f': ({name_box[2][0]}, {name_box[2][1]}, {name_box[2][2]}, {name_box[2][3]}),\n '
            character = character[:-4] + '.\n'
            multi_imgs.append({'img': base64_string, 'names': character})
        if flag == 0:
            continue

        img_size = img_info['imgs'][0]['img_size']

        instruct = ["Given the images", f"The image size is ({img_size[0]}, {img_size[1]}).\n"]
        for img in multi_imgs:
            instruct.append({"image": img['img']})
            instruct.append(img['names'])
        instruct.append(f"Please provide the description of the above images about {names_str} concisely and briefly.")

        error_sample = 0
        while True:
            try:
                response = client.chat.completions.create(
                    model=model, 
                    messages=[ 
                        { "role": "system", 
                            "content": [
                                "You are a helpful and precise assistant for providing a description for some continuous images. You can treat it as video caption generation. You need give an overall story description for these images according to the following rules. \
                                User will give the image size (width, height) and some images. For each image, user will give some character names and their positions in the image. The position is expressed with bounding box, which is the person left-top corner coordinates and right-bottom corner coordinates (left, top, right, bottom). \
                                Different characters will be splited by '\n', you must remember the right people in the right position. Maybe there are some other people without name in the image, your caption need contain them if necessary, but the main subject should be ahout chracters with names. \
                                The description should be accurate and brief. The answer should be less than 60 words. Please pay more attention to people's action. Your answer should be in accordance with temporal order of the images. Your description should be coherent and have some logical connections. Your answer should be only about the visual content, don't include your own speculation. \
                                You should describe the changes in the characters' states and interactions throughout the entire images sequence as much as possible, avoiding fragmented descriptions for each individual image. Especially refrain from using phrases like 'in the image 1' and so on. \
                                Then you should give an accurate and brief description of the images with given character names. \
                                A good example: 'Carol watches as Hal, in a white tank top, looks at his shirt. As she approaches, they engage in a close conversation, and eventually, Hal looks at Carol while gesturing, continuing their discussion.' "
                                ]
                        },
                        { "role": "user", 
                            "content": instruct
                        } 
                    ] 
                ) 
                gpt_result = response.choices[0].message.content
                break
            except Exception as e:
                print(e)
                if 'Error code: 429' in str(e):
                    time.sleep(5)
                    continue
                else:
                    error_sample = 1
                    break
        # print(gpt_result)
        if error_sample == 0:
            res.append({'idx':idx, 'img_idx': img_info['idx'], 'caption': gpt_result})
            with open('movie_gpt4v_caption_mi_jyt.json', 'w') as f:
                json.dump(res, f, indent=4)

    