      
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
        azure_endpoint="https://search.bytedance.net/gpt/openapi/online/multimodal/crawl", 
        api_version="gpt-4-vision", 
        api_key="xxxxxxx") 

    movie_imgs = json.load(open('/mnt/bn/automl-aigc/yatai/data/movie/movienet/files/movie_gpt4v.json', 'r'))

    resume = json.load(open('movie_gpt4v_caption.json', 'r'))
    cur_len = resume[-1]['idx']

    res = resume
    for idx, img_info in tqdm(enumerate(movie_imgs)): # len(movie_imgs): 80000+
        if idx <= cur_len:
            continue
        if not os.path.exists(img_info['img_path']):
            continue
        img = cv2.imread(img_info['img_path'])
        success, encoded_image = cv2.imencode('.jpg', img)
        content = encoded_image.tobytes() 
        base64_image = base64.b64encode(content) 
        base64_string = base64_image.decode('utf-8')

        img_size = img_info['img_size']

        character = 'The characters: '
        for cast in img_info['name_box']:
            character = character + cast[1] + f': ({cast[2][0]}, {cast[2][1]}, {cast[2][2]}, {cast[2][3]}),\n '
        character = character[:-4] + '.\n'

        instruct = ["Given the image", {"image": base64_string}, f"The image size is ({img_size[0]}, {img_size[1]}).\n {character} Describe the image concisely and briefly."]

        error_sample = 0
        while True:
            try:
                response = client.chat.completions.create(
                    model=model, 
                    messages=[ 
                        { "role": "system", 
                            "content": [
                                "You are a helpful and precise assistant for providing description for an appropriate image. \
                                User will give an image and the image size (width, height). Then user will give some character names and their position in the image. The position is expressed with bounding box, which is the person left-top corner coordinates and right-bottom corner coordinates (left, top, right, bottom). \
                                Firstly, you need to judge if it is approprite. An appropriate image should be clear, should be easy for you to give a caption, the people in it should be easy to recognize. \
                                If the image is not appropriate, you should answer 'no', if it is appropriate, you should give an accurate description of the image with given character names according to following rules. \
                                Different characters will be splited by '\n', you must remember the right people in the right position. Maybe there are some other people without name in the image, your caption need contain them if necessary, but the main subject should be ahout chracters with names. \
                                The description should be accurate and brief. The answer should be less than 60 words. Please pay more attention to people's action. Your answer should be only about the visual content, don't include your own speculation. \
                                Then you should give an accurate description of the image with given character names."
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
        
        if error_sample == 0:
            res.append({'idx':idx, 'img_path': img_info['img_path'], 'caption': gpt_result})
            with open('movie_gpt4v_caption.json', 'w') as f:
                json.dump(res, f, indent=4)

    