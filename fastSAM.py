from PIL import Image
# from flask_cors import CORS, cross_origin #flask 보안 및 cross 
import cv2 # opencv
from diffusers import StableDiffusionInpaintPipeline # 인페인팅 모델 
from rembg import remove
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import numpy as np
import torch, torchvision






# REST API 호출, 이미지 파일 처리에 필요한 라이브러리
import requests
import json
import io
import base64
import urllib
from PIL import Image
REST_API_KEY= '9b22d611c336e7a4d4e647a0a3c40a96'
# 이미지 변환하기
def inpainting(image, mask, prompt):
    r = requests.post(
        'https://api.kakaobrain.com/v2/inference/karlo/inpainting',
        json = {
            'prompt': prompt,
            'image': image,
            'mask': mask
        },
        headers = {
            'Authorization': f'KakaoAK {REST_API_KEY}',
            'Content-Type': 'application/json'
        }
    )
    # 응답 JSON 형식으로 변환
    response = json.loads(r.content)
    return response

# Base64 인코딩
def imageToString(img):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return my_encoded_img

# 이미지 파일 불러오기
img = Image.open('urbanbrush-20221108214712319041.jpg')
mask = Image.open('Test.png')
prompt = "background"


# 이미지를 Base64 인코딩하기
img_base64 = imageToString(img)
mask_base64 = imageToString(mask)

# 이미지 변환하기 REST API 호출
response = inpainting(img_base64,mask_base64,prompt)
print(response)

# 응답의 첫 번째 이미지 생성 결과 출력하기
result = Image.open(urllib.request.urlopen(response.get("images")[0].get("image")))
result.show()

# #내 카카오 어플리케이션 키 
# REST_API_KEY= '9b22d611c336e7a4d4e647a0a3c40a96'
# #FastSam 모델 활용하기
# model = FastSAM('FastSAM-x.pt')  # or FastSAM-x.pt
# source = "./urbanbrush-20221108214712319041.jpg"
# image1 = Image.open(source)
# everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
# prompt_process = FastSAMPrompt(source, everything_results, device='cpu')
# prompt_process.point_prompt(points=[[500, 350]], pointlabel=[1])
# res = prompt_process.results #마스킹한 데이터를 담고있는 객체
# masked_array = np.array(res[0].masks.data[0])
# mask_image = Image.fromarray(masked_array) #PIL 객체
# mask_image.save("TEST.png")




