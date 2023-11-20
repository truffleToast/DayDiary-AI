from flask import Flask, request, jsonify
import json
import os
from PIL import Image
import requests 
from flask_cors import CORS, cross_origin #flask 보안 및 cross 
from rembg import remove
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import numpy as np
import io
import base64
import tempfile
def eraseMyImg():
    # 폼 데이터를 변수 data에 저장
    data= request.form
    source =request.files.get('image', None) #None은 파일이 없을 경우 반환할 기본값을 지정하는 것
    print(source)
    #임시 파일 생성 및 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file: # 임시폴더 만들어서 png로 저장시키고 
        image = Image.open(source.stream)
        image.save(temp_file.name)
        temp_file_path = temp_file.name
    vertical = int(data['y'])
    horizion = int(data['x'])
    print(temp_file_path, vertical, horizion)
    # image안에서 객체찾기 실행 -> 즉 model.compile
    everything_results = model(temp_file_path, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
    # model.compile2 -> 어디서 실행할 것인가 , cpu, 객체 둘
    prompt_process = FastSAMPrompt(temp_file_path, everything_results, device='cpu')
    #사용자가 지정한 위치에서 모든 범위를 확인해야함 
    # points default [[0,0]] [[x1,y1],[x2,y2]] 포인트의 default는 [[0,0]] , [[x1, y1] , [x2,y2]] 
    # point_label default [0] [1,0] 0:background, 1:foreground
    # point_lable default는 0: 배경 1: 배경이   아닌 객체 탐지
    prompt_process.point_prompt(points=[[horizion, vertical]], pointlabel=[1])
    mask_result = prompt_process.results #마스킹한 데이터를 담고있는 객체
    print(mask_result) 
    masked_array = np.array(mask_result[0].masks.data[0]) # 객체에서 0번 데이터를 numpy_array로 처리  -> 여기서 오류 도와줘요 명훈쌤
    print(masked_array)
    mask_image = Image.fromarray(masked_array) 
    #그다음에는 생성 mask한 부분만 prompt를 background로 해서 생성하면 됨 -> 즉 생성모델
    #Karlo api에 보낼 수 있게 디코딩/인코딩
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
    def imageToString(img): #Karlo API 코드이므로 건들 필요 X
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii') 
        return my_encoded_img

    # prompt 설정 -> 여기서는 background
    prompt = "castle"

    # 이미지를 Base64 인코딩하기
    img_base64 = imageToString(image)
    mask_base64 = imageToString(mask_image)

    # 이미지 변환하기 REST API 호출
    response = inpainting(img_base64,mask_base64,prompt)
    
    print(response)
    # 응답의 첫 번째 이미지 생성 결과 출력하기
    image_url = response["images"][0].get("image")
    res = {'image_url': image_url}
    return jsonify(res) # 여기서 경로 -> eclipse로 가서 사용자에게 보여주기 

if __name__ == '__main__':
    # 외부에서 접근 가능하도록 호스트 설정 ('0.0.0.0'으로 설정하면 모든 네트워크 인터페이스에서 접근 가능)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)