#required
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
from io import BytesIO
import base64
import tempfile


# app.py
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from werkzeug.utils import secure_filename
import boto3
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET, REST_API_KEY

import random
import time
def randomTime():
    # 현재 시간으로 시드 설정
    current_time = time.time()
    random.seed(current_time)

    # 이제 난수 생성
    random_number = random.random()
    print("Random number:", random_number)

    # 범위 내 난수 생성 예시
    random_number_in_range = random.randint(1, 100)
    print("Random number between 1 and 100:", random_number_in_range)
    return str(random_number)



app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'your_secret_key'

class UploadForm(FlaskForm):
    image = FileField('Image', validators=[FileRequired(), FileAllowed(['jpg', 'png', 'jpeg'], 'Images only!')])

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)




#여기서부터는 karlo api 사용법 -> api키를 발급 받고 json파일을 보내면 됨, 기본확장자는 webp 이므로 png로 변경했습니다.
def t2i(prompt, negative_prompt): 
    r = requests.post(
        'https://api.kakaobrain.com/v2/inference/karlo/t2i',
        json = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'image_format' : 'png'
            },
        headers = {
            'Authorization': f'KakaoAK {REST_API_KEY}',
            'Content-Type': 'application/json' 
        }
    )
    
    # 응답 JSON 형식으로 변환
    response = json.loads(r.content)
    return response 

#inpainting
def inpainting(image, mask, prompt):
    
        r = requests.post(
            'https://api.kakaobrain.com/v2/inference/karlo/inpainting',
            json = {
                'prompt': prompt,
                'image': image,
                'mask': mask # 편집할 영역이 검정 즉 -> 거꾸로 바꿔야 함
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



#1024 * 1024 비율에 맞게 조정하기 사용자 위치 수정하기
def adjust_coordinates(orginal_width, orginal_height, target_size,  horizion, vertical):
    if orginal_width > target_size:
        x_ratio = target_size / orginal_width
    else:
        x_ratio = 1    
    if orginal_height >1024:    
        y_ratio = target_size / orginal_height
    else:
        y_ratio = 1 
    adjusted_x = int(horizion * x_ratio)
    adjusted_y = int(vertical * y_ratio)
    return adjusted_x, adjusted_y

def image_resize(image_path,original_width, original_height):
    with Image.open(image_path) as img:
        # 원본 이미지의 가로와 세로 비율 계산
        ratio = min(1024 / original_width, 1024 / original_height)
        # 새로운 크기 계산
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        # 이미지 크기 조정
        resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)

    return resized_img


#FastSam 모델 활용하기
Samodel = FastSAM('FastSAM-x.pt')  # or FastSAM-x.pt

#자바 실제 파일 위치  -> 여기에 temp 만들어서 진행할것

tempPath ="./images/temp"


# 옮기기
# 로컬에서 실제 폴더에 접근이 안되는 현상. 
#CORS 플라스크 보안해제 -> localhost:8081에서나 localhost8080의 경우는 허락해준다. 원래는 SOP에 의해 하나의 프로토콜에서 오는것만 허락하게됨 
# CORS(app, resources={r"/*": {"origins": ["http://localhost:8080", "http://localhost:8081"]}}) # localhost8081 localhost 8080 모두 가능
#임시 이미지 저장 경로 설정
# 임시 폴더 경로 설정
# 이후에 파일 저장 로직 수행

@app.route("/rembg", methods = ['POST'])
def removeBg():
    image_file = request.files['image']
    if image_file: #이미지 파일이 있다면
        filename = image_file.filename 
        file_extension = filename.split('.')[-1].lower()
        file_name =randomTime() #자바는 클라이언트이므로 불가 -> 서버에서 처리하는게 좋음
        image_path = os.path.join(tempPath, file_name +"."+ file_extension) # 자바에서 이런형식으로 저장되게 설정해야함

        image_file.save(image_path)
        # 배경 제거
        input_image = Image.open(image_path) # IMAGE -> PIL library에서 제공하는 PIL 구조의 형태로 열어준다.
        output_image = remove(input_image) # rembg에 담겨있는 remove를 통해 배경을 제거 
        # 결과 이미지 경로
        # image_url = os.path.join(javaPath,temp_folder, file_name + '_nobg.' +file_extension) #결과 이미지를 저장 
        # output_image.save(image_url) #같은 폴더에 '+_nobg.png'만 붙여서 저장

        img_byte_arr = BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        # img_byte_arr = img_byte_arr.getvalue()

        # aws 업로드 하는 로직
        folder_path = 'EditPage/Flask_img/'     
      
        # secure_filename(input.filename)
        key = folder_path + file_name +"_nobg.png"

        # S3에 이미지 업로드
        s3.upload_fileobj(BytesIO(img_byte_arr.getvalue()), S3_BUCKET, key)
        s3image_url = f'https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}'
        
        #로컬에 있는 데이터 삭제    
        os.remove(image_path)
       # 클라이언트에 이미지 url
        res= {'image_url': s3image_url} #client는 json 객체를 뜯어서 src를 확인하고 그것을 유저에게 띄워줌
        return jsonify(res) #ajax로 돌아가 함수 구현
    else:
        return jsonify({"error": "No image file provided."}), 400 # 오류


@app.route("/makeImg" ,methods=['POST']) #POST로 객체를 보내서 이미지를 만들어주라
def makeimg(): #Karlo ai 모델 -> diffusion 기반 카카오 api
    data = request.form # data는 request.form  
    prompt = data['prompt'] #name이 prompt인 값 
    negative_prompt =data['negative_prompt'] # name이 negative prompt인 값
    #prompt = 'sunset and ocean by Renoir' # 예시:  sunset and ocean , 르누아르 화가 스타일
    #negative_prompt ='nsfw' #예시: 성인, 부적합 이미지 제거
    # 이미지 생성하기
    response = t2i(prompt, negative_prompt) #response는 이제 만들어진 이미지 default위에 json에 담겨있는 값에 이미지 생성값이 1이므로 하나만 생성됨.
    # print(response.get("images")[0].get("image")) #print 디버깅용 코드 
    # result = Image.open(urllib.request.urlopen(response.get("images")[0].get("image"))) #PIL이미지로 OPEN flask환경에서는 작동되지 않음.
    # result가 있다면 spring에 result를 반환
    if response:
        image_url = response["images"][0].get("image")
        print(image_url)
        res= {'image_url': image_url} #client는 json 객체를 뜯어서 src를 확인하고 그것을 유저에게 띄워줌
        return jsonify(res) #ajax로 돌아가 함수 구현
            # 이거는 출력결과 -> 예시
            #응답 방식 HTTP/1.1 200 OK 
            # Content-Type: application/json
            # {
            #     "id": "3d6fb820-9845-4b3e-8d6d-7cfd01db5649",
            #     "model_version": "${MODEL_VERSION}",
            #     "images": [
            #         {
            #             "id": "a80108f8-b9c6-4bf4-aa38-957a38ced4a8",
            #             "seed": 3878985944,
            #             "image": "https://mk.kakaocdn.net/dna/karlo/image/..."
            #         }
            #     ]
            # }
    else:
        print("실패")
        # 요청이 실패했을 경우 오류 처리
        return jsonify({'error': '이미지 생성에 실패했습니다.'})
@app.route("/removeObject", methods = ['POST'])
# 이미지 소스 가져오기 -> fastSAM(source) -> 아미지 제거 요청(png, jpg,webp 등) -> 최종 결과 보여주기
def eraseMyImg():   
    # 폼 데이터를 변수 data에 저장
    data= request.form
    image_file = request.files['image']
    #파일 저장 -> 자바 경로에 저장 한 후 삭제하는 방향
    filename = image_file.filename 
    file_extension = filename.split('.')[-1].lower()    
    file_name =randomTime() #자바는 클라이언트이므로 불가 -> 서버에서 처리하는게 좋음
    image_path = os.path.join(tempPath, file_name +"."+ file_extension) # 자바에서 이런형식으로 저장되게 설정해야함
    # 로컬 파일에 잠시 세이브
    image_file.save(image_path)
    vertical = int(data['y'])
    horizion = int(data['x'])
    original_width =int(data['OriginalX'])
    original_height=int(data['OriginalY'])

    target_size =1024
    adjusted_horizon, adjusted_vertical = adjust_coordinates(original_width, original_height, target_size, horizion, vertical)

    # image안에서 객체찾기 실행 -> 즉 model.compile
    everything_results = Samodel(image_path, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
    # model.compile2 -> 어디서 실행할 것인가 , cpu, 객체 둘
    prompt_process = FastSAMPrompt(image_path, everything_results, device='cpu')
    #사용자가 지정한 위치에서 모든 범위를 확인해야함 
    # points default [[0,0]] [[x1,y1],[x2,y2]] 포인트의 default는 [[0,0]] , [[x1, y1] , [x2,y2]] 
    # point_label default [0] [1,0] 0:background, 1:foreground
    # point_lable default는 0: 배경 1: 배경이 아닌 객체 탐지
    prompt_process.point_prompt(points=[[adjusted_horizon, adjusted_vertical]], pointlabel=[1])
    mask_result = prompt_process.results #마스킹한 데이터를 담고있는 객체
    print(mask_result) 
    masked_array = np.array(mask_result[0].masks.data[0]) # 객체에서 0번 데이터를 numpy_array로 처리  -> 여기서 오류 도와줘요 명훈쌤
    masked_array = masked_array == False #편집할 영역이 검정이어야 하므로 이렇게 처리합니다.
    mask_image = Image.fromarray(masked_array) 
    #그다음에는 생성 mask한 부분만 prompt를 background로 해서 생성하면 됨 -> 즉 생성모델
    #Karlo api에 보낼 수 있게 디코딩/인코딩
    # prompt 설정 -> 여기서는 background
    prompt = "background"




    #이미지 가능 부분이 1024 * 1024 이므로 이미지를 그에 맞게 축소하는 로직이 필요할 듯 함
    # 원본이미지를 1024 * 1024로 바꿔야
    #이미지 원본의 크기를
    #이미지 원본의 크기를 보내고
    # 그걸 python 변수로 저장 한다
    # 이미지 처리 후
    # 해당 이미지 원본 크기만큼 늘려서 저장하면
    
    # 11.30 일 추가부분


    resized_img=image_resize(image_path,original_width, original_height)

    # 이미지를 Base64 인코딩하기
    img_base64 = imageToString(resized_img)
    mask_base64 = imageToString(mask_image)

    # 이미지 변환하기 REST API 호출
    response = inpainting(img_base64,mask_base64,prompt)
    
    #로컬에 있는 데이터 삭제    
    os.remove(image_path)

    print(response)
    # 응답의 첫 번째 이미지 생성 결과 출력하기
    image_url = response["images"][0].get("image")
    #최종에서 찌그러지지 않게 다시 처리 //TODO 
    
    res = {'image_url': image_url}
    return jsonify(res) # 여기서 경로 -> eclipse로 가서 사용자에게 보여주기


@app.route("/changeBackground", methods = ['POST'])
# 이미지 소스 가져오기 -> fastSAM(source) -> 아미지 제거 요청(png, jpg,webp 등) -> 최종 결과 보여주기
def changeBack():
    # 폼 데이터를 변수 data에 저장
    data= request.form
    image_file = request.files['image']
    #임시 파일 생성 및 저장
    # with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file: # 임시폴더 만들어서 png로 저장시키고 
    #    image = Image.open(image_file.stream)
    #    image.save(temp_file.name)
    #     temp_file_path = temp_file.name
    filename = image_file.filename 
    file_extension = filename.split('.')[-1].lower()
    file_name =randomTime() #자바는 클라이언트이므로 불가 -> 서버에서 처리하는게 좋음
    image_path = os.path.join(tempPath, file_name +"."+ file_extension) # 자바에서 이런형식으로 저장되게 설정해야함
    image_file.save(image_path)

    prompt = data['prompt']
    vertical = int(data['y'])
    horizion = int(data['x'])
    original_width =int(data['OriginalX'])
    original_height=int(data['OriginalY'])

    target_size =1024
    adjusted_horizon, adjusted_vertical = adjust_coordinates(original_width, original_height, target_size, horizion, vertical)
    # image안에서 객체찾기 실행 -> 즉 model.compile
    everything_results = Samodel(image_path, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
    # model.compile2 -> 어디서 실행할 것인가 , cpu, 객체 둘
    prompt_process = FastSAMPrompt(image_path, everything_results, device='cpu')
    #사용자가 지정한 위치에서 모든 범위를 확인해야함 
    # points default [[0,0]] [[x1,y1],[x2,y2]] 포인트의 default는 [[0,0]] , [[x1, y1] , [x2,y2]] 
    # point_label default [0] [1,0] 0:background, 1:foreground
    # point_lable default는 0: 배경 1: 배경이 아닌 객체 탐지
    prompt_process.point_prompt(points=[[adjusted_horizon, adjusted_vertical]], pointlabel=[1])
    mask_result = prompt_process.results #마스킹한 데이터를 담고있는 객체
    print(mask_result) 
    masked_array = np.array(mask_result[0].masks.data[0]) # 객체에서 0번 데이터를 numpy_array로 처리  -> 여기서 오류 도와줘요 명훈쌤
#   masked_array = masked_array == False    
    print(masked_array)
    mask_image = Image.fromarray(masked_array) 
    #그다음에는 생성 mask한 부분만 prompt를 background로 해서 생성하면 됨 -> 즉 생성모델
    #Karlo api에 보낼 수 있게 디코딩/인코딩
    # 이미지를 Base64 인코딩하기
    resized_img=image_resize(image_path,original_width, original_height)

    img_base64 = imageToString(resized_img)
    mask_base64 = imageToString(mask_image)

    # 이미지 변환하기 REST API 호출
    response = inpainting(img_base64,mask_base64,prompt)
    #로컬에 있는 데이터 삭제    
    os.remove(image_path)
    print(response)
    # 응답의 첫 번째 이미지 생성 결과 출력하기
    image_url = response["images"][0].get("image")
    res = {'image_url': image_url}
    return jsonify(res) # 여기서 경로 -> eclipse로 가서 사용자에게 보여주기 

if __name__ == '__main__':
    # 외부에서 접근 가능하도록 호스트 설정 ('0.0.0.0'으로 설정하면 모든 네트워크 인터페이스에서 접근 가능)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)



