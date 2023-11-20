from flask import Flask, request, jsonify
import json
import os
from PIL import Image
import requests 
from flask_cors import CORS, cross_origin #flask 보안 및 cross 
from diffusers import StableDiffusionInpaintPipeline # 인페인팅 모델 
from rembg import remove
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import numpy as np
import torch
import io
#내 카카오 어플리케이션 키 
REST_API_KEY= '9b22d611c336e7a4d4e647a0a3c40a96'

#FastSam 모델 활용하기
model = FastSAM('FastSAM-s.pt')  # or FastSAM-x.pt

#자바 실제 파일 위치  -> 여기에 temp 만들어서 진행할것
javaPath = r"C:\eGovFrame-4.0.0\workspace.edu\.metadata\.plugins\org.eclipse.wst.server.core\tmp1\webapps"
# 로컬에서 실제 폴더에 접근이 안되는 현상. 
#CORS 플라스크 보안해제 -> localhost:8081에서나 localhost8080의 경우는 허락해준다. 원래는 SOP에 의해 하나의 프로토콜에서 오는것만 허락하게됨 
app = Flask(__name__)
CORS(app)
# CORS(app, resources={r"/*": {"origins": ["http://localhost:8080", "http://localhost:8081"]}}) # localhost8081 localhost 8080 모두 가능
#임시 이미지 저장 경로 설정
temp_folder='temp' #테스트용 temp 폴더 만들어서 저장 ->실제로는 이클립스에 temp 만들어서 저장하고 서버를 나갈때 지금 있는 이미지를 지울 수 있게 처리해야함
# 임시 폴더 경로 설정
temp_folder_path = os.path.join(javaPath, temp_folder)
# 이후에 파일 저장 로직 수행

# 랜덤 스트링 
@app.route("/removeBg", methods = ['post'])
def imgEdit():
    image_file =request.files.get('myfile', None) #form 태그에 input 태그에 name이 myfile인 객체 담기
    if image_file: #이미지 파일이 있다면
        file_name =image_file.filename #자바는 클라이언트이므로 불가 -> 서버에서 처리하는게 좋음
        image_path = os.path.join(javaPath, temp_folder, file_name + ".png") # 자바에서 이런형식으로 저장되게 설정해야함
        image_file.save(image_path)
        # 배경 제거
        input_image = Image.open(image_path) # IMAGE -> PIL library에서 제공하는 PIL 구조의 형태로 열어준다.
        output_image = remove(input_image) # rembg에 담겨있는 remove를 통해 배경을 제거 
        # 결과 이미지 경로
        image_url = os.path.join(javaPath,temp_folder, file_name + '_no bg.png') #결과 이미지를 저장 
        output_image.save(image_url) #같은 폴더에 '+_nobg.png'만 붙여서 저장
       # 클라이언트에 이미지 url
        res= jsonify(image_url = image_url) #client는 json 객체를 뜯어서 src를 확인하고 그것을 유저에게 띄워줌
        return res #ajax로 돌아가 함수 구현
    else:
        return jsonify({"error": "No image file provided."}), 400 # 오류
        
@app.route("/makeImg" ,methods=['POST']) #POST로 객체를 보내서 이미지를 만들어주라
def makeimg(): #Karlo ai 모델 -> diffusion 기반 카카오 api
    data = request.form # data는 request.form  
    prompt = data['prompt'] #name이 prompt인 값 
    negative_prompt =data['negative_prompt'] # name이 negative prompt인 값
    #prompt = 'sunset and ocean by Renoir' # 예시:  sunset and ocean , 르누아르 화가 스타일
    #negative_prompt ='nsfw' #예시: 성인, 부적합 이미지 제거
    
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
    # 이미지 생성하기
    response = t2i(prompt, negative_prompt) #response는 이제 만들어진 이미지 default위에 json에 담겨있는 값에 이미지 생성값이 1이므로 하나만 생성됨.
    # print(response.get("images")[0].get("image")) #print 디버깅용 코드 
    # result = Image.open(urllib.request.urlopen(response.get("images")[0].get("image"))) #PIL이미지로 OPEN flask환경에서는 작동되지 않음.
    # result가 있다면 spring에 result를 반환
    if response:
        image_url = response["images"][0].get("image")
        print(image_url)
        res= jsonify(image_url = image_url) #client는 json 객체를 뜯어서 src를 확인하고 그것을 유저에게 띄워줌
        return res # 여기서 경로 -> eclipse로 가서 사용자에게 보여주기 
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
# 이미지 소스 가져오기 -> fastSAM -> 아미지 제거 요청 -> 최종 결과 보여주기
def eraseMyImg():
    data= request.form
    source =request.files.get('image', None) #None은 파일이 없을 경우 반환할 기본값을 지정하는 것
    image_file =request.files.get('image', None) #form 태그에 input 태그에 name이 myfile인 객체 담기
    # image = Image.open(io.BytesIO(source.read()))  # BytesIO를 사용하여 PIL 이미지로 변환
    image_bytes = source.read()
    image = Image.open(image_file.stream)  #받은 이미지를 PILImage로 변경
# 이미지 읽기 시도
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # PIL 이미지가 정상적으로 읽어졌을 때의 코드
        print(f"Width: {image.width}, Height: {image.height}, Mode: {image.mode}")
        source = request.files['image']
    
    except Exception as e:
        print(f"Error reading image: {e}")

    # image = Image.open(source.stream) #받은 이미지를 PILImage로 변경
    vertical = data['y']
    horizion = data['x']
    # image안에서 객체찾기 실행 -> 즉 model.compile
    everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
    # model.compile2 -> 어디서 실행할 것인가 , cpu, 객체 둘
    
    prompt_process = FastSAMPrompt(source, everything_results, device='cpu')
    #사용자가 지정한 위치에서 모든 범위를 확인해야함 
    # points default [[0,0]] [[x1,y1],[x2,y2]] 포인트의 default는 [[0,0]] , [[x1, y1] , [x2,y2]] 
    # point_label default [0] [1,0] 0:background, 1:foreground
    # point_lable default는 0: 배경 1: 배경이 아닌 객체 탐지
    prompt_process.point_prompt(points=[[horizion, vertical]], pointlabel=[1])
    res = prompt_process.results #마스킹한 데이터를 담고있는 객체 
    masked_array = np.array(res[0].masks.data[0]) # 객체에서 0번 데이터를 numpy_array로 처리
    mask_image = Image.fromarray(masked_array) #PIL 이미지로 변경
    #그다음에는 생성 mask한 부분만 prompt를 background로 해서 생성하면 됨 -> 즉 생성모델
    print("마스킹 완료")
    pipe = StableDiffusionInpaintPipeline.from_pretrained( 
    "runwayml/stable-diffusion-inpainting", 
    revision="fp32", 
    #fp16은 half-precision의 준말로, 16비트 부동 소수점 형식을 나타냄. 
    # 이 형식은 메모리를 적게 사용하므로 딥 러닝 분야에서 매우 인기가 있음. 
    # 그러나 16비트의 정밀도가 낮아서 모델의 정확도가 떨어질 수 있음. 
    # 따라서 모델을 훈련할 때는 일반적으로 fp32(32비트 부동 소수점 형식)를 사용하고,
    # 추론(inference) 단계에서는 fp16을 사용하여 연산 속도를 높이는 경우가 많음.  => 즉 추론 하는 단계 -> 이미지생성
    torch_dtype=torch.float32 #torch 파일로 만들돼, 16진수 소수점 으로 만든다. 
    ,)
    prompt = "background" # background를 입력해서 배경으로 채워줘 라고 하는 것.
    #image and mask_image should be PIL images. => 둘다 PIL 이미지로 만듣어서 처리 해야합니다.
    #The mask structure is white for inpainting and black for keeping as is => mask된 부분은 흰색 바깥은 검은색으로 처리하기
    new_image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
    new_image.save(f"{source.filename}removed.png")
    image_url =f"{source.filename}removed.png"
    res= jsonify(image_url = image_url) #client는 json 객체를 뜯어서 src를 확인하고 그것을 유저에게 띄워줌
    print(res)
    return res # 여기서 경로 -> eclipse로 가서 사용자에게 보여주기 
    
if __name__ == '__main__':
    # 외부에서 접근 가능하도록 호스트 설정 ('0.0.0.0'으로 설정하면 모든 네트워크 인터페이스에서 접근 가능)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)



