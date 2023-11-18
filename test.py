from PIL import Image
# from flask_cors import CORS, cross_origin #flask 보안 및 cross 
import cv2 # opencv
from diffusers import StableDiffusionInpaintPipeline # 인페인팅 모델 
from rembg import remove
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import numpy as np
import torch
#FastSam 모델 활용하기
model = FastSAM('FastSAM-x.pt')  # or FastSAM-x.pt
source = "./urbanbrush-20221108214712319041.jpg"
image = cv2.imread(source)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
arr_image= np.array(image)
P_img = Image.fromarray(arr_image) 
# vertical = data['y']
# horizion = data['x']
# image안에서 객체찾기 실행 -> 즉 model.compile
everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
# model.compile2 -> 어디서 실행할 것인가 , cpu, 객체 둘
prompt_process = FastSAMPrompt(source, everything_results, device='cpu')
#사용자가 지정한 위치에서 모든 범위를 확인해야함 
# points default [[0,0]] [[x1,y1],[x2,y2]] 포인트의 default는 [[0,0]] , [[x1, y1] , [x2,y2]] 
# point_label default [0] [1,0] 0:background, 1:foreground
# point_lable default는 0: 배경 1: 배경이 아닌 객체 탐지
prompt_process.point_prompt(points=[[500, 350]], pointlabel=[1])
res = prompt_process.results #마스킹한 데이터를 담고있는 객체
masked_array = np.array(res[0].masks.data[0]) # 객체에서 0번 데이터를 numpy_array로 처리
mask_image = Image.fromarray(masked_array) #PIL 이미지로 변경
#그다음에는 생성 mask한 부분만 prompt를 background로 해서 생성하면 됨 -> 즉 생성모델
pipe = StableDiffusionInpaintPipeline.from_pretrained( 
"runwayml/stable-diffusion-inpainting", 
revision="fp16", 
#fp16은 half-precision의 준말로, 16비트 부동 소수점 형식을 나타냄. 
# 이 형식은 메모리를 적게 사용하므로 딥 러닝 분야에서 매우 인기가 있음. 
# 그러나 16비트의 정밀도가 낮아서 모델의 정확도가 떨어질 수 있음. 
# 따라서 모델을 훈련할 때는 일반적으로 fp32(32비트 부동 소수점 형식)를 사용하고,
# 추론(inference) 단계에서는 fp16을 사용하여 연산 속도를 높이는 경우가 많음.  => 즉 추론 하는 단계 -> 이미지생성
torch_dtype=torch.float32, #torch 파일로 만들돼, 16진수 소수점 으로 만든다. 
)
prompt = "" # background를 입력해서 배경으로 채워줘 라고 하는 것.
#image and mask_image should be PIL images. => 둘다 PIL 이미지로 만듣어서 처리 해야합니다.
#The mask structure is white for inpainting and black for keeping as is => mask된 부분은 흰색 바깥은 검은색으로 처리하기
new_image = pipe(prompt=prompt, image=P_img, mask_image=mask_image).images[0] 
new_image.save("removed.png")
