from PIL import Image
# from flask_cors import CORS, cross_origin #flask 보안 및 cross 
import cv2 # opencv
from diffusers import StableDiffusionInpaintPipeline # 인페인팅 모델 
from rembg import remove
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import numpy as np
import torch, torchvision
#FastSam 모델 활용하기
model = FastSAM('FastSAM-x.pt')  # or FastSAM-x.pt
source = "./urbanbrush-20221108214712319041.jpg"
image1 = Image.open(source)
everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
prompt_process = FastSAMPrompt(source, everything_results, device='cpu')
prompt_process.point_prompt(points=[[500, 350]], pointlabel=[1])
res = prompt_process.results #마스킹한 데이터를 담고있는 객체
masked_array = np.array(res[0].masks.data[0])
mask_image = Image.fromarray(masked_array) #PIL 객체
mask_image.save("TEST.png")