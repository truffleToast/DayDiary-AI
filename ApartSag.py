from PIL import Image
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import numpy as np
import torch

model = FastSAM('FastSAM-x.pt')  # or FastSAM-x.pt
source = "./apart.jpg"


everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
# model.compile2 -> 어디서 실행할 것인가 , cpu, 객체 둘
prompt_process = FastSAMPrompt(source, everything_results, device='cpu')
#사용자가 지정한 위치에서 모든 범위를 확인해야함 
# points default [[0,0]] [[x1,y1],[x2,y2]] 포인트의 default는 [[0,0]] , [[x1, y1] , [x2,y2]] 
# point_label default [0] [1,0] 0:background, 1:foreground
# point_lable default는 0: 배경 1: 배경이 아닌 객체 탐지
prompt_process.point_prompt(points=[[450, 211]], pointlabel=[1])
res = prompt_process.results #마스킹한 데이터를 담고있는 객체
masked_array = np.array(res[0].masks.data[0]) # 객체에서 0번 데이터를 numpy_array로 처리
mask_image = Image.fromarray(masked_array) #PIL 이미지로 변경
mask_image.save("./test6.png")


