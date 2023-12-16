import torch
from torchvision import transforms
from transformers import TrainingArguments, Trainer
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification,VideoMAEConfig
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2
def load_model():
    image_processor = VideoMAEImageProcessor.from_pretrained("checkpoint-4440")
    trained_model = VideoMAEForVideoClassification.from_pretrained("checkpoint-4440")
    trained_model = trained_model.to(device)
    image_processor1 = VideoMAEImageProcessor.from_pretrained("checkpoint-158")
    trained_model1 = VideoMAEForVideoClassification.from_pretrained("checkpoint-158")
    trained_model1 = trained_model1.to(device1)
    return image_processor,trained_model,image_processor1,trained_model1

processor,model,processor1,model1 = load_model()

l2b = {0:"B01:손동작/머리",1:"B02:손동작/얼굴",2:"B03:손동작/몸긁기",
       3:"B04:손동작/손톱",4:"B05:머리동작/고개흔들기",5:"B06:머리동작/좌우흔들기"
       ,6:"B07:머리동작/숙이기",7:"B08:팔동작/뒷짐",8:"B09:팔동작/무의미반동",9:"B10:자세/좌우흔들기",10:"B11:자세/비스듬히",11:"B12:자세/비비꼬기"}
def korea_txt(text,frame):
    img = np.full((50,300,3),0, np.uint8)
    font = ImageFont.truetype("NanumGothic.ttf",20)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((0,0),text,(255,255,255),font=font)
    img = np.array(img_pil)
    height, width, channels = img.shape
    offset = np.array((40, 50)) #top-left point from which to insert the smallest image. height first, from the top of the window
    frame[offset[0]:offset[0] + height, offset[1]:offset[1] + width] = img
    # return img
def run_inference(video):
    """Utility to run inference given a model and test video.
    
    The video is assumed to be preprocessed already.
    """
    # (num_frames, num_channels, height, width)
    video = torch.tensor(video).reshape(10,3,224,224)

    # perumuted_sample_test_video = video.permute(1, 0, 2, 3)
    perumuted_sample_test_video = video.permute(0, 1, 3, 2)

    inputs = {
        "pixel_values": perumuted_sample_test_video.unsqueeze(0)
    }
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    return logits.argmax()

# img = read_img()
# outputs, img = run_inference(img)
# keypoint_img = draw_keypoints(output, img)
def run_inference1(video):
    """Utility to run inference given a model and test video.
    
    The video is assumed to be preprocessed already.
    """
    # (num_frames, num_channels, height, width)

    video = torch.tensor(video).reshape(10,3,224,224)

    # perumuted_sample_test_video = video.permute(1, 0, 2, 3)
    perumuted_sample_test_video = video.permute(0, 1, 3, 2)

    inputs = {
        "pixel_values": perumuted_sample_test_video.unsqueeze(0)
    }
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device1) for k, v in inputs.items()}
    
    # forward pass
    with torch.no_grad():
        outputs = model1(**inputs)
        logits = outputs.logits

    return logits.argmax()
def pose_estimation_video(filename):
    cap = cv2.VideoCapture(filename)
    set_frame = 24
    cap.set(cv2.CAP_PROP_POS_FRAMES,set_frame)
    # VideoWriter for saving the vide
    img_list = []
    label_list = []
    frame_rate = 6
    i = 0
    output = -2
    seq = 6
    while cap.isOpened():
        (ret, frame) = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if seq == frame_rate:
                seq = 0
                img_list.append(processor(frame)['pixel_values'])
            if len(img_list) == 10:
                label = run_inference1(img_list)
                if label == 1:
                    output = run_inference(img_list)
                    label_list.append(l2b[output.item()])
                else:
                    label_list.append("부정 아님")
                img_list = []
                
            # cv2.putText(frame,str(output+1), 
            #     bottomLeftCornerOfText, 
            #     font, 
            #     fontScale,
            #     fontColor,
            #     thickness,
            #     lineType)
            # frame = cv2.resize(frame, (int(cap.get(3)), int(cap.get(4))))
            # out.write(frame)
            # cv2.imshow('Pose estimation', frame)
        else:
            break     
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        seq +=1
    cap.release()
    i_x= 0
    i = 0
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 24)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('ted_video_output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    print(len(label_list))
    seq = 6
    while True:
        (ret, frame) = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if seq == frame_rate:
                seq=0
                i_x +=1
            if i_x == 10:
                i_x = 0
                i +=1
            # try:
                # cv2.putText(frame,str(label_list[i]), 
                #     bottomLeftCornerOfText, 
                #     font, 
                #     fontScale,
                #     fontColor,
                #     thickness,
                #     lineType)
            if len(label_list)-1 >= i :
                korea_txt(str(label_list[i]),frame)
            else:print(i)
            # except:print(i)
            frame = cv2.resize(frame, (int(cap.get(3)), int(cap.get(4))))
            out.write(frame)
            # cv2.imshow('Pose estimation', frame)
        else:
            break
        seq +=1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
pose_estimation_video("ted.mp4")