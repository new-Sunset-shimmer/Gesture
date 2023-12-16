import torch
from torchvision import transforms
import random
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from lagcn import Model as Model_gcn
import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import OrderedDict
import math
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
from torchvision import transforms
from transformers import TrainingArguments, Trainer
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification,VideoMAEConfig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2

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
        outputs = trained_model(**inputs)
        logits = outputs.logits

    return logits.argmax()
def rand_view_transform(X, agx, agy, s):
    agx = math.radians(agx)
    agy = math.radians(agy)
    Rx = np.asarray([[1,0,0], [0,math.cos(agx),math.sin(agx)], [0, -math.sin(agx),math.cos(agx)]])
    Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0,1,0], [math.sin(agy), 0, math.cos(agy)]])
    Ss = np.asarray([[s,0,0],[0,s,0],[0,0,s]])
    X0 = np.dot(np.reshape(X,(-1,3)), np.dot(Ry,np.dot(Rx,Ss)))
    X = np.reshape(X0, X.shape)
    return X
def get(value):
    random.random()
    agx = 0
    agy = 0
    s = 1.0

    center = value[0,1,:]
    value = value - center
    scalerValue = rand_view_transform(value, agx, agy, s)

    scalerValue = np.reshape(scalerValue, (-1, 3))
    scalerValue = (scalerValue - np.min(scalerValue,axis=0)) / (np.max(scalerValue,axis=0) - np.min(scalerValue,axis=0))
    scalerValue = scalerValue*2-1

    scalerValue = np.reshape(scalerValue, (-1, 18, 3))

    data = np.zeros( (52, 18, 3) )

    value = scalerValue[:,:,:]
    length = value.shape[0]

    idx = np.linspace(0,length-1,52).astype(np.int64)
    data[:,:,:] = value[idx,:,:] # T,V,C
    data_motion = np.zeros_like(data)
    data_motion[:-1, :, :] = data[1:, :, :] - data[:-1, :, :]
    data = data_motion
    data = np.transpose(data, (2, 0, 1))
    C,T,V = data.shape
    # data = np.reshape(data,(C,T,V,1))
    data = np.reshape(data,(1,C,T,V,1))
    return data
mp_pose = mp.solutions.pose
def load_model():
    
    image_processor = VideoMAEImageProcessor.from_pretrained("checkpoint-158")
    trained_model = VideoMAEForVideoClassification.from_pretrained("checkpoint-158")
    trained_model = trained_model.to(device1)
    # Setting up the Pose function.
    model = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

    # Initializing mediapipe drawing class, useful for annotation.
    mp_drawing = mp.solutions.drawing_utils 
    
    model_class = Model_gcn()
    # model_class = model_class.load_state_dict('./runs-137-44936.pt')
    weights = torch.load('./runs-137-44936.pt')
    weights = OrderedDict([[k.split('module.')[-1], v.cuda(device)] for k, v in weights.items()])
    try:
        model_class.load_state_dict(weights)
    except:
        state = model_class.state_dict()
        diff = list(set(state.keys()).difference(set(weights.keys())))
        print('Can not find these weights:')
        for d in diff:
            print('  ' + d)
        state.update(weights)
        model_class.load_state_dict(state)

    # Put in inference mode
    model_class.eval()
    trained_model.eval()
    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model_class.to(device)
    return model,model_class,mp_drawing,image_processor,trained_model

model,model_class,mp_drawing,image_processor,trained_model = load_model()

def run_inference(image):
    # Resize and pad image
    image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (567, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image) # torch.Size([3, 567, 960])
    if torch.cuda.is_available():
      image = image.half().to(device)
    # Turn image into batch
    image = image.unsqueeze(0) # torch.Size([1, 3, 567, 960])
    with torch.no_grad():
      output, _ = model(image)
    return output, image
soft_max = torch.nn.Softmax(dim=1)
def class_inference(skeletons):
    skeletons = torch.tensor(skeletons,dtype=torch.float32)
    skeletons = get(skeletons)
    skeletons = torch.tensor(skeletons,dtype=torch.float32)
    with torch.no_grad():
        skeletons = skeletons.float().cuda(device)

    with torch.no_grad():
      output = model_class(skeletons)
      output1 = soft_max(output[1])
      output = soft_max(output[0])
      x = (output+output1)/2
    # return output.argmax(),output1
    return x.argmax(),x
def draw_keypoints(output, image):
  output = non_max_suppression_kpt(output, 
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=model.yaml['nc'], # Number of Classes
                                     nkpt=model.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
  
  with torch.no_grad():
        output = output_to_keypoint(output)
  nimg = image[0].permute(1, 2, 0) * 255
  nimg = nimg.cpu().numpy().astype(np.uint8)
  nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
  skeletons = []
  for idx in range(output.shape[0]):
      plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
      skeletons.append(output[idx, 4:].T)
  return nimg,skeletons
# img = read_img()
# outputs, img = run_inference(img)
# keypoint_img = draw_keypoints(output, img)
def getxyz(x):
    list = []
    for _ in x:
        if type(_) != tuple:
            list.append([_.x,_.y,_.z])
        else:
            list.append([(_[0].x*_[1].x)/2,(_[0].y*_[1].y)/2,(_[0].z*_[1].z)/2])
    return list
def pose_estimation_video(filename):
    cap = cv2.VideoCapture(filename)
    # VideoWriter for saving the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 24)
    label_list=[]
    skeleton_list = []
    i = 0
    label = -2
    frame_rate = 6
    seq = 6
    image_list = []
    while cap.isOpened():
        (ret, frame) = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if seq == frame_rate:
                seq = 0
                results = model.process(frame)
                if(results.pose_landmarks==None):
                    results = np.zeros((18,3))
                else:
                    results = np.array(getxyz((results.pose_landmarks.landmark[0],results.pose_landmarks.landmark[2],results.pose_landmarks.landmark[5],
                                       results.pose_landmarks.landmark[7],results.pose_landmarks.landmark[8],
                                       (results.pose_landmarks.landmark[11],results.pose_landmarks.landmark[12]),
                                       results.pose_landmarks.landmark[11],results.pose_landmarks.landmark[12],results.pose_landmarks.landmark[13],
                                       results.pose_landmarks.landmark[14],results.pose_landmarks.landmark[15],results.pose_landmarks.landmark[16],
                                       results.pose_landmarks.landmark[23],results.pose_landmarks.landmark[24],results.pose_landmarks.landmark[25],
                                       results.pose_landmarks.landmark[26],results.pose_landmarks.landmark[27],results.pose_landmarks.landmark[28])))
                skeleton_list.append(results.reshape(18,3))
                image_list.append(image_processor(frame)['pixel_values'])
                i += 1
            if i == 10:
                label = run_inference1(image_list)
                if label == 1:
                    label,lable_temp = class_inference(skeleton_list)
                    label_list.append(l2b[label.item()])
                else:
                    label_list.append(["부정 아님"]) 
                skeleton_list = []
                image_list = []
                i = 0
            # frame = cv2.resize(frame, (int(cap.get(3)), int(cap.get(4))))
            # out.write(frame)
            # cv2.imshow('Pose estimation', frame)
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        seq +=1
    print("start")
    cap.release()
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 24)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(filename + '_output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    seq = 6
    i = 0
    i_x = 0
    while cap.isOpened():
        (ret, frame) = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if seq == frame_rate:
                seq = 0
                i_x += 1
            # results = model.process(frame)
            # mp_drawing.draw_landmarks(image=frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
            if i_x == 10:
                i_x = 0
                i +=1
            # try:
            #     cv2.putText(frame,str(label_list[i]), 
            #         bottomLeftCornerOfText, 
            #         font, 
            #         fontScale,
            #         fontColor,
            #         thickness,
            #         lineType)
            # except:
            #     print(i)
            if len(label_list)-1 >= i :
                korea_txt(str(label_list[i]),frame)
            else:print(i)
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
    
pose_estimation_video("real.mp4")