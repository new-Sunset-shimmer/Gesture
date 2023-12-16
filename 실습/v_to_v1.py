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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2

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

def load_model():
    model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
    model_class = Model_gcn()
    # model_class = model_class.load_state_dict('./runs-137-44936.pt')
    weights = torch.load('./runs-137-44936.pt')
    weights = OrderedDict([[k.split('module.')[-1], v.cuda(device1)] for k, v in weights.items()])
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
    model.float().eval()
    model_class.eval()
    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)
        model_class.to(device1)
    return model,model_class

model,model_class = load_model()

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
        skeletons = skeletons.float().cuda(device1)

    with torch.no_grad():
      output = model_class(skeletons)
      output1 = soft_max(output[1])
      output = soft_max(output[0])
    return output.argmax(),output1
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
    while cap.isOpened():
        (ret, frame) = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if seq == frame_rate:
                seq = 0
                output, frame = run_inference(frame)
                frame,output1 = draw_keypoints(output, frame)
                if(len(output1)==0):
                    output1 = [np.zeros((18,3)),0]
                skeleton_list.append(output1[0].reshape(18,3))
                i += 1
            if i == 10:
                label,lable_temp = class_inference(skeleton_list)
                label_list.append(label)
                skeleton_list = []
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
    out = cv2.VideoWriter('ice_skating_output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
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
            output, frame = run_inference(frame)
            frame,output1 = draw_keypoints(output, frame)
            if i_x == 10:
                i_x = 0
                i +=1
            try:
                cv2.putText(frame,str(label_list[i]), 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
            except:
                print(i)
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