# 사용한 모델들.
<hr>

[![VIDEO-MAE](https://arxiv.org/pdf/2203.12602.pdf)](https://github.com/MCG-NJU/VideoMAE)(https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics)
<br>
[![LA-GCN](https://arxiv.org/pdf/2305.12398v1.pdf)](https://github.com/damnull/lagcn)

## 모델 성능
<hr>

### VIDEO-MAE
<img width="955" alt="Screenshot 2023-12-16 at 20 00 00" src="https://github.com/new-Sunset-shimmer/Gesture/assets/77263106/94d2eb82-d552-4312-b80f-a2dd35817db2">

### LA-GCN
<img width="955" alt="Screenshot 2023-12-16 at 20 00 08" src="https://github.com/new-Sunset-shimmer/Gesture/assets/77263106/afe07a5d-62de-495b-9a2e-d17b87939357">
![Screenshot_2023-11-16_at_22 25 31](https://github.com/new-Sunset-shimmer/Gesture/assets/77263106/bdd0df94-fb77-46f8-af38-19d435b8aecf)


## 실행
### 모델 학습
VIDEO-MAE
    -python ./모델/VIDEO-MAE.py
LA-GCN
    -python ./모델/models/LAGCN-master/main.py --config ./모델/models/LAGCN-master/configs/nwucla/joint_motion.yaml --work-dir ./모델/work_dir/motions/csub/lagcn_joint --device 0
### 상용화
VIDEO-MAE
  -python ./실습/VIDEO-MAE.py  
LA-GCN
  -python ./실습/LA_GCN.py  

## 상용화 비쥬얼화
![c7bdac6c3a593679](https://github.com/new-Sunset-shimmer/Gesture/assets/77263106/f3c17646-4a90-47e5-8c36-7626ff93438e)
![스크린샷 2023-12-11 14 50 09](https://github.com/new-Sunset-shimmer/Gesture/assets/77263106/bd4c715f-a1b2-4c72-81be-acda15781989)

