# 사용한 모델들.
<hr>

[![VIDEO-MAE]](https://arxiv.org/pdf/2203.12602.pdf)[![코드]](https://github.com/MCG-NJU/VideoMAE)[![Hugginface]](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics)
<br>
[![LA-GCN]](https://arxiv.org/pdf/2305.12398v1.pdf)[![코드]](https://github.com/damnull/lagcn)

## 모델 성능
<hr>

### VIDEO-MAE
<img width="955" alt="Screenshot 2023-12-16 at 20 00 00" src="https://github.com/new-Sunset-shimmer/Gesture/assets/77263106/94d2eb82-d552-4312-b80f-a2dd35817db2">
<br>
### LA-GCN
<img width="955" alt="Screenshot 2023-12-16 at 20 00 08" src="https://github.com/new-Sunset-shimmer/Gesture/assets/77263106/afe07a5d-62de-495b-9a2e-d17b87939357">
<img width="955" alt="Screenshot 2023-12-16 at 20 00 08" src="https://github.com/new-Sunset-shimmer/Gesture/assets/77263106/bdd0df94-fb77-46f8-af38-19d435b8aecf">


## 실행
### 모델 학습
VIDEO-MAE
    <br>
    -python ./모델/VIDEO-MAE.py
    <br>
LA-GCN
    <br>
    -python ./모델/models/LAGCN-master/main.py --config ./모델/models/LAGCN-master/configs/nwucla/joint_motion.yaml --work-dir ./모델/work_dir/motions/csub/lagcn_joint --device 0
    
### 상용화
VIDEO-MAE
    <br>
    <b>학습된 모델 필요. 본 깃허브 페이지에는 없음</b> <br>
  -python ./실습/VIDEO-MAE.py  
      <br>
LA-GCN
    <br>
    <b>학습된 모델 필요. 본 깃허브 페이지에는 없음</b> <br>
  -python ./실습/LA_GCN.py  

## 상용화 비쥬얼화
![스크린샷 2023-12-11 14 50 09](https://github.com/new-Sunset-shimmer/Gesture/assets/77263106/bd4c715f-a1b2-4c72-81be-acda15781989)
<img width="955" alt="Screenshot 2023-12-16 at 20 00 00" src="https://github.com/new-Sunset-shimmer/Gesture/assets/77263106/14d85700-b3d0-4bf4-b08d-2b30c7faf30d">


