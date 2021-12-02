# :zero: Get Our Results

1. Conda Environments *(Recommended)* 

   ```bash
   $ conda create -n lightweight  python=3.7.11
   $ conda activate lightweight 
   $ pip install -r requirements.txt
   ```

2. Train model : `train.py`
   
   ```bash
   python train.py --model configs/model/mobilenetv3_cut.yaml --data configs/data/taco_im156.yaml
   ```

3. Evaluate model : inference.py

   ```bash
   python inference.py --model_dir /opt/ml/code/exp/latest --weight best.pt --img_root /opt/ml/data/test/
   ```
   
# :one: The purpose of the project

1. 쓰레기 분류기를 만들며, 실제 로봇에 탑재가능한 경량화 된 모델을 제작한다.
2. 경량화 기법을 도입하여, 동일 파라메터 대비 고성능 모델, 동일 성능 대비 적은 파라메터를 가지는 모델을 탐색한다.
3. 해당 목적달성을 위해 AutoML 라이브러리인 optuna 를 사용하여 search space를 설정, 원하는 모델을 탐색한다.


# :two: Project Outline

## 2.0. Topic


## 2.1. Envirionments


## 2.2. Collaboration Tools


## 2.3. Structure


## 2.4. Benefits



# :three: Carry Out the Project

## 3.1. Problem Definition

-


## 3.2. Experiments and Observation

-


# :four: Results of Project Execution

## 4.1. Final Model

-


