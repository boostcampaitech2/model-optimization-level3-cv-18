# 0️⃣ Get Our Results

1. Conda Environment *(Recommended)*
   
    ```bash
    $ conda create -n lightweight python=3.7.11
    $ conda activate lightweight 
    $ pip install -r requirements.txt
    ```
    
2. Train model : `train.py`
   
    ```bash
    python train.py --model configs/model/mobilenetv3_cut.yaml --data configs/data/taco_im156.yaml
    ```
    
3. Evaluate model : `inference.py`
   
    ```bash
    python inference.py --model_dir /opt/ml/code/exp/latest --weight best.pt --img_root /opt/ml/data/test/
    ```
    

# 1️⃣ Overview

## 1.0. Aim

1. Make the garbage classifier and a lightweight model that can be mounted on real industrial robots.
2. Explore a high performance model compared to the same parameters and a model with fewer parameters compared to the same performance by using optimization techniques.
3. Use `optuna`, an AutoML llibrary, to configure search space and hyperparameters.

# 2️⃣ Project Outline

## 2.0. Topic

Recently, AI has shown remarkable performance in various industries. For example, there is an AI, called [supercube](https://www.superbin.co.kr/new/contents/supercube.php), which decides whether trash is recyclable or not. It is crucial that these models complete tasks rapidly enough to be actually deployable.

So, in this project, we build a light-weight AI model that can be mounted on real industrial robots. 

## 2.1. Envirionment

V100 GPU (32GB)

## 2.2. Collaboration Tools

We use Github, Notion, WandB for our collaboration.

## 2.3. Structure

- I/O
    - Input: Cropped images from TACO dataset (COCO format)
    - Output: Classification score <img src="https://render.githubusercontent.com/render/math?math=(F_1)"/> and Inference time <img src="https://render.githubusercontent.com/render/math?math=(t)"/>.
        - Evaluation metric
        - <img src="https://render.githubusercontent.com/render/math?math=score = 0.5*score_{\text{submit time}}" /> + <img src="https://render.githubusercontent.com/render/math?math=score_{\text{F}_1}" />
            - <img src="https://render.githubusercontent.com/render/math?math=score_{\text{submit time}}=\frac{thismodel_{\text{submit time}}}{baseline_{\text{submit time}}}}"/>
            - <img src="https://render.githubusercontent.com/render/math?math=score_{\text{F}_1}=\text{sigmoid}\big(20*(baseline_{\text{F}_1} - this model_{\text{F}_1})\big)"/>
- Model that we use.
    - Model: Pruned MobileNetV3

## 2.4. Benefits

We often seek ways to maintain or improve model performance while facing certain constraints. In this competition, we learned how to find a model that meets the desired conditions. Specifically, we use model optimization techniques in our final project to build a lightweight classifier that can be deployed on the client side.

# 3️⃣ Carry Out the Project

## 3.1. Problem Definition

![LB-result](https://user-images.githubusercontent.com/87659486/144399995-bcb93cae-97ae-4b20-bf65-6d81f599b9bd.png)

The figure above represents the target value. F1-Score should be 0.7 or more and Time Score should be within 60.

- Since the `TACO` dataset is designed for segmentation/detection tasks, use images cropped with a bounding box to convert to classification tasks.
- Class imbalnce problem, width and height of each image are different.
- For purposes of this competition, external datasets are prohibited.
- Loading weights from pertained models is permitted.

## 3.2. Experiments and Observation

```jsx

```

# 4️⃣ Results of Project Execution

## 4.1. Final Model

| Key | Value | 비고 |
| --- | --- | :-: |
| Backbone  | MobileNetV3 | **[1]** |
| Optimizer  | SGD |  |
| Init Learning Rate | 0.1 |  |
| LR Scheduler | OneCycleLR |  |
| Loss | Custom Loss (fixed) |  |
| Image Size | 156 |  |
| Image augmentation | RandAugment | **[2]** |
| Batch Size | 64 |  |
| Epochs | 150 / 200 |  |
| Val Ratio | 0.1 / 0.5 |  |
| FP16 | True |  |

**[1]** MobileNetV3의 파라미터를 축소해서 사용

**[2]** 학습 데이터에만 적용

## 4.2. Final Metric in the Competition

|  | Target LB | Public LB | Private LB  |
| --- | --- | --- | --- |
| Score | 1.1950 | 1.1676 |  |
|  F1 Score | 0.7000 | 0.7012 |  |
| Time   | 60.0000 | 58.1225 |  |

# 5️⃣ Participants

| Name | Github | Role |
| --- | --- | --- |
| 김서기 (T2035) | [link](https://github.com/seogi98) | Model experiment(MobileNetV3) |
| 김승훈 (T2042) | [link](https://github.com/lead-me-read-me) | Model experiment(MobileNetV3) |
| 배민한 (T2260) | [link](https://github.com/Minhan-Bae) | Model experiment(MobileNetV3) |
| 손지아 (T2113) | [link](https://github.com/oikosohn) | Model experiment(MobileNetV3, ShuffleNetV2), Set the WandB log |
| 이상은 (T2157) | [link](https://github.com/lisy0123) | Model experiment(MobileNetV3) |
| 조익수 (T2213) | [link](https://github.com/projectcybersyn2) | Model experiment(MobileNetV3) |
