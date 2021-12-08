# 0️⃣ Get Our Results
1. Conda Environment *(Recommended)*

	```bash
	$ conda create -n lightweight python=3.7.11
	$ conda activate lightweight
	$ pip install -r requirements.txt
	```
	
2. Train model: `train.py`
	
	```bash
	$ python train.py \
		--model configs/model/mobilenetv3-480.yaml \
		--data configs/data/taco-im156-ep200.yaml
	```
	
3. Evaluate model: `inference.py`

	```bash
	python inference.py \
		--model_dir exp/latest \
		--weight best.pt \
		--img_root data/test/
	```
	
# 1️⃣ Overview
## 1.0. Objectives
1. Build a lightweight model that can contribute to automated garbage collection and classification.
2. Optimize the neural model by model pruning and hyperparameter tuning.
3. Use `optuna`, an AutoML llibrary, to configure search space and hyperparameters.

# 2️⃣ Project Outline

## 2.0. Description

Recently, AI has shown remarkable performance across various industries. For example, there is an AI, called [supercube](https://www.superbin.co.kr/new/contents/supercube.php), that determines whether a waste material is recyclable or not. It is crucial that these models complete tasks rapidly enough to be actually deployable.

So, in this project, we build a light-weight classifier that can be mounted on real industrial robots. 

## 2.1. Environment

V100 GPU (32GB)

## 2.2. Collaboration Tools

Github, Notion,  WandB

## 2.3. Structure

- I/O
    - Input: Cropped images from TACO dataset (COCO format)
    - Output: Classification score ( **F<sub>1</sub>** ) and Inference time(***t*** ).
        - Evaluation metric
            - score = 0.5*score<sub>submit time</sub> + score<sub>{F<sub>1</sub>}</sub>
                - score<sub>submit time</sub>= this model<sub>submit time</sub> &divide; baseline<sub>submit time</sub>
                - score<sub>F<sub>1</sub></sub> = sigmoid(20*(baseline<sub>F<sub>1</sub></sub> - this model<sub>F<sub>1</sub></sub>))
- Our model: (Compressed) MobileNetV3

## 2.4. Benefits

We often seek to maintain or improve model performance while facing certain constraints. In this competition, we studied how to find a model that meets the required conditions. Our goal is to apply the techniques we learned along the way to build a lightweight, client-side model for our final project.


# 3️⃣ Execution Details

## 3.1. Problem Definition

![Untitled](https://user-images.githubusercontent.com/87659486/144399995-bcb93cae-97ae-4b20-bf65-6d81f599b9bd.png)

Target performance level. As depicted by the figure above, the `F1 Score` should be 0.7 or higher, and the `Time Score` should be below 60.

- Since the `TACO` dataset was originally designed for segmentation/detection tasks, crop the images along the bounding box to make them suitable for classification.
- Class imbalance problem; width and height of each image are different.
- For purposes of this competition, use of external datasets is prohibited.
- Loading weights from pretrained models is permitted.

## 3.2. Experiments and Observation

MobileNetV3 Tuning

- Apply RandAugment (n_level=2) in all experiments.

	| No. | Nums of Layers | Parameters | Image Size | Epoch | Score | F1 score | Time |
	| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
	| 1 | 179 | 2,127,374 | 224 | 100 | `1.5460` | `0.6880` | `84.9850` |
	| 2 | 163 | 1,741,254 | 224 | 100 | `1.7950` | `0.6530` | `92.9260` |
	| 3 | 79 | 294,854 | 156 | 100 | `1.2950` | `0.6870` | `63.0780` |
	| 4 | 79 | 294,854 | 156 | 150 | `1.1732` | `0.7005` | `58.3412` |
	| 5 | 79 | 294,854 | 156 | 200 | `1.1676` | `0.7012` | `58.1225` |


# 4️⃣ Results

## 4.1. Final Model

| Key | Value | Remarks |
| --- | --- | :-: |
| Backbone  | MobileNetV3 | **[1]** |
| Optimizer  | SGD |  |
| Init Learning Rate | 0.1 |  |
| LR Scheduler | OneCycleLR |  |
| Loss | Custom Loss (fixed) |  |
| Image Size | 156 |  |
| Augmentation | RandAugment | **[2]** |
| Batch Size | 64 |  |
| Epochs | 200 |  |
| Val Ratio | 0.1  |  |
| FP16 | *True* |  |
> **[1]** Compressed model.  
> **[2]** Applied on train set only.

## 4.2. Final Metric in the Competition

We achieved the target performance level on both Public and Private Leaderboards (LBs).

- **The final submitted model :** the best performing model on Public LB.

	|  | Target LB | Public LB | Private LB  |
	| --- | :-: | :-: | :-: |
	| Score | `1.1950` | `1.1676` | `1.1804` |
	|  F1 Score | `0.7000` | `0.7012` | `0.6986` |
	| Time   | `60.0000` | `58.1225` | `58.1225` |


# 5️⃣ Participants

| Name | Github | Role |
| :-: | :-: | --- |
| 김서기 (T2035) | [link](https://github.com/seogi98) | Model experiment(MobileNetV3) |
| 김승훈 (T2042) | [link](https://github.com/lead-me-read-me) | Model experiment(MobileNetV3) |
| 배민한 (T2260) | [link](https://github.com/Minhan-Bae) | Model experiment(MobileNetV3) |
| 손지아 (T2113) | [link](https://github.com/oikosohn) | Model experiment(MobileNetV3, ShuffleNetV2), Set the WandB log |
| 이상은 (T2157) | [link](https://github.com/lisy0123) | Model experiment(MobileNetV3) |
| 조익수 (T2213) | [link](https://github.com/projectcybersyn2) | Model experiment(MobileNetV3) |
