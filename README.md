#  InfantinO

InfantinO is an online learning framework targeting infants' facial expression classification which provides management about unconventional machine learning lifecycle including annotation & model update. Label ambiguity, catastrophic forgetting under frequent domain shift, insufficient dataset is a main challenge of infants' facial expression classification. These challenges can be overcome with our framework, which utilizes domain adaptation & online learning.


## About our team
|Github||
|---|---|
|[YBIGTA 20기 박준하](https://github.com/hahajjjun)|Lead, Modeling, Data pipeline|
|[YBIGTA 20기 이주훈](https://github.com/giovanlee)|Modeling, Data acquisition|
|[YBIGTA 20기 정정훈](https://github.com/JugJugIE)|Modeling, Dataset acquisition|
|[YBIGTA 21기 국호연](https://github.com/brightsky77)|Data pipeline, Web Backend|
|[YBIGTA 21기 박제연](https://github.com/bonapark00)|Data pipeline, Web Backend|
|[YBIGTA 21기 장동현](https://github.com/rroyc20)|Web Frontend, Annotation GUI|

## Usage
    
``` python
from generic_onlinelearner import CustomTrainer
trainer = CustomTrainer()
trainer.run_full() # Train feature extractor
trainer.run_partial() # Train online learner : single image is required for model update
with open('recent_model_uri.log', 'r') as f:
    model_uri = f.readline()
img_path = 'example_image.png'
prediction, uncertainty = trainer.inference_partial(path=img_path, model_uri=model_uri) # Inference for single image input
```

Transfer to other domain: place your images to 
[Train Dataset](https://github.com/hahajjjun/InfantinO/tree/master/modeling/src/data/online_raw), with all classes included.

## Project Procedure

  (1) Collect infant face expression data
  - Facial expression to 7 categories : {angry, disgust, fear, happy, sad, surprise, neutral}
  - Collect Dataset through google
  - Image augmentation
  
  (2) Train Feature Extractor
  - Using timm, create 'efficientnet-b0' model with 64 output layers
  - Add classifying layer with 7 outputs
  - Train the model with infant data
  - Extract only feature extration structure and weights of the model
  - Connect the Feature Extactor with scikitlearn MLP Classifier

  (3) Conduct Online learning through MLFlow
  - Using MLFlow, train the model with Oniline Learning method
  - Online Learning enables superfast re-training without catastrophic forgetting

    ![baby image](https://velog.velcdn.com/images/jugjug/post/0e3adb7e-a59b-40a2-91e0-2163a588558a/image.png)
  
    
## Relevant Materials

- [continual-learning-in-practice](https://assets.amazon.science/8e/63/5bfdb1bb419491ba26ce3b219369/continual-learning-in-practice.pdf)
- [bentoml docs](https://docs.bentoml.org/en/latest/)
- [MLflow docs](https://mlflow.org/)
