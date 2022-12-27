
#  InfantinO
In a medical emergency situations, the description of the patient's situation, especially before opertating physical examination, plays an important role. 

For the case of infants, therefore, parents' or the protector's explanation plays an important role to the reamining medical procedure.

However, with un-sufficient or wrong informations, the remaining procedure might suffer waste of time or catastrophic results. To prevent these situations, the need of tools to assist precise measurment rises.

InfantinO, the Online learning framework for infants' facial expression recognition using DL models, can be the solution to this task.

## Project Procedure

  (1) Collect infant face expression data
  -
  - Divide face expression to 7 categories
    
    {angry, disgust, fear, happy, sad, surprise, neutral}

  - Collect Dataset through google
  - Using Pytorch conduct various Data Augmenations.


  (2) Train Feature Extractor through data
  -

  - Using timm, create 'efficientnet-bo' model with 64 output layers
  - Add classifying layer with 7 outputs
  - Train the model with infant data
  - Extract only feature extration structure and weights of the model
  - Connect the Feature Extactor with scikitlearn MLP Classifier

  (3) Conduct Online learning through MLFlow
  - 
  - Using MLFlow, train the model with Oniline Learning method
  - Online Learning enables superfast re-training without catastrophic forgetting

  (4) Performance
  -  

  - Distinguish face expression through softmax

    ![Alt text](https://velog.velcdn.com/images/jugjug/post/0e3adb7e-a59b-40a2-91e0-2163a588558a/image.png)


  
## How to transfer to other domains

To use for infant face classification, run the following command.


    python3 generic_onlinelearner.py


To use for other domain, place your images to 

[Train Dataset](https://github.com/hahajjjun/InfantinO/tree/master/modeling/src/data/online_raw), with all classes included.

Train method is same as above.

    python3 generic_onlinelearner.py
## Relevant Materials

- [continual-learning-in-practice](https://assets.amazon.science/8e/63/5bfdb1bb419491ba26ce3b219369/continual-learning-in-practice.pdf)
- [bentoml docs](https://docs.bentoml.org/en/latest/)
- [MLflow docs](https://mlflow.org/)


## Authors
![Alt text](https://velog.velcdn.com/images/jugjug/post/46b435c4-765f-40c2-b4fb-e6a68f490bca/image.png)

## Team InfantinO
- [YBIGTA 20기 박준하👑](https://github.com/hahajjjun)
- [YBIGTA 20기 이주훈](https://github.com/giovanlee)
- [YBIGTA 20기 정정훈](https://github.com/JugJugIE)
- [YBIGTA 21기 국호연](https://github.com/brightsky77)
- [YBIGTA 21기 박제연](https://github.com/bonapark00)
- [YBIGTA 21기 장동현](https://github.com/rroyc20)
