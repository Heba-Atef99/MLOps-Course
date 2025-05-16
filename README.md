# MLOps-Course
<h3>MLOps is the process of taking ML models from an Idea into production in the shortest possible way with as little risk as possible.</h3>

![image](https://github.com/user-attachments/assets/4e6f2505-768f-439b-b0bf-4022e6d3c135)


## MLOPs vs DevOps
MLOps is the application of DevOps principles and practices to machine learning models.


## Why MLOps:
### 1. Experimentation:
ML = trial and error. You try different models, features, or set of parameters
MLOps help in tracking what you’ve tried and what was best
Ex. If you tested 5 models to predict sales. MLOps lets you compare them and remember which one was best.

### 2. Track Metrics:
ML models produce accuracy, precision, recall, etc. 
MLOps helps you automatically log and compare these metrics over time
	EX. if your model accuracy dropped from 90% to 70%. MLOps helps you spot and fix that rapidly

### 3. Automate Validation and deployment:
Every time you train a model, it gets tested and deployed automatically without manual effort using pipelines - sequence of automated steps.
Validation tests examples: Accuracy > 90%? No Missing Values? etc.

### 4. Model Drift:
Detect when your model starts giving worse results overtime
Ex. a model to predict which customer will cancel their subscription but you added new products → so customer behaviour will change so the 

### 5. Data Drift:
The data that your model trained on, is different from data in production
Ex. your model trained on English reviews, suddenly the input changes and become spanish



