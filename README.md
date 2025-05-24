# 📘 Course Overview

This repository was created as part of a course delivered to ITI students. It provides a practical, hands-on journey through the MLOps lifecycle, from model development to deployment and monitoring. Each module is designed to build upon the previous, offering a comprehensive understanding of MLOps practices.



# 🤖 MLOps Introduction
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


## 🗂️ Repository Structure

The course is organized into sequential modules that reflect the stages of a typical MLOps pipeline:

```
MLOps-Course/
├── 0_versioning/         # Experiment tracking using MLflow
├── 1_packaging/          # Techniques for packaging ML models into reusable components
├── 2_serving/            # Methods for serving models using FastAPI & other techniques
├── 3_unit_testing/       # Practices for unit testing ML code to ensure reliability
├── 4_containerization/   # Using Docker to containerize ML applications
├── 5_ci_cd/              # Implementing Continuous Integration and Deployment pipelines using GitHub Actions
├── 6_load_testing/       # Assessing the performance of ML services under load using locust
├── 7_monitoring/         # Monitoring deployed models for performance and drift with Graphana & Prometheus
└── README.md             # Course overview and instructions
```



Each module includes hands-on labs and practical examples to reinforce learning.



## 🚀 Getting Started

To begin working with this course:([GitHub Docs][1])

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Heba-Atef99/MLOps-Course.git
   cd MLOps-Course
   ```



2. **Set up your environment**:
   Ensure you have the necessary tools installed, such as Python, Docker, and any other dependencies specified in the module requirements.

3. **Explore the modules**:
   Navigate through each module folder in order, starting with `0_versioning`, to follow the course progression.



## 🛠️ Lab Boilerplate

For those interested in applying the labs or seeking a starting point for their own MLOps projects, we have prepared a boilerplate repository:

👉 [MLOps-Course-Labs](https://github.com/Heba-Atef99/MLOps-Course-Labs)

This companion repository provides a foundational structure to help you implement the concepts covered in this course.


