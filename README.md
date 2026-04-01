# AI-Assisted Academic Integrity Risk Detection System

This project is developed as part of our MCA final year industry project (Xebia).  
The goal of this system is to detect possible academic integrity risks using AI and data analysis.

---

## Project Overview

Traditional plagiarism tools mainly focus on copied content, but they fail to detect:
- AI-generated answers
- Sudden unusual improvement in student performance
- Behavioral inconsistencies

So, in this project, we tried to build a system that:
- Detects AI-generated text  
- Analyzes student performance behavior  
- Gives a risk indication instead of direct accusation  

---

##  Modules in the Project

###  Module 1 – AI Content Detection (Transformer-Based)

- Uses HuggingFace RoBERTa model  
- Detects whether content is AI-generated or human-written  

**Status:**
- Model training completed on Google Colab (GPU)
- Achieved around **98% accuracy**
- Integration part is still pending

**Challenges faced:**
- GPU requirement for training  
- Version conflicts in libraries  
- Difficult to run locally  

---

### Module 2 – AI Detection using ML

- Uses TF-IDF and machine learning  
- Classifies text as AI or Human  

✔ Lightweight  
✔ Easy to run locally  
✔ Gives good accuracy  

---

### Module 3 – Behavioral Anomaly Detection

- Uses Random Forest  
- Detects unusual patterns in student marks  

**Features used:**
- G1, G2 (previous marks)
- G3 (final marks)
- Absences
- Failures

**Approach:**
- Machine Learning + Rule-based logic  

---

## Results

### Module 2 Output

![Module 2 Result](images/module2_result.png)

- Good accuracy  
- Predicts AI vs Human with probability  

---

### Module 3 Output

![Module 3 Result](images/module3_result.png)

- Accuracy: around **86–87%**
- Detects:
  - Sudden grade jumps  
  - Unusual performance patterns  

---

### Module 1 Output

![Module 1 Result](images/module1_result.png)

- High accuracy during training (~98%)
- Not fully integrated due to system constraints  

---

## 💡 Key Features

- Detects AI-generated content  
- Detects abnormal academic behavior  
- Uses both ML and rule-based logic  
- Gives explainable output  

---

## What makes this project different?

- It does not directly accuse students  
- It only highlights **risk level**  
- Combines:
  - Text analysis  
  - Behavioral analysis  

---

## Method Used

- TF-IDF for text features  
- Random Forest for behavior analysis  
- Transformer model (RoBERTa) for deep learning  
- Rule-based logic for improving predictions  

---

## Example Outputs

- AI-written answer → detected as AI  
- Normal student → no issue  
- Sudden marks jump → flagged as anomaly  

---

## Future Work

- Combine all modules into one system  
- Create a dashboard (Streamlit)  
- Improve model performance with more data  

---

## Team Members
 
- Abhinandan Kumar
- Tushar Mogha 
- Stuti Mishra

---

## Conclusion

This project shows how AI and data analysis can be used together to detect academic integrity risks in a better and more practical way.
