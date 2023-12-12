# Leveraging Big Data Analytics for Disease Classification Using Chatbot-Generated Features

## Aditya Balasubramanian
## Boston University
## December 7, 2023

## Abstract

In the rapidly evolving landscape of healthcare, the influx of medical data has become unprecedented, paving the way for transformative advancements. This project proposal delves into the realm of leveraging big data analytics to enhance disease classification, facilitated by the burgeoning capabilities of chatbot technology. The core focus is on constructing a robust classification model that harnesses features generated through chatbot interactions. The medical dataset is accessible at [HuggingFace](https://huggingface.co/).

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset](#2-dataset)
3. [Data Processing](#3-data-processing)
4. [Model Building](#4-model-building)
   - [Traditional Machine Learning Model (Random Forest)](#41-traditional-machine-learning-model-random-forest)
   - [BERT Model](#42-bert-model)
   - [Support Vector Machine (SVM) Model](#43-support-vector-machine-svm-model)
5. [Chatbot](#5-chatbot)
6. [Conclusion and Future Scope](#7-conclusion-and-future-scope)

## 1 Introduction

The contemporary landscape of the healthcare industry is undergoing a transformative shift, primarily attributed to the burgeoning availability of medical data. This surge in data, ranging from electronic health records to real-time chatbot interactions, presents an unprecedented opportunity to revolutionize disease classification. In tandem with these advancements, chatbot technology has emerged as a pivotal player in healthcare, streamlining patient interactions and generating valuable textual data in the process.

The integration of chatbots into healthcare not only facilitates efficient communication but also yields a wealth of textual features derived from patient queries, responses, and explanations. Leveraging the power of big data analytics in this context holds immense promise for enhancing disease classification methodologies.

## 2 Dataset

The dataset used for this project is sourced from Hugging Face Datasets, a repository known for its diverse and comprehensive collection of datasets. The medical dataset at our disposal is tailored to the medical question-answering domain and aligns with the project’s objectives.

### 2 Key Features

- **Question:** Represents queries related to patient symptoms or concerns.
- **Options (opa, opb, opc, opd):** Multiple-choice options associated with each question.
- **Explanation (exp):** Provides additional context or elaboration on the correct answer.
- **Topic Name:** Denotes the medical topic or category of the question.
- **Choice Type:** Indicates the type of choice, providing context on the format of answer options.
- **COP (Target Variable):** Stands for Correct Option Position, serving as the target variable for disease classification.

## 3 Data Processing

Data preprocessing is a crucial step in the data analysis and machine learning pipeline that involves cleaning and transforming raw data into a format suitable for analysis or model training. It’s often said that the success of a machine learning project is heavily dependent on the quality of the data, and data prepossessing plays a key role in enhancing this quality.

We prepossessed the dataset using the following steps:

### 3.1 Tokenization, Lowercasing, and Stopword Removal

Tokenization is the process of breaking down text into individual tokens or words. This step is essential for converting raw text into meaningful units, allowing for further analysis. Lowercasing is employed to ensure uniformity by converting all text to lowercase, reducing the complexity of the dataset. Stopword removal involves eliminating common words (e.g., "the," "and," "is") that do not contribute significant meaning to the text, focusing on content-bearing words.

### 3.2 Lemmatization and Special Character Removal

Lemmatization involves reducing words to their base or root form, enhancing feature extraction by standardizing variations of a word. Special character removal is performed to ensure text cleanliness by eliminating non-alphanumeric characters.

### 3.3 Handling Missing Values

Addressing missing values is crucial for ensuring model robustness. Depending on the nature of the missing data, strategies include imputation, where missing values are replaced with estimated values based on the available data, or removal, where rows with missing values are excluded from the dataset.

## 4 Model Building

Model building involves creating a mathematical algorithm to make predictions or decisions based on input data in machine learning. The process includes defining the problem, collecting and preparing representative data, splitting it into training, validation, and test sets, selecting a suitable model architecture, tuning hyperparameters, training the model, and evaluating its performance on validation and test sets.

### 4.1 Traditional Machine Learning Model (Random Forest)

#### 4.1.1 TF-IDF Vectorization

TF-IDF vectorization is applied to convert the text data into numerical vectors for input to the Random Forest classifier. The TfidfVectorizer from scikit-learn is employed, with a maximum of 5000 features. This process transforms the text features into a format suitable for traditional machine learning models.

#### 4.1.2 Model Training and Evaluation

A Random Forest classifier is trained using the TF-IDF-transformed text data and the ’cop’ column as the target variable. The classifier is configured with 100 decision trees and a random seed of 42 for reproducibility. The model is evaluated on the validation set using the accuracy metric, which measures the percentage of correctly predicted ’cop’ values.

#### 4.1.3 Model Saving

After training and evaluation, the Random Forest model is saved to Google Cloud Storage (GCS) for future use. This enables easy retrieval and deployment of the trained model without the need for retraining.

### 4.2 BERT Model

#### 4.2.1 Tokenization and Encoding

The BERT model, based on the ’bert-baseuncased’ architecture, is employed for sequence classification. The text data is tokenized and encoded using the BERT tokenizer, and special tokens are added to mark the beginning and end of each sequence. The resulting tensors are then used as input to the BERT model.

#### 4.2.2 Training Loop

The model is trained over multiple epochs, with each epoch consisting of iterations through the training data. The training loop uses PyTorch DataLoaders to efficiently load batches of data. The AdamW optimizer is employed for optimization, and a learning rate scheduler is used to adjust the learning rate during training.

#### 4.2.3 Monitoring Training and Validation Losses

During training, both training and validation losses are monitored. This information is crucial for understanding how well the model is learning from the data and whether there is overfitting or underfitting. The losses are printed after each epoch, providing insights into the model’s performance.

### 4.3 Support Vector Machine (SVM) Model

#### 4.3.1 Feature Extraction and Vectorization

For the SVM model, the text data undergoes feature extraction and vectorization. The process involves converting the textual information in the ’question’ column into a format suitable for SVM training. In this case, TF-IDF (Term Frequency-Inverse Document Frequency) vectorization is commonly employed. Each question is represented as a numerical vector based on the importance of words within that question.

#### 4.3.2 Model Training

The SVM model is trained using the vectorized features of the text data. The scikit-learn library provides the SVC (Support Vector Classification) class for SVM classification tasks. During training, the model learns the optimal hyperplane that separates different classes in the feature space. The ’cop’ column, representing the correct option, serves as the target variable during training.

#### 4.3.3 Hyperparameter Tuning

SVM models have hyperparameters, such as the choice of kernel (e.g., RBF), regularization parameter (C), and kernel-specific parameters (e.g., gamma for RBF). Hyperparameter tuning may be performed to find the optimal combination that maximizes the model’s performance on the training data.

#### 4.3.4 Evaluation on Validation Set

After training, the SVM model is evaluated on the validation set to assess its performance on unseen data. Metrics such as accuracy are computed to quantify the model’s ability to correctly classify instances in the validation set. This evaluation helps in understanding how well the SVM model generalizes to new, previously unseen questions.

## 5 Chatbot

We built two different kinds of chatbots. One as explained below, using the Replicate API, and another using Llama 2. Llama gave limited flexibility for us to work on as it was a paid service, and we ran out of free tokens.

### 5.1 Importing Modules

The script imports the following modules:
- **os:** Provides interaction with the operating system, primarily used to set environment variables.
- **pickle:** A module for serializing and deserializing Python objects, employed to load a pre-trained scikit-learn model.
- **replicate:** Presumably, a custom or third-party library interacting with an API (likely named "REPLICATE").

### 5.2 Loading the Scikit-Learn Model

The script loads a pre-trained scikit-learn model from a file named finalized_model_2.sav using the pickle.load() function. The loaded model is stored in the variable model.

### 5.3 Defining the Chatbot Response Function

A function named chatbot_response takes a symptom as input, utilizes the loaded model to predict a health disorder based on the symptom, and returns a response string.

### 5.4 Setting Environment Variable

The script sets the environment variable REPLICATE_API_TOKEN to a specific value, likely an authentication token required to access an API (possibly the REPLICATE API).

### 5.5 Chat Loop

The script initiates a simple chat loop, continuously prompting the user to enter a symptom until the user types ’exit’. Inside the loop:

- User input is collected.
- If the user types ’exit’, the loop breaks, and the program ends.
- Otherwise, the chatbot generates a response using the scikit-learn model and prints it.
- The script uses the REPLICATE API (presumably) by calling the replicate.run() function, passing parameters including the user input as a prompt. The output from this API call is then printed.

### 5.6 REPLICATE API Call

The replicate.run() function appears to make a request to the REPLICATE API using a specific model version. The API call includes parameters like top_k, top_p, temperature, max_new_tokens, and system_prompt, controlling the behavior of the language model used by the API.

## 6 Conclusion and Future Scope

The envisioned goals and scope of our project encapsulate a forward-looking approach to technology and user engagement. Our primary objective is to achieve enhanced personalization, fostering stronger connections and satisfaction for users by tailoring experiences to individual preferences. Additionally, we aspire to integrate telehealth services, aiming to provide comprehensive and accessible healthcare solutions. The project also emphasizes the implementation of multimodal interaction and improved Natural Language Understanding (NLU) to create more intuitive and inclusive user interfaces. As we forge ahead, the overarching goal is to contribute to global health awareness and crisis response, utilizing technology as a catalyst for disseminating timely information and facilitating swift, impactful actions. In essence, our project’s future goals align with harnessing technology to elevate user experiences, advance healthcare solutions, and contribute meaningfully to societal well-being.

---

