# 🌟 SENTIMENT SCOPE 🌟

[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/RAJA-2004/SENTIMENT-SCOPE?logo=github&style=for-the-badge)](https://github.com/RAJA-2004/SENTIMENT-SCOPE/)
[![GitHub last commit](https://img.shields.io/github/last-commit/RAJA-2004/SENTIMENT-SCOPE?style=for-the-badge&logo=git)](https://github.com/RAJA-2004/SENTIMENT-SCOPE/)
[![GitHub stars](https://img.shields.io/github/stars/RAJA-2004/SENTIMENT-SCOPE?style=for-the-badge)](https://github.com/RAJA-2004/SENTIMENT-SCOPE/stargazers)
[![My stars](https://img.shields.io/github/stars/RAJA-2004?affiliations=OWNER%2CCOLLABORATOR&style=for-the-badge&label=My%20stars)](https://github.com/RAJA-2004/SENTIMENT-SCOPE/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/RAJA-2004/SENTIMENT-SCOPE?style=for-the-badge&logo=git)](https://github.com/RAJA-2004/SENTIMENT-SCOPE/network)
[![Code size](https://img.shields.io/github/languages/code-size/RAJA-2004/SENTIMENT-SCOPE?style=for-the-badge)](https://github.com/RAJA-2004/SENTIMENT-SCOPE)
[![Languages](https://img.shields.io/github/languages/count/RAJA-2004/SENTIMENT-SCOPE?style=for-the-badge)](https://github.com/RAJA-2004/SENTIMENT-SCOPE)
[![Top](https://img.shields.io/github/languages/top/RAJA-2004/SENTIMENT-SCOPE?style=for-the-badge&label=Top%20Languages)](https://github.com/RAJA-2004/SENTIMENT-SCOPE)
[![Issues](https://img.shields.io/github/issues/RAJA-2004/SENTIMENT-SCOPE?style=for-the-badge&label=Issues)](https://github.com/RAJA-2004/SENTIMENT-SCOPE)
[![Watchers](https://img.shields.io/github/watchers/RAJA-2004/SENTIMENT-SCOPE?label=Watch&style=for-the-badge)](https://github.com/RAJA-2004/SENTIMENT-SCOPE/)

Analyze social media sentiment using Python and machine learning techniques.

<img src="https://www.travelmediagroup.com/wp-content/uploads/2022/04/bigstock-Market-Sentiment-Fear-And-Gre-451706057-2880x1800.jpg" alt="Market Sentiment - Fear And Greed">

## 📖 Project Overview
This project involves developing a sentiment analysis tool that analyzes social media posts to determine the sentiment (positive or negative) expressed in the text. The goal is to understand public opinion and trends by analyzing large volumes of text data from social media platforms.

## ✨ Features

- **🔍 Data Collection**: Downloads the Sentiment 140 dataset using Kaggle API.
  
- **📥 Data Loading**: Loads the dataset into a Pandas DataFrame for further processing.

- **🧼 Data Preprocessing**: Cleans and prepares text data by:
  - Converting text to lowercase.
  - Removing non-alphabetic characters.
  - Tokenizing and stemming words using NLTK's PorterStemmer.
  - Removing stopwords to reduce complexity.

- **🧠 Sentiment Analysis**: Utilizes machine learning techniques, specifically Logistic Regression, for sentiment classification:
  - Converts preprocessed text into numerical vectors using TF-IDF vectorization.
  - Trains a Logistic Regression model on the vectorized data to predict sentiment (positive or negative).

- **📊 Model Evaluation**: Evaluates the model's performance using accuracy score metrics:
  - Computes accuracy scores on both training and test datasets to assess model performance.
  - Visualizes accuracy scores in graphical formats to illustrate model effectiveness.

- **📥 Dataset Management**: Manages the dataset:
  - Downloads and extracts the dataset using Kaggle.
  - Handles dataset loading, including renaming columns and addressing missing values.

- **🧪 Model Persistence**: Saves the trained Logistic Regression model using pickle for future use or deployment.


## ⚙️ Installation
To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/raja-2004/Social-Media-Sentiment-Analysis.git
cd Social-Media-Sentiment-Analysis
pip install numpy
pip install pandas
pip install scikit-learn
pip install nltk

```
For NLTK's additional data download (specifically stopwords):

```bash
import nltk
nltk.download('stopwords')
```


## 📊 Dataset

This project utilizes the **Sentiment 140** dataset, comprising 1.6 million tweets labeled as 0 (negative) or 1 (positive). <br>
Create new API Token <br>
Uplod the JSON file to the project folder
```bash
https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.kaggle.com/datasets/kazanova/sentiment140&ved=2ahUKEwjYj6Oj0YeHAxVmSmwGHe5WA6QQFnoECBUQAQ&usg=AOvVaw3FGJ5Ag61V7UFBoTg4qUjV
```

## 📈 Results

**Accuracy of Training Data :  81%** <br>
**Accuracy of Testing Data  :  77%**

## 🤝 Contributing

Contributions are welcome! Fork the repository and submit a pull request to contribute.

## Need help?

Feel free to contact me on [LinkedIn](https://www.linkedin.com/in/rajadigvijaysingh/) 

---------

```python

if youEnjoyed:
    starThisRepository()


```

-----------
