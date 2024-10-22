Stock News Sentiment Analysis
Using Machine Learning

Welcome to the Stock news sentiment analysis project! This repository contains the code and resources for a cutting-edge approach that combines machine learning algorithms with sentiment analysis to accurately predict stock news sentiment analysis.

## Project Overview
Project Description: Sentiment analysis is a powerful technique for determining the attitude or emotion behind textual data. News, relevant articles, discussions, and other public communications can significantly affect stock prices in financial markets. This project uses machine learning algorithms such as logistic regression, Naïve Bayes, SVM, logistic models, sentiment analysis from news articles, and other data sources to develop a robust prediction model.
## The project will be divided into the following phases:
•	Data Collection:
o	Textual data relevant to stocks is gathered from various sources like news headlines, Twitter posts, or   stock market forums. This data is often labeled with sentiment scores (positive, negative, or neutral, data from newsapi.org to search worldwide news and locate articles and breaking news headlines from sources across the web.
•	Data Preprocessing:
o	Clean and format the textual data by removing irrelevant information (e.g., stop words, special characters, and wordnet).
o	Align stock price data with news and blog articles data by date.
•	Sentiment Analysis:
o	Use Natural Language Processing (NLP) techniques and sentiment analysis libraries (e.g., NLTK, TextBlob ) to assign sentiment scores (positive, negative, neutral)  Logistic Regression, Naive Bayes, Support Vector Machines to the collected textual data.
•	Feature Engineering:
o	Integration of machine learning algorithms (Svm, logistic regression, Naïve Bayes) with sentiment analysis for accurate predictions.
•	Modeling and Prediction:
o	Preprocessing of data to eliminate noise, normalize text, and extract relevant features for sentiment analysis.
o	Train models to predict stock price movements based on sentiment features.
•	Evaluation and Visualization:
o	Evaluate the models using appropriate metrics to gauge accuracy and effectiveness.
o	Perform historical price data analysis and sentiment scores derived from news article analysis.

After running the machine learning code, its generate 3 different files ending with pkl



## Features
Integration of machine learning algorithms (Svm, logistic regression, Naïve Bayes) with sentiment analysis for accurate predictions.
Preprocessing of data to eliminate noise, normalize text, and extract relevant features for sentiment analysis.
Evaluation of machine learning models using appropriate metrics to gauge accuracy and effectiveness.
Historical price data analysis and sentiment scores obtained from news article  analysis.
Practical implications for traders and investors, enabling them to make informed decisions based on comprehensive analysis.

## For the streamlt
The code builds a Streamlit web app for sentiment analysis of news articles based on a user query. Here's a breakdown of the key sections:
1. Imports:
•	The code starts by importing necessary libraries, such as streamlet, requests, JSON, pickle, and others, for data manipulation, model loading, and visualization.
•	It also imports functions from the preprocessing module (likely containing text cleaning functions) and nltk library for natural language processing.
2. Preprocessing Functions:
•	nltk.download('stopwords'): Downloads stop words from NLTK for text cleaning.
•	preprocess_text(text): This function (likely defined elsewhere in the preprocessing module) takes text as input and performs the following: 
o	Lemmatization: Converts words to their base form (e.g., "running" becomes "run").
o	Stop word removal: Removes common words like "the", "a", "an" that don't contribute to sentiment.
•	The commented-out section (# Make prediction using the loaded model) suggests there might be code for prediction functionality that's currently disabled.
3. Streamlit App (main function):
•	st.title: Sets the app title to "Stock News Sentiment Analyzer".
•	User Input: 
o	query = st.text_input("Enter a news query:"): Creates a text input box for users to enter their desired news topic.
o	model_type = st.selectbox('Type of Model',('LogisticRegression', 'MultinomialNB','SVM')): Provides a dropdown menu for users to select the model type for sentiment analysis (Logistic Regression, Multinomial Naive Bayes, or Support Vector Machine).
4. Button Click and News Retrieval:
•	if st.button("Analyze News"):: This block executes when the user clicks the "Analyze News" button. 
o	newsapi_key: Stores your News API key (replace with your own).
o	newsapi = NewsApiClient(api_key=newsapi_key): Creates a News API client using your key.
o	articles = newsapi.get_everything(q=query): Fetches news articles based on the user's query.
5. Data Extraction and DataFrame Creation:
•	Loops through retrieved articles and extracts relevant data (title, description, content) into a list of dictionaries.
•	Creates a Pandas DataFrame (df) from the extracted data list.
•	Displays the article titles using st.dataframe(df[['title']]).
6. Model Loading and Sentiment Prediction:
•	Loads the pre-trained model based on the user's selection from the dropdown menu: 
o	models dictionary maps model type to its corresponding saved model file path.
o	model = load(models[model_type]): Loads the chosen model using joblib.
•	Loops through each row of the DataFrame (df). 
o	Preprocesses the title text using the preprocess_text function.
o	Transforms the processed text into a format suitable for the model using the loaded vectorizer (vectorizer.transform).
o	Makes a sentiment prediction using the loaded model (prediction = model.predict(input_features)[0]).
o	Assigns a sentiment label ("Positive" or "Negative") to the current row in the DataFrame based on the prediction (df.at[index, 'Sentiment'] = Sentiment).
7. Results Display and Sentiment Distribution:
•	Creates a separator using Markdown (st.markdown("------------------------------------------------------")).
•	Displays the DataFrame again now including the predicted sentiment (st.dataframe(df[['title','Sentiment']])).
•	Calculates the percentage of positive and negative articles for the chosen model.
•	Displays the calculated percentages with labels (st.write).
•	Creates a pie chart using Matplotlib to visualize the sentiment distribution.
•	Displays the pie chart in Streamlit using st.pyplot(fig).
8. Handling No Data Case:
•	If no positive or negative sentiment is found in the results, displays a message indicating no data available for percentages.
9. Running the App:
•	The code uses the standard Streamlit boilerplate (if __name__ == "__main__": main()) to ensure the main function runs only when the script is executed directly (not imported as a module).

This code demonstrates how to build a user-friendly web app for sentiment analysis using Streamlit and various machine learning libraries. It allows users to explore different models and visualize the sentiment distribution of news articles based on their queries.


Overall, the results suggest that all three models are capable of performing sentiment analysis on news articles with reasonable accuracy. The choice of model will depend on factors like computational resources, interpretability requirements, and the specific characteristics of your dataset. Further analysis and experimentation can help you select the most suitable model for your particular use case.

## stock_price_visualization.py   
 A user-friendly web application is developed using Streamlit to allow end-users to perform
sentiment analysis on their text inputs. 



## Make predictions:
Use the trained models to make predictions on new or unseen data.
Analyze the predictions and gain insights into stock news sentiment analysis.

## Contributing
We welcome contributions to enhance the project and make it even more robust. To contribute, please follow these steps:

### Fork the repository.
Create a new branch for your contribution.
Make your changes and commit them.
Push your changes to your fork.
Submit a pull request, explaining the changes you have made.




## Acknowledgments
We would like to acknowledge the contributions and resources from various open-source projects and the research community that have helped in the development of this project.

This project showcases the application of machine learning for sentiment analysis on news article data. 
By utilizing the 3 models model and deploying it via Streamlit, we developed an interactive tool that enables users to analyze sentiments in real-time. This tool can be adapted to other datasets and integrated into larger systems to offer sentiment analysis as a service. 


Feel free to explore the exciting world of stock market trend prediction using machine learning and sentiment analysis!
