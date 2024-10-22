# Import Libraries
# !pip install newsapi-python
import streamlit as st
import requests
import json
import pickle
import emoji
import preprocess
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
# Load trained models
from joblib import load
from newsapi import NewsApiClient
import pandas as pd
import matplotlib.pyplot as plt

# Load pre-trained models and vectorizer
# Download NLTK resources
nltk.download('stopwords')

# Replace with your NewsAPI API key
newsapi_key = "0a0a7649aef9455789c3bb304d3ea6ed"

# Load vectorizer
vectorizer = load('tfidf_vectorizer.joblib')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

    # Make prediction using the loaded model

        #prediction = model.predict(input_features)[0]

# Streamlit app
def main():

    st.title("Stock News Sentiment Analyzer😊😐😕😡")
    # Fetch news articles based on the query
    query = st.text_input("Enter a news query:")
    model_type = st.selectbox('Type of Model',('LogisticRegression', 'MultinomialNB','SVM'))
    models = {"LogisticRegression":"Logistic_Regression_model.pkl","MultinomialNB":"Multinomial_Naive_Bayes_model.pkl", "SVM":"Support_Vector_Machine_model.pkl"}
    
    # Button to fetch and analyze news
    if st.button("Analyze News"):
        # Initialize NewsAPI client
        newsapi = NewsApiClient(api_key=newsapi_key)
        articles = newsapi.get_everything(q=query)



        # Extract relevant data from articles
        data = []
        for article in articles['articles']:
            data.append({
                        'title': article['title'],
                        'description': article['description'],
                        'content': article['content'],
                    })

        # Create a DataFrame from the extracted data
        df = pd.DataFrame(data)
        # Display the DataFrame
        st.dataframe(df[['title']])

        # Load trained models
        from joblib import load

        # Load vectorizer
        vectorizer = load('tfidf_vectorizer.joblib')

        # Create a new column to store the predicted sentiment
        df['Sentiment'] = None
        for index, row in df.iterrows():
            processed_input = preprocess_text(row['title'])
            input_features = vectorizer.transform([processed_input])
            if model_type in models.keys():
                model= load(models[model_type])
                prediction = model.predict(input_features)[0]
                Sentiment = "Positive" if prediction == 1 else "Negative"
                df.at[index, 'Sentiment'] = Sentiment
                #print(f"{model_name}: {sentiment}")
                
        st.markdown("------------------------------------------------------")
        st.dataframe(df[['title','Sentiment']])
        st.markdown("------------------------------------------------------")
        # Calculate positive and negative percentages for each model
        if len(df['Sentiment'] == 'Positive') > 0 and len(df['Sentiment'] == 'Negative') > 0:
            model_positive_percentage = (df['Sentiment'] == 'Positive').mean() * 100
            model_negative_percentage = (df['Sentiment'] =='Negative').mean() * 100
            st.write(f"Positive Percentage: {model_positive_percentage:.2f}%")
            st.write(f"Negative Percentage: {model_negative_percentage:.2f}%")

            # Create the pie chart data
            labels = ['Positive', 'Negative']
            data = [model_positive_percentage, model_negative_percentage]
            colors = ['gold', 'lightskyblue']
            explode = (0.1, 0)  # Explode the first slice to highlight

            # Create the pie chart

            # Create a figure and axes objects
            fig, ax = plt.subplots()

            # Plot the pie chart
            ax.pie(data, labels=labels, autopct="%1.1f%%",colors=colors, startangle=140)

            # Set the title
            ax.set_title('Sentiment Distribution')

            # Display the pie chart in Streamlit
            st.pyplot(fig)
          
        else:
            st.write("No data available for positive or negative percentages.")

if __name__ == "__main__":
    main()
