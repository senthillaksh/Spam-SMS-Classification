
import pickle
import streamlit as st

# Load the Multinomial Naive Bayes model and CountVectorizer
classifier = pickle.load(open('spam_classification_model.pkl', 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

st.title("Spam SMS Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):

  # Cleaning special character from the sms
  review = re.sub('[^a-zA-Z]', ' ', input_sms)

  # Converting the entire sms into lower case
  review = review.lower()

  # Tokenizing the sms by words
  review = review.split()
    
  # Removing the stop words
  filtered_words = [word for word in review if not word in stopwords.words('english')]

  # stemming the words
  stemmed_words = [ps.stem(word) for word in filtered_words]

  # Joining the stemmed words
  review = ' '.join(stemmed_words)

  # vectorize
  vect = cv.transform([review])

  # predict
  result = classifier.predict(vect)[0]

  # 4. Display
  if result == 1:
    st.success("Oops! This is a SPAM message.")
  else:
    st.success("This is a HAM (normal) message.")
