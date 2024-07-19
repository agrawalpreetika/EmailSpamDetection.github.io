import streamlit as st
import streamlit.components.v1 as components
import pickle
import string
from sklearn.feature_extraction.text import CountVectorizer

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def main():
    
    
    st.title("Email/SMS Spam Classifier")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image("mail-1454734_640.webp",width=200)

    with col3:
        st.write(' ')
    
    
    # Description of the detection and accuracy
    st.markdown("""
    This app uses a machine learning model to classify emails and SMS messages as spam or not spam.
    The model has been trained on a dataset and achieved an accuracy of 98.01 % on test data.
    """)

    spam = st.text_area("Enter the message:")

    if st.button('Predict'):

        if spam:
            spam_vec = vectorizer.transform([spam])
            prediction = model.predict(spam_vec)
            if prediction[0] == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
            st.empty()  # Clear the text area after prediction

if __name__ == '__main__':
    main()

