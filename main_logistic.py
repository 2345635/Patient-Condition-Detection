import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from pywebio import start_server
from pywebio.input import input, TEXT
from pywebio.output import put_text

# Load the data
df = pd.read_excel('data.xlsx', engine='openpyxl')

# Tokenize the text
df['TOKENIZED_EXAMINATIONS'] = df['EXAMNINATIONS'].apply(nltk.word_tokenize)

# Download stopwords
nltk.download('stopwords')

# Remove stopwords
stopwords = nltk.corpus.stopwords.words('english')
df['STOPWORDS_REMOVED_EXAMINATIONS'] = df['TOKENIZED_EXAMINATIONS'].apply(lambda tokens: [token for token in tokens if token.lower() not in stopwords])

# Prepare the features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['STOPWORDS_REMOVED_EXAMINATIONS'].apply(' '.join))

# Prepare the target variable
target = df['DIAGNOSIS']

# Train the model
model = LogisticRegression()
model.fit(X, target)

# Define the prediction function
def predict_diagnosis():
    patient = input("Enter the Patient Name:", type=TEXT)
    text = input("Enter the examinations:", type=TEXT)
    text_preprocessed = preprocess_text(text)
    features = vectorizer.transform([text_preprocessed])
    prediction = model.predict(features)[0]
    put_text(f"{patient} has ", prediction)

# Preprocess the input text
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token.lower() not in stopwords]
    return ' '.join(tokens)

# Start the PyWebIO server
if __name__ == '__main__':
    start_server(predict_diagnosis, port=8080)
