import sys

# import libraries
#Read db
from sqlalchemy import create_engine
#Data Manipulation
import pandas as pd
import numpy as np
#process text
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')
#Natural Language Processing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#Build Model
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
#Test Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    inputs: 
    database_filepath = (str) filepath that contains databse
    outputs:
    X: Independent Variable (the message), 
    y: Dependent Variables (the Category tags)
    category_names: list of names of the categories
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Categorized', con = engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns.tolist()
    return X,y,category_names

def tokenize(text):
    """
    inputs:
    text: (str) message to be tokenized and lemmatized
    output:
    cleaned: (str) message after tokenization and lemmatization
    """
    text = text.lower()
    sentences = nltk.sent_tokenize(text)
    cleaned = []
    for sentence in sentences:
        #remove non alpha_numeric characters
        sentence = re.sub(pattern = "\W", repl = " ", string = sentence)        
        #split sentences into words
        words = nltk.word_tokenize(sentence)
        #remove stop words
        words = [x for x in words if x not in stopwords.words("english")]
        #lemmatize text
        lemmatizer = WordNetLemmatizer()
        for word in words:
            clean_word = lemmatizer.lemmatize(word)
            cleaned.append(clean_word)
    return cleaned

def build_model():
    """
    Inputs: None
    Outputs: 
    cv : Grid Search Model Object that has not yet been fitted
    
    This will create a pipeline using vectorization, dfidf, and a MultiOutputClassifier that uses Logistic Regression
    Then it will use gridsearch to find the best parameters
    """
    
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('multioutput' ,MultiOutputClassifier(LogisticRegression(random_state = 42))),
    ])
    parameters = {
    'vect__max_features': [None,1,2]
    ,'multioutput__estimator__solver' : ['liblinear', 'sag']
    ,'multioutput__estimator__max_iter': [100, 200]
             }
    #Note...This grid will cause computation time of the .fit() to be very slow. At least 1 hour to complete with 4 processors
    cv = GridSearchCV(pipeline, parameters, n_jobs = -1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Inputs:
    
    model: fitted model
    X_test: Indepent Variable (message) test set
    Y_test: Dependent Variables test set
    category_names: Names of Dependent Variables
    
    This will use the model to predict on the test set- Then, print the classification report of each dependent variable against the test dataset.
    """
    #get model predictions
    y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(category)
        print(classification_report(Y_test[category], y_pred[:,i]))
    pass

def save_model(model, model_filepath):
    """
    Save model as pickle file
    Inputs:
    model: Fitted Model
    model_filepath: filepath to save pickle file
    Outputs:
    None

    Help with pickling adopted from below blog post
    https://ianlondon.github.io/blog/pickling-basics/
    published April 14, 2016; assessed 1/4/2023
    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model,file)
    pass


def main():
    """
    Run entire python file
    """
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
