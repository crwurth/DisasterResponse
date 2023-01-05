import sys

# import libraries
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Categorized', con = engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns.tolist()
    return X,y,category_names

def tokenize(text):
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
        
        lemmatizer = WordNetLemmatizer()
        for word in words:
            clean_word = lemmatizer.lemmatize(word)
            cleaned.append(clean_word)
    return cleaned

def build_model():
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('multioutput' ,MultiOutputClassifier(LogisticRegression(random_state = 42))),
    ])
    from sklearn.model_selection import GridSearchCV
    
    parameters = {
    'vect__max_features': [None,1,2]
    ,'multioutput__estimator: [LogisticRegression(random_state = 42), DecisionTreeClassifier(random_state = 42),
    ,'multioutput__estimator__solver' : ['liblinear', 'sag']
    ,'multioutput__estimator__max_iter': [100, 200]
             }

    #Computation time very slow, so these grid searches were done separately to prevent workspace errors. 
#Each result showed the original parameter as best 
#(vect__max_features = none, multioutput__estimator = Logistic Regression, multioutput__estimator__solver = 'liblinear',
#multioutput__estimator__max_iter = 100)
    cv = GridSearchCV(pipeline, parameters, n_jobs = -1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    #get model predictions
    y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(category)
        print(classification_report(Y_test[category], y_pred[:,i]))
    pass


def save_model(model, model_filepath):
    import pickle

#Help with pickling adopted from below blog post              #https://ianlondon.github.io/blog/picklingbasics/#:~:text=To%20save%20a%20pickle%2C%20use,name%20it%20whatever%20you%20want
#published April 14, 2016; assessed 1/4/2023
    with open(model_filepath, 'wb') as file:
        pickle.dump(model,file)
    pass


def main():
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