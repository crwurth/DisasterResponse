import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram

from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/Disasters.db')
df = pd.read_sql_table('Categorized', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    categories = df.iloc[:,4:]
    top_3_category_counts = categories.sum().sort_values(ascending =False)[:3]
    tags = categories.sum(axis = 1).sort_values(ascending = False)
   
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=top_3_category_counts.index,
                    y=top_3_category_counts
                )
            ],

            'layout': {
                'title': 'Top 3 Category Tags',
                'yaxis': {
                    'title': "Total Tags"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Histogram(
                    x=tags
                )
            ],

            'layout': {
                'title': 'Distribution of Tags per Message',
                'yaxis': {
                    'title': "Total Messages"
                },
                'xaxis': {
                    'title': "Tag Count"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
