import sys

import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine

import re
import nltk
nltk.download(['omw-1.4', 'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, roc_auc_score

from xgboost import XGBClassifier




def load_data(database_filepath):
	engine = create_engine('sqlite:///../data/DisasterResponse.db')
	df = pd.read_sql_table(table_name='message_table', con=engine)
	X = df['message']
	Y = df.iloc[:, 4:]
	return (X, Y, Y.columns)



def tokenize(text):
    
    # Normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    
    # Tokenization
    words = word_tokenize(text)
    
    # Stop words removal
    words = [w for w in words if words not in stopwords.words("english")]
    
    # Stemming and Lemmatizing
    tokens = [PorterStemmer().stem(w) for w in words]
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
	pipeline_xgb = Pipeline([
		('vect', CountVectorizer(tokenizer=tokenize)),
		('tfidf', TfidfTransformer()),
		('clf', MultiOutputClassifier(XGBClassifier(random_state = 42,
											  max_depth=3,
											  colsample_bytree=0.7,
											  min_child_weight=1,
											  subsample=1)))
	])
	
	parameters_xgb = {
		'clf__estimator__n_estimators': (100, 200),
		#'clf__estimator__max_depth': (3, 6),
		'clf__estimator__learning_rate': (0.01, 0.1),
		#'clf__estimator__colsample_bytree': (0.7, 1),
		#'clf__estimator__min_child_weight': (1, 5, 10),
		#'clf__estimator__subsample': (0.7, 1)
	}
	
	cv_xgb = GridSearchCV(pipeline_xgb, param_grid=parameters_xgb, n_jobs=1, verbose=3, cv=5)
	print(cv_xgb)
	
	return cv_xgb



def displayAUC(Y_test, Y_pred_proba, category_names):
    # Save probability in a 2d numpy array (n_category * n_test)
    Y_pred_proba_array = np.array([1-x[0] for x in Y_pred_proba[0]])
    for i in range(1,len(Y_pred_proba)):
        prob = np.array([1-x[0] for x in Y_pred_proba[i]])
        Y_pred_proba_array = np.vstack((Y_pred_proba_array, prob))
    
    # AUC score
    avg_auc = 0
    num_auc = 0
    for i in range(len(Y_pred_proba)):
        try:
            auc = roc_auc_score(Y_test.values[:,i], Y_pred_proba_array[i,:])
            avg_auc += auc
            num_auc += 1
            print(str(i) + " "  + category_names[i] + " " + str(round(auc,4)))
        except ValueError:
            print(str(i) + " "  + category_names[i] + " NA")  
    avg_auc /= num_auc
    print("Average AUC is: ", avg_auc)



def evaluate_model(model, X_test, Y_test, category_names):
	Y_pred = model.predict(X_test)
	Y_pred_proba = model.predict_proba(X_test)
	print(classification_report(Y_test.values, Y_pred, target_names=category_names))
	displayAUC(Y_test, Y_pred_proba, category_names)


def save_model(model, model_filepath):
	pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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