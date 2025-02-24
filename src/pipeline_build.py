from enum import Enum
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from src.dataset_cleanup import remove_empty_rows


class Classifier(Enum):
    SVM='SVM',
    NB='NB'


def find_best_parameters_svm(frame: pd.DataFrame, feature_labels: str | list, estimator=Classifier.SVM):
    kf = KFold(n_splits=2, random_state=7, shuffle=True)
    vectorizer = TfidfVectorizer(max_features=1000, lowercase=True, ngram_range=(1,2))
    frame = frame.reset_index(drop=True)
    if isinstance(feature_labels, list):
        data_x = pd.DataFrame
        list_length = len(feature_labels)
        temp_frame = pd.DataFrame()
        for index, feature  in enumerate(feature_labels):
            if index <= list_length - 2:
               temp_frame['unified'] = frame[feature] + frame[feature_labels[index + 1]]
            else:
                data_x = frame
                data_x['unified'] = temp_frame['unified']
                for feat in feature_labels:
                    data_x.drop(feat, axis=1, inplace=True)
                data_x = data_x['unified']
    else:
        data_x = frame[feature_labels]

    param_grid_svm = {
        "C": [1.0, 0.5, 1.5],
        "degree": [3, 4, 5]
    }

    param_grid = {
        "alpha": [0.1, 1.0]
       # "fit_prior" = fit_prior,
       # "class_prior" = class_prior,
       # "force_alpha" = force_alpha,
    }

    if estimator == Classifier.SVM:
        grid = GridSearchCV(estimator=svm.SVC(), param_grid=param_grid_svm)
    else:
        grid = GridSearchCV(estimator=MultinomialNB(), param_grid=param_grid)
    frame = frame.reset_index(drop=True)
    data_y = frame['category']

    for train_index, test_index in kf.split(data_x):
        x_train, x_test = data_x[train_index], data_x[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]

        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        grid.fit(x_train.toarray(), y_train)
        y_pred = grid.predict(x_test.toarray())
        score = grid.score(x_test.toarray(), y_test)

        print(score)
        print(y_pred)

    return grid

def save_model(model: GridSearchCV, label):
    with open(f'../data/trained/model-{label}.pkl', 'wb') as f:
        pickle.dump(model, f)

def load_model(label) -> GridSearchCV:
    with open(f'../data/trained/model-{label}.pkl', 'rb') as f:
            return pickle.load(f)

def predict(data):
    return trained_model.predict(data)


trained_model = GridSearchCV
#save_model(find_best_parameters_svm(remove_empty_rows(pd.read_json('../data/processed/lab_lcn_lnp.json'), 'lcn'), ['lab', 'lcn', 'lnp'], estimator=Classifier.NB), Classifier.NB)
#trained_model = load_model(Classifier.NB)
