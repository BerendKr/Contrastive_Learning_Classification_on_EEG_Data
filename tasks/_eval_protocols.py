import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

def fit_svm(features, y, MAX_SAMPLES, random_seed):
    nb_classes = np.unique(y, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    pipe = make_pipeline(
        StandardScaler(),
        SVC()
    )
    if train_size // nb_classes < 5 or train_size < 50:
        return pipe.fit(features, y)
    else:
        grid_search = GridSearchCV(
            pipe, {
                'svc__C': [0.01, 0.1, 1, 10, 100],
                'svc__kernel': ['rbf'],
                'svc__gamma': ['auto'],
                # 'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'svc__shrinking': [True], # False?
                'svc__probability': [True],
                'svc__tol': [0.001],
                'svc__cache_size': [200],
                'svc__class_weight': ['balanced', None],
                'svc__max_iter': [10000],
                'svc__decision_function_shape': ['ovr'],
                'svc__random_state': [random_seed]
            },
            cv=5, n_jobs=-2, verbose=1, scoring='roc_auc'
        )
        # If the training set is too large, subsample MAX_SAMPLES examples
        if train_size > MAX_SAMPLES:
            split = train_test_split(
                features, y,
                train_size=MAX_SAMPLES, random_state=random_seed, stratify=y
            )
            features = split[0]
            y = split[2]
        print('start grid search for SVM')
        grid_search.fit(features, y)

        print(grid_search.best_params_)

        def extract_params(pipeline_params, step_name='svc'):
            prefix = f"{step_name}__"
            return {
                k[len(prefix):]: v for k, v in pipeline_params.items()
                if k.startswith(prefix)
            }
        return grid_search.best_estimator_, extract_params(grid_search.best_params_)



def fit_knn(features, y, random_seed):
    n_classes = len(np.unique(y))
    scoring_mode = 'accuracy' if n_classes > 2 else 'roc_auc'
    pipe = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            # n_neighbors=1, # uncomment if not grid search
            # weights='uniform',
            # p=2,
            # algorithm='auto'
        )
    )
    # pipe.fit(features, y)
    grid_search = GridSearchCV(
    pipe, {
            'kneighborsclassifier__n_neighbors': [1,2,3,4,5,6,7,8,9,10,15,20,25],
            'kneighborsclassifier__weights': ['uniform', 'distance'],
            'kneighborsclassifier__p': [1, 2],
            'kneighborsclassifier__algorithm': ['auto']
        },
        scoring=scoring_mode, cv=5, n_jobs=-1, verbose=0
    )
    print('starting grid search for KNN')
    grid_search.fit(features, y)
    print("grid_search.best_params_: ", grid_search.best_params_)
    # return pipe

    def extract_params(pipeline_params, step_name='kneighborsclassifier'):
            prefix = f"{step_name}__"
            return {
                k[len(prefix):]: v for k, v in pipeline_params.items()
                if k.startswith(prefix)
            }
    
    return grid_search.best_estimator_, extract_params(grid_search.best_params_)



def fit_random_forest(features, y, random_seed):
    pipe = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(random_state=random_seed, n_jobs=-2)
    )

    param_grid = {
        'randomforestclassifier__n_estimators': [50, 100, 200],  # Number of trees
        'randomforestclassifier__max_depth': [None, 10, 30],  # Tree depth
        'randomforestclassifier__min_samples_split': [5, 10],  # Minimum samples per split
        'randomforestclassifier__min_samples_leaf': [2],  # Minimum samples per leaf
        'randomforestclassifier__bootstrap': [True],  # Whether to use bootstrap sampling
    }

    grid_search = GridSearchCV(
        pipe, param_grid,
        scoring='roc_auc', cv=5, n_jobs=-2, verbose=1
    )

    print('Starting Random Forest grid search')
    grid_search.fit(features, y)

    def extract_params(pipeline_params, step_name='randomforestclassifier'):
            prefix = f"{step_name}__"
            return {
                k[len(prefix):]: v for k, v in pipeline_params.items()
                if k.startswith(prefix)
            }
    print(grid_search.best_params_)
    return grid_search.best_estimator_, extract_params(grid_search.best_params_)

