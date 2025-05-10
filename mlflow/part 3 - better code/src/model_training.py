from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import logging

def train_random_forest(X_train, y_train):
    logging.info("Training Random Forest model...")
    rf = RandomForestClassifier(random_state=6)
    param_grid = {
        'n_estimators': [200, 400, 700],
        'max_depth': [10, 20, 30],
        'criterion': ["gini", "entropy"],
        'max_leaf_nodes': [50, 100]
    }
    grid = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=0)
    model = grid.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    logging.info("Training Logistic Regression model...")
    lr = LogisticRegression(random_state=6)
    param_grid = {
        'C': [100, 10, 1.0, 0.1, 0.01],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    grid = GridSearchCV(lr, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=0)
    model = grid.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    logging.info("Training Decision Tree model...")
    dt = DecisionTreeClassifier(random_state=6)
    param_grid = {
        "max_depth": [3, 5, 7, 9, 11, 13],
        'criterion': ["gini", "entropy"]
    }
    grid = GridSearchCV(dt, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=0)
    model = grid.fit(X_train, y_train)
    return model