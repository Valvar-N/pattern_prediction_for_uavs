from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=5000), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("En iyi parametreler:", grid.best_params_)
print("En iyi doğruluk:", grid.best_score_)