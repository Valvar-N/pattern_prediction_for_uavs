from sklearn.model_selection import learning_curve
import numpy as np

def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y,
                                                            cv=5,
                                                            scoring='accuracy',
                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                            random_state=42)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Eğitim Skoru')
    plt.plot(train_sizes, test_mean, 'o-', label='Doğrulama Skoru')
    plt.title(title)
    plt.xlabel('Eğitim Seti Boyutu')
    plt.ylabel('Doğruluk')
    plt.legend()
    plt.grid()
    plt.show()

# Logistic Regression için learning curve
plot_learning_curve(LogisticRegression(random_state=42), X, y_encoded, "Logistic Regression Öğrenme Eğrisi")

# Random Forest için learning curve
plot_learning_curve(RandomForestClassifier(random_state=42), X, y_encoded, "Random Forest Öğrenme Eğrisi")