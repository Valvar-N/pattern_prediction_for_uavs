import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import utils 

# Veriyi yükle
file_path = utils.find_file_path("structured_datas", "all_processed_with_labels_combined.xlsx")
df = pd.read_excel(file_path)

# 2. LABEL DÖNÜŞÜMÜ (non-loiter: 0, loiter: 1)
df["label_encoded"] = df["label"].map({"non-loiter": 0, "loiter": 1})

# 3. FEATURE ENGINEERING: loiter_radius → log dönüşümü
df["loiter_radius_log"] = np.log1p(df["loiter_radius"])

# 4. ÖZELLİK VE HEDEF TANIMI
features = ["average_roll", "speed_std", "loiter_radius_log"]
X = df[features]
y = df["label_encoded"]

# Özelliklerin Dağılımı
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
sns.histplot(df["average_roll"], kde=False, color="orange", bins=30)
plt.title("average_roll dağılımı")
plt.subplot(1, 3, 2)
sns.histplot(df["speed_std"], kde=False, color="orange", bins=30)
plt.title("speed_std dağılımı")
plt.subplot(1, 3, 3)
sns.histplot(df["loiter_radius"], kde=False, color="orange", bins=30)
plt.title("loiter_radius dağılımı")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))
for i, feature in enumerate(features):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(x="label", y=feature, data=df)
    plt.title(f"{feature} değerleri label bazında")
plt.tight_layout()
plt.show()

corr = df[features].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Feature'lar Arası Korelasyon Matrisi")
plt.show()

# 5. EĞİTİM/TEST AYIRIMI
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. MODEL TANIMLARI
models = {
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", random_state=42)),
        ]
    ),
    "Random Forest": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(class_weight="balanced", random_state=42)),
        ]
    ),
}

# 7. EĞİTİM & DEĞERLENDİRME
results = {}
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {"classification_report": report, "confusion_matrix": cm}

# 8. SONUÇLARI YAZDIRMA
for model_name, result in results.items():
    print(f"\n🔍 Model: {model_name}")
    print("Classification Report:")
    print(pd.DataFrame(result["classification_report"]).T)
    print("Confusion Matrix:")
    print(result["confusion_matrix"])

# 9. RANDOM FOREST ÖZELLİK ÖNEMLERİ
rf_pipeline = models["Random Forest"]
rf_model = rf_pipeline.named_steps["clf"]
feature_names = X.columns
importances = rf_model.feature_importances_

plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=feature_names, palette="viridis")
plt.title("🎯 Özellik Önemleri (Random Forest)")
plt.xlabel("Önem Skoru")
plt.ylabel("Özellikler")
plt.tight_layout()
plt.show()

def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=42,
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, "o-", label="Eğitim Skoru")
    plt.plot(train_sizes, test_mean, "o-", label="Doğrulama Skoru")
    plt.title(title)
    plt.xlabel("Eğitim Seti Boyutu")
    plt.ylabel("Doğruluk")
    plt.legend()
    plt.grid()
    plt.show()

# Logistic Regression için learning curve
plot_learning_curve(
    LogisticRegression(random_state=42),
    X,
    y,
    "Logistic Regression Öğrenme Eğrisi",
)

# Random Forest için learning curve
plot_learning_curve(
    RandomForestClassifier(random_state=42),
    X,
    y,
    "Random Forest Öğrenme Eğrisi",
)

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"],
}

grid = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=5000),
    param_grid,
    cv=5,
    scoring="accuracy",
)
grid.fit(X_train, y_train)

print("En iyi parametreler:", grid.best_params_)
print("En iyi doğruluk:", grid.best_score_)




import numpy as np

# Sadece 2 feature seçiyoruz
X_2d = df[['average_roll', 'loiter_radius']]
y_2d = y_encoded

# Eğitim-test bölünmesi
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y_2d, test_size=0.2, random_state=42)

# Modeli eğit
lr_2d = LogisticRegression(random_state=42)
lr_2d.fit(X_train_2d, y_train_2d)

# Karar sınırı için meshgrid oluştur
x_min, x_max = X_2d['average_roll'].min() - 0.5, X_2d['average_roll'].max() + 0.5
y_min, y_max = X_2d['loiter_radius'].min() - 0.01, X_2d['loiter_radius'].max() + 0.01
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z = lr_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Grafik çizimi
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_train_2d['average_roll'], X_train_2d['loiter_radius'], c=y_train_2d, cmap='coolwarm', edgecolor='k', label='Train')
plt.scatter(X_test_2d['average_roll'], X_test_2d['loiter_radius'], c=y_test_2d, cmap='coolwarm', marker='x', label='Test')
plt.xlabel('average_roll')
plt.ylabel('loiter_radius')
plt.title('Logistic Regression Karar Sınırı (Train ve Test Veri Seti)')
plt.legend()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi yükle


# Özelliklerin Dağılımı
plt.figure(figsize=(18, 5))

# Average Roll Dağılımı
plt.subplot(1, 3, 1)
sns.histplot(df['average_roll'], kde=False, color='orange', bins=30)
plt.title('average_roll dağılımı')

# Speed Std Dağılımı
plt.subplot(1, 3, 2)
sns.histplot(df['speed_std'], kde=False, color='orange', bins=30)
plt.title('speed_std dağılımı')

# Loiter Radius Dağılımı
plt.subplot(1, 3, 3)
sns.histplot(df['loiter_radius'], kde=False, color='orange', bins=30)
plt.title('loiter_radius dağılımı')

plt.tight_layout()
plt.show()



import seaborn as sns

sns.pairplot(df, hue='label', vars=features, diag_kind='kde', plot_kws={'alpha':0.6})
plt.suptitle('Özelliklerin Pairplot’u (Loiter / Non-loiter)', y=1.02)
plt.show()



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Model oluştur
lr_model = LogisticRegression(random_state=42)

# Eğit
lr_model.fit(X_train, y_train)

# Tahmin yap
y_pred_lr = lr_model.predict(X_test)

# Performans raporu
print("Logistic Regression Performans:")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))




from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Performans:")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))




