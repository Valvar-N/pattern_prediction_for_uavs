import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 1. VERİ YÜKLEME
df = pd.read_excel("all_processed_with_labels_combined.xlsx")

# 2. LABEL DÖNÜŞÜMÜ (non-loiter: 0, loiter: 1)
df['label_encoded'] = df['label'].map({'non-loiter': 0, 'loiter': 1})

# 3. FEATURE ENGINEERING: loiter_radius → log dönüşümü
df['loiter_radius_log'] = np.log1p(df['loiter_radius'])

# 4. ÖZELLİK VE HEDEF TANIMI
X = df[['average_roll', 'speed_std', 'loiter_radius_log']]
y = df['label_encoded']

# 5. EĞİTİM/TEST AYIRIMI
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. MODEL TANIMLARI
models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', random_state=42))
    ]),
    "Random Forest": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])
}

# 7. EĞİTİM & DEĞERLENDİRME
results = {}
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {
        'classification_report': report,
        'confusion_matrix': cm
    }

# 8. SONUÇLARI YAZDIRMA
for model_name, result in results.items():
    print(f"\n🔍 Model: {model_name}")
    print("Classification Report:")
    print(pd.DataFrame(result['classification_report']).T)
    print("Confusion Matrix:")
    print(result['confusion_matrix'])

# 9. RANDOM FOREST ÖZELLİK ÖNEMLERİ
rf_model = models["Random Forest"].named_steps['clf']
feature_names = X.columns
importances = rf_model.feature_importances_

# Görselleştirme
plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=feature_names, palette="viridis")
plt.title("🎯 Özellik Önemleri (Random Forest)")
plt.xlabel("Önem Skoru")
plt.ylabel("Özellikler")
plt.tight_layout()
plt.show()
