import seaborn as sns

plt.figure(figsize=(15, 5))
for i, feature in enumerate(features):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x='label', y=feature, data=df)
    plt.title(f'{feature} değerleri label bazında')
plt.tight_layout()
plt.show()