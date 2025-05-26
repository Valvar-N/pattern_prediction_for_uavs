corr = df[features].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Feature'lar Arası Korelasyon Matrisi")
plt.show()