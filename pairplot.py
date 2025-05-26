import seaborn as sns

sns.pairplot(df, hue='label', vars=features, diag_kind='kde', plot_kws={'alpha':0.6})
plt.suptitle('Özelliklerin Pairplot’u (Loiter / Non-loiter)', y=1.02)
plt.show()