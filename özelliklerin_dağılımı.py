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
