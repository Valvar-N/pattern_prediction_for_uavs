# 📡 İHA Davranış Tahmini - Loitering Algılama Projesi

## 🧠 Proje Amacı  
Bu projenin amacı, **insansız hava araçlarının (İHA)** hareket örüntülerini analiz ederek **loitering (dairesel bekleme) davranışını** tespit edebilen bir model geliştirmektir. Özellikle askeri ve sivil uygulamalarda, bir İHA'nın niyetini (keşif mi yapıyor, saldırı mı hazırlıyor, bekliyor mu) anlamak büyük stratejik avantajlar sağlar.  

## 📌 Hedeflenen Davranış: Loitering  
Loitering, bir İHA'nın belirli bir bölge etrafında dairesel şekilde veya düşük hızda bekleyerek o bölgeyi gözlemlemesi anlamına gelir. Bu proje, bu tür davranışları otomatik olarak tespit edebilmek için veri analizi ve makine öğrenmesi yöntemleri kullanır.

---

## 📁 Veri Seti Özeti  

- Toplam veri süresi: Yaklaşık birkaç dakika (örnekleme: 10 Hz)
- Kullanılan özellikler:
  - `roll` (yatış açısı)
  - `speed` (hız)
  - `heading` (burun yönü)
- Veriler her **10 saniyede bir** (100 satır) gruplanarak ortalama alınır ve bu özetlenmiş veri ile model beslenir.
- Her örnek için çıkarılan sütunlar:
  - `average_roll` (ortalama yatış açısı)
  - `average_speed` (ortalama yer hızı (m/s))
  - `loiter_radius` (tahmini dönüş yarıçapı)
  - `label` (loitering mi değil mi)

---

## 📐 Loitering Yarıçapı Hesabı  

Dönüş yarıçapı şu formülle hesaplanır:

```
Loiter Radius = V^2 / (g * tan(φ))
```

- `V` : Ortalama hız (m/s)
- `φ` : Ortalama roll açısı (radyan)
- `g` : Yerçekimi ivmesi (9.81 m/s²)

Bu formül sabit kanatlı hava araçları için klasik dönüş yarıçapı hesaplamasıdır.

---

## ⚙️ Çıkış Formatı  

Model girişleri, 10 saniyelik zaman dilimlerine ait özetlerle oluşturulur:

| average_roll | average_speed | loiter_radius | label |
|--------------|---------------|----------------|--------|
| 12.4         | 21.3          | 34.2           | 1      |
| 3.1          | 24.9          | 130.5          | 0      |
| ...          | ...           | ...            | ...    |

- `label`: 1 → loiter, 0 → non-loiter

---

## 🤖 Kullanılan Makine Öğrenimi Yöntemleri

- **Logistic Regression:** Basit ama etkili bir sınıflandırıcı, baseline model olarak kullanıldı.
- **Random Forest Classifier:** Daha kompleks yapıları öğrenebilen, karar ağaçları topluluğu.
- **GridSearchCV:** Parametre optimizasyonu için kullanıldı.

---


## 📚 Atıf Verilen Kaynaklar

1. Tareen, A. et al. (2024). *Loitering Munition and Autonomous UAV Behavior Recognition: A Survey*. Drones.  
   [https://www.mdpi.com/2504-446X/8/6/255](https://www.mdpi.com/2504-446X/8/6/255)

2. Asad, M. et al. (2021). *A Survey of Deep Learning Methods for UAV Detection*. arXiv.  
   [https://arxiv.org/abs/2109.15158](https://arxiv.org/abs/2109.15158)

---

## ✨ Katkıda Bulunanlar

- Yiğit Elarslan
- Abdulkadir Turan
