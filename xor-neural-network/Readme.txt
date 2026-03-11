"""
Ağı 100 kez eğittiğimizde çıktı: [[0.49466996]
                                  [0.50305173]
                                  [0.50335617]
                                  [0.51008834]]

Ağı 1000 kez eğittiğimizde çıktı: [[0.47709913]
                                   [0.50469944]
                                   [0.50943575]
                                   [0.52600513]]

Ağı 10000 kez eğittiğimizde çıktı: [[0.05587752]
                                    [0.94888694]
                                    [0.94881863]
                                    [0.05495454]]

Ağı 100000 kez eğittiğimizde çıktı: [[0.01298684]
                                     [0.98890291]
                                     [0.98889676]
                                     [0.01141451]]

"""
"""
Biz modele şu dört girdiyi veriyoruz:

[0,0]
[0,1]
[1,0]
[1,1]

Yani örneğin:

input = [0,1] Bu iki özellikli bir veri.

Peki 0.55 gibi sayı nasıl çıkıyor?

Çünkü model doğrudan 0 veya 1 üretmez. Önce bir olasılık değeri üretir.

Mesela model hesaplıyor:

output = 0.55 Bu şu anlama gelir:

%55 ihtimalle 1  %45 ihtimalle 0, Yani model emin değil.

Bu sayı nereden geliyor? Sinir ağı aslında şu işlemi yapıyor:

input × ağırlık + bias  Sonra bunu sigmoid fonksiyonundan geçiriyoruz.

Sigmoid fonksiyonu:

sigmoid(x) = 1 / (1 + e^-x)  Bu fonksiyonun özelliği: çıktı her zaman 0 ile 1 arasındadır 
Örnek:

giriş	sigmoid çıkışı
-3	0.047
0	0.5
3	0.95
Küçük bir örnek yapalım

Diyelim input: [0,1]

Ağırlıklar:[0.4, 0.7]

İşlem: 0×0.4 + 1×0.7 = 0.7

Sigmoid:  sigmoid(0.7) ≈ 0.66

Modelin çıktısı: 0.66

Yani model diyor ki: %66 ihtimalle sonuç = 1

Eğitimden sonra ne olur? 

Başta tahminler şöyle olabilir:

0.52
0.55
0.48
0.50

Ama eğitimden sonra:

0.02
0.97
0.96
0.03

Yani model artık çok daha emin.

Kısaca olay

Input: [0,1]

Model içinde olanlar: çarpma + toplama + sigmoid

Output: 0.55 (olasılık)

"""