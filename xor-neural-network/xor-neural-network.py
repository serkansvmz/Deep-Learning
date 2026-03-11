# XOR problemini çözen 2 katmanlı sinir ağı projesi
# Author: Serkan Sevmez
#--------------------------------

import numpy as np

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

np.random.seed(42)                    # Random sayıların her çalıştırmada aynı çıkmasını sağlar.

W1 = np.random.rand(2,2)              # 2 giriş, 2 gizli nöron
b1 = np.zeros((1,2))                  # nöronlara [0 0] sabit değerleri atıyoruz

W2 = np.random.rand(2,1)             
b2 = np.zeros((1,1))

def sigmoid(x):                 
    return 1/(1+np.exp(-x))           # değeri 0 ile 1 arasına sıkıştırıyoruz, örnek; 5 = 0.99,  -5 = 0.01

def sigmoid_derivative(x):            # türev alarak hatayı azaltıyoruz
    return x*(1-x)

learning_rate = 0.1                   # ağ öğrenme hızı 

for epoch in range(4000):            # ağı 10000 kez eğitiyoruz her döngü 1

    # Forward
    hidden_input = np.dot(X,W1) + b1            # input -> gizli katman Z = XW + b
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output,W2) + b2  # çıkış nöronuna veri gönderilir
    final_output = sigmoid(final_input)          # modelin tahmini

    # Hata
    error = y - final_output                     # hata payı hesaplama

    # Backprop
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Ağı güncelle
    W2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    print(final_output)
