import tensorflow as tf

# MNIST veri setini yükledik
mnist_veri = tf.keras.datasets.mnist
(egitim_verisi, egitim_etiketleri), (test_verisi, test_etiketleri) = mnist_veri.load_data()

# veriyi normalleştirdik
veri_egitim = tf.keras.utils.normalize(egitim_verisi, axis=1)
veri_test = tf.keras.utils.normalize(test_verisi, axis=1)

# model oluşturduk
model = tf.keras.models.Sequential()

# katman ekledik
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# modeli derledik
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# modeli eğittik
model.fit(veri_egitim, egitim_etiketleri, epochs=5)

# modeli kaydettik
model.save('el_yazisi_model.h5')

# model testinin doğruluk oranı
kayip, dogruluk = model.evaluate(veri_test, test_etiketleri)
print(f"Doğruluk Oranı: {dogruluk}")