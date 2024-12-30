import cv2
import numpy as np
import tensorflow as tf

# modeli yükledik
model = tf.keras.models.load_model('el_yazisi_model.h5')

# boş bir çizim alanı olulşurduk
canvas = np.zeros((350, 350), dtype=np.uint8)
cizim_modu = False
ilk_x, ilk_y = -1, -1

# fare olayları için fonksiyon yazdık
def fare_olaylari(olay, x, y, bayraklar, param):
    global ilk_x, ilk_y, cizim_modu
    
    if olay == cv2.EVENT_LBUTTONDOWN:
        cizim_modu = True
        ilk_x, ilk_y = x, y

    elif olay == cv2.EVENT_MOUSEMOVE:
        if cizim_modu:
            cv2.line(canvas, (ilk_x, ilk_y), (x, y), 255, kalinlik)
            ilk_x, ilk_y = x, y

    elif olay == cv2.EVENT_LBUTTONUP:
        cizim_modu = False

# pencere oluşturup ve fareyi bağladık
cv2.namedWindow("Rakam Ciz")
cv2.setMouseCallback("Rakam Ciz", fare_olaylari)

print("Rakam çizip 'q' tuşuna basarak tahmin alın.")
kalinlik = 15

while True:
    cv2.imshow("Rakam Ciz", canvas)
    tus = cv2.waitKey(1) & 0xFF

    if tus == ord('q'):
        break

#çizim alanını 28-28 e getirip işledik
kucuk_resim = cv2.resize(canvas, (28, 28))
kucuk_resim = kucuk_resim / 255.0
model_girdisi = np.reshape(kucuk_resim, (1, 28, 28))

#tahmin yaptık
sonuc = model.predict(model_girdisi)
tahmin = np.argmax(sonuc)

print(f"Tahmin: {tahmin}")

# en son açılır alanı kapadık
cv2.destroyAllWindows()
