import tensorflow as tf
import matplotlib.pyplot as plt
from model import CALSigNet


data = tf.keras.preprocessing.image_dataset_from_directory(
    "../dataset",
    image_size=(128,128),
    color_mode="grayscale",
    label_mode="binary",
    batch_size=32
)


data = data.map(lambda x, y: (x/255.0, y))


train_size = int(len(data)*0.8)
val_size = int(len(data)*0.2)

train_data = data.take(train_size)
val_data = data.skip(train_size).take(val_size)


model = CALSigNet()


history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)


model.save("../model/calsignet.h5")


plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()
plt.grid(True)


plt.savefig("../static/accuracy_graph.png")

plt.show()