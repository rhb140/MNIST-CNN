from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


#load Data set
(XTrain, yTrain), (XTest, yTest) = mnist.load_data()

#normalize the data - before the pixel values range from 0-255, now 0-1
XTrain = XTrain / 255.0
XTest = XTest / 255.0

#one-hot encode the y data
yTrain = to_categorical(yTrain)
yTest = to_categorical(yTest)

#create the model
model = Sequential([
    Conv2D(64, (3,3), activation = "relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), activation = "relu"), 
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(10, activation = "softmax")
])

#compile the model
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

#create an early stopping call back
earlyStop = EarlyStopping(monitor = "val_loss", patience = 3, restore_best_weights = True)

#train the model
history = model.fit(XTrain, yTrain, epochs = 50, batch_size = 64, validation_data = (XTest, yTest), callbacks = [earlyStop])

#evaluate the model
loss, accuracy = model.evaluate(XTest, yTest)
print(f"accuracy: {accuracy}\nloss: {loss}")

# Plotting Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


