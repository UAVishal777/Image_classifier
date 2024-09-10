import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define the class names here so it's available in all functions
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_and_preprocess_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    return (train_images, train_labels), (test_images, test_labels)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model

def plot_image(i, predictions_array, true_labels, img):
    true_label = true_labels[i]
    img = img[i]
    
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'
    
    # Safeguard against index errors
    predicted_class_name = class_names[predicted_label] if predicted_label < len(class_names) else 'Unknown'
    true_class_name = class_names[true_label] if true_label < len(class_names) else 'Unknown'
    
    plt.xlabel(f"{predicted_class_name} {100 * np.max(predictions_array):2.0f}% ({true_class_name})", color=color)

def plot_value_array(i, predictions_array, true_labels):
    true_label = true_labels[i]
    
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    
    predicted_label = np.argmax(predictions_array)
    
    # Ensure the predicted and true labels are within the class_names range
    if predicted_label < len(class_names):
        thisplot[predicted_label].set_color('red')
    if true_label < len(class_names):
        thisplot[true_label].set_color('blue')
