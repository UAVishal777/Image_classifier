import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import load_and_preprocess_data, build_model, plot_image, plot_value_array

def main():
    # Load and preprocess data
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
    
    # Build and compile the model
    model = build_model()
    
    # Train the model
    model.fit(train_images, train_labels, epochs=5)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc}')
    
    # Make predictions
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    
    # Plot the first X test images, their predicted labels, and the true labels
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()
    
    # Visualize a single prediction
    img = test_images[1]
    img = np.expand_dims(img, 0)
    predictions_single = probability_model.predict(img)

    print(predictions_single)
    
    plt.figure(figsize=(6, 3))
    plot_value_array(1, predictions_single[0], test_labels)
    _ = plt.xticks(range(10), ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'], rotation=45)
    plt.show()

if __name__ == "__main__":
    main()