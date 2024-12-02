import tensorflow as tf
import numpy as np
import cv2

# Constants
MODEL_PATH = 'model.h5'
MNIST_PATH = 'mnist.npz'
THRESHOLD_INIT = 150
DETECTION_BOX_SIZE = 150  # Size of the square box for the detected region

# Load MNIST Data
def get_mnist_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=MNIST_PATH)
    return (x_train / 255.0, y_train, x_test / 255.0, y_test)

# Build and Train Model
def build_and_train_model(x_train, y_train, x_test, y_test):
    class StopTrainingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs and logs.get('accuracy') > 0.99:
                print("\nReached 99% accuracy, stopping training.")
                self.model.stop_training = True

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[StopTrainingCallback()])
    return model

# Digit Prediction
def predict_digit(model, img):
    img = img.reshape(1, 28, 28)
    predictions = model.predict(img)
    return str(np.argmax(predictions))

# OpenCV Logic
def start_opencv_loop(model):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Digit Recognizer')

    def toggle_inference(event, x, y, flags, param):
        nonlocal start_inference
        if event == cv2.EVENT_LBUTTONDOWN:
            start_inference = not start_inference

    def update_threshold(value):
        nonlocal threshold
        threshold = value

    cv2.setMouseCallback('Digit Recognizer', toggle_inference)
    cv2.createTrackbar('Threshold', 'Digit Recognizer', THRESHOLD_INIT, 255, update_threshold)

    # Variables
    start_inference = False
    threshold = THRESHOLD_INIT
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray_frame)  # Black mask for outside the box

        if start_inference:
            _, thr = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY_INV)

            # Extract central ROI for prediction
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
            half_box = DETECTION_BOX_SIZE // 2
            roi = thr[center_y - half_box:center_y + half_box, center_x - half_box:center_x + half_box]

            # Apply mask to keep only the center box visible
            mask[center_y - half_box:center_y + half_box, center_x - half_box:center_x + half_box] = \
                thr[center_y - half_box:center_y + half_box, center_x - half_box:center_x + half_box]

            # Resize ROI to 28x28 for prediction
            resized_img = cv2.resize(roi, (28, 28))
            prediction = predict_digit(model, resized_img)

            # Draw detection box and prediction
            cv2.rectangle(mask, (center_x - half_box, center_y - half_box),
                          (center_x + half_box, center_y + half_box), (255, 255, 255), 2)
            cv2.putText(mask, prediction, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

            # Display the masked frame
            cv2.imshow('Digit Recognizer', mask)
        else:
            # Show normal frame if not inferring
            cv2.imshow('Digit Recognizer', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Execution
def main():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print('Loaded saved model.')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training a new model...")
        x_train, y_train, x_test, y_test = get_mnist_data()
        model = build_and_train_model(x_train, y_train, x_test, y_test)
        model.save(MODEL_PATH)

    print("Starting OpenCV loop...")
    start_opencv_loop(model)

if __name__ == '__main__':
    main()