from keras.models import load_model
import numpy as np
from keras.preprocessing import image

# Load the trained model
classifier = load_model('Trained_model.h5')

# Prediction of single image
img_name = input('Enter Image Name: ')
image_path = './{}'.format(img_name)

test_image = image.load_img(image_path, target_size=(200, 200))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Predict the class probabilities for the test image
result = classifier.predict(test_image)

# Get the predicted class label
class_indices = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
                 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
                 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}

predicted_class_index = np.argmax(result)
predicted_class = [key for key, value in class_indices.items() if value == predicted_class_index][0]

print('Predicted Sign is:', predicted_class)

