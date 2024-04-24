import tensorflow as tf
import streamlit as st
import math
import numpy as np

CRACK_PASSWORD_PER_SECOND = 1000000000

model = tf.keras.models.load_model('passwordModel.keras')

def calc_entropy(password: str) -> float:
    """Calculate the entropy of a given password.

    Args:
        password (str): The password to calculate entropy for.

    Returns:
        float: The calculated entropy.
    """
    cardinality = calc_cardinality(password)
    length = len(password)
    sample_space = (cardinality) ** (length)
    return math.log(sample_space, 2)


def calc_cardinality(password: str) -> int:
    """Calculate the cardinality of a password.

    Args:
        password (str): The password to calculate cardinality for.

    Returns:
        int: The calculated cardinality.
    """
    lower, upper, digits, symbols = 0, 0, 0, 0
    for char in password:
        if char.islower():
            lower += 1
        elif char.isdigit():
            digits += 1
        elif char.isupper():
            upper += 1
        else:
            symbols += 1
    return lower + digits + upper + symbols


def entropy_to_crack_time(entropy: float) -> float:
    """Convert entropy to estimated crack time.

    Args:
        entropy (float): The entropy value.

    Returns:
        float: The estimated crack time in seconds.
    """
    return (0.5 * math.pow(2, entropy)) / CRACK_PASSWORD_PER_SECOND

def classify(password):
  length = len(password)
  entropy = calc_entropy(password)
  crackTime = entropy_to_crack_time(entropy)

  data = [[length, entropy, crackTime]]

  results = model.predict(data)
  category_mapping = {0: 'Average', 1: 'Strong', 2: 'Very strong', 3: 'Very weak', 4: 'Weak'}

  # Convert one-hot encoded predictions back to original categories
  decoded_predictions = []
  for prediction in results:
      category_index = np.argmax(prediction)  # Find the index of the maximum value (1)
      category = category_mapping[category_index]  # Get the corresponding category
      decoded_predictions.append(category)

  return decoded_predictions[0]

st.title("Password Strength Classifier")
password = st.text_input("Enter password here")
results = ""
if st.button('Predict'): 
  st.header(classify(password)) 
