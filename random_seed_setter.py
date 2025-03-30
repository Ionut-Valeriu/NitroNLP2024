import random

def reset_numpy_seed(seed_value=42):
  try:
    # Set NumPy random seed
    import numpy as np
    np.random.seed(seed_value)
    print(f'NumPy random seed set with value: {seed_value}')
  except Exception as e:
    print(f'NumPy random seed was not set: {e}')
  return


def reset_tensorflow_seed(seed_value=42):
  try:
    # Set TensorFlow random seed
    import tensorflow as tf
    success = False
    # Here we have 2 different ways to set the seed
    # depending on the version of TensorFlow
    try:
      tf.random.set_seed(seed_value)
      success = True
    except Exception as e:
      pass
    try:
      tf.set_random_seed(seed_value)
      success = True
    except Exception as e:
      pass
    if success:
      print(f'TensorFlow random seed set with value: {seed_value}')
    else:
      print(f'TensorFlow random seed was not set')
  except Exception as e:
    print(f'TensorFlow random seed was not set: {e}')
  return


def reset_torch_seed(seed_value=42):
  try:
    # Set PyTorch random seed
    import torch
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed_value)
      torch.cuda.manual_seed_all(seed_value)  # if you are using multiple GPUs
    print(f'PyTorch random seed set with value: {seed_value}')
  except Exception as e:
    print(f'PyTorch random seed was not set: {e}')
  return


def set_random_seeds(seed_value=42):
  # Set Python random seed
  random.seed(seed_value)
  reset_numpy_seed(seed_value)
  reset_tensorflow_seed(seed_value)
  reset_torch_seed(seed_value)
  return


if __name__ == '__main__':
  # Set the desired seed value
  seed = 42

  # Set random seeds
  set_random_seeds(seed)
