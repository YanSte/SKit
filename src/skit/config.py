try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


try:
    import tensorflow as tf
    IS_TENSORFLOW_IMPORTED = True
except ImportError:
    IS_TENSORFLOW_IMPORTED = False
