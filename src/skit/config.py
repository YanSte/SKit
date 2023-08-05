# ==============================
#            Env
# ==============================

# Env Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# ==============================
#           Library
# ==============================

# Env with library tensorflow
try:
    import tensorflow as tf
    IS_TENSORFLOW_IMPORTED = True
except ImportError:
    IS_TENSORFLOW_IMPORTED = False
