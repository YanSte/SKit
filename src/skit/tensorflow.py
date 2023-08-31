from skit.config import IS_TENSORFLOW_IMPORTED

if IS_TENSORFLOW_IMPORTED:
    import tensorflow as tf

    def configure_gpu_memory(memory_limit=12288):
        """
        Configure GPU memory for TensorFlow.

        Parameters:
            memory_limit (int): The memory limit in megabytes. Defaults to 12288 (12 GB).

        Returns:
            None
        """
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            except RuntimeError as e:
                print("Virtual devices must be set before GPUs have been initialized.")
                print(e)
