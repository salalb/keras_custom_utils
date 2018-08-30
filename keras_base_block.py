import warnings
from keras.models import Model
from keras.layers import Input, Dense

class KerasBlock:
    """
        Base class for custom Keras blocks.

        It generates a Keras block (stack of layers) that are callable as in the Keras functional
        API. It can also be returned as a Model instance from keras.model calling the as_model
        method.
    """

    def __init__(self, *args, **kargs):
        raise NotImplementedError("__init__ method not implemented")

    def __call__(self, X):
        raise NotImplementedError("__call__ method not implemented")

    def as_model(self, input_shape, use_dense=True, n_output=None, activation=None, **kargs):
        """
            Returns a keras.models.Model instance from this stack of layersself.

            input_shape: shape of the input tensorself.
            use_dense:   False by default. If True, will add an extra Dense layer at the end,
                         with output dimension equal to self.output_dim or n_output (default)
                         if passed.
            n_output:    output dimension if using an extra Dense layer at the end. If n_output
                         is False, self.output_dim will be used. If both are defined, n_output
                         is used.
            activation:  None (linear) by default. Only used if use_dense=True, is the Activation
                         of the last Dense layer.
        """
        input = Input(shape=input_shape)
        output = self.__call__(input)
        if use_dense:
            if self.output_dim and not n_output:
                output = Dense(self.output_dim, activation=activation, trainable=self.trainable)(output)
            elif not self.output_dim and n_output:
                output = Dense(n_output, activation=activation, trainable=self.trainable)(output)
            elif self.output_dim and n_output:
                warnings.warn("Both self.output_dim and n_output are not None, using the later.")
                output = Dense(n_output, activation=activation, trainable=self.trainable)(output)
            elif not self.output_dim and not n_output:
                raise ValueError("output dimension not initialised nor passed with n_output")

        return Model(inputs=input, outputs=output)
