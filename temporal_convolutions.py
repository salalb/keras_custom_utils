import keras.backend as K
from keras.layers import Dense, Activation, Conv1D
from keras.layers import Lambda, SpatialDropout1D, Add

from .keras_base_block import KerasBlock


class DilatedCausalConv1D(KerasBlock):
    """
        Single layer block of a 1D conv layer with dilation.
        When calling as_model, it's expecting tensors of shape (m, timesteps, filters/features),
        thus input_shape=(timesteps, filters).
    """

    def __init__(self,
                name,
                filters,
                kernel_size,
                strides=1,
                dilation_rate=1,
                output_dim=None,
                activation=None,
                use_bias=True,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True
                ):
        self.name = name
        self.output_dim = output_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.trainable = trainable
        self.conv = Conv1D(
                filters = self.filters,
                kernel_size = self.kernel_size,
                strides = self.strides,
                padding = "causal",
                data_format = "channels_last",
                dilation_rate = self.dilation_rate,
                activation = self.activation,
                use_bias = self.use_bias,
                kernel_initializer = self.kernel_initializer,
                bias_initializer = self.bias_initializer,
                kernel_regularizer = self.kernel_regularizer,
                bias_regularizer = self.bias_regularizer,
                activity_regularizer = self.activity_regularizer,
                kernel_constraint = self.kernel_constraint,
                bias_constraint =self.bias_constraint,
                trainable = self.trainable,
                name = self.name
            )

    def __call__(self, input_X):
        X = self.conv(input_X)
        return X

    # def as_model(self, input_shape, use_dense=True, n_output=None, activation=None, **kargs):
    # This is inherited from the base class (KerasBlock)

class TemporalBlock(KerasBlock):
    """
        Temporal residual block as in Temporal Convolutional Networks
    """

    def __init__(self,
                filters,
                kernel_size,
                strides,
                dilation_rate,
                dropout=0.2,
                output_dim=None,
                trainable=True,
                name=None,
                activity_regularizer=None
                ):
        self.dropout = dropout
        self.filters = filters
        self.output_dim = output_dim
        self.trainable = trainable
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.strides = strides
        self.activity_regularizer = activity_regularizer
        self.name = name
        self.conv1 = DilatedCausalConv1D(
            name="{}_conv1".format(self.name), filters=self.filters, kernel_size=self.kernel_size,
            strides=self.strides, dilation_rate=dilation_rate, activation="relu",
            trainable=self.trainable, activity_regularizer=self.activity_regularizer
        )
        self.conv2 = DilatedCausalConv1D(
            name="{}_conv2".format(self.name), filters=self.filters, kernel_size=self.kernel_size,
            strides=self.strides, dilation_rate=dilation_rate, activation="relu",
            trainable=self.trainable, activity_regularizer=self.activity_regularizer
        )

    def __call__(self, input_X):
        shortcut = input_X
        X = self.conv1(input_X)
        X = Lambda(self.__channel_normalization, name="{}_norm1".format(self.name))(X)
        X = SpatialDropout1D(self.dropout, name="{}_dropout1".format(self.name))(X)
        X = self.conv2(X)
        X = Lambda(self.__channel_normalization, name="{}_norm2".format(self.name))(X)
        X = SpatialDropout1D(self.dropout, name="{}_dropout2".format(self.name))(X)
        shortcut = Conv1D(self.filters, 1, padding="same", name="{}_1x1".format(self.name))(shortcut)
        X = Add()([X, shortcut])
        X = Activation("relu", name="{}_relu".format(self.name))(X)
        return X

    def __channel_normalization(self, input_X):
        max_values = K.max(K.abs(input_X), 2, keepdims=True) + 1e-6
        X = input_X / max_values
        return X

    # def as_model(self, input_shape, use_dense=True, n_output=None, activation=None, **kargs):
    # This is inherited from the base class (KerasBlock)

class TemporalConvNet(KerasBlock):
    """
        Temporal Convolutional Network
    """
    def __init__(self,
                filters_set,
                kernel_size=2,
                dropout=0.2,
                trainable=True,
                name=None,
                output_dim=None,
                activity_regularizer=None
                ):

        self.layers = []
        self.filters_set = filters_set
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.trainable = trainable
        self.name = name
        self.output_dim = output_dim
        self.activity_regularizer = activity_regularizer
        self.dilations = []
        num_levels = len(filters_set)
        if not num_levels:
            raise ValueError("Empty filters_set")
        for i in range(num_levels):
            dilation_size = 2 ** i
            self.dilations.append(dilation_size)
            filters = filters_set[i]
            self.layers.append(
                TemporalBlock(
                    filters, kernel_size, strides=1, dilation_rate=dilation_size,
                    dropout=dropout, trainable=trainable, name="tblock_{}".format(i),
                    activity_regularizer=activity_regularizer
                )
            )
        self.receptive_field = 1 + 2*(self.kernel_size - 1)*(2**num_levels - 1)

    def __call__(self, input_X):
        X = input_X
        for layer in self.layers:
            X = layer(X)
        return X

    # def as_model(self, input_shape, use_dense=True, n_output=None, activation=None, **kargs):
    # This is inherited from the base class (KerasBlock)
    
