import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

def crop_and_concat(net1, net2):
    """
    the size(net1) <= size(net2)
    """
    net1_shape = net1.get_shape().as_list()
    net2_shape = net2.get_shape().as_list()
      
    offsets = [0, (net2_shape[1] - net1_shape[1]) // 2, (net2_shape[2] - net1_shape[2]) // 2, 0]
    size = [-1, net1_shape[1], net1_shape[2], -1]
    net2_resize = tf.slice(net2, offsets, size)
    return tf.concat([net1, net2_resize], 3)

class Model:
    def __init__(self, config):
        self.depths = config.depths
        self.filters_root = config.filters_root
        self.kernel_size = config.kernel_size
        self.dilation_rate = config.dilation_rate
        self.pool_size = config.pool_size
        self.X_shape = config.X_shape
        self.Y_shape = config.Y_shape
        self.n_channel = config.n_channel
        self.n_class = config.n_class
        self.drop_rate = config.drop_rate

    def get_model(self):
        
        logging.info("Model: depths {depths}, filters {filters}, "
           "filter size {kernel_size[0]}x{kernel_size[1]}, "
           "pool size: {pool_size[0]}x{pool_size[1]}, "
           "dilation rate: {dilation_rate[0]}x{dilation_rate[1]}".format(
            depths=self.depths,
            filters=self.filters_root,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            pool_size=self.pool_size))
        
        convs = [None]*self.depths # store output of each depth
        
        inputs = keras.Input(shape=self.X_shape)
        
        ### [First half of the network: downsampling inputs] ###
        # Entry block
        net = layers.Conv2D(filters=self.filters_root,kernel_size=self.kernel_size,activation=None,padding='same',dilation_rate=self.dilation_rate,kernel_initializer='glorot_uniform',
                       kernel_regularizer=None)(inputs)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)
        net = layers.Dropout(self.drop_rate)(net)
        # Down sample layers
        for depth in range(0, self.depths):
            filters = int(2**(depth) * self.filters_root)
            net = layers.Conv2D(filters=filters,kernel_size=self.kernel_size,activation=None,use_bias=False,padding='same',dilation_rate=self.dilation_rate,kernel_initializer='glorot_uniform',
                       kernel_regularizer=None)(net)
            net = layers.BatchNormalization()(net)
            net = layers.Activation("relu")(net)
            net = layers.Dropout(self.drop_rate)(net)
            
            convs[depth] = net
            
            if depth < self.depths-1:
                net = layers.Conv2D(filters=filters,kernel_size=self.kernel_size,strides=self.pool_size,activation=None,use_bias=False,padding='same',dilation_rate=self.dilation_rate,kernel_initializer='glorot_uniform',
                       kernel_regularizer=None)(net)
                net = layers.BatchNormalization()(net)
                net = layers.Activation("relu")(net)
                net = layers.Dropout(self.drop_rate)(net)
    
        # Up sample layers
        for depth in range(self.depths - 2, -1, -1):
            filters = int(2**(depth) * self.filters_root)
            net = layers.Conv2DTranspose(filters=filters,kernel_size=self.kernel_size,strides=self.pool_size,activation=None,use_bias=False,padding='same',kernel_initializer='glorot_uniform',
                       kernel_regularizer=None)(net)
            net = layers.BatchNormalization()(net)
            net = layers.Activation("relu")(net)
            net = layers.Dropout(self.drop_rate)(net)
            
    #         net = layers.add([convs[depth], net]) # Skip connection
            net = crop_and_concat(convs[depth], net) # Skip connection
            
            net = layers.Conv2D(filters=filters,kernel_size=self.kernel_size,activation=None,use_bias=False,padding='same',dilation_rate=self.dilation_rate,kernel_initializer='glorot_uniform',
                       kernel_regularizer=None)(net)
            net = layers.BatchNormalization()(net)
            net = layers.Activation("relu")(net)
            net = layers.Dropout(self.drop_rate)(net)
        # Output Map
        net = layers.Conv2D(filters=self.n_class,kernel_size=(1,1),activation=None,padding='same',kernel_initializer='glorot_uniform',
                       kernel_regularizer=None)(net)
        outputs = layers.Softmax()(net)
        
        # Define model
        model = keras.Model(inputs, outputs)
        return model

