from tensorflow.keras import layers,models


def add_batchnorm_relu(inp):
    inp = layers.BatchNormalization()(inp)
    inp = layers.ReLU()(inp)
    return inp


class Resnext:

    def __init__(self,block_strength = [3,4,6,3],cardinality=32):
        self.block_strength = block_strength
        self.cardinality = cardinality

        self.img_height = 224
        self.img_width = 224
        self.img_channels = 3

    def build_resnext(self):
        index = 0
        image_tensor = layers.Input(shape=(self.img_height, self.img_width, self.img_channels))

        # conv1
        first_conv = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(image_tensor)
        first_conv = add_batchnorm_relu(first_conv)

        #Max Pool
        max_pool = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(first_conv)

        #Block1
        block1 = BottleneckBlock(max_pool,64,128,32)
        block1 = block1.build_bottleneck_block(self.block_strength[index])
        index = index+1

        # Block1
        block2 = BottleneckBlock(block1, 128, 256, 32)
        block2 = block2.build_bottleneck_block(self.block_strength[index])
        index = index + 1

        # Block1
        block3 = BottleneckBlock(block2, 256, 512, 32)
        block3 = block3.build_bottleneck_block(self.block_strength[index])
        index = index + 1

        # Block1
        block4 = BottleneckBlock(block3, 512, 1024, 32)
        block4 = block4.build_bottleneck_block(self.block_strength[index])

        #print(block4.shape)

        gap = layers.GlobalAveragePooling2D()(block4)
        #print(gap.shape)

        fc = layers.Dense(10)(gap)
        #print(fc.shape)

        model = models.Model(inputs=[image_tensor], outputs=[fc])
        #print(model.summary())
        return model


class BottleneckBlock:
    def __init__(self,input_previous,input_channels,output_channels,cardinality):

        self.cardinality = cardinality
        self.input_previous = input_previous
        self.input_channels = input_channels
        self.output_channels = output_channels

    def build_bottleneck_block(self,size):
        for i in range(size):
            project_shortcut = True if i == 0  else False
            #print('Debug:',i,project_shortcut)
            self.input_previous = self.residual_block(_project_shortcut=project_shortcut)
        return self.input_previous


    def residual_block(self, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = self.input_previous

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(self.input_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(self.input_previous)
        y = add_batchnorm_relu(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = self.grouped_convolution(y, self.input_channels, _strides=_strides)
        y = add_batchnorm_relu(y)

        y = layers.Conv2D(self.output_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        y = self.shortcut_connect(_project_shortcut,y,shortcut)

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.ReLU()(y)

        return y

    def shortcut_connect(self,_project_shortcut,residual,shortcut):
        #Projection Shortcut
        if _project_shortcut:
            #print('Project shortcut called')
            #print(residual.shape,shortcut.shape)
            shortcut = layers.Conv2D(self.output_channels, kernel_size=(1, 1), strides=(1,1), padding='same')(
                shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # Identity Shortcut
        y = layers.add([shortcut, residual])
        return y

    def grouped_convolution(self,inp, nb_channels, _strides):


        group_ch = nb_channels // self.cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(self.cardinality):
            group = layers.Lambda(
                lambda z: z[:, :, :, j * group_ch:j * group_ch + group_ch])(inp)
            groups.append(layers.Conv2D(group_ch, kernel_size=(3, 3), strides=_strides, padding='same')(group))

        # the grouped convolutional layer concatenates them as the outputs of the layer
        inp = layers.concatenate(groups)

        return inp


if __name__=='__main__':
    resnext = Resnext()
    model = resnext.build_resnext()
    print(model.summary())