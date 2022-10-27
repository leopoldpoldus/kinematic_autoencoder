from keras_applications import get_submodules_from_kwargs
import os
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet101, ResNet152, MobileNet, \
    MobileNetV2  # , InceptionV3, InceptionResNetV2

import tensorflow.keras.layers as k_layers
from tensorflow.keras.models import load_model
import tensorflow

backend = tensorflow.keras.backend
layers = tensorflow.keras.layers
models = tensorflow.keras.models
keras_utils = tensorflow.keras.utils


def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }


def Conv2dBn(
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_batchnorm=False,
        **kwargs):
    """Extension of Conv2D layer with batchnorm"""

    conv_name, act_name, bn_name = None, None, None
    block_name = kwargs.pop('name', None)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if block_name is not None:
        conv_name = block_name + '_conv'

    if block_name is not None and activation is not None:
        act_str = activation.__name__ if callable(activation) else str(activation)
        act_name = block_name + '_' + act_str

    if block_name is not None and use_batchnorm:
        bn_name = block_name + '_bn'

    bn_axis = 3

    def wrapper(input_tensor):

        x = k_layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=not (use_batchnorm),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=conv_name,
        )(input_tensor)

        if use_batchnorm:
            x = k_layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        if activation:
            x = k_layers.Activation(activation, name=act_name)(x)

        return x

    return wrapper


def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper


def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = 3

    def wrapper(input_tensor, skip=None):
        x = k_layers.UpSampling2D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            x = k_layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)

        return x

    return wrapper


def DecoderTransposeX2Block(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = bn_axis = 3

    def layer(input_tensor, skip=None):

        x = k_layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = k_layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = k_layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            x = k_layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer


def freeze_model(model, **kwargs):
    """Set all layers non-trainable, excluding BatchNormalization layers"""
    _, layers, _, _ = get_submodules_from_kwargs(kwargs)
    for layer in model.layers:
        if not isinstance(layer, k_layers.BatchNormalization):
            layer.trainable = False
    return


def filter_keras_submodules(kwargs):
    """Selects only arguments that define keras_application submodules. """
    submodule_keys = kwargs.keys() & {'backend', 'layers', 'models', 'utils'}
    return {key: kwargs[key] for key in submodule_keys}


default_feature_layers = {
    # VGG
    'vgg16': ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
    'vgg19': ('block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2'),
    # ResNets
    'resnet18': ('conv4_block1_1_relu', 'conv3_block1_1_relu', 'conv2_block1_1_relu', 'conv1_relu'),
    'resnet34': ('conv4_block1_1_relu', 'conv3_block1_1_relu', 'conv2_block1_1_relu', 'conv1_relu'),
    'resnet50': ('conv4_block1_1_relu', 'conv3_block1_1_relu', 'conv2_block1_1_relu', 'conv1_relu'),
    'resnet101': ('conv4_block1_1_relu', 'conv3_block1_1_relu', 'conv2_block1_1_relu', 'conv1_relu'),
    'resnet152': ('conv4_block1_1_relu', 'conv3_block1_1_relu', 'conv2_block1_1_relu', 'conv1_relu'),
    # Mobile Nets
    'mobilenet': ('conv_pw_11_relu', 'conv_pw_5_relu', 'conv_pw_3_relu', 'conv_pw_1_relu'),
    'mobilenetv2': ('block_13_expand_relu', 'block_6_expand_relu', 'block_3_expand_relu',
                    'block_1_expand_relu'),
    # Inception
    'inceptionv3': (228, 86, 16, 9),
    'inceltionresnetv2': (596, 260, 16, 9)
}
"""
# ResNets
'resnet18': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
'resnet34': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
'resnet50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
'resnet101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
'resnet152': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
"""

backbones = {
    'vgg16': VGG16,
    'vgg19': VGG19,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
    'mobilenet': MobileNet,
    'mobilenetv2': MobileNetV2,
    'kinematic': extract_kinematic_features,
    # 'inceptionv3': InceptionV3,
    # 'inceptionresnetv2': InceptionResNetV2
}

"""
UNET DECODER
"""


def build_unet(
        backbone,
        decoder_block,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True):
    input_ = backbone.input
    x = backbone.output

    # x = tensorflow.keras.layers.Multiply()([x, tensorflow.fill(tensorflow.shape(x), classifier_output)])

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # add center block if previous operation was max-pooling (for vgg models)

    if isinstance(backbone.layers[-1].__class__.__name__, k_layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)

    # building decoder blocks
    for i in range(n_upsample_blocks):
        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)

    avg = tensorflow.keras.layers.GlobalAveragePooling2D()(x)
    # classifier_output = tensorflow.keras.layers.Dense(1, activation='sigmoid', name='classification_output')(avg)

    # model head (define number of output classes)
    x = k_layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)
    x = k_layers.Activation(activation, name='segmentation_output')(x)

    # create keras model instance
    model = tensorflow.keras.models.Model(input_, [x])

    return model


def Unet(backbone_name='vgg16',
         input_shape=(None, None, 3),
         classes=1,
         activation='sigmoid',
         weights=None,
         encoder_weights='imagenet',
         encoder_freeze=False,
         encoder_features='default',
         decoder_block_type='upsampling',
         decoder_filters=(256, 128, 64, 32, 16),
         decoder_use_batchnorm=True,
         continue_training=False,
         model_path='',
         **kwargs):
    """
    Args:
        backbone_name: name of the encoder / feature extractor
        input_shape: (H, W, C). If H & W are not set, it can process any size (must be divisible by 32)
        classes: nr of classes for output
        activation: activation fn.
        weights: optional, path to model weights
        encoder_weights: None (random init) or imagenet (pre-trained on imagenet)
        encoder_freeze: if True, backbone layers are not trainable
        encoder_features: a list of layer numbers or names starting from top of the model.
            Each of these layers will be concatenated with corresponding decoder block. If ``default`` is used
            layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
        decoder_block_type:
            - `upsampling`:  ``UpSampling2D`` -> ``Conv2D`` -> ``Conv2D``
            - `transpose`:   ``Transpose2D`` -> ``Conv2D`
        decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.

    """
    if continue_training:
        if os.path.exists(model_path):
            model = load_model(model_path, compile=False)
            model.summary()
            return model
        else:
            print('Failed to load existing model at: {}'.format(model_path))

    global backend, layers, models, keras_utils
    submodule_args = filter_keras_submodules(kwargs)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)

    if decoder_block_type == 'upsampling':
        decoder_block = DecoderUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        decoder_block = DecoderTransposeX2Block
    else:
        raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_block_type))

    backbone = backbones[backbone_name](
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs)

    if encoder_features == 'default':
        encoder_features = default_feature_layers[backbone_name][:4]

    model = build_unet(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        activation=activation,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
    )

    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    if weights is not None:
        model.load_weights(weights)

    return model
