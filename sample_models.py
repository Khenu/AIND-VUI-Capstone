from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM,
    Multiply, Add, Dropout)


# Model 0: RNN
def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True,
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


# Model 1: Model 1: RNN + TimeDistributed Dense
def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_simp_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    # linear or relu?
    time_dense = TimeDistributed(Dense(output_dim, activation='linear'),
        name='timedist')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


# For Model 2
def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


# Model 2: CNN + RNN + TimeDistributed Dense
def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_simp_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, activation='linear'),
        name='timedist')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

# OLD
# # Model 3: Deeper RNN + TimeDistributed Dense
# def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
#     """ Build a deep recurrent network for speech
#     """
#     # Main acoustic input
#     input_data = Input(name='the_input', shape=(None, input_dim))
#     # TODO: Add recurrent layers, each with batch normalization
#     # First recurrent layer
#     simp_rnn = SimpleRNN(units, activation='relu',
#                          return_sequences=True, implementation=2,
#                          name='rnn')(input_data)
#     bn_rnn = BatchNormalization(name='bn_simp_rnn')(simp_rnn)
#     # Additional recurrent layers
#     if recur_layers > 1:
#         for i in range(2, recur_layers+1):
#             simp_rnn = SimpleRNN(units, activation='relu',
#                                  return_sequences=True, implementation=2,
#                                  name='rnn_{}'.format(i))(bn_rnn)
#             bn_rnn = BatchNormalization(name='bn_rnn_{}'.format(i))(simp_rnn)
#
#     # TODO: Add a TimeDistributed(Dense(output_dim)) layer
#     time_dense = TimeDistributed(Dense(output_dim, activation='relu'),
#                                  name='timedist')(bn_rnn)
#     # Add softmax activation layer
#     y_pred = Activation('softmax', name='softmax')(time_dense)
#     # Specify the model
#     model = Model(inputs=input_data, outputs=y_pred)
#     model.output_length = lambda x: x
#     print(model.summary())
#     return model




# Model 3: Deeper RNN + TimeDistributed Dense
def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    # First recurrent layer
    simp_rnn = SimpleRNN(units, activation='tanh',
                         return_sequences=True, implementation=2,
                         name='rnn')(input_data)
    bn_rnn = BatchNormalization(name='bn_simp_rnn')(simp_rnn)
    # Additional recurrent layers
    if recur_layers > 1:
        for i in range(2, recur_layers+1):
            simp_rnn = SimpleRNN(units, activation='tanh',
                                 return_sequences=True, implementation=2,
                                 name='rnn_{}'.format(i))(bn_rnn)
            bn_rnn = BatchNormalization(name='bn_rnn_{}'.format(i))(simp_rnn)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, activation='relu'),
                                 name='timedist')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model




# Model 4: Bidirectional RNN + TimeDistributed Dense
def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(
        SimpleRNN(units, activation='relu', return_sequences=True, implementation=2),
        merge_mode='concat', name='bidir_rnn')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, activation='linear'),
        name='timedist')(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model



# Final model
def final_model():
    """ Build a deep network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    ...
    # TODO: Add softmax activation layer
    y_pred = ...
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = ...
    print(model.summary())
    return model



# residual block
def res_block(input_, filters, kernel_size, dilation_rate, layer_grp_num):
    '''
    Residual block of our model, which is stacked many times in the network.
    Produces both residual and parameterised skip connections.
    '''

    prefix = 'block_%d_%d_' % (layer_grp_num, dilation_rate)

    residual = input_

    # tanh/filter convolution
    tanh_out = Conv1D(filters, kernel_size=kernel_size,
                      dilation_rate=dilation_rate, padding='causal', activation='tanh',
                      name=''.join((prefix, 'conv_tanh')))(input_)

    bn_tanh_out = BatchNormalization(name=''.join((prefix, 'bn_tanh_out')))(tanh_out)

    # sigmoid/gate convolution
    sigmoid_out = Conv1D(filters, kernel_size=kernel_size,
                         dilation_rate=dilation_rate, padding='causal', activation='sigmoid',
                         name=''.join((prefix, 'conv_sigmoid')))(input_)

    bn_sigmoid_out = BatchNormalization(name=''.join((prefix, 'bn_sigmoid_out')))(sigmoid_out)

    # Element-wise multiplication
    merged = Multiply(name=''.join((prefix, 'multi')))([bn_tanh_out, bn_sigmoid_out])

    # Send through 1x1 convolution to create skip_connection, F(x, {W_i}), the
    # residual mapping to be learned.
    skip_out = Conv1D(filters=filters, kernel_size=1, activation='tanh', padding='causal',
                      name=''.join((prefix, 'conv_out')))(merged)

    # Batch normalization
    bn_skip_out = BatchNormalization(name=''.join((prefix, 'bn_skip_out')))(skip_out)

    # final output, y = F(x) + x, element-wise addition
    out = Add(name=''.join((prefix, 'add')))([bn_skip_out, residual])

    # Return y and the skip_connection
    return out, bn_skip_out



def wavenet_model(input_dim, output_dim=29, num_layer_grps=3, filters=128):

    # Main acoustic input
    input_data = Input(shape=(None, input_dim), name='the_input')

    # Expand dimensions from the 13 of MFCC to filters
    conv_in = Conv1D(filters=filters, kernel_size=1, padding='valid',
               activation='tanh', name='conv_in')(input_data)

    # Batch normalization
    z = BatchNormalization(name='bn_conv_in')(conv_in)

    # Dilated convolution block loop
    # skip connections
    # Create the skip connections and place them in a list
    skip_connections_list = []
    for i in range(num_layer_grps):
        for r in [1, 2, 4, 8, 16]:
            z, skip = res_block(z, filters=filters, kernel_size=7,
                                dilation_rate=r, layer_grp_num=i)
            skip_connections_list.append(skip)
    # This is the "+" adding all the skip connections in the diagram.
    skip_sum = Add(name='skip_sum')(skip_connections_list)

    # Final logit layers
    # 1 x 1 with tanh
    conv_1 = Conv1D(filters=filters, kernel_size=1, strides=1,
                    padding='same', activation='tanh',
                    name='conv_1')(skip_sum)

    # Batch normalization
    bn_conv_1 = BatchNormalization(name='bn_conv_1')(conv_1)

    # 1 x 1
    y_pred = Conv1D(filters=output_dim, kernel_size=1, strides=1,
                    padding='same', activation='softmax',
                    name='conv_2')(bn_conv_1)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model



#########################################


# residual block
def res_block_2(input_, filters, kernel_size, dilation_rate, layer_grp_num):
    '''
    Residual block of our model, which is stacked many times in the network.
    Produces both residual and parameterised skip connections.
    '''

    prefix = 'block_%d_%d_' % (layer_grp_num, dilation_rate)

    residual = input_

    # tanh/filter convolution
    tanh_out = Conv1D(filters, kernel_size=kernel_size,
                      dilation_rate=dilation_rate, padding='causal', activation='tanh',
                      name=''.join((prefix, 'conv_tanh')))(input_)

    bn_tanh_out = BatchNormalization(name=''.join((prefix, 'bn_tanh_out')))(tanh_out)

    # sigmoid/gate convolution
    sigmoid_out = Conv1D(filters, kernel_size=kernel_size,
                         dilation_rate=dilation_rate, padding='causal', activation='sigmoid',
                         name=''.join((prefix, 'conv_sigmoid')))(input_)

    bn_sigmoid_out = BatchNormalization(name=''.join((prefix, 'bn_sigmoid_out')))(sigmoid_out)

    # Element-wise multiplication
    merged = Multiply(name=''.join((prefix, 'multi')))([bn_tanh_out, bn_sigmoid_out])

    # Send through 1x1 convolution to create skip_connection, F(x, {W_i}), the
    # residual mapping to be learned.
    skip_out = Conv1D(filters=filters, kernel_size=1, activation='tanh', padding='causal',
                      name=''.join((prefix, 'conv_out')))(merged)

    # Batch normalization
    bn_skip_out = BatchNormalization(name=''.join((prefix, 'bn_skip_out')))(skip_out)

    # final output, y = F(x) + x, element-wise addition
    out = Add(name=''.join((prefix, 'add')))([bn_skip_out, residual])

    # Return y and the skip_connection
    return out, bn_skip_out




def wavenet_model_2(input_dim, output_dim=29, num_layer_grps=3, filters=128):

    # Main acoustic input
    input_data = Input(shape=(None, input_dim), name='the_input')

    # Expand dimensions from the 13 of MFCC to filters
    conv_in = Conv1D(filters=filters, kernel_size=1, padding='valid',
               activation='tanh', name='conv_in')(input_data)

    # Batch normalization
    z = BatchNormalization(name='bn_conv_in')(conv_in)

    # Dilated convolution block loop
    # skip connections
    # Create the skip connections and place them in a list
    skip_connections_list = []
    for i in range(num_layer_grps):
        for r in [1, 2, 4, 8, 16]:
            z, skip = res_block_2(z, filters=filters, kernel_size=7,
                                dilation_rate=r, layer_grp_num=i)
            skip_connections_list.append(skip)
    # This is the "+" adding all the skip connections in the diagram.
    skip_sum = Add(name='skip_sum')(skip_connections_list)

    # Final logit layers
    # 1 x 1 with tanh
    conv_1 = Conv1D(filters=filters, kernel_size=1, strides=1,
                    padding='same', activation='tanh',
                    name='conv_1')(skip_sum)

    # Batch normalization
    bn_conv_1 = BatchNormalization(name='bn_conv_1')(conv_1)

    # Dropout layer to reduce overfitting
    drop_conv_1= Dropout(0.2)(bn_conv_1)

    # 1 x 1
    y_pred = Conv1D(filters=output_dim, kernel_size=1, strides=1,
                    padding='same', activation='softmax',
                    name='conv_2')(drop_conv_1)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model
