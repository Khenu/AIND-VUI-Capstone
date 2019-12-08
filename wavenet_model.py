

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
