#!/usr/bin/env python
from paddle.trainer_config_helpers import *

use_dummy = get_config_arg("use_dummy", bool, True)
batch_size = get_config_arg('batch_size', int, 1)
is_predict = get_config_arg("is_predict", bool, False)
is_test = get_config_arg("is_test", bool, False)
layer_num = get_config_arg('layer_num', int, 6)

####################Data Configuration ##################
# 10ms as one step
dataSpec = dict(
    uttLengths = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500],
    counts = [3, 10, 11, 13, 14, 13, 9, 8, 5, 4, 3, 2, 2, 2, 1],
    lblLengths = [7, 17, 35, 48, 62, 78, 93, 107, 120, 134, 148, 163, 178, 193, 209],
    freqBins = 161,
    charNum = 29, # 29 chars
    scaleNum = 1280
    )
num_classes = dataSpec['charNum']
if not is_predict:
    train_list = 'data/train.list' if not is_test else None
    test_list = None #'data/test.list'
    args = {
        'uttLengths': dataSpec['uttLengths'],
        'counts': dataSpec['counts'],
        'lblLengths': dataSpec['lblLengths'],
        'freqBins': dataSpec['freqBins'],
        'charNum': dataSpec['charNum'],
        'scaleNum': dataSpec['scaleNum'],
        'batch_size': batch_size
    }
    define_py_data_sources2(
        train_list,
        test_list,
        module='dummy_provider' if use_dummy else 'image_provider',
        obj='process',
        args=args)

###################### Algorithm Configuration #############
settings(
    batch_size=batch_size,
    learning_rate=1e-3,
#    learning_method=AdamOptimizer(),
#    regularization=L2Regularization(8e-4),
)

####################### Deep Speech 2 Configuration #############
def conv_bn_relu(input, kh, kw, sh, sw, ic, oc = 32, clipped = 20):
    tmp = img_conv_layer(
        input = input,
        num_filters = oc,
        num_channels = ic,
        filter_size_y = kh,
        filter_size = kw,
        stride_y = sh,
        stride = sw
    )
    return batch_norm_layer(input = tmp, act = BReluActivation()) # TODO: change clipped 24 -> 20

def bdrnn(input, dim_out):
    tmp = fc_layer(input=input, size=dim_out, bias_attr=False, act=LinearActivation()) #act=None
    tmp = batch_norm_layer(input = tmp, num_channels = dim_out, act = None)
    rnn = recurrent_layer(input=tmp, act=BReluActivation()) # TODO: change clipped 24 -> 20
    rnn_inv = recurrent_layer(input=tmp, act=BReluActivation(), reverse=True)
    return addto_layer(input = [rnn, rnn_inv])

######## DS2
tmp = data_layer(name = 'data', size = dataSpec['freqBins'])

# change to non-seq and transpose
tmp = view_layer(input=tmp,
                name="view_to_noseq",
                view_type=ViewType.SEQUENCE_TO_NONE,
                width = dataSpec['freqBins'],
                height = 100) # TODO:-1
tmp = img_trans_layer(input = tmp, height = dataSpec['freqBins']) #the height after trans

# conv
tmp = conv_bn_relu(tmp, 5, 20, 2, 2, 1, 32)
tmp = conv_bn_relu(tmp, 5, 10, 1, 2, 32, 32)

# reshape and transpose
tmp = view_layer(input=tmp,
                name="reshape",
                view_type=ViewType.NO_CHANGE,
                channel = 1,
                height = 2400,
                width = 16) # TODO:-1
tmp = img_trans_layer(input = tmp, width = 2400) # the width after transpose
tmp = view_layer(input=tmp,
                name="view_to_seq",
                view_type=ViewType.NONE_TO_SEQUENCE,
                seq_len = -1,
                channel = 2400,
                height = 1,
                width = 1)

tmp = bdrnn(tmp, 1760) # at least one

for i in xrange(layer_num):
    tmp = bdrnn(tmp, 1760)

output = fc_layer(input=tmp, size=num_classes + 1, act=LinearActivation()) #act=None

if not is_predict:
    lbl = data_layer(name='label', size=num_classes)
    cost = warp_ctc_layer(input=output, name = "WarpCTC", blank = 0, label=lbl, size = num_classes + 1) # CTC size should +1
    outputs(cost)
else:
    outputs(output)
