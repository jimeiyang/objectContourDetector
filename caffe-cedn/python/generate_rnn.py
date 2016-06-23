# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=no-self-use
# pylint: disable=too-few-public-methods

from caffe.proto import caffe_pb2
from caffe import Net

from google.protobuf import text_format

# pylint: disable=invalid-name
# pylint: disable=no-member
LayerType = caffe_pb2.LayerParameter.LayerType
EltwiseOp = caffe_pb2.EltwiseParameter.EltwiseOp
PoolMethod = caffe_pb2.PoolingParameter.PoolMethod
DBType = caffe_pb2.DataParameter.DB
# pylint: enable=invalid-name
# pylint: enable=no-member

class NetworkBuilder(object):

  def __init__(self, training_batch_size=20, testing_batch_size=20, **kwargs):
    self.training_batch_size = training_batch_size
    self.testing_batch_size = testing_batch_size
    self.other_args = kwargs


  def _make_inception(self, network, x1x1, x3x3r, x3x3, x5x5r, x5x5, proj,
                      name_generator):
    """Make Inception submodule."""

    layers = []

    split = self._make_split_layer(network)
    layers.append(split)

    context1 = self._make_conv_layer(network, kernel_size=1, num_output=x1x1,
                                     bias_value=0)
    layers.append(context1)

    relu1 = self._make_relu_layer(network)
    layers.append(relu1)

    context2a = self._make_conv_layer(network, kernel_size=1, num_output=x3x3r,
                                      bias_value=0)
    layers.append(context2a)

    relu2a = self._make_relu_layer(network)
    layers.append(relu2a)

    context2b = self._make_conv_layer(network, kernel_size=3, num_output=x3x3,
                                      pad=1)
    layers.append(context2b)

    relu2b = self._make_relu_layer(network)
    layers.append(relu2b)

    context3a = self._make_conv_layer(network, kernel_size=1, num_output=x5x5r,
                                      bias_value=0)
    layers.append(context3a)

    relu3a = self._make_relu_layer(network)
    layers.append(relu3a)

    context3b = self._make_conv_layer(network, kernel_size=5, num_output=x5x5,
                                      pad=2)
    layers.append(context3b)

    relu3b = self._make_relu_layer(network)
    layers.append(relu3b)

    context4a = self._make_maxpool_layer(network, kernel_size=3)
    layers.append(context4a)

    relu4a = self._make_relu_layer(network)
    layers.append(relu4a)

    context4b = self._make_conv_layer(network, kernel_size=1, num_output=proj,
                                      pad=1, bias_value=0)
    layers.append(context4b)

    relu4b = self._make_relu_layer(network)
    layers.append(relu4b)

    concat = self._make_concat_layer(network)
    layers.append(concat)

    connections = [
      (split.name, (split.top, context1.bottom)),
      (split.name, (split.top, context2a.bottom)),
      (split.name, (split.top, context3a.bottom)),
      (split.name, (split.top, context4a.bottom)),
      (context2a.name,
          (context2a.top, relu2a.bottom, relu2a.top, context2b.bottom)),
      (context3a.name,
          (context3a.top, relu3a.bottom, relu3a.top, context3b.bottom)),
      (context4a.name,
          (context4a.top, relu4a.bottom, relu4a.top, context4b.bottom)),
      (context1.name, (context1.top, relu1.bottom, relu1.top, concat.bottom)),
      (context2b.name,
          (context2b.top, relu2b.bottom, relu2b.top, concat.bottom)),
      (context3b.name,
          (context3b.top, relu3b.bottom, relu3b.top, concat.bottom)),
      (context4b.name,
          (context4b.top, relu4b.bottom, relu4b.top, concat.bottom)),
    ]

    for connection in connections:
      self._tie(connection, name_generator)

    return layers

  def _make_prod_layer(self, network, coeff=None):
    layer = network.layers.add()
    layer.name = 'prod'
    layer.type = LayerType.Value('ELTWISE')
    params = layer.eltwise_param
    params.operation = EltwiseOp.Value('PROD')
    if coeff:
      for c in coeff:
        params.coeff.append(c)
    return layer

  def _make_sum_layer(self, network, coeff=None):
    layer = network.layers.add()
    layer.name = 'sum'
    layer.type = LayerType.Value('ELTWISE')
    params = layer.eltwise_param
    params.operation = EltwiseOp.Value('SUM')
    if coeff:
      for c in coeff:
        params.coeff.append(c)
    return layer

  def _make_upsampling_layer(self, network, stride):
    layer = network.layers.add()
    layer.name = 'upsample'
    layer.type = LayerType.Value('UPSAMPLING')
    params = layer.upsampling_param
    params.kernel_size = stride
    return layer

  def _make_folding_layer(self, network, channels, height, width, prefix=''):
    layer = network.layers.add()
    layer.name = '%sfolding' % (prefix)
    layer.type = LayerType.Value('FOLDING')
    params = layer.folding_param
    params.channels_folded = channels
    params.height_folded = height
    params.width_folded = width
    return layer

  def _make_conv_layer(self, network, kernel_size, num_output, stride=1, pad=0,
                       bias_value=0.1, shared_name=None, wtype='xavier', std=0.01):
    """Make convolution layer."""

    layer = network.layers.add()
    layer.name = 'conv_%dx%d_%d' % (kernel_size, kernel_size, stride)

    layer.type = LayerType.Value('CONVOLUTION')
    params = layer.convolution_param
    params.num_output = num_output
    params.kernel_size = kernel_size
    params.stride = stride
    params.pad = pad
    weight_filler = params.weight_filler
    weight_filler.type = wtype
    if weight_filler.type == 'gaussian':
      weight_filler.mean = 0
      weight_filler.std = std
    bias_filler = params.bias_filler
    bias_filler.type = 'constant'
    bias_filler.value = bias_value

    layer.blobs_lr.append(1)
    layer.blobs_lr.append(2)

    layer.weight_decay.append(1)
    layer.weight_decay.append(0)

    if shared_name:
      layer.param.append('%s_w' % shared_name)
      layer.param.append('%s_b' % shared_name)

    return layer

  def _make_maxpool_layer(self, network, kernel_size, stride=1):
    """Make max pooling layer."""

    layer = network.layers.add()
    layer.name = 'maxpool_%dx%d_%d' % (kernel_size, kernel_size, stride)

    layer.type = LayerType.Value('POOLING')
    params = layer.pooling_param
    params.pool = PoolMethod.Value('MAX')
    params.kernel_size = kernel_size
    params.stride = stride

    return layer

  def _make_avgpool_layer(self, network, kernel_size, stride=1):
    """Make average pooling layer."""

    layer = network.layers.add()
    layer.name = 'avgpool_%dx%d_%d' % (kernel_size, kernel_size, stride)

    layer.type = LayerType.Value('POOLING')
    params = layer.pooling_param
    params.pool = PoolMethod.Value('AVE')
    params.kernel_size = kernel_size
    params.stride = stride

    return layer

  def _make_lrn_layer(self, network, name='lrn'):
    """Make local response normalization layer."""

    layer = network.layers.add()
    layer.name = name

    layer.type = LayerType.Value('LRN')
    params = layer.lrn_param
    params.local_size = 5
    params.alpha = 0.0001
    params.beta = 0.75

    return layer

  def _make_concat_layer(self, network, dim=1):
    """Make depth concatenation layer."""

    layer = network.layers.add()
    layer.name = 'concat'

    layer.type = LayerType.Value('CONCAT')
    params = layer.concat_param
    params.concat_dim = dim

    return layer

  def _make_dropout_layer(self, network, dropout_ratio=0.5):
    """Make dropout layer."""

    layer = network.layers.add()
    layer.name = 'dropout'

    layer.type = LayerType.Value('DROPOUT')
    params = layer.dropout_param
    params.dropout_ratio = dropout_ratio

    return layer

  def _make_tensor_layer(self, network, num_output, weight_lr=1,
                         bias_lr=2, bias_value=0.1, prefix='',
                         shared_name=None,
                         wtype='xavier', std=0.01):
    """Make tensor product layer."""

    layer = network.layers.add()
    layer.name = '%stensor_product' % prefix

    layer.type = LayerType.Value('TENSOR_PRODUCT')
    params = layer.inner_product_param
    params.num_output = num_output
    weight_filler = params.weight_filler
    weight_filler.type = wtype
    if wtype == 'gaussian':
      weight_filler.mean = 0
      weight_filler.std = std
    bias_filler = params.bias_filler
    bias_filler.type = 'constant'
    bias_filler.value = bias_value

    layer.blobs_lr.append(weight_lr)
    layer.blobs_lr.append(bias_lr)

    layer.weight_decay.append(1)
    layer.weight_decay.append(0)

    if shared_name:
      layer.param.append('%s_w' % shared_name)
      layer.param.append('%s_b' % shared_name)

    return layer

  def _make_inner_product_layer(self, network, num_output, weight_lr=1,
                                bias_lr=2, bias_value=0.1, prefix='',
                                shared_name=None,
                                wtype='xavier', std=0.01):
    """Make inner product layer."""

    layer = network.layers.add()
    layer.name = '%sinner_product' % prefix

    layer.type = LayerType.Value('INNER_PRODUCT')
    params = layer.inner_product_param
    params.num_output = num_output
    weight_filler = params.weight_filler
    weight_filler.type = wtype
    if wtype == 'gaussian':
      weight_filler.mean = 0
      weight_filler.std = std
    bias_filler = params.bias_filler
    bias_filler.type = 'constant'
    bias_filler.value = bias_value

    layer.blobs_lr.append(weight_lr)
    layer.blobs_lr.append(bias_lr)

    layer.weight_decay.append(1)
    layer.weight_decay.append(0)

    if shared_name:
      layer.param.append('%s_w' % shared_name)
      layer.param.append('%s_b' % shared_name)

    return layer

  def _make_split_layer(self, network):
    """Make split layer."""

    layer = network.layers.add()
    layer.name = 'split'

    layer.type = LayerType.Value('SPLIT')

    return layer

  def _make_relu_layer(self, network):
    """Make ReLU layer."""

    layer = network.layers.add()
    layer.name = 'relu'

    layer.type = LayerType.Value('RELU')

    return layer

  def _tie(self, layers, name_generator):
    """Generate a named connection between layer endpoints."""

    name = 'ep_%s_%d' % (layers[0], name_generator.next())
    for layer in layers[1]:
      layer.append(name)

  def _connection_name_generator(self):
    """Generate a unique id."""

    index = 0
    while True:
      yield index
      index += 1

  def _build_rnn_network(self, wtype='xavier', std=0.01, batchsize=100, numstep=24):
    network = caffe_pb2.NetParameter()
    network.force_backward = True
    network.name = 'rotation_rnn'
    network.input.append('images')
    network.input_dim.append(batchsize)
    network.input_dim.append(3)
    network.input_dim.append(64)
    network.input_dim.append(64)
    for t in range(numstep):
      network.input.append('rotations%d' % t)
      network.input_dim.append(batchsize)
      network.input_dim.append(3)
      network.input_dim.append(1)
      network.input_dim.append(1)

    layers = []
    name_generator = self._connection_name_generator()

    tensor_view = []
    relu2_view = []
    relu2_view_split = []
    concat = []
    dec_fc1 = []
    dec_relu1 = []
    dec_fc2 = []
    dec_relu2 = []
    dec_relu2_split = []
    dec_img_fc1 = []
    dec_img_relu1 = []
    dec_img_fold = []
    dec_img_up1 = []
    dec_img_conv1 = []
    dec_img_relu2 = []
    dec_img_up2 = []
    dec_img_conv2 = []
    dec_img_relu3 = []
    dec_img_up3 = []
    dec_img_conv3 = []
    dec_mask_fc1 = []
    dec_mask_relu1 = []
    dec_mask_fold = []
    dec_mask_up1 = []
    dec_mask_conv1 = []
    dec_mask_relu2 = []
    dec_mask_up2 = []
    dec_mask_conv2 = []
    dec_mask_relu3 = []
    dec_mask_up3 = []
    dec_mask_conv3 = []

    conv1 = self._make_conv_layer(network, kernel_size=5, stride=2, pad=2, num_output=64, shared_name='conv1')
    conv1.bottom.append('images')
    relu1 = self._make_relu_layer(network)
    conv2 = self._make_conv_layer(network, kernel_size=5, stride=2, pad=2, num_output=128, shared_name='conv2')
    relu2 = self._make_relu_layer(network)
    conv3 = self._make_conv_layer(network, kernel_size=5, stride=2, pad=2, num_output=256, shared_name='conv3')
    relu3 = self._make_relu_layer(network)
    fc1 = self._make_inner_product_layer(network, num_output=1024, shared_name='fc1')
    relu4 = self._make_relu_layer(network)
    fc2 = self._make_inner_product_layer(network, num_output=1024, shared_name='fc2')
    relu5 = self._make_relu_layer(network)

    enc_split = self._make_split_layer(network)

    fc1_id = self._make_inner_product_layer(network, num_output=512, shared_name='fc1_id')
    relu1_id = self._make_relu_layer(network)
    id_split = self._make_split_layer(network)

    fc1_view = self._make_inner_product_layer(network, num_output=512, shared_name='fc1_view')
    relu1_view = self._make_relu_layer(network)

    tensor_view.append(self._make_tensor_layer(network, num_output=512, shared_name='tensor_view'))
    tensor_view[-1].bottom.append('rotations0')
    relu2_view.append(self._make_relu_layer(network))
    relu2_view_split.append(self._make_split_layer(network))

    connections = []
    connections.append((conv1.name, (conv1.top, relu1.bottom, relu1.top, conv2.bottom)))
    connections.append((conv2.name, (conv2.top, relu2.bottom, relu2.top, conv3.bottom)))
    connections.append((conv3.name, (conv3.top, relu3.bottom, relu3.top, fc1.bottom)))
    connections.append((fc1.name, (fc1.top, relu4.bottom, relu4.top, fc2.bottom)))
    connections.append((fc2.name, (fc2.top, relu5.bottom)))
    connections.append((relu5.name, (relu5.top, enc_split.bottom)))
    connections.append((enc_split.name, (enc_split.top, fc1_id.bottom)))
    connections.append((fc1_id.name, (fc1_id.top, relu1_id.bottom, relu1_id.top, id_split.bottom)))
    connections.append((enc_split.name, (enc_split.top, fc1_view.bottom)))
    connections.append((fc1_view.name, (fc1_view.top, relu1_view.bottom, relu1_view.top, tensor_view[-1].bottom)))

    for t in range(numstep):
      # Action.
      if t > 0:
        tensor_view.append(self._make_tensor_layer(network, num_output=512, shared_name='tensor_view'))
        tensor_view[-1].bottom.append('rotations%d' % t)
        relu2_view.append(self._make_relu_layer(network))
        relu2_view_split.append(self._make_split_layer(network))
      # Decoder.
      concat.append(self._make_concat_layer(network))
      dec_fc1.append(self._make_inner_product_layer(network, num_output=1024, shared_name='dec_fc1'))
      dec_relu1.append(self._make_relu_layer(network))
      dec_fc2.append(self._make_inner_product_layer(network, num_output=1024, shared_name='dec_fc2'))
      dec_relu2.append(self._make_relu_layer(network))
      dec_relu2_split.append(self._make_split_layer(network))
      # Dec img path.
      dec_img_fc1.append(self._make_inner_product_layer(network, num_output=16384, shared_name='dec_img_fc1'))
      dec_img_relu1.append(self._make_relu_layer(network))
      dec_img_fold.append(self._make_folding_layer(network,256,8,8))
      dec_img_up1.append(self._make_upsampling_layer(network,stride=2))
      dec_img_conv1.append(self._make_conv_layer(network, kernel_size=5, stride=1, pad=2, num_output=128, shared_name='dec_img_conv1'))
      dec_img_relu2.append(self._make_relu_layer(network))
      dec_img_up2.append(self._make_upsampling_layer(network,stride=2))
      dec_img_conv2.append(self._make_conv_layer(network, kernel_size=5, stride=1, pad=2, num_output=64, shared_name='dec_img_conv2'))
      dec_img_relu3.append(self._make_relu_layer(network))
      dec_img_up3.append(self._make_upsampling_layer(network,stride=2))
      dec_img_conv3.append(self._make_conv_layer(network, kernel_size=5, stride=1, pad=2, num_output=3, shared_name='dec_img_conv3'))
      # Dec mask path.
      dec_mask_fc1.append(self._make_inner_product_layer(network, num_output=8192, shared_name='dec_mask_fc1'))
      dec_mask_relu1.append(self._make_relu_layer(network))
      dec_mask_fold.append(self._make_folding_layer(network,128,8,8))
      dec_mask_up1.append(self._make_upsampling_layer(network,stride=2))
      dec_mask_conv1.append(self._make_conv_layer(network, kernel_size=5, stride=1, pad=2, num_output=64, shared_name='dec_mask_conv1'))
      dec_mask_relu2.append(self._make_relu_layer(network))
      dec_mask_up2.append(self._make_upsampling_layer(network,stride=2))
      dec_mask_conv2.append(self._make_conv_layer(network, kernel_size=5, stride=1, pad=2, num_output=32, shared_name='dec_mask_conv2'))
      dec_mask_relu3.append(self._make_relu_layer(network))
      dec_mask_up3.append(self._make_upsampling_layer(network,stride=2))
      dec_mask_conv3.append(self._make_conv_layer(network, kernel_size=5, stride=1, pad=2, num_output=1, shared_name='dec_mask_conv3'))

      # dec connections.
      if t > 0:
        connections.append((relu2_view_split[-2].name, (relu2_view_split[-2].top, tensor_view[-1].bottom)))
      connections.append((tensor_view[-1].name, (tensor_view[-1].top, relu2_view[-1].bottom)))
      connections.append((relu2_view[-1].name, (relu2_view[-1].top, relu2_view_split[-1].bottom)))
      connections.append((id_split.name, (id_split.top, concat[-1].bottom)))
      connections.append((relu2_view_split[-1].name, (relu2_view_split[-1].top, concat[-1].bottom)))
      connections.append((concat[-1].name, (concat[-1].top, dec_fc1[-1].bottom)))
      connections.append((dec_fc1[-1].name, (dec_fc1[-1].top, dec_relu1[-1].bottom, dec_relu1[-1].top, dec_fc2[-1].bottom)))
      connections.append((dec_fc2[-1].name, (dec_fc2[-1].top, dec_relu2[-1].bottom)))
      connections.append((dec_relu2[-1].name, (dec_relu2[-1].top, dec_relu2_split[-1].bottom)))
      # dec image connections.
      connections.append((dec_relu2_split[-1].name, (dec_relu2_split[-1].top, dec_img_fc1[-1].bottom)))
      connections.append((dec_img_fc1[-1].name, (dec_img_fc1[-1].top, dec_img_relu1[-1].bottom, dec_img_relu1[-1].top, dec_img_fold[-1].bottom)))
      connections.append((dec_img_fold[-1].name, (dec_img_fold[-1].top, dec_img_up1[-1].bottom)))
      connections.append((dec_img_up1[-1].name, (dec_img_up1[-1].top, dec_img_conv1[-1].bottom)))
      connections.append((dec_img_conv1[-1].name, (dec_img_conv1[-1].top, dec_img_relu2[-1].bottom, dec_img_relu2[-1].top, dec_img_up2[-1].bottom)))
      connections.append((dec_img_up2[-1].name, (dec_img_up2[-1].top, dec_img_conv2[-1].bottom)))
      connections.append((dec_img_conv2[-1].name, (dec_img_conv2[-1].top, dec_img_relu3[-1].bottom, dec_img_relu3[-1].top, dec_img_up3[-1].bottom)))
      connections.append((dec_img_up3[-1].name, (dec_img_up3[-1].top, dec_img_conv3[-1].bottom)))
      # dec mask connections.
      connections.append((dec_relu2_split[-1].name, (dec_relu2_split[-1].top, dec_mask_fc1[-1].bottom)))
      connections.append((dec_mask_fc1[-1].name, (dec_mask_fc1[-1].top, dec_mask_relu1[-1].bottom, dec_mask_relu1[-1].top, dec_mask_fold[-1].bottom)))
      connections.append((dec_mask_fold[-1].name, (dec_mask_fold[-1].top, dec_mask_up1[-1].bottom)))
      connections.append((dec_mask_up1[-1].name, (dec_mask_up1[-1].top, dec_mask_conv1[-1].bottom)))
      connections.append((dec_mask_conv1[-1].name, (dec_mask_conv1[-1].top, dec_mask_relu2[-1].bottom, dec_mask_relu2[-1].top, dec_mask_up2[-1].bottom)))
      connections.append((dec_mask_up2[-1].name, (dec_mask_up2[-1].top, dec_mask_conv2[-1].bottom)))
      connections.append((dec_mask_conv2[-1].name, (dec_mask_conv2[-1].top, dec_mask_relu3[-1].bottom, dec_mask_relu3[-1].top, dec_mask_up3[-1].bottom)))
      connections.append((dec_mask_up3[-1].name, (dec_mask_up3[-1].top, dec_mask_conv3[-1].bottom)))

    layers = [ conv1, relu1, conv2, relu2, conv3, relu3, fc1, relu4, fc2, relu5, enc_split, fc1_id, relu1_id, id_split ]
    layers += tensor_view
    layers += relu2_view
    layers += relu2_view_split
    layers += concat
    layers += dec_fc1
    layers += dec_relu1
    layers += dec_fc2
    layers += dec_relu2
    layers += dec_relu2_split
    layers += dec_img_fc1
    layers += dec_img_relu1
    layers += dec_img_fold
    layers += dec_img_up1
    layers += dec_img_conv1
    layers += dec_img_relu2
    layers += dec_img_up2
    layers += dec_img_conv2
    layers += dec_img_relu3
    layers += dec_img_up3
    layers += dec_img_conv3
    layers += dec_mask_fc1
    layers += dec_mask_relu1
    layers += dec_mask_fold
    layers += dec_mask_up1
    layers += dec_mask_conv1
    layers += dec_mask_relu2
    layers += dec_mask_up2
    layers += dec_mask_conv2
    layers += dec_mask_relu3
    layers += dec_mask_up3
    layers += dec_mask_conv3

    final_img_concat = self._make_concat_layer(network)
    for idx,l in enumerate(dec_img_conv3):
      l.name = 't%d_%s' % (idx,l.name)
      connections.append((l.name, (l.top, final_img_concat.bottom)))
    final_img_concat.top.append('images_concat')
    final_img_concat.loss_weight.append(10.0)

    final_mask_concat = self._make_concat_layer(network)
    for idx,l in enumerate(dec_mask_conv3):
      l.name = 't%d_%s' % (idx,l.name)
      connections.append((l.name, (l.top, final_mask_concat.bottom)))
    final_mask_concat.top.append('masks_concat')
    final_mask_concat.loss_weight.append(1.0)

    layers += [ final_img_concat, final_mask_concat ]

    # make connections.
    for connection in connections:
      self._tie(connection, name_generator)

    for l in tensor_view[0:]:
      tmp = reversed(l.bottom)
      l.ClearField('bottom')
      l.bottom.extend(tmp)

    # Fix up the names based on the connections that were generated.
    for pos, layer in enumerate(layers):
      layer.name += '_%d' % pos

    return network


  def build_network(self, netname, batchsize=100, numstep=2):
    """main method."""

    if netname == 'rnn':
      network = self._build_rnn_network(batchsize=batchsize, numstep=numstep)
    else:
      print('unknown netname: %s' % netname)
      return

    network_filename = '%s.prototxt' % netname
    print network
    with open(network_filename, 'w') as network_file:
      network_file.write(text_format.MessageToString(network))
    return Net(network_filename)


if __name__ == '__main__':
  __Network_builder__ = NetworkBuilder()
  __Network_builder__.build_network(netname='rnn', batchsize=31, numstep=4)
