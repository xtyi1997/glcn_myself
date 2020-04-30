from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
import copy
import json

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    # x, 1-self.dropout, self.num_features_nonzero
    # 对稀疏矩阵做drop_out
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

class SparseGraphLearn(object):
    """Sparse Graph learning layer."""
    def __init__(self, input_dim, output_dim, edge, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.num_nodes = placeholders['num_nodes']
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.edge = edge

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            # 这个属于glorot 变量初始化
            self.vars['weights'] = glorot([input_dim, output_dim], name='weights')
            self.vars['a'] = glorot([output_dim, 1], name='a')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

    def __call__(self, inputs):
        x = inputs
        # dropout  sparse——dropout是自己定义的
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # graph learning  dot也是自己定义
        h = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
        N = self.num_nodes
        edge_v = tf.abs(tf.gather(h,self.edge[0]) - tf.gather(h,self.edge[1]))
        edge_v = tf.squeeze(self.act(dot(edge_v, self.vars['a'])))
        sgraph = tf.SparseTensor(indices=tf.transpose(self.edge), values=edge_v, dense_shape=[N, N])
        sgraph = tf.sparse_softmax(sgraph)
        return h, sgraph


class GraphConvolution(object):
    """Graph convolution layer provided by Thomas N. Kipf, Max Welling, 
    [Semi-Supervised Classification with Graph Convolutional Networks]
    (http://arxiv.org/abs/1609.02907) (ICLR 2017)"""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                                    name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs, adj):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        pre_sup = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
        output = dot(adj, pre_sup, sparse=True)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

    def __call__(self, inputs, adj):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs, adj)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])



# myself
def psum_output_layer(x, num_classes):
  num_segments = int(x.shape[1]) / num_classes
  if int(x.shape[1]) % num_classes != 0:
    print('Wasted psum capacity: %i out of %i' % (
        int(x.shape[1]) % num_classes, int(x.shape[1])))
  sum_q_weights = tf.get_variable(
      'psum_q', shape=[num_segments], initializer=tf.zeros_initializer, dtype=tf.float32, trainable=True)
  tf.losses.add_loss(tf.reduce_mean((sum_q_weights ** 2)) * 1e-3 )
  softmax_q = tf.nn.softmax(sum_q_weights)  # softmax
  psum = 0
  for i in range(int(num_segments)):
    segment = x[:, i*num_classes : (i+1)*num_classes]
    psum = segment * softmax_q[i] + psum
  return psum


def adj_times_x(adj, x, adj_pow=1):
  """Multiplies (adj^adj_pow)*x."""
  for i in range(adj_pow):
    x = tf.sparse_tensor_dense_matmul(adj, x)
  return x
MODULE_REFS = {
    'tf': tf,
    'tf.layers': tf.layers,
    'tf.nn': tf.nn,
    'tf.sparse': tf.sparse,
    'tf.contrib.layers': tf.contrib.layers
}
class MixHopModel(object):
    """Builds MixHop architectures. Used as architectures can be learned.

    Use like:
      model = MixHopModel(sparse_adj, x, is_training, kernel_regularizer)
      ...
      model.add_layer('<module_name>', '<fn_name>', args_to_fn)
      model.add_layer( ... )
      ...

    Where <module_name> must be a string defined in MODULE_REFS, and <fn_name>
    must be a function living inside module indicated by <module_name>, finally,
    args_to_fn are passed as-is to the function (with name <fn_name>), with the
    exception of arguments:
      pass_kernel_regularizer: if argument is present, then we pass
        kernel_regularizer argument with value given to the constructor.
      pass_is_training: if argument is present, then we pass is_training argument
        with value given to the constructor.
      pass_training: if argument is present, then we pass training argument with
        value of is_training given to the constructor.

    In addition <module_name> can be:
      'self': invokes functions in this class. 调用此类中的函数
      'mixhop_model': invokes functions in this file.

    See example_pubmed_model() for reference.
    """

    def __init__(self, sparse_adj, sparse_input, is_training, kernel_regularizer):
        # true
        self.is_training = is_training
        # 正则化
        self.kernel_regularizer = kernel_regularizer
        # 稀疏临街矩阵
        self.sparse_adj = sparse_adj
        # 稀疏输入矩阵
        self.sparse_input = sparse_input
        # 层数
        self.layer_defs = []
        # 激活
        self.activations = [sparse_input]

    def save_architecture_to_file(self, filename):
        with open(filename, 'w') as fout:
            fout.write(json.dumps(self.layer_defs, indent=2))

    def load_architecture_from_file(self, filename):
        if self.layer_defs:
            raise ValueError('Model is (partially) initialized. Cannot load.')
        layer_defs = json.loads(open(filename).read())
        for layer_def in layer_defs:
            self.add_layer(layer_def['module'], layer_def['fn'], *layer_def['args'],
                           **layer_def['kwargs'])

    # 这个函数比较重要
    def add_layer(self, module_name, layer_fn_name, *args, **kwargs):
        #
        self.layer_defs.append({
            'module': module_name,
            'fn': layer_fn_name,
            'args': args,
            # 深拷贝对象
            'kwargs': copy.deepcopy(kwargs),
        })
        #
        if 'pass_training' in kwargs:
            kwargs.pop('pass_training')
            kwargs['training'] = self.is_training
        if 'pass_is_training' in kwargs:
            kwargs.pop('pass_is_training')
            kwargs['is_training'] = self.is_training
        if 'pass_kernel_regularizer' in kwargs:
            kwargs.pop('pass_kernel_regularizer')
            kwargs['kernel_regularizer'] = self.kernel_regularizer
        #
        fn = None
        if module_name == 'mixhop_model':
            # globals() 函数会以字典类型返回当前位置的全部全局变量。
            # 这里不太明白啊
            fn = globals()[layer_fn_name]
            print(dir(fn))
        elif module_name == 'self':
            fn = getattr(self, layer_fn_name)
            print(fn)
        elif module_name in MODULE_REFS:
            fn = getattr(MODULE_REFS[module_name], layer_fn_name)
            print(fn)
        else:
            raise ValueError(
                'Module name %s not registered in MODULE_REFS' % module_name)
        self.activations.append(
            fn(self.activations[-1], *args, **kwargs))

    def mixhop_layer(self, x, adjacency_powers, dim_per_power,
                     kernel_regularizer=None, layer_id=None, replica=None):
        return mixhop_layer(x, self.sparse_adj, adjacency_powers, dim_per_power,
                            kernel_regularizer, layer_id, replica)

def mixhop_layer(x, sparse_adjacency, adjacency_powers, dim_per_power,
                 kernel_regularizer=None, layer_id=None, replica=None):
  """Constructs MixHop layer.

  Args:
    sparse_adjacency: Sparse tensor containing square and normalized adjacency
      matrix.
    adjacency_powers: list of integers containing powers of adjacency matrix.
    dim_per_power: List same size as `adjacency_powers`. Each power will emit
      the corresponding dimensions.
    layer_id: If given, will be used to name the layer
  """
  #
  replica = replica or 0
  layer_id = layer_id or 0
  segments = []
  for p, dim in zip(adjacency_powers, dim_per_power):
    net_p = adj_times_x(sparse_adjacency, x, p)

    with tf.variable_scope('r%i_l%i_p%s' % (replica, layer_id, str(p))):
      layer = tf.layers.Dense(
          dim,
          kernel_regularizer=kernel_regularizer,
          activation=None, use_bias=False)
      net_p = layer.apply(net_p)

    segments.append(net_p)
  return tf.concat(segments, axis=1)

def example_pubmed_model(
        sparse_adj, x, num_x_entries, is_training, kernel_regularizer, input_dropout,
        layer_dropout, num_classes=3):
    """Returns PubMed model with test performance ~>80.4%.

    Args:
      sparse_adj: Sparse tensor of normalized adjacency matrix.
      x: Sparse tensor of feature matrix.
      num_x_entries: number of non-zero entries of x. Used for sparse dropout.
      is_training: boolean scalar Tensor.
      kernel_regularizer: Keras regularizer object.
      input_dropout: Float in range [0, 1.0). How much to drop out from input.
      layer_dropout: Dropout value for dense layers.
    """
    model = MixHopModel(sparse_adj, x, is_training, kernel_regularizer)

    model.add_layer('mixhop_model', 'sparse_dropout', input_dropout,
                    num_x_entries, pass_is_training=True)
    model.add_layer('tf', 'sparse_tensor_to_dense')
    model.add_layer('tf.nn', 'l2_normalize', axis=1)

    # MixHop Conv layer
    model.add_layer('self', 'mixhop_layer', [0, 1, 2], [17, 22, 21], layer_id=0,
                    pass_kernel_regularizer=True)

    model.add_layer('tf.contrib.layers', 'batch_norm')
    model.add_layer('tf.nn', 'tanh')

    model.add_layer('tf.layers', 'dropout', layer_dropout, pass_training=True)
    # MixHop Conv layer
    model.add_layer('self', 'mixhop_layer', [0, 1, 2], [3, 1, 6], layer_id=1,
                    pass_kernel_regularizer=True)
    model.add_layer('tf.layers', 'dropout', layer_dropout, pass_training=True)
    # MixHop Conv layer
    model.add_layer('self', 'mixhop_layer', [0, 1, 2], [2, 4, 4], layer_id=2,
                    pass_kernel_regularizer=True)
    model.add_layer('tf.contrib.layers', 'batch_norm')
    model.add_layer('tf.nn', 'tanh')
    model.add_layer('tf.layers', 'dropout', layer_dropout, pass_training=True)

    # Classification Layer
    model.add_layer('tf.layers', 'dense', num_classes, use_bias=False,
                    activation=None, pass_kernel_regularizer=True)
    return model


