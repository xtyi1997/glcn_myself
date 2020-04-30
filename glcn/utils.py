import numpy as np
import scipy
import scipy.sparse as sp
import sys
import scipy.io as sio
import tensorflow as tf
import pickle
import os
import collections

flags = tf.app.flags
FLAGS = flags.FLAGS

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_x(filename):
  if sys.version_info > (3, 0):
    return pickle.load(open(filename, 'rb'), encoding='latin1')
  else:
    return np.load(filename)

def concatenate_csr_matrices_by_rows(matrix1, matrix2):
  """Concatenates sparse csr matrices matrix1 above matrix2.
  # 连接稀疏矩阵
  Adapted from:
  https://stackoverflow.com/questions/6844998/is-there-an-efficient-way-of-concatenating-scipy-sparse-matrices
  """
  new_data = np.concatenate((matrix1.data, matrix2.data))
  new_indices = np.concatenate((matrix1.indices, matrix2.indices))
  new_ind_ptr = matrix2.indptr + len(matrix1.data)
  new_ind_ptr = new_ind_ptr[1:]
  new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

  return scipy.sparse.csr_matrix((new_data, new_indices, new_ind_ptr))

def load_data_my(dataset_name='cora',dataset_dir="../data/"):
    # path = path + dataset_str + "/"
    dataset_name='ind.'+dataset_name
    base_path = os.path.join(dataset_dir, dataset_name)
    # 每个节点的连接情况
    edge_lists = pickle.load(open(base_path + '.graph', 'rb'))
    # 1708 1403
    # allx = np.array(np.load(base_path + '.allx', allow_pickle=True), dtype='float32')
    allx = load_x(base_path + '.allx')
    # change it allow_pickle=True  1708 7
    ally = np.array(np.load(base_path + '.ally', allow_pickle=True), dtype='float32')
    testx = load_x(base_path + '.tx')
    # 1000  1433
    # Add test  下标
    test_idx = list(map(int, open(base_path + '.test.index').read().split('\n')[:-1]))
    # 自己添加 1000
    print(len(test_idx))
    num_test_examples = max(test_idx) - min(test_idx) + 1
    # 生成1000  1433 大小的空矩阵
    sparse_zeros = scipy.sparse.csr_matrix((num_test_examples, allx.shape[1]),
                                           dtype='float32')
    # 1000*1433
    allx = concatenate_csr_matrices_by_rows(allx, sparse_zeros)
    # llallx = allx.tolil()  # 2708 1433
    # llallx[test_idx] = testx
    allx[test_idx] = testx
    # 这个顺序不知道有没有用
    llallx=allx.tocsc()
    # allx = scipy.vstack([allx, sparse_zeros])
    # test_idx_set = set(test_idx)
    testy = np.array(np.load(base_path + '.ty', allow_pickle=True), dtype='float32')
    ally = np.concatenate(
        [ally, np.zeros((num_test_examples, ally.shape[1]), dtype='float32')],
        0)
    ally[test_idx] = testy
    # num_nodes = len(edge_lists)
    # Will be used to construct (sparse) adjacency matrix.  构建邻接矩阵 先转换为set
    # edge_sets = collections.defaultdict(set)
    # for node, neighbors in edge_lists.items():
    #     edge_sets[node].add(node)  # Add self-connections
    #     for n in neighbors:
    #         edge_sets[node].add(n)
    #         edge_sets[n].add(node)  # Assume undirected.
    adj_indices=[]
    adj_indptr=[]
    adj_indptr.append(0)
    len_num=-1
    for node, neighbors in edge_lists.items():
        len_num=len(neighbors)+len_num
        adj_indptr.append(len_num)
        for i in neighbors:
            adj_indices.append(i)
    print(max(adj_indices))
    adj_data=np.ones(len(adj_indices))
    adj_indices=np.array(adj_indices)
    adj_indptr=np.array(adj_indptr)
    # adj=scipy.sparse.csr_matrix((adj_data, adj_indices, adj_indptr))

    if dataset_name == "ind.cora":
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
        adj = scipy.sparse.csr_matrix((adj_data, adj_indices, adj_indptr))
    elif dataset_name == "ind.citeseer":
        idx_test = list(map(int, open(base_path + '.test.index').read().split('\n')[:-1]))
        # 会检查数据类型
        idx_test = np.asarray(idx_test).flatten()
        idx_test=np.sort(idx_test)
        # idx_test=np.array(np.load(base_path + '.test.index', allow_pickle=True), dtype='float32')
        # idx_test = sio.loadmat(path + "test.mat")
        # idx_test = idx_test['array'].flatten()
        idx_train = range(120)
        idx_val = range(120, 620)
        adj = scipy.sparse.csc_matrix((adj_data, adj_indices, adj_indptr))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    else:
        # idx_test = sio.loadmat(path + "test.mat")
        # idx_test = idx_test['matrix']
        idx_test = list(map(int, open(base_path + '.test.index').read().split('\n')[:-1]))
        idx_test = np.asarray(idx_test).flatten()
        idx_test = np.sort(idx_test)
        idx_train = range(60)
        idx_val = range(200, 500)
        adj = scipy.sparse.csc_matrix((adj_data, adj_indices, adj_indptr))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    train_mask = sample_mask(idx_train, ally.shape[0])
    # 200-500
    val_mask = sample_mask(idx_val, ally.shape[0])
    # 500-1500
    test_mask = sample_mask(idx_test, ally.shape[0])
    y_train = np.zeros(ally.shape)
    y_val = np.zeros(ally.shape)
    y_test = np.zeros(ally.shape)
    y_train[train_mask, :] = ally[train_mask, :]
    y_val[val_mask, :] = ally[val_mask, :]
    y_test[test_mask, :] = ally[test_mask, :]
    features=llallx
    # features=scipy.sparse.csr_matrix(llallx.)
    return adj,features,y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_data(dataset_str,path="../data/"):
    path = path + dataset_str +"/"
    if dataset_str == "cora":
        # 里边有这个参数，返回的是一个字典
        features = sio.loadmat(path + "feature")
        features = features['matrix']
        adj = sio.loadmat(path + "adj")
        adj = adj['matrix']
        labels = sio.loadmat(path + "label")
        labels = labels['matrix']
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
    elif dataset_str == "citeseer":
        features = sio.loadmat(path + "feature")
        features = features['matrix']
        adj = sio.loadmat(path + "adj")
        adj = adj['matrix']
        labels = sio.loadmat(path + "label")
        labels = labels['matrix']
        idx_test = sio.loadmat(path + "test.mat")
        idx_test = idx_test['array'].flatten()
        idx_train = range(120)
        idx_val = range(120, 620)
    else:
        features = sio.loadmat(path + "feature")
        features = features['matrix']
        adj = sio.loadmat(path + "adj")
        adj = adj['matrix']
        labels = sio.loadmat(path + "label")
        labels = labels['matrix']
        idx_test = sio.loadmat(path + "test.mat")
        idx_test = idx_test['matrix']
        idx_train = range(60)
        idx_val = range(200, 500)

    # 做一下比较  1>2 false bool 矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 前60  feature2708*1433  lable 2708*7
    train_mask = sample_mask(idx_train, labels.shape[0])
    # 200-500
    val_mask = sample_mask(idx_val, labels.shape[0])
    # 500-1500
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # D^{-1/2}AD^{1/2}    L = I_N - D^{-1/2}AD^{1/2}
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # matric 的todense（）返回稀疏矩阵的np.matrix形式
    # 返回此矩阵的密集矩阵表示
    edge = np.array(np.nonzero(adj_normalized.todense()))
    return sparse_to_tuple(adj_normalized), edge


def construct_feed_dict(features, adj, labels, labels_mask, epoch, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj})
    feed_dict.update({placeholders['step']: epoch})
    feed_dict.update({placeholders['num_nodes']: features[2][0]})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

