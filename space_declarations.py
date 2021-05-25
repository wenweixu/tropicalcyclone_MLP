from hyperopt import hp
'''
Declare search space for hyperopt.  Imported by hyperopt_search.py.
'''

n_nodes_options = [2 ** i for i in range(2, 12)] #range(2, 12)
activation_options = ['relu', 'sigmoid']

mlp_space = {
    'n_nodes_layer1': hp.choice('n_nodes_layer1', n_nodes_options),
    'layer1_activation': hp.choice('layer1_activation', activation_options),
    'num_layers': hp.choice('num_layers', [{'layers': 'one'},
                                           {'layers': 'two',
                                            'nodes2': hp.choice('nodes2.2', n_nodes_options),
                                            'activation2': hp.choice('activation2.2', activation_options)},
                                            {'layers': 'three',
                                             'nodes2': hp.choice('nodes3.2', n_nodes_options),
                                             'activation2': hp.choice('activation3.2', activation_options),
                                             'nodes3': hp.choice('nodes3.3', n_nodes_options),
                                             'activation3': hp.choice('activation3.3', activation_options)},
                                            {'layers': 'four',
                                             'nodes2': hp.choice('nodes4.2', n_nodes_options),
                                             'activation2': hp.choice('activation4.2', activation_options),
                                             'nodes3': hp.choice('nodes4.3', n_nodes_options),
                                             'activation3': hp.choice('activation4.3', activation_options),
                                             'nodes4': hp.choice('nodes4.4', n_nodes_options),
                                             'activation4': hp.choice('activation4.4', activation_options)},
                                            {'layers': 'five',
                                             'nodes2': hp.choice('nodes5.2', n_nodes_options),
                                             'activation2': hp.choice('activation5.2', activation_options),
                                             'nodes3': hp.choice('nodes5.3', n_nodes_options),
                                             'activation3': hp.choice('activation5.3', activation_options),
                                             'nodes4': hp.choice('nodes5.4', n_nodes_options),
                                             'activation4': hp.choice('activation5.4', activation_options),
                                             'nodes5': hp.choice('nodes5.5', n_nodes_options),
                                             'activation5': hp.choice('activation5.5', activation_options)}
                                           ]),
    'learning_rate': hp.choice('learning_rate', [0.01, 0.001, 0.0001, 0.00001]),
    'batch_size': hp.choice('batch_size', [2 ** i for i in range(2, 7)]),
}


n_nodes_options = [2 ** i for i in range(2, 8)]
activation_options = ['relu', 'sigmoid']

fused_space = {
    'n_nodes_layer1': hp.choice('n_nodes_layer1', n_nodes_options),
    'layer1_activation': hp.choice('layer1_activation', activation_options),
    'num_layers': hp.choice('num_layers', [{'layers': 'one'},
                                           {'layers': 'two',
                                            'nodes2': hp.choice('nodes2.2', n_nodes_options),
                                            'activation2': hp.choice('activation2.2', activation_options)}
                                           ]),
    'learning_rate': hp.choice('learning_rate', [0.01, 0.001, 0.0001, 0.00001]),
    'loss': hp.choice('loss', ['mae','mse']),
    'batch_size': hp.choice('batch_size', [2 ** i for i in range(3, 7)]),
    'l2_2': hp.choice('l2_2', [ 0.1, 0.01, 0.001, 0.0001,0.00001, 0]),
    'l2_1': hp.choice('l2_1', [ 0.1, 0.01, 0.001, 0.0001,0.00001, 0]),
    'np_seed': hp.choice('np_seed', [0,1,2]),
    'tf_seed': hp.choice('tf_seed', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
}

n_convs_options = [16,32,64,128]
kernal_sz_options = [(2,2),(4,4),(8,8)]

cnn_space = {
    'n_nodes_layer1': hp.choice('n_convs_layer1', n_convs_options),
    'num_layers': hp.choice('num_layers', [{'layers': 'one',
                                            'n_convs1': hp.choice('n_convs1.1', n_convs_options),
                                            'kernal_sz1': hp.choice('kernal_sz1.2', kernal_sz_options)},
                                            {'layers': 'two',
                                             'n_convs1': hp.choice('n_convs2.1', n_convs_options),
                                             'n_convs2': hp.choice('n_convs2.2', n_convs_options),
                                             'kernal_sz1': hp.choice('kernal_sz2.1', kernal_sz_options),
                                             'kernal_sz2': hp.choice('kernal_sz2.2', kernal_sz_options)},
                                            {'layers': 'three',
                                             'n_convs1': hp.choice('n_convs3.1', n_convs_options),
                                             'n_convs2': hp.choice('n_convs3.2', n_convs_options),
                                             'n_convs3': hp.choice('n_convs3.3', n_convs_options),
                                             'kernal_sz1': hp.choice('kernal_sz3.1', kernal_sz_options),
                                             'kernal_sz2': hp.choice('kernal_sz3.2', kernal_sz_options),
                                             'kernal_sz3': hp.choice('kernal_sz3.3', kernal_sz_options)},
                                            {'layers': 'four',
                                             'n_convs1': hp.choice('n_convs4.1', n_convs_options),
                                             'n_convs2': hp.choice('n_convs4.2', n_convs_options),
                                             'n_convs3': hp.choice('n_convs4.3', n_convs_options),
                                             'n_convs4': hp.choice('n_convs4.4', n_convs_options),
                                             'kernal_sz1': hp.choice('kernal_sz4.1', kernal_sz_options),
                                             'kernal_sz2': hp.choice('kernal_sz4.2', kernal_sz_options),
                                             'kernal_sz3': hp.choice('kernal_sz4.3', kernal_sz_options),
                                             'kernal_sz4': hp.choice('kernal_sz4.4', kernal_sz_options)}
                                           ]),
    'batch_norm': hp.choice('batch_norm', [True, False]),
    # 'pooling': hp.choice('pooling', [True, False]),
    'learning_rate': hp.choice('learning_rate', [0.01, 0.001, 0.0001])
}
