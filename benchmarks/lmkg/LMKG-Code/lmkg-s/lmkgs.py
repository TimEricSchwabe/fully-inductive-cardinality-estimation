import argparse
import glob
import json
import re
import os, time, numpy as np
from tensorflow.keras.models import Sequential, Model, load_model  # type: ignore[import-untyped]
from tensorflow.keras.layers import Flatten, Dense, concatenate  # type: ignore[import-untyped]
import tensorflow as tf  # type: ignore[import-untyped]
from tensorflow.keras import backend as K  # type: ignore[import-untyped]
try:
    from tensorflow.keras.optimizers.legacy import Adam  # type: ignore[import-untyped]
except ImportError:
    from tensorflow.keras.optimizers import Adam  # type: ignore[import-untyped]
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from data_processor_release import read_star_graph_pattern, read_chain_graph_pattern, read_combined, read_combined_all_sizes_star_or_chain
from estimates_file import check_network_estimates
from tensorflow.keras import backend as K  # type: ignore[import-untyped]
import complex_reader

MIN: float = 0.0
MAX: float = 0.0

with tf.device("GPU:0"):
    def inverse_transform_minmax(y):
        """ Returns the inverse transform of the number """
        y = denormalize_MINMAX(y)
        y = K.exp(y)
        return y

    def q_loss(y_true, y_pred):
        """ Calculation of q_loss with the original values """
        y_true = inverse_transform_minmax(y_true)
        y_pred = inverse_transform_minmax(y_pred)
        return K.maximum(y_true, y_pred) / K.minimum(y_true, y_pred)

    def write_MIN_MAX(file_name, MIN, MAX):
        os.makedirs("final_datasets/minmax", exist_ok=True)
        with open("final_datasets/minmax/" + file_name, "w") as f:
            f.write(str(MIN)+" "+str(MAX))

    def normalize(y, min, max):
        """ Normalization used for the cardinality """
        y = (y - min) / (max - min)
        return y

    def denormalize(y, min, max):
        """ Denormalization used for the cardinality """
        y = y * (max - min) + min
        return y


    def create_flatten_mlp(shape, include_layer = False, regress = False, additional_layers = [256, 256]):
        """
        Creates Flattening and possible adds an MLP Layer for X, A, E separately
        :param shape: [input_shape] of the input that needs to be passed by the layers
        :param include_layer: boolean, indicates whether we are adding MLP for the specific input
        :param regress: boolean, parameter used mainly for testing
        :param additional_layers: array of neurons for the additional MLP
        :return: model
        """

        total = 1
        for s in shape:
            total *= s
        model = Sequential()
        model.add(Flatten(input_shape=shape))

        if include_layer:
            for i in range(1, len(additional_layers)):
                model.add(Dense(additional_layers[i], activation="relu"))
            # model.add(Dense(128, activation="relu"))
        # check to see if the regression node should be added
        if regress:
            model.add(Dense(1, activation="linear"))
        # return our model
        return model


    def normalize_MINMAX(y):
        """ Normalization with MIN and MAX value """
        if 'MIN' not in globals():
            global MIN, MAX
            MIN = min(y)
            MAX = max(y)
        if MAX == MIN:
            return np.zeros_like(y)
        y = (y - MIN) / (MAX - MIN)
        return y

    def denormalize_MINMAX(y):
        """ Denormalization with MIN and MAX value """
        y = (y * (MAX - MIN)) + MIN
        return y

    def read_MIN_MAX(file_name):
        """ Reading of MIN and MAX value needed for prediction time since we require the same scaling as during training"""
        with open("final_datasets/minmax/" + file_name, "r") as f:
            line = f.readline()
            min_y = int(line.split(" ")[0])
            max_y = int(line.split(" ")[1])
            return min_y, max_y

    def create_model(matrix_mode, d, b, n, e, layers = [1024, 1024]):
        """
        Creates Flattening and possible adds an MLP Layer for X, A, E separately
        :param matrix_mode: 0,1 being two matrices 2 and 3 being with 3 matrices as reported in paper, one-hot and binary encoding, 4 works only for a specific query
        :param d: int, the number of distinct nodes (subjects + objects) in KG
        :param b: int, the number of distinct edges (predicates) in KG
        :param n: int, the number of nodes in the subgraph
        :param e: int, the number of edges in the subgraph
        :param layers: array of neurons for the additional MLP
        :return: model
        """

        """ Currently supported 2 and 3. 3 as measured in the experiments """
        if matrix_mode == 0:
            # 2 matrices, one-hot encoding
            # create the MLP models
            nn1 = create_flatten_mlp((n, d), regress=False)
            nn2 = create_flatten_mlp((b, n, n), regress=False)
            # create the input to our final set of layers as the *output* of previous layer
            combinedInput = concatenate([nn1.output, nn2.output])
        elif matrix_mode == 1:
            # 2 matrices, binary encoding
            binary_d = int(np.ceil(np.log2(d))) + 1
            # create the MLP models
            nn1 = create_flatten_mlp((n, binary_d), regress=False)
            nn2 = create_flatten_mlp((b, n, n), regress=False)
            # create the input to our final set of layers as the *output* of previous layer
            combinedInput = concatenate([nn1.output, nn2.output])
        elif matrix_mode == 2:
            # 3 matrices, one-hot encoding
            # create the MLP models
            nn1 = create_flatten_mlp((n, d), include_layer=False, regress=False)
            nn2 = create_flatten_mlp((e, b), include_layer=False, regress=False)
            nn3 = create_flatten_mlp((e, n, n), include_layer=False, regress=False)
            # create the input to our final set of layers as the *output* of previous layer
            combinedInput = concatenate([nn1.output, nn2.output, nn3.output])
        elif matrix_mode == 3:
            # 2 matrices, binary encoding
            binary_d = int(np.ceil(np.log2(d))) + 1
            binary_b = int(np.ceil(np.log2(b))) + 1
            # create the MLP models
            nn1 = create_flatten_mlp((n, binary_d), False, regress=False)
            nn2 = create_flatten_mlp((e, binary_b), False, regress=False)
            nn3 = create_flatten_mlp((e, n, n), False, regress=False)
            # create the input to our final set of layers as the *output* of previous layer
            combinedInput = concatenate([nn1.output, nn2.output, nn3.output])
        elif matrix_mode == 4:
            # can be used for a single query type
            # create the MLP and CNN models
            binary_d = int(np.ceil(np.log2(d))) + 1
            binary_b = int(np.ceil(np.log2(b))) + 1
            # create the MLP models
            nn1 = create_flatten_mlp((n, binary_d), False, regress=False)
            nn2 = create_flatten_mlp((e, binary_b), False, regress=False)
            # create the input to our final set of layers as the *output* of previous layer
            combinedInput = concatenate([nn1.output, nn2.output])
        else:
            print("Non-supported")
            exit(1)

        # our final dense layers
        x = Dense(layers[0], activation="relu")(combinedInput)
        # x = Dropout(.1)(x)
        for i in range(1, len(layers)):
            x = Dense(layers[i], activation="relu")(x)
            # x = Dropout(.3)(x)
        x = Dense(1, activation="sigmoid")(x)

        if matrix_mode == 0 or matrix_mode == 1 or matrix_mode == 4:
            model = Model(inputs=[nn1.input, nn2.input], outputs=x)
        elif matrix_mode == 2 or matrix_mode == 3:
            model = Model(inputs=[nn1.input, nn2.input, nn3.input], outputs=x)
        else:
            print("Matrix mode unsupported")
            exit(1)
        return model




    def train_model(model, X, A, E, y, batch_size, epochs, matrix_mode, learning_rate = 1e-3, decay = False, use_q_loss = True, scale = 0, do_training = False, encoding_avg_time = 0, query_set_name = "unknown"):
        """
        Actual training of the model
        :param model: the created model for training
        :param X: nodes matrix
        :param A: adjacency tensor
        :param E: edges matrix
        :param y: cardinality
        :param batch_size:
        :param epochs:
        :param matrix_mode:
        :param learning_rate:
        :param decay:
        :param use_q_loss:
        :param scale: type of scaling, recommended and best is the currently selected scale = 3
        :param do_training: to train or to predict
        :param encoding_avg_time: time needed for encoding, if needed for storing
        :return:
        """
        if decay:
            opt = Adam(learning_rate=learning_rate, decay=learning_rate / 300)
        else:
            opt = Adam(learning_rate=learning_rate)

        max_cardinality = max(y)
        min_cardinality = min(y)

        if do_training:
            write_MIN_MAX(query_type + "_min_max_" + dataset_name + ".txt", min(y), max(y))

        """ Other explored strategies, LMKG currently uses scale 3 """
        if scale == 0:
            print("We are using log1p")
            y = np.log(y + 1) + 1
        elif scale == 1:
            print("We are normalizing")
            y = normalize(y, min_cardinality, max_cardinality)
        elif scale == 2:
            scaler_y = MinMaxScaler()
            print("We are standardizing")
            y = np.log(y)
            scaler_y.fit(np.reshape(y, (-1, 1)))
            y = scaler_y.transform(np.reshape(y, (-1, 1)))
        elif scale == 3:
            global MIN, MAX

            y = np.log(np.maximum(y, 1))
            if do_training:
                MIN = float(np.min(y))
                MAX = float(np.max(y))
            else:
                """ We need the same min and max from the training data during evaluation """
                raw_min, raw_max = read_MIN_MAX(query_type+"_min_max_"+dataset_name+".txt")
                MIN = np.log(max(raw_min, 1))
                MAX = np.log(max(raw_max, 1))
            y = normalize_MINMAX(y)
        else:
            print("Scale not supported")

        """ As loss there is the option to have q_loss or mse, q_loss is currently better """
        print("As loss we are using qloss: " + str(use_q_loss))
        loss = q_loss if use_q_loss else 'mse',
        metrics = ['mae', 'mse', 'accuracy']
        model.compile(loss=loss, optimizer=opt, metrics = metrics)

        x=[]
        if matrix_mode == 0:
            x = [X, A]
        elif matrix_mode == 1:
            x = [X]
        elif matrix_mode == 2 or matrix_mode == 3:
            x = [X, E, A]
        elif matrix_mode == 4:
            x = [X, E]
        if do_training:
            for _ in range(epochs):
                model.fit(
                    x=x, y=y,
                    epochs=1, batch_size=batch_size, shuffle=True)
        print_time = ""

        avg_time_prediction_ms = 0
        per_query_prediction_ms = None
        if do_training:
            preds = model.predict(x, verbose=0)
        else:
            preds_rows = []
            per_query_prediction_ms = []
            N_REPEATS = 5
            for idx in range(len(y)):
                x_single = [arr[idx:idx + 1] for arr in x]
                repeat_preds = []
                t0 = time.time()
                for _ in range(N_REPEATS):
                    repeat_preds.append(model.predict(x_single, verbose=0, batch_size=1)[0])
                preds_rows.append(np.mean(repeat_preds, axis=0))

                total_time = (time.time() - t0) * 1000.0 / N_REPEATS
                per_query_prediction_ms.append(total_time)

            preds = np.array(preds_rows)
            avg_time_prediction_ms = float(np.mean(per_query_prediction_ms))
        print(f"The average prediction time is {avg_time_prediction_ms:.3f} ms")
        print(f"The average encoding time is {avg_time_encoding_ms:.3f} ms")


        if scale == 0:
            preds = np.round(np.exp(preds-1)-1).astype(int)
            y = np.round(np.exp(y-1)-1).astype(int)
        elif scale == 1:
            preds = denormalize(preds, min_cardinality, max_cardinality)
            y = denormalize(y, min_cardinality, max_cardinality)
        elif scale == 2:
            preds = np.reshape(preds, (-1, 1))
            y = np.reshape(y, (-1, 1))
            preds = scaler_y.inverse_transform(preds)
            y = scaler_y.inverse_transform(y)
            preds = np.exp(preds)
            y = np.exp(y)
        elif scale == 3:
            preds = denormalize_MINMAX(preds)
            y = denormalize_MINMAX(y)
            preds = np.exp(preds)
            y = np.exp(y)
        else:
            print("Scale not supported")

        # removing outliers if needed
        nb_outliers = 0
        if nb_outliers > 0:
            sorted_y = y
            sorted_y = sorted(sorted_y, reverse = True)
            outliers = sorted_y[:nb_outliers]
            preds_outlier = []
            y_outlier = []
            for i in range(len(y)):
                if y[i] not in outliers:
                    y_outlier.append(y[i])
                    preds_outlier.append(preds[i])
            y = y_outlier
            preds = preds_outlier

        if do_training:
            query_to_estimate_time_ms = None
        elif per_query_prediction_ms is not None:
            query_to_estimate_time_ms = [avg_time_encoding_ms + pred_ms for pred_ms in per_query_prediction_ms]
        else:
            query_to_estimate_time_ms = (avg_time_encoding_ms + avg_time_prediction_ms)

        # accuracy statistics for the network

        correct_estimate = check_network_estimates(
            preds, y, statistics_file_name=statistics_file_name, print_time=print_time,
            nb_outliers=nb_outliers, dataset_name=dataset_name, query_type=query_type,
            query_set_name=query_set_name,
            epoch=epochs if do_training else 1,
            stage="train" if do_training else "eval",
            write_csv=not do_training,
            query_to_estimate_time_ms=query_to_estimate_time_ms,
            per_query_meta=per_query_meta,
        )
        # correct_estimate = store_statistics.check_network_estimates(preds, y, statistics_file_name = statistics_file_name, print_time=print_time, nb_outliers = nb_outliers)

        return correct_estimate



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', dest='training', action='store_true')
    parser.add_argument('--eval', dest='training', action='store_false')
    parser.set_defaults(training=True)
    parser.add_argument('--batch-size', type=int, default=512, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=100, help='# epochs')
    parser.add_argument('--dataset', type=str, default='swdf', help='The dataset name')
    parser.add_argument('--query-type', type=str, default ='combined', help='Supports star, chain, combined or complex')
    parser.add_argument('--datasets-path', type=str, default='final_datasets', help='Path to the train and eval queries')
    parser.add_argument('--query-join', type=int, default=2, help='The size of the join.')
    parser.add_argument('--layers', type=int, default=2, help='The number of layers in the NN.')
    parser.add_argument('--neurons', type=int, default=256, help='The number of neurons in each of the layers.')
    parser.add_argument('--decay', action="store_true", help='Decay of the learning rate.')
    parser.add_argument('--test-mode', type=str, default = "all", help='[all, star, chain], Used for testing. In case we have trained a combined model (for star and chain) '
                                                        'we can evaluate only on star, only on chain or on all')


    args = parser.parse_args()

    print("GPU AVAILABLE")
    print(tf.test.is_gpu_available())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    ''' parameters for the network'''
    # Fixed parameters
    matrix_mode = 3
    train_tuples = 1_000_000_000
    # Changable parameters
    do_training = args.training
    create_new_model = True
    if not do_training:
        create_new_model = False
    batch_size = args.batch_size
    epochs = args.epochs
    datasets_path = args.datasets_path
    decay = args.decay


    '''dimension parameters for the network'''
    global dataset_name
    dataset_name = args.dataset
    if dataset_name == 'swdf':
        # SWDF
        d = 76712
        b = 171
        n = 4
        e = 4
    elif dataset_name == 'yago':
        # YAGO
        d = 12811192
        b = 91
        n = 4
        e = 4
    elif dataset_name == 'lubm1M':
        # LUBM1M
        d = 664050
        b = 19
        n = 4
        e = 4
    else:
        stats_path = os.path.join("statistics", dataset_name + "_stats.json")
        if not os.path.exists(stats_path):
            print(f"Unknown dataset '{dataset_name}'. No stats file found at {stats_path}.")
            exit(1)
        with open(stats_path) as _f:
            _stats = json.load(_f)
        d = _stats["d"]
        b = _stats["b"]
        n = args.query_join + 1
        e = args.query_join

    # Load stats JSON for query size discovery (used by all* modes)
    _stats_path = os.path.join("statistics", dataset_name + "_stats.json")
    _stats = None
    if os.path.exists(_stats_path):
        with open(_stats_path) as _f:
            _stats = json.load(_f)

    layers = [args.neurons for i in range(args.layers)]
    nn_size_name = '_'.join([str(neuron) for neuron in layers])
    scale = 3

    query_join = args.query_join
    # Can be [star chain - combined (query-size grouping) allstar allchain (query-type grouping) alltypessizes (single model), complex (single model)]
    query_type = args.query_type



    print(" The model is trained: %r \n Batch size: %s "
          "\n Epochs: %d \n Dataset: %s \n Query type: %s \n "
          "Datasets path: %s \n Query Join: %d \n NN Size: %s \n"
          "Learning rate decay: %r \n"
          % (do_training, batch_size, epochs, dataset_name, query_type, datasets_path, query_join, nn_size_name, decay))

    if "complex" not in query_type and "all" not in query_type:
        query_type = query_type + '_' + str(query_join)


    n = query_join + 1
    e = query_join

    if "complex" in query_type:
        n = 8
        e = 6
    elif 'allstar' in query_type or 'allchain' in query_type or 'alltypessizes' in query_type:
        if _stats is None:
            print(f"Stats file required for '{query_type}' mode but not found at {_stats_path}.")
            exit(1)
        _all_sizes = []
        if 'allstar' in query_type or 'alltypessizes' in query_type:
            _all_sizes.extend(_stats.get("star_sizes", []))
            if not do_training:
                _all_sizes.extend(_stats.get("eval_star_sizes", []))
        if 'allchain' in query_type or 'alltypessizes' in query_type:
            _all_sizes.extend(_stats.get("path_sizes", []))
            if not do_training:
                _all_sizes.extend(_stats.get("eval_path_sizes", []))
        if not _all_sizes:
            print(f"No query sizes found in stats for '{query_type}' mode.")
            exit(1)
        max_join = max(_all_sizes)
        n = max_join + 1
        e = max_join
        print(f"  all* mode: using n={n}, e={e} (max query size {max_join})")

    model_name = 'LMKGS-' + query_type + '-q_loss-' + nn_size_name + '-' + dataset_name + 'scale-' + str(scale) + '_decay'+str(decay) + '.h5'

    weights_name = model_name
    statistics_file_name = 'statistics-'+ model_name

    _base_type = re.sub(r'_\d+$', '', query_type)  # "star_2" → "star"
    query_set_name = f"{dataset_name}_{_base_type}_queries"

    if do_training:
        if "star" in query_type and 'all' not in query_type:
            print("entering for star queries")
            file_name = datasets_path + '/generated_' + dataset_name + '/star/'+query_type+'.txt'
        elif 'chain' in query_type and 'all' not in query_type:
            print("entering for chain queries")
            file_name = datasets_path + '/generated_' + dataset_name + '/path/' + query_type + '.txt'
        elif 'combined' in query_type:
            print("entering for query size grouping")
            file_name_star = datasets_path + '/generated_' + dataset_name + '/star/star_' + str(query_join) + '.txt'
            file_name_chain = datasets_path + '/generated_' + dataset_name + '/path/chain_' + str(query_join) + '.txt'
        elif 'allstar' in query_type or 'allchain' in query_type or 'alltypessizes' in query_type:
            all_file_names = []
            if 'allstar' in query_type or 'alltypessizes' in query_type:
                for i in _stats.get("star_sizes", []):
                    all_file_names.append(datasets_path + '/generated_' + dataset_name + '/star/star_' + str(i) + '.txt')
            if 'allchain' in query_type or 'alltypessizes' in query_type:
                for i in _stats.get("path_sizes", []):
                    all_file_names.append(datasets_path + '/generated_' + dataset_name + '/path/chain_' + str(i) + '.txt')
            print(f"  Training files: {all_file_names}")
        elif 'complex' in query_type:
            path = datasets_path + '/generated_' + dataset_name + '/complex/'
            complex_file_names = []
            complex_file_names.append(path + 'swdf_complex_star1_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star2_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star3_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star3_chain3.txt')
        else:
            print("Query type not supported.")
            exit(1)
    else:
        if "star" in query_type and 'all' not in query_type:
            print("entering for star")
            matches = glob.glob(datasets_path + '/generated_' + dataset_name + '/star/eval_*_'+query_type+'.txt')
            if not matches:
                raise FileNotFoundError(f"No eval file found for star/{query_type} in dataset {dataset_name}")
            file_name = matches[0]
        elif 'chain' in query_type and 'all' not in query_type:
            print("entering for chain")
            matches = glob.glob(datasets_path + '/generated_' + dataset_name + '/path/eval_*_'+query_type+'.txt')
            if not matches:
                raise FileNotFoundError(f"No eval file found for path/{query_type} in dataset {dataset_name}")
            file_name = matches[0]
        elif 'combined' in query_type:
            print("entering for combined")
            test_mode = args.test_mode
            star_matches = glob.glob(datasets_path + '/generated_' + dataset_name + '/star/eval_*_star_' + str(query_join) + '.txt')
            chain_matches = glob.glob(datasets_path + '/generated_' + dataset_name + '/path/eval_*_chain_' + str(query_join) + '.txt')
            file_name_star = None
            file_name_chain = None
            if test_mode in ("all", "star"):
                if not star_matches:
                    raise FileNotFoundError(f"No star eval file found for {dataset_name} size {query_join}")
                file_name_star = star_matches[0]
            if test_mode in ("all", "chain"):
                if not chain_matches:
                    raise FileNotFoundError(f"No chain eval file found for {dataset_name} size {query_join}")
                file_name_chain = chain_matches[0]
        elif 'allstar' in query_type or 'allchain' in query_type or 'alltypessizes' in query_type:
            all_file_names = []
            if 'allstar' in query_type or 'alltypessizes' in query_type:
                eval_star_sizes = _stats.get("eval_star_sizes", _stats.get("star_sizes", []))
                for i in eval_star_sizes:
                    ms = glob.glob(datasets_path + '/generated_' + dataset_name + '/star/eval_*_star_' + str(i) + '.txt')
                    if ms: all_file_names.append(ms[0])
            if 'allchain' in query_type or 'alltypessizes' in query_type:
                eval_path_sizes = _stats.get("eval_path_sizes", _stats.get("path_sizes", []))
                for i in eval_path_sizes:
                    mc = glob.glob(datasets_path + '/generated_' + dataset_name + '/path/eval_*_chain_' + str(i) + '.txt')
                    if mc: all_file_names.append(mc[0])
            print(f"  Eval files: {all_file_names}")
        elif 'complex' in query_type:
            path = datasets_path + '/generated_' + dataset_name + '/complex/'
            complex_file_names = []
            complex_file_names.append(path + 'swdf_complex_star1_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star2_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star3_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star3_chain3.txt')
        else:
            print("Query type not supported.")
            exit(1)


    per_query_meta = None
    if "star" in query_type and 'all' not in query_type:
        X, A, E, y, encoding_time = complex_reader.read_queries(file_name, d, b, n, e, query_type ="star")
        # X, A, E, y, encoding_time = read_star_graph_pattern(d, b, n, e, file_name, train_tuples, matrix_mode, 0)
    elif "chain" in query_type and 'all' not in query_type:
        X, A, E, y, encoding_time = complex_reader.read_queries(file_name, d, b, n, e, query_type ="chain")
        # X, A, E, y, encoding_time = read_chain_graph_pattern(d, b, n, e, file_name, train_tuples, matrix_mode, 0)
    elif 'combined' in query_type:
        test_mode = args.test_mode
        statistics_file_name += test_mode
        X, A, E, y, encoding_time = complex_reader.read_combined(d, b, n, e, file_name_star, file_name_chain, train_tuples, test_mode)
    elif 'allstar' in query_type or 'allchain' in query_type or 'alltypessizes' in query_type:
        X, A, E, y, encoding_time, per_query_meta = complex_reader.read_combined_all_sizes_star_or_chain(d, b, n, e, all_file_names, train_tuples)
    elif 'complex' in query_type:
        if not do_training:
            X, A, E, y, encoding_time = complex_reader.read_complex_queries(complex_file_names, d, b, n, e, 100)
        else:
            X, A, E, y, encoding_time = complex_reader.read_complex_queries(complex_file_names, d, b, n, e)
    else:
        print("Query type not supported.")
        exit(1)

    avg_time_encoding_ms = (encoding_time / len(y)) * 1000.0


    '''training/testing/predicting'''
    '''create the model'''
    if create_new_model:
        '''create new model for the data'''
        nn_model = create_model(matrix_mode, d, b, n, e, layers = layers)
    else:
        #load the model
        save_path = 'models/'+dataset_name+"/"
        nn_model = load_model(save_path + model_name, custom_objects={"q_loss": q_loss})


    '''train the model'''
    train_model(nn_model, X, A, E, y, batch_size, epochs, decay = decay, matrix_mode = matrix_mode, scale = scale, do_training=do_training, encoding_avg_time=avg_time_encoding_ms, query_set_name=query_set_name)


    '''storing of the model'''
    if do_training:
        save_path = 'models/'+dataset_name+"/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        nn_model.save(save_path + model_name)

        '''storing just the weights of the model'''
        weights = nn_model.get_weights()
        weights_path = 'weights/'
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        with open(weights_path + model_name + '_size.txt', "w") as f:
            f.write(str(weights))