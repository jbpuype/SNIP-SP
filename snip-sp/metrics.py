import tensorflow as tf
import numpy as np
from copy import deepcopy
from sklearn import metrics as ms
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import os, sys
PATH = os.getcwd()
sys.path.insert(1, 'PATH')
classes=[]
sparse_layers=[]

def metrics(args, model, sess, dataset, path):
    global classes
    classes=[]
    print('|========= START METRICS =========|')
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    bool = True
    try:
        original_classes, pred_classes = get_labels(args, model, sess, dataset)
    except TypeError:
        print("Network is pruned too much, auwch!")
        bool = False
    if bool:
        np.savetxt(path + 'labels.txt', np.vstack([original_classes, pred_classes]), delimiter=" ", newline = "\n", fmt="%s")
        confusion_matrix(original_classes, pred_classes, path)
        print('Precision score (micro-averaged):', ms.precision_score(original_classes, pred_classes, average='micro'))
        # print('Precision score (macro-averaged):', ms.precision_score(original_classes, pred_classes, average='macro'))
        print('Recall score (micro-averaged):', ms.recall_score(original_classes, pred_classes, average='micro'))
        # print('Recall score (macro-averaged):', ms.recall_score(original_classes, pred_classes, average='macro'))

#The next function calculates where SNIP prunes:
#The first output line is the number of parameters of the layer and the number of parameters that remains after pruning,
# for each layer of the model
#Followed by the ratio of parameters pruned for each layer
#Followed by the number of remaining parameters, the number of total parameters and the ratio of the remaining paramaters
def check_layers(args, model, sess, dataset):
    print('|========= PARAMETERS PER LAYER =========|')
    sp=sparse_layers[0][:len((sparse_layers[0]))//2]
    print(sp)
    s = len(sp)//2
    j = []
    k = [0, 0]
    for i in range(s):
        j.append(sp[2 * i + 1][0] / sparse_layers[0][2 * i][0])
        k[0] += sp[2 * i + 1][0]
        k[1] += sp[2 * i][0]
    print(j)
    print(k[0], k[1], k[0] / k[1])

def confusion_matrix(original_classes, pred_classes, path):
    matrix = ms.confusion_matrix(original_classes, pred_classes)
    matrix = np.asmatrix(matrix)
    file_matrix = path +'matrix.txt'
    np.savetxt(file_matrix, matrix)
    matrix_normalized = np.divide(matrix, np.sum(matrix))
    df_cm = pd.DataFrame(matrix_normalized, range(len(matrix_normalized)), range(len(matrix_normalized)))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})  # font size
    file_figure = path + 'matrix.png'
    if len(matrix_normalized) > 15:
        figure = plt.gcf()
        figure.set_size_inches(40, 50)
    plt.savefig(file_figure, dpi=100)
    plt.show()

#Get the correct labels and the predicted labels for so one can calculate the performance of the model
def get_labels(args, model, sess, dataset):
    # tf.reset_default_graph()
    saver = tf.train.Saver(max_to_keep=10)
    # Identify which checkpoints are available.
    state = tf.train.get_checkpoint_state(args.path_model)
    model_files = {int(s[s.index('itr')+4:]): s for s in state.all_model_checkpoint_paths}
    itrs = sorted(model_files.keys())
    # Subset of iterations.
    itr_subset = itrs
    assert itr_subset
    _evaluate(args, model, saver, model_files[itr_subset[-1]], sess,dataset, 10000)
    if np.amax(classes[1])==0:
        return None
    L=[]
    for i in classes[1]:
        l = np.where(i==np.amax(i))[0]
        if len(l)>1:
            l = l[0]
        L.append(l.item())
    return classes[0][0]['label'], L

def _evaluate(args, model, saver, model_file, sess, dataset, batch_size):
    global classes
    global sparse_layers
    # load model
    if saver is not None and model_file is not None:
        saver.restore(sess, model_file)
    else:
        raise FileNotFoundError
    # load test set; epoch generator
    generator = dataset.generate_example_epoch(mode='test')

    empty = False
    while not empty:
        # construct a batch of test examples
        keys = ['input', 'label']
        batch = {key: [] for key in keys}
        for i in range(batch_size):
            try:
                example = next(generator)
                for key in keys:
                    batch[key].append(example[key])
            except StopIteration:
                empty = True
        classes.append([batch])
        # run the batch
        if batch['input'] and batch['label']:
            # stack and padding (if necessary)
            for key in keys:
                batch[key] = np.stack(batch[key])
            feed_dict = {}
            feed_dict.update({model.inputs[key]: batch[key] for key in keys})
            feed_dict.update({model.compress: False, model.is_train: False, model.pruned: True, model.sp:args.sparse_pruning, model.pred_cl_bool:True})
            result = sess.run([model.outputs, model.logits, model.pred_cl, model.layers], feed_dict)
            classes.append(deepcopy(result[0]['logits']))
            sparse_layers.append(result[3])
