import os
import argparse
import tensorflow as tf
import sys
import numpy
import time
import glob

PATH = os.getcwd()
sys.path.insert(1, 'PATH')
import dataset as dtst
import model as mdl
import prune
import train
import test
import metrics as mtrs

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    parser = argparse.ArgumentParser()
    # Data options
    parser.add_argument('--datasource', type=str, default='mnist', help='dataset to use')
    parser.add_argument('--path_data', type=str, default='./', help='location to dataset')
    parser.add_argument('--aug_kinds', nargs='+', type=str, default=[], help='augmentations to perform')
    # Model options
    parser.add_argument('--arch', type=str, default='lenet5', help='network architecture to use')
    parser.add_argument('--target_sparsity', type=float, default=0.90, help='level of sparsity to achieve')
    #parser.add_argument('--target_sparsity_sp', type=float, default=0, help='level of sparsity to achieve')
    #the target sparsity is set manually in model.py
    # Train options
    parser.add_argument('--batch_size', type=int, default=100, help='number of examples per mini-batch')
    parser.add_argument('--train_iterations', type=int, default=100, help='number of training iterations')
    parser.add_argument('--train_iterations_sp', type=int, default=100, help='number of training iterations')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer of choice')
    parser.add_argument('--lr_decay_type', type=str, default='constant', help='learning rate decay type')
    parser.add_argument('--lr', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument('--decay_boundaries', nargs='+', type=int, default=[], help='boundaries for piecewise_constant decay')
    parser.add_argument('--decay_values', nargs='+', type=float, default=[], help='values for piecewise_constant decay')
    #pruning options
    parser.add_argument('--snip', type=str2bool, default=True, help='True if you want to do snip')
    parser.add_argument('--sparse_pruning', type=str2bool, default=True, help='True if you want to do sparse pruning')
    # Initialization
    parser.add_argument('--initializer_w_bp', type=str, default='vs', help='initializer for w before pruning')
    parser.add_argument('--initializer_b_bp', type=str, default='zeros', help='initializer for b before pruning')
    parser.add_argument('--initializer_w_ap', type=str, default='vs', help='initializer for w after pruning')
    parser.add_argument('--initializer_b_ap', type=str, default='zeros', help='initializer for b after pruning')
    parser.add_argument('--seed', type=int, default='4404589', help='seed for the random functions')
    # Logging, saving, options
    parser.add_argument('--save_output', type =str2bool, default = True, help='False if you want to see the output directly, when True is you want it in a .txt file')
    parser.add_argument('--logdir', type=str, default='logs', help='location for summaries and checkpoints')
    parser.add_argument('--check_interval', type=int, default=100, help='check interval during training')
    parser.add_argument('--save_interval', type=int, default=100, help='save interval during training')
    args = parser.parse_args()
    # Add more to args
    args.path_summary = os.path.join(args.logdir, 'summary')
    args.path_model = os.path.join(args.logdir, 'model')
    args.path_assess = os.path.join(args.logdir, 'assess')
    return args

path_output=""
def main(args):
    tf.set_random_seed(args.seed)
    numpy.random.seed(args.seed)

    # Dataset
    dataset = dtst.Dataset(**vars(args))

    # Reset the default graph and set a graph-level seed
    tf.reset_default_graph()
    tf.disable_eager_execution()

    # Model
    model = mdl.Model(num_classes=dataset.num_classes, **vars(args))
    model.construct_model()

    # Session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    def clear():
        files = glob.glob('./logs/summary/train/*')
        for f in files:
            os.remove(f)
        files = glob.glob('./logs/summary/val/*')
        for f in files:
            os.remove(f)
        files = glob.glob('./logs/model/*')
        for f in files:
            os.remove(f)
    clear()

    # Prune: SNIP
    if args.snip:
        prune.prune(args, model, sess, dataset)

    # Train and test
    train.train(args, model, sess, dataset, False)
    test.test(args, model, sess, dataset, False)

    # Prune: SPARSE PRUNING
    if args.sparse_pruning:
        clear()
        prune.sparse_pruning(args, model, sess, dataset)
        train.train(args, model, sess, dataset, True)
        test.test(args, model, sess, dataset, True)

    #Calculate the metrics
    path_metrics =  path_output
    mtrs.metrics(args, model, sess, dataset, path_metrics)

    #set sparse_pruning False to use the following method
    if args.snip and not args.sparse_pruning:
        mtrs.check_layers(args, model, sess, dataset)

    sess.close()
    sys.exit()

if __name__ == "__main__":
    args = parse_arguments()
    if args.save_output:
        orig_stdout = sys.stdout
        if not os.path.exists("./logs/output"):
            os.makedirs("./logs/output")
        path_output = "./logs/output" + "/" + str(time.strftime("%Y%m%d-%H%M%S")) + "_" + str(args.datasource) + "_" + str(args.snip) + "_" + str(args.target_sparsity) + \
                      "_" + str(args.sparse_pruning) + "_" +  str(args.train_iterations) \
                      + "_" + str(args.train_iterations_sp) + "/"
        os.mkdir(path_output)
        d = path_output +"/" + 'output.txt'
        f = open(d, 'w')
        sys.stdout = f
        main(args)
        sys.stdout = orig_stdout
        f.close()
    else:
        main()

