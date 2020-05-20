import os
import tensorflow as tf
import numpy as np
import sys
PATH = os.getcwd()
sys.path.insert(1, 'PATH')

def test(args, model, sess, dataset, sp_bool):
    print('|========= START TEST =========|')
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    # tf.reset_default_graph()
    saver = tf.train.Saver(max_to_keep=10)
    # Identify which checkpoints are available.
    state = tf.train.get_checkpoint_state(args.path_model)
    model_files = {int(s[s.index('itr')+4:]): s for s in state.all_model_checkpoint_paths}
    itrs = sorted(model_files.keys())
    # Subset of iterations.
    itr_subset = itrs
    assert itr_subset
    # Evaluate.
    acc = []
    for itr in itr_subset:
        print('evaluation: {} | itr-{}'.format(dataset.datasource, itr))
        result = _evaluate(model, saver, model_files[itr], sess, dataset, args.batch_size, sp_bool)
        print('Accuracy: {:.5f} (#examples:{})'.format(result['accuracy'], result['num_example']))
        acc.append(result['accuracy'])
    print('Max: {:.5f}, Min: {:.5f} (#Eval: {})'.format(max(acc), min(acc), len(acc)))
    print('Error: {:.3f} %'.format((1 - max(acc))*100))

def _evaluate(model, saver, model_file, sess, dataset, batch_size, sp_bool):
    # load model
    if saver is not None and model_file is not None:
        saver.restore(sess, model_file)
    else:
        raise FileNotFoundError
    # load test set; epoch generator
    generator = dataset.generate_example_epoch(mode='test')

    accuracy = []
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
        # run the batch
        if batch['input'] and batch['label']:
            # stack and padding (if necessary)
            for key in keys:
                batch[key] = np.stack(batch[key])
            feed_dict = {}
            feed_dict.update({model.inputs[key]: batch[key] for key in keys})
            feed_dict.update({model.compress: False, model.is_train: False, model.pruned: True, model.sp:sp_bool})
            result = sess.run([model.outputs], feed_dict)
            accuracy.extend(result[0]['acc_individual'])
    results = { # has to be JSON serialiazable
        'accuracy': np.mean(accuracy).astype(float),
        'num_example': len(accuracy),
    }
    assert results['num_example'] == dataset.dataset['test']['input'].shape[0]
    return results
