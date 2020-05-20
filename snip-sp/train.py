import os
import tensorflow as tf
import time
import numpy as np
import sys
PATH = os.getcwd()
sys.path.insert(1, 'PATH')
import augment


def train(args, model, sess, dataset,sp_bool):
    print('|========= START TRAINING =========|')
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    if not os.path.isdir(args.path_summary): os.makedirs(args.path_summary)
    if not os.path.isdir(args.path_model): os.makedirs(args.path_model)
    saver = tf.train.Saver()
    random_state = np.random.RandomState(args.seed)
    writer = {}
    writer['train'] = tf.summary.FileWriter(args.path_summary + '/train', sess.graph)
    writer['val'] = tf.summary.FileWriter(args.path_summary + '/val')
    t_start = time.time()
    check_convergence=[]
    convergence = False
    iters = args.train_iterations_sp if sp_bool else args.train_iterations
    for itr in range(iters):
        if convergence:
            break
        batch = dataset.get_next_batch('train', args.batch_size)
        batch = augment.augment(batch, args.aug_kinds, random_state)
        feed_dict = {}
        feed_dict.update({model.inputs[key]: batch[key] for key in ['input', 'label']})
        feed_dict.update({model.compress: False, model.is_train: True, model.pruned: True, model.sp: sp_bool})
        input_tensors = [model.outputs] # always execute the graph outputs
        if (itr+1) % args.check_interval == 0:
            input_tensors.extend([model.summ_op, model.sparsity])
        input_tensors.extend([model.train_op])
        result = sess.run(input_tensors, feed_dict)

        # Check on validation set.
        if (itr+1) % args.check_interval==0: #args.check_interval == 0:
            batch = dataset.get_next_batch('val', args.batch_size)
            batch = augment.augment(batch, args.aug_kinds, random_state)
            feed_dict = {}
            feed_dict.update({model.inputs[key]: batch[key] for key in ['input', 'label']})
            feed_dict.update({model.compress: False, model.is_train: False, model.pruned: True, model.sp: sp_bool})
            input_tensors = [model.outputs, model.summ_op, model.sparsity]
            result_val = sess.run(input_tensors, feed_dict)
            check_convergence.append(result_val[0]['los'])
            #Check if the model is converged, if yes, the training will stop
            if len(check_convergence)>20:
                if min(check_convergence[-20:])>min(check_convergence[:-20]):
                    convergence = True

        # Check summary and print results
        if (itr+1) % args.check_interval == 0:
            writer['train'].add_summary(result[1], itr)
            writer['val'].add_summary(result_val[1], itr)
            pstr = '(train/val) los:{:.3f}/{:.3f} acc:{:.3f}/{:.3f} spa:{:.3f}'.format(
                result[0]['los'], result_val[0]['los'],
                result[0]['acc'], result_val[0]['acc'],
                result[2],
            )
            print('itr{}: {} (t:{:.1f})'.format(itr+1, pstr, time.time() - t_start))
            t_start = time.time()

        # Save model
        if (itr+1) % args.save_interval == 0:
            saver.save(sess, args.path_model + '/itr-' + str(itr))
