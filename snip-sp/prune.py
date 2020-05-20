import time

def prune(args, model, sess, dataset):
    print('|========= START SNIP PRUNING =========|')
    t_start = time.time()
    batch = dataset.get_next_batch('train', args.batch_size)
    feed_dict = {}
    feed_dict.update({model.inputs[key]: batch[key] for key in ['input', 'label']})
    feed_dict.update({model.compress: True, model.is_train: False, model.pruned: False, model.sp: False})
    result = sess.run([model.outputs, model.sparsity], feed_dict)
    print('Pruning: {:.3f} global sparsity (t:{:.1f})'.format(result[-1], time.time() - t_start))

def sparse_pruning(args, model, sess, dataset):
    print('|========= START SPARSE PRUNING =========|')
    t_start = time.time()
    batch = dataset.get_next_batch('train', args.batch_size)
    feed_dict={}
    feed_dict.update({model.inputs[key]: batch[key] for key in ['input', 'label']})
    feed_dict.update({model.compress: False, model.is_train: False, model.pruned: True, model.sp: True})
    result = sess.run([model.outputs, model.sparsity], feed_dict)
    print('Pruning: {:.3f} global sparsity (t:{:.1f})'.format(result[-1], time.time() - t_start))
