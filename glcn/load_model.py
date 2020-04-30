import tensorflow as tf
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.ckpt.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
  # var_list = [v.name for v in tf.all_variables()]
  var_list = [v.name for v in tf.contrib.framework.get_variables_to_restore()]
  print(var_list)
  print(sess.run(var_list))
