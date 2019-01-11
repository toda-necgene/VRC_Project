
class Gan():
	pass

import os
import tensorflow as tf

if __name__ = '__main__':
	use_tpu = True

	iterations = 3
	num_shards = 8
	model_dir = 'tpu_models'

	os.makedirs(model_dir, exist_ok=True)

	tpu_run_config = tf.contrib.tpu.RunConfig(
                        model_dir=model_dir,
                        session_config=tf.ConfigProto(
                            allow_soft_placement=True,
									 log_device_placement=True),
                        tpu_config=tf.contrib.tpu.TPUConfig(iterations, num_shards)

	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	if use_tpu:
  		optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
		  
	estimator = tf.contrib.tpu.TPUEstimator(
					    model_fn=model,
						 config=tpu_run_config,
						 use_tpu=use_tpu,
						 params={'optimizer':optimizer})