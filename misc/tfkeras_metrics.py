import tensorflow as tf

y_true = [[0, 0, 1, 0, 0],
          [0, 1, 0, 0, 0],
          [0, 0, 0, 0, 1]]

y_pred = [[.01, .02, .9, .05, .02], # correct!          
          [.6, .3, .01, .01, .08], # wrong
          [.1, 0, .1, 0, .8]] # correct!



for metric in [tf.keras.metrics.Accuracy,
               tf.keras.metrics.BinaryAccuracy,
               tf.keras.metrics.CategoricalAccuracy,
               tf.keras.metrics.CategoricalCrossentropy,
               tf.keras.metrics.Precision,
               tf.keras.metrics.Recall]:
    m = metric()
    m.update_state(y_true=y_true,
                   y_pred=y_pred)
    score = m.result().numpy()
    print("%s - %.2f" % (m.name, score))

'''
accuracy - 0.13
binary_accuracy - 0.87
categorical_accuracy - 0.67
categorical_crossentropy - 0.51
precision - 0.67
recall - 0.67
'''
