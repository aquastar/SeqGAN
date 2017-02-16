import tensorflow as tf

import utils
import vgg16

img = utils.load_image('./test.jpg').reshape((1, 224, 224, 3))

tf.reset_default_graph()
with tf.Session() as sess:
    images = tf.placeholder("float", [1, 224, 224, 3])
    feed_dict = {images: img}

    vgg = vgg16.Vgg16()
    vgg.build(images)

    prob = sess.run(vgg.fc8, feed_dict=feed_dict)
    print prob
