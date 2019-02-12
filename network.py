import tensorflow as tf


def build_network(non_spatial, screen, minimap, screen_dimensions, minimap_dimensions):
    non_spatial_fc = tf.contrib.layers.fully_connected(
        inputs=tf.contrib.layers.flatten(non_spatial),
        num_outputs=256,
        activation_fn=tf.tanh,
        scope='non_spatial_fc'
    )

    screen_conv1 = tf.contrib.layers.conv2d(
        inputs=tf.transpose(screen, []),
        num_outputs=16,
        kernel_size=5,
        strides=1,
        scope='screen_conv1'
    )
    screen_conv2 = tf.contrib.layers.conv2d(
        inputs=screen_conv1,
        num_outputs=32,
        kernel_size=3,
        strides=1,
        scope='screen_conv2'
    )

    minimap_conv1 = tf.contrib.layers.conv2d(
        inputs=tf.transpose(minimap, []),
        num_outputs=16,
        kernel_size=5,
        strides=1,
        scope='minimap_conv1'
    )
    minimap_conv2 = tf.contrib.layers.conv2d(
        inputs=minimap_conv1,
        num_outputs=32,
        kernel_size=3,
        strides=1,
        scope='minimap_conv2'
    )

    state_representation = tf.concat([
        tf.contrib.layers.flatten(screen_conv2),
        tf.contrib.layers.flatten(minimap_conv2),
        non_spatial_fc
    ], axis=1)

    spatial_action = None
    non_spatial_action = None
    value = None

    return non_spatial_action, spatial_action, value
