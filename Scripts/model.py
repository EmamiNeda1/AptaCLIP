import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def cosine_triplet_loss(margin=0.3):
    def loss(y_true, y_pred):
        dim = y_pred.shape[1] // 3

        anchor = tf.math.l2_normalize(y_pred[:, :dim], axis=1)
        positive = tf.math.l2_normalize(y_pred[:, dim:2 * dim], axis=1)
        negative = tf.math.l2_normalize(y_pred[:, 2 * dim:], axis=1)

        pos_dist = 1 - tf.reduce_sum(anchor * positive, axis=1)
        neg_dist = 1 - tf.reduce_sum(anchor * negative, axis=1)

        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))

    return loss


def protein_projection_head(x):
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128)(x)

    return layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1)
    )(x)


def aptamer_projection_head(input_dim):
    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128)(x)

    outputs = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1)
    )(x)

    return keras.Model(inputs, outputs, name="aptamer_projection")


def build_aptaclip_model(protein_dim, aptamer_dim):
    protein_input = layers.Input(shape=(protein_dim,), name="protein_input")
    positive_input = layers.Input(shape=(aptamer_dim,), name="positive_aptamer_input")
    negative_input = layers.Input(shape=(aptamer_dim,), name="negative_aptamer_input")
    rna_flag_input = layers.Input(shape=(1,), name="rna_flag")

    aptamer_projector = aptamer_projection_head(aptamer_dim)

    protein_embedding = protein_projection_head(protein_input)
    positive_embedding = aptamer_projector(positive_input)
    return model
