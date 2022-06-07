import warnings
from itertools import product

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action="ignore", category=FutureWarning)


CMAP_BOLD = ["#ff7603", "#bf5cca", "#0f7075"]


def plot_data(data, y):

    # Set context
    sns.set_context("talk")

    X_data = data.values.copy()

    # Put the result into a color plot
    sns.set_style("dark")
    plt.figure(figsize=(8, 8))

    # Plot also the training points
    x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
    y_min, y_max = X_data[:, 1].min() - 1, X_data[:, 1].max() + 1
    sns.scatterplot(
        x=X_data[:, 0],
        y=X_data[:, 1],
        hue=y.map({0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}),
        palette=CMAP_BOLD,
        alpha=1.0,
        edgecolor="black",
        marker="o",
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.title("Penguins: Bill length and depth")
    sns.set_style("white")


def plot_decision_boundaries(data, y, clf, species, scaled=False, N=1000, title=""):

    # Set context
    sns.set_context("talk")

    X_data = data.values.copy()
    X_tr = data.values.copy()

    # Create color maps
    cmap_light = ListedColormap(CMAP_BOLD)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
    y_min, y_max = X_data[:, 1].min() - 1, X_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / N), np.arange(y_min, y_max, (x_max - x_min) / N))

    scaler = StandardScaler()
    if scaled:
        X_tr = scaler.fit_transform(X_tr)
    clf = clf.fit(X_tr, y)
    if scaled:
        grid = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    else:
        grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 8))
    plt.contourf(xx, yy, Z, cmap=cmap_light, levels=len(species) - 1, alpha=0.5)

    # Plot also the training points
    correct_idx = clf.predict(X_tr) == y
    sns.scatterplot(
        x=X_data[correct_idx, 0],
        y=X_data[correct_idx, 1],
        hue=y[correct_idx].map({0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}),
        palette=list(np.array(CMAP_BOLD)[np.unique(y[correct_idx])]),
        alpha=1.0,
        edgecolor="black",
        marker="o",
    )
    if hasattr(clf, "support_vectors_"):
        plt.scatter(*list(scaler.inverse_transform(clf.support_vectors_).T), c="k", marker=".", s=20)
        title += " %d support vectors" % len(clf.support_vectors_)
    sns.scatterplot(
        x=X_data[~correct_idx, 0],
        y=X_data[~correct_idx, 1],
        hue=y[~correct_idx].map({0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}),
        palette=list(np.array(CMAP_BOLD)[np.unique(y[~correct_idx])]),
        alpha=1.0,
        edgecolor="white",
        marker="X",
        legend=False,
    )
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.title(title + "\nAccuracy = {:.1f}%".format(100 * np.mean(clf.predict(X_tr) == y)))


def plot_grid_search(param_grid, cv_results):

    # Set context
    sns.set_context("notebook")

    # Establish combinations of different hyperparameters, that isn't the one
    # we want to plot on the x-axis
    combinations = list(product(param_grid["prep__num__scaler"], param_grid["prep__num__pca"]))

    # Creates a figure with multiple subplot
    fig, axs = plt.subplots(
        len(param_grid["prep__num__scaler"]), len(param_grid["prep__num__pca"]), figsize=(12, 6), sharex=True
    )

    # Extract useful information about max performance
    max_score = cv_results["mean_test_score"].max()
    c_values = cv_results["regressor__alpha"]

    # Loop through the subplots and populate them
    for i, (s, p) in enumerate(combinations):

        # Select subplot relevant grid search results
        mask = np.logical_and(
            cv_results["num__pca"].astype("str") == str(p),
            cv_results["num__scaler"].astype("str") == str(s),
        )
        df_cv = cv_results[mask].sort_values("regressor__alpha").set_index("regressor__alpha")

        # Select relevant axis
        ax = axs.flatten()[i]

        # Plot train and test curves
        df_cv[["mean_train_score", "mean_test_score"]].plot(logx=True, title=f"{s} | {p}", ax=ax)
        ax.fill_between(
            df_cv.index,
            df_cv["mean_train_score"] - df_cv["std_train_score"],
            df_cv["mean_train_score"] + df_cv["std_train_score"],
            alpha=0.3,
        )
        ax.fill_between(
            df_cv.index,
            df_cv["mean_test_score"] - df_cv["std_test_score"],
            df_cv["mean_test_score"] + df_cv["std_test_score"],
            alpha=0.3,
        )

        # Plot best performance metric as dotted line
        ax.hlines(max_score, c_values.min(), c_values.max(), color="gray", linestyles="dotted")

    # Limit y-axis
    plt.tight_layout()
    plt.show()


def plot_convnet_history(history):

    # Store history in a dataframe
    df_history = pd.DataFrame(history)

    # Visualize training history
    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    df_history.iloc[:, df_history.columns.str.contains("loss")].plot(title="Loss during training", ax=axs[0])
    df_history.iloc[:, df_history.columns.str.contains("accuracy")].plot(title="Accuracy during training", ax=axs[1])
    axs[0].set_xlabel("Epoch [#]")
    axs[1].set_xlabel("Epoch [#]")
    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("Accuracy")
    plt.show()


def plot_convnet_weights(model):

    # Extract first hidden convolutional layers
    conv_layer = model.layers[1]

    # Transform the layer weights to a numpy array
    kernels = conv_layer.weights[0].numpy()

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(4, 4))

    # Remove gaps between suplots
    plt.subplots_adjust(wspace=0, hspace=0)

    # Plot the 64 kernels from the first convolutional layer
    for i, axis in enumerate(axes.flatten()):
        # Get i-th kernel (shape: 5x5x3)
        kernel = kernels[:, :, :, i].copy()

        # Rescale values between 0 and 1
        kernel -= kernel.min()  # Rescale between 0 and max
        kernel /= kernel.max()  # Rescale between 0 and 1

        # Plot kernel with imshow()
        axis.imshow(kernel)
        axis.get_xaxis().set_visible(False)  # disable x-axis
        axis.get_yaxis().set_visible(False)  # disable y-axis

    plt.show()


def plot_convnet_activation_map(img, model):

    # Extract first hidden convolutional layers
    conv_layer = model.layers[1]

    # Pick a sample image and get 1st conv. activations
    activation_maps = conv_layer(np.expand_dims(img, axis=0))

    # Create figure with subplots
    fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))

    # Plot the activation maps of the 1st conv. layer for the sample image
    for i, axis in enumerate(axes.flatten()):
        # Get activation map of the i-th filter
        activation = activation_maps[0, :, :, i]

        # Plot it with imshow()
        axis.set_title("map {}".format(i + 1))
        axis.imshow(activation, cmap="gray")
        axis.get_xaxis().set_visible(False)  # disable x-axis
        axis.get_yaxis().set_visible(False)  # disable y-axis

    plt.tight_layout()
    plt.show()


def plot_2d_grid(X_data, y_label, labels, d1=0, d2=1):

    # Set context
    sns.set_context("talk")

    embedding = X_data[:, [d1, d2]]

    fig, ax = plt.subplots(1, figsize=(16, 7))
    ax = plt.subplot(aspect="equal")

    plt.scatter(*embedding.T, s=0.4, c=y_label, cmap="Spectral", alpha=0.8)
    plt.xticks([])
    plt.xlabel("Dimension 1")
    plt.yticks([])
    plt.ylabel("Dimension 2")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    cbar = plt.colorbar(boundaries=np.arange(11) - 0.5)
    cbar.set_ticks(np.arange(10))
    cbar.set_ticklabels(labels)

    for i, t in enumerate(labels):
        # Position of each label
        xtext, ytext = np.median(embedding[y_label == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(t), fontsize=24)
        txt.set_path_effects([path_effects.Stroke(linewidth=5, foreground="w"), path_effects.Normal()])

    plt.tight_layout()


def plot_pca_decomposition(X_data, y_label, n_dim=100):

    # Cloths idx
    idx_numbers = [np.where(y_label == i)[0][3] for i in range(10)]
    idx_numbers

    fig, ax = plt.subplots(nrows=2, figsize=(16, 4))

    numbers = np.reshape(X_data, (-1, 28, 28)).astype("uint8")
    ax[0].imshow(np.concatenate(numbers[idx_numbers], axis=1), cmap="binary")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_ylabel("784-D (original)")

    pca_transform = PCA(n_dim)
    X_embed = pca_transform.fit_transform(X_data / 255.0)

    # Cloths
    idx_numbers = [np.where(y_label == i)[0][3] for i in range(10)]

    numbers = np.reshape(
        (np.clip(pca_transform.inverse_transform(X_embed[idx_numbers]), 0, 1)) * 255, (-1, 28, 28)
    ).astype("uint8")

    ax[1].imshow(np.concatenate(numbers, axis=1), cmap="binary")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_ylabel("{}-D".format(n_dim))

    plt.tight_layout()
    plt.show()


def get_variational_autoencoder(n_dim=2):

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    class VAE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
                )
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                total_loss = reconstruction_loss + kl_loss
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }


    class Sampling(layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon


    encoder_inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu")(x)
    z_mean = layers.Dense(n_dim, name="z_mean")(x)
    z_log_var = layers.Dense(n_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(n_dim,))
    x = layers.Dense(32, activation="relu")(latent_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(7 * 7 * 64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())

    return vae


def plot_rgb_space(X, nth=1):

    X_colors = X[::nth, :]

    # Plot color values in 3D space
    fig = plt.figure(figsize=(16, 5))

    # Loop through 3 different views
    for i, view in enumerate([[-45, 10], [40, 80], [60, 10]]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(X_colors[:, 0], X_colors[:, 1], X_colors[:, 2], facecolors=X_colors, s=2)
        ax.set_xlabel("R")
        ax.set_ylabel("G")
        ax.set_zlabel("B")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        ax.view_init(azim=view[0], elev=view[1], vertical_axis="z")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
    plt.suptitle("Colors in RGB space", fontsize=20)
    plt.show()


def create_synthethic_dataset(n_points_per_cluster=250):

    # Generate sample data
    np.random.seed(0)
    C1 = [-5, -2] + 0.8 * np.random.randn(n_points_per_cluster, 2)
    C2 = [4, -1] + 0.1 * np.random.randn(n_points_per_cluster, 2)
    C3 = [1, -2] + 0.2 * np.random.randn(n_points_per_cluster, 2)
    C4 = [-2, 3] + 0.3 * np.random.randn(n_points_per_cluster, 2)
    C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
    C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
    X = np.vstack((C1, C2, C3, C4, C5, C6))
    return X


def plot_clustering(X, y, labels, title=""):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(8, 6))
    for digit in range(10):
        if np.all(y == labels):
            c = "gray"
        else:
            c = plt.cm.tab10(labels[y == digit] / 10)
        plt.scatter(
            *X[y == digit].T,
            marker=f"${digit}$",
            s=50,
            c=c,
            alpha=0.5,
        )
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.title(title)


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
