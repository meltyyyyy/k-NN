import matplotlib.pyplot as plt
import mglearn.plots
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


def execute():
    X, y = mglearn.datasets.make_wave(n_samples=40)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    reg = KNeighborsRegressor(n_neighbors=3)
    reg.fit(X_train, y_train)

    print("Test set predictions:\n{}".format(reg.predict(X_test)))
    print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    line = np.linspace(-3, 3, 1000).reshape(-1, 1)
    for n_neighbors, ax in zip([1, 3, 9], axes):
        # predict in near 1, 3, 9
        reg = KNeighborsRegressor(n_neighbors=n_neighbors)
        reg.fit(X_train, y_train)
        ax.plot(line, reg.predict(line))
        ax.plot(X_train, y_train, "^", c=mglearn.cm2(0), markersize=8)
        ax.plot(X_test, y_test, "v", c=mglearn.cm2(1), markersize=8)

        ax.set_title("{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)))
        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")

    axes[0].legend(["Model predictions", "Training data/target", "Test data/target"], loc="best")
    fig.savefig("k_nr/knr_regression.png")
