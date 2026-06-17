import numpy as np


class CustomKMeansPlusPlus:
    def __init__(
        self,
        n_clusters=3,
        max_iter=300,
        tol=1e-4,
        n_init=10,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        if self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters cannot be greater than number of samples")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self.n_init < 1:
            raise ValueError("n_init must be at least 1")

        rng = np.random.default_rng(self.random_state)
        best_result = None

        for _ in range(self.n_init):
            centers, labels, inertia, n_iter = self._fit_once(X, rng)
            if best_result is None or inertia < best_result["inertia"]:
                best_result = {
                    "centers": centers,
                    "labels": labels,
                    "inertia": inertia,
                    "n_iter": n_iter,
                }

        self.cluster_centers_ = best_result["centers"]
        self.labels_ = best_result["labels"]
        self.inertia_ = best_result["inertia"]
        self.n_iter_ = best_result["n_iter"]

        return self

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("The model is not fitted yet")

        X = np.asarray(X, dtype=float)
        distances = self._squared_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_

    @staticmethod
    def _squared_distances(X, centers):
        return ((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2)

    def _fit_once(self, X, rng):
        centers = self._init_centers(X, rng)

        for iteration in range(1, self.max_iter + 1):
            distances = self._squared_distances(X, centers)
            labels = np.argmin(distances, axis=1)

            new_centers = centers.copy()
            for cluster_id in range(self.n_clusters):
                cluster_points = X[labels == cluster_id]
                if len(cluster_points) == 0:
                    new_centers[cluster_id] = X[rng.integers(0, X.shape[0])]
                else:
                    new_centers[cluster_id] = cluster_points.mean(axis=0)

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers

            if shift < self.tol:
                break

        distances = self._squared_distances(X, centers)
        labels = np.argmin(distances, axis=1)
        inertia = distances[np.arange(X.shape[0]), labels].sum()

        return centers, labels, inertia, iteration

    def _init_centers(self, X, rng):
        centers = np.empty((self.n_clusters, X.shape[1]), dtype=float)
        first_index = rng.integers(0, X.shape[0])
        centers[0] = X[first_index]

        closest_distances = self._squared_distances(X, centers[:1]).ravel()

        for center_id in range(1, self.n_clusters):
            total_distance = closest_distances.sum()

            if total_distance == 0:
                next_index = rng.integers(0, X.shape[0])
            else:
                probabilities = closest_distances / total_distance
                next_index = rng.choice(X.shape[0], p=probabilities)

            centers[center_id] = X[next_index]
            new_distances = self._squared_distances(
                X, centers[center_id : center_id + 1]
            ).ravel()
            closest_distances = np.minimum(closest_distances, new_distances)

        return centers
