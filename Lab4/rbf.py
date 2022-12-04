import numpy as np

class RBF:
    def __init__(self, k=24, num_classes=10, std_from_clusters=True) -> None:
        self.k = k 
        self.num_classes = num_classes
        self.std_from_clusters = std_from_clusters
        self.RBF_M = None


    def get_dist(self, x_samp, center):
        return np.sqrt(np.sum((x_samp-center)**2))

    def kMeans(self, X, max_iters=1000):
        
        replace = False
        if self.k >= 5000:
            replace = True
        centroids = X[np.random.choice(range(len(X)), self.k, replace=replace)]

        converged = False
        
        current_iter = 0

        while (not converged) and (current_iter < max_iters):

            cluster_list = [[] for _ in range(len(centroids))]

            for x in X:
                distances_list = []
                for c in centroids:
                    distances_list.append(self.get_dist(c, x))
                cluster_list[int(np.argmin(distances_list))].append(x)

            cluster_list = list((filter(None, cluster_list)))

            prev_centroids = centroids.copy()

            centroids = []

            for j in range(len(cluster_list)):
                centroids.append(np.mean(cluster_list[j], axis=0))

            pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids))

            # print('K-MEANS: ', int(pattern))

            converged = (pattern == 0)

            current_iter += 1

        return np.array(centroids), [np.std(x) for x in cluster_list]

    def rbf(self, x_samp, center, std):
        d = self.get_dist(x_samp, center)
        return 1/np.exp(-d / std ** 2)  

    def rbf_matrix(self, X, centers_list, std_list):
        self.RBF_M = []
        for x_s in X:
            self.RBF_M.append([self.rbf(x_s, c, std) for (c, std) in zip(centers_list, std_list)])

        self.RBF_M = np.array(self.RBF_M)

    def prediction(self):
        pred = self.RBF_M @ self.w

        return np.array([np.argmax(p) for p in pred])

    def validate(self, preds, res):
        acc = 0

        for (p, r) in zip(preds, res):
            if p == np.argmax(r):
                acc+=1

        return acc

    def train(self, input, output):
        self.centroids, self.std_list = self.kMeans(input)

        if not self.std_from_clusters:
            dist_max = np.max([self.get_dist(c1, c2) for c1 in self.centroids for c2 in self.centroids])
            self.std_list = np.repeat(dist_max / np.sqrt(2 * self.k), self.k)

        self.rbf_matrix(input, self.centroids, self.std_list)

        RBF_M_T = self.RBF_M.T
        # print(RBF_M_T)
        self.w = np.linalg.pinv(RBF_M_T @ self.RBF_M) @ RBF_M_T @ output 

        preds = self.prediction()
        
        acc = self.validate(preds, output)
        return acc / len(output)

    def test(self, input, output):
        self.rbf_matrix(input, self.centroids, self.std_list)
        
        RBF_M_T = self.RBF_M.T
        self.w = np.linalg.pinv(RBF_M_T @ self.RBF_M) @ RBF_M_T @ output
        
        preds = self.prediction()
        
        acc = self.validate(preds, output)
        return acc / len(output)