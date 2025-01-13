
#Radial Basis Function Networks (RBFNs)
#It is a type of neural network that uses radial basis functions as activation functions.
#RBFNs have three Layers: Input Layer, Hidden Layer and Output Layer.




import numpy as np


def gaussian_rbf(x, center, spread):
    return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * spread ** 2))


class RBFN:
    def __init__(self, num_centers, spread):
        self.num_centers = num_centers
        self.spread = spread
        self.centers = None
        self.weights = None

    def fit(self, X, y):
        
        random_indices = np.random.choice(X.shape[0], self.num_centers, replace=False)
        self.centers = X[random_indices]
        
       
        G = np.zeros((X.shape[0], self.num_centers))
        for i, x in enumerate(X):
            for j, center in enumerate(self.centers):
                G[i, j] = gaussian_rbf(x, center, self.spread)
        
       
        self.weights = np.linalg.pinv(G).dot(y)
        
        print("Training completed.")
        print("Centers (RBF Neurons):", self.centers)
        print("Weights:", self.weights)
        print("-" * 50)

    def predict(self, X):
        
        G = np.zeros((X.shape[0], self.num_centers))
        for i, x in enumerate(X):
            for j, center in enumerate(self.centers):
                G[i, j] = gaussian_rbf(x, center, self.spread)
        
        
        return G.dot(self.weights)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        mse = np.mean((y - predictions) ** 2)
        print("Evaluation Results:")
        print(f"Mean Squared Error: {mse:.6f}")
        print("Predictions:", predictions.flatten())
        print("Actual:", y.flatten())
        print("-" * 50)
        return mse


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


num_centers = 2  
spread = 1.0     


rbfn = RBFN(num_centers=num_centers, spread=spread)
rbfn.fit(X, y)


print("Predictions on training data:")
predictions = rbfn.predict(X)
print(predictions)

rbfn.evaluate(X, y)
