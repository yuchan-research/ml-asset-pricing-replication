from sklearn.linear_model import ElasticNet

def train_enet(X, y, alpha=0.0001, l1_ratio=0.3):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    model.fit(X, y)
    return model
