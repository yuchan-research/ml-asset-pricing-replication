from sklearn.ensemble import RandomForestRegressor

def train_rf(X, y, n_estimators=200, max_depth=6, random_state=42):
    model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,          # shallow
    min_samples_leaf=5,   # 작게
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

    model.fit(X, y)
    return model
