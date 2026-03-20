import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
def generate_data(seed=0, n=50):
    np.random.seed(seed)
    distance = np.random.uniform(5, 40, n)
    load = np.random.uniform(10, 100, n)
    congestion = np.random.randint(0, 5, n)
    time = 1.8 * distance + 0.3 * load + 5.0 * congestion + 10 + np.random.normal(0, 5, n)
    X = np.column_stack([distance, load, congestion])
    y = time
    return X, y

# 특성 정규화 함수
def normalize_features(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# MSE 계산 함수
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent 함수 (Linear Regression)
def gradient_descent(X, y, alpha=0.1, n_iter=1000):
    X_norm, mu, sigma = normalize_features(X)
    n_samples, n_features = X_norm.shape
    w = np.zeros(n_features)
    b = 0.0
    losses = []
    for _ in range(n_iter):
        y_pred = X_norm @ w + b
        loss = mse_loss(y, y_pred)
        losses.append(loss)
        # Gradient 계산
        residual = y - y_pred
        grad_w = -2 / n_samples * (X_norm.T @ residual)
        grad_b = -2 / n_samples * np.sum(residual)
        # 파라미터 업데이트
        w -= alpha * grad_w
        b -= alpha * grad_b
    return w, b, losses, mu, sigma

# 1. 기본 gradient descent 실행 및 MSE 출력
def run_basic_gd():
    X, y = generate_data()
    w, b, losses, mu, sigma = gradient_descent(X, y, alpha=0.1, n_iter=1000)
    print(f"최종 MSE: {losses[-1]:.4f}")
    return X, y, w, b, losses, mu, sigma

# 2. Loss curve plot
def plot_loss_curve(losses, title="Loss Curve"):
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.title(title)
    plt.grid()
    plt.show()

# 3. Learning rate 실험 및 plot
def learning_rate_experiment(X, y, alphas, n_iter=500):
    plt.figure()
    for alpha in alphas:
        _, _, losses, _, _ = gradient_descent(X, y, alpha=alpha, n_iter=n_iter)
        plt.plot(losses, label=f"alpha={alpha}")
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.title('Learning Rate Sensitivity')
    plt.legend()
    plt.grid()
    plt.show()

# 4. L2 Regularization이 적용된 Gradient Descent
def gradient_descent_l2(X, y, alpha=0.1, n_iter=1000, lmbd=0.1):
    X_norm, mu, sigma = normalize_features(X)
    n_samples, n_features = X_norm.shape
    w = np.zeros(n_features)
    b = 0.0
    losses = []
    for _ in range(n_iter):
        y_pred = X_norm @ w + b
        loss = mse_loss(y, y_pred) + lmbd * np.sum(w ** 2)
        losses.append(loss)
        residual = y - y_pred
        grad_w = -2 / n_samples * (X_norm.T @ residual) + 2 * lmbd * w
        grad_b = -2 / n_samples * np.sum(residual)
        w -= alpha * grad_w
        b -= alpha * grad_b
    return w, b, losses, mu, sigma

# 5. Logistic Regression용 Gradient Descent
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y_true, y_pred):
    # y_true: 0 또는 1, y_pred: 확률
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def gradient_descent_logistic(X, y, alpha=0.1, n_iter=1000):
    X_norm, mu, sigma = normalize_features(X)
    n_samples, n_features = X_norm.shape
    w = np.zeros(n_features)
    b = 0.0
    losses = []
    for _ in range(n_iter):
        z = X_norm @ w + b
        y_pred = sigmoid(z)
        loss = cross_entropy_loss(y, y_pred)
        losses.append(loss)
        residual = y - y_pred
        grad_w = -1 / n_samples * (X_norm.T @ residual)
        grad_b = -1 / n_samples * np.sum(residual)
        w -= alpha * grad_w
        b -= alpha * grad_b
    return w, b, losses, mu, sigma

def run_all():
    # 1. 기본 gradient descent
    X, y = generate_data()
    w, b, losses, mu, sigma = gradient_descent(X, y, alpha=0.1, n_iter=1000)
    print(f"[Linear] 최종 MSE: {losses[-1]:.4f}")
    plot_loss_curve(losses, title="Linear Regression Loss Curve")

    # 2. Learning rate 실험
    alphas = [0.001, 0.01, 0.1, 0.5, 1.0]
    learning_rate_experiment(X, y, alphas, n_iter=500)

    # 3. L2 Regularization 실험
    for lmbd in [0.0, 0.01, 0.1, 1.0, 10.0]:
        w_l2, b_l2, losses_l2, _, _ = gradient_descent_l2(X, y, alpha=0.1, n_iter=1000, lmbd=lmbd)
        print(f"[L2={lmbd}] 최종 MSE+Reg: {losses_l2[-1]:.4f}, w={w_l2}")

    # 4. Logistic Regression
    # 이진 분류 레이블 생성
    y_class = (y > np.median(y)).astype(int)
    w_log, b_log, losses_log, _, _ = gradient_descent_logistic(X, y_class, alpha=0.1, n_iter=1000)
    print(f"[Logistic] 최종 Cross-Entropy: {losses_log[-1]:.4f}")
    plot_loss_curve(losses_log, title="Logistic Regression Loss Curve")
    # 분류 정확도
    X_norm, mu, sigma = normalize_features(X)
    y_pred_prob = sigmoid(X_norm @ w_log + b_log)
    y_pred_label = (y_pred_prob > 0.5).astype(int)
    acc = np.mean(y_pred_label == y_class)
    print(f"[Logistic] Training Accuracy: {acc:.4f}")

if __name__ == "__main__":
    run_all()
