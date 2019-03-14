import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
n_feature = 2   #гиперпараметры для тестов
n_class = 2
n_iter = 100
eps=1e-8
minibatch_size = 100
n_experiment = 3
learning_rate = float(input("Введите learning rate:"))
gamma=.9
def shuffle(X, y):
    Z = np.column_stack((X, y))
    np.random.shuffle(Z)
    return Z[:, :-1], Z[:, -1]
def make_network(n_hidden=100):
    model = dict(W1=np.random.randn(n_feature, n_hidden),W2=np.random.randn(n_hidden, n_class))
    return model
def softmax(x):
    return np.exp(x) / np.exp(x).sum()
def forward(x, model):
    h = x @ model['W1']
    h[h < 0] = 0
    prob = softmax(h @ model['W2'])
    return h, prob
def backward(model, xs, hs, errs):
    dW2 = hs.T @ errs
    dh = errs @ model['W2'].T
    dh[hs <= 0] = 0
    dW1 = xs.T @ dh
    return dict(W1=dW1, W2=dW2)
def get_minibatch(X, y, minibatch_size):
    minibatches = []
    X,y=shuffle(X,y)
    for i in range(0, X.shape[0], minibatch_size):
        X_mini = X[i:i + minibatch_size]
        y_mini = y[i:i + minibatch_size]
        minibatches.append((X_mini, y_mini))
    return minibatches
def sgd(model, X_train, y_train, minibatch_size):
    for iter in range(n_iter):
        X_train,y_train=shuffle(X_train,y_train)  #рандомим
        for i in range(0, X_train.shape[0], minibatch_size):
            X_train_mini = X_train[i:i + minibatch_size]  # х и у для данного минипакета
            y_train_mini = y_train[i:i + minibatch_size]
            model = sgd_step(model, X_train_mini, y_train_mini)
    return model
def sgd_step(model, X_train, y_train):
    grad = get_minibatch_grad(model, X_train, y_train)
    model = model.copy()
    for layer in grad: #обновление параметров w1 w2
        model[layer] += 1e-4 * grad[layer]   # learning rate=1e-4
    return model
def get_minibatch_grad(model, X_train, y_train):
    xs, hs, errs = [], [], []
    for x, cls_idx in zip(X_train, y_train):
        h, y_pred = forward(x, model)
        y_true = np.zeros(n_class)
        y_true[int(cls_idx)] = 1.
        err = y_true - y_pred #расчет выходного градиента
        xs.append(x)  #входящая информация
        hs.append(h)  #промежуточное состояние
        errs.append(err)  #Градиент выходного слоя
    return backward(model, np.array(xs), np.array(hs), np.array(errs)) # Метод обратного распространения ошибки с информацией из минипакетов
def momentum(model, X_train, y_train, minibatch_size):
    velocity = {k: np.zeros_like(v) for k, v in model.items()}
    minibatches = get_minibatch(X_train, y_train, minibatch_size)
    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]
        grad = get_minibatch_grad(model, X_mini, y_mini)
        for layer in grad:
            velocity[layer] = gamma * velocity[layer] + learning_rate * grad[layer]
            model[layer] += velocity[layer]
    return model
def nesterov(model, X_train, y_train, minibatch_size):
    velocity = {k: np.zeros_like(v) for k, v in model.items()}
    minibatches = get_minibatch(X_train, y_train, minibatch_size)
    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]
        model_ahead = {k: v + gamma * velocity[k] for k, v in model.items()}
        grad = get_minibatch_grad(model_ahead, X_mini, y_mini)
        for layer in grad:
            velocity[layer] = gamma * velocity[layer] + learning_rate * grad[layer]
            model[layer] += velocity[layer]
    return model
def adagrad(model, X_train, y_train, minibatch_size):
    cache = {k: np.zeros_like(v) for k, v in model.items()}
    minibatches = get_minibatch(X_train, y_train, minibatch_size)
    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]
        grad = get_minibatch_grad(model, X_mini, y_mini)
        for k in grad:
            cache[k] += grad[k]**2
            model[k] += learning_rate * grad[k] / (np.sqrt(cache[k]) + eps)
    return model
def rmsprop(model, X_train, y_train, minibatch_size):
    cache = {k: np.zeros_like(v) for k, v in model.items()}
    minibatches = get_minibatch(X_train, y_train, minibatch_size)
    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]
        grad = get_minibatch_grad(model, X_mini, y_mini)
        for k in grad:
            cache[k] = gamma * cache[k] + (1 - gamma) * (grad[k]**2)
            model[k] += learning_rate * grad[k] / (np.sqrt(cache[k]) + eps)
    return model
def adam(model, X_train, y_train, minibatch_size):
    M = {k: np.zeros_like(v) for k, v in model.items()}
    R = {k: np.zeros_like(v) for k, v in model.items()}
    beta1 = .9
    beta2 = .999
    minibatches = get_minibatch(X_train, y_train, minibatch_size)
    for iter in range(1, n_iter + 1):
        t = iter
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]
        grad = get_minibatch_grad(model, X_mini, y_mini)
        for k in grad:
            M[k] = beta1 * M[k] + (1. - beta1) * grad[k]
            R[k] = beta2 * R[k] + (1. - beta2) * grad[k]**2
            m_k_hat = M[k] / (1. - beta1**(t))
            r_k_hat = R[k] / (1. - beta2**(t))
            model[k] += learning_rate * m_k_hat / (np.sqrt(r_k_hat) + eps)
    return model
X, y = make_moons(n_samples=5000, random_state=42, noise=0.1)                  #Берем стандартный датасет и рандомно спилитим на тестовую и тренировочную выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
accs = np.zeros(n_experiment)  # Для сбора информации о точности
alg_choose=int(input('Выберите алгоритм который вы хотите использовать, и напишите его номер 1.SGD 2.SGD_step 3.Nesterov 4.Momentum 5.AdaGrad 6.RMSProp 7.Adam'))
for k in range(n_experiment):
    model = make_network()# Пересоздание модели
    if alg_choose==1:
        model = sgd(model, X_train, y_train,minibatch_size) # тренируем модель
    if alg_choose==2:
        model = sgd_step(model, X_train, y_train)
    if alg_choose==3:
        model = nesterov(model, X_train, y_train,minibatch_size)
    if alg_choose==4:
        model = momentum(model, X_train, y_train,minibatch_size)
    if alg_choose==5:
        model = adagrad(model, X_train, y_train,minibatch_size)
    if alg_choose==6:
        model = rmsprop(model, X_train, y_train,minibatch_size)
    if alg_choose==7:
        model = adam(model, X_train, y_train,minibatch_size)
    y_pred = np.zeros_like(y_test)
    for i, x in enumerate(X_test):
        _, prob = forward(x, model) #предсказываем
        y = np.argmax(prob)
        y_pred[i] = y
    accs[k] = (y_pred == y_test).sum() / y_test.size #усредняем
print('Mean accuracy: {}'.format(accs.mean()))