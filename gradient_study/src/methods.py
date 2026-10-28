"""MSE optimizers and lecture statistics. No estimator from sklearn is used."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit
from scipy import stats

METHODS = ('VGD', 'SGD', 'SAG', 'Momentum', 'Adam')


def mse(y, prediction):
    return float(np.mean((np.asarray(y) - prediction) ** 2))


def r2_score_manual(y, prediction):
    y = np.asarray(y, dtype=float)
    total = np.sum((y - y.mean()) ** 2)
    if total <= 0:
        raise ValueError('R² не определён для постоянной целевой переменной.')
    return float(1 - np.sum((y - prediction) ** 2) / total)


def learning_rate(k, initial, schedule):
    if schedule not in ('constant', 'time_decay') or initial <= 0 or k < 0:
        raise ValueError('Нужны k>=0, положительный шаг и известное расписание.')
    # Lecture 2: s0=1, p=0.5; first update uses k=0.
    return initial if schedule == 'constant' else initial / np.sqrt(1 + k)


@njit(cache=True)
def _kernel(X, y, method, initial, decay, max_iter, tolerance, batch_size,
            seed, check_every, divergence_limit):
    """Compiled loops only accelerate the explicit lecture update equations."""
    n, d = X.shape
    np.random.seed(seed)
    w = np.zeros(d)
    momentum = np.zeros(d)
    first = np.zeros(d)
    second = np.zeros(d)
    memory = np.zeros((n, d))
    average = np.zeros(d)
    gram = 2.0 * (X.T @ X) / n
    rhs = 2.0 * (X.T @ y) / n
    gradient = np.zeros(d)
    # Stored losses use standardized y. Conversion to original units is external.
    history = np.empty((max_iter // check_every + 2, 3))
    initial_loss = np.mean(y ** 2)
    history[0] = (0., initial_loss, np.sqrt(np.sum(rhs ** 2)))
    used = 1
    status = 1  # 0 converged, 1 budget exhausted, 2 diverged
    iterations = 0
    beta1_power, beta2_power = 1.0, 1.0
    for k in range(max_iter):
        eta = initial / np.sqrt(1.0 + k) if decay else initial
        gradient[:] = 0.
        if method == 1:
            # Random mini-batch, sampled independently with replacement.
            for _ in range(min(batch_size, n)):
                i = np.random.randint(n)
                residual = np.dot(X[i], w) - y[i]
                for j in range(d):
                    gradient[j] += 2.0 * residual * X[i, j] / min(batch_size, n)
        elif method == 2:
            i = np.random.randint(n)
            residual = np.dot(X[i], w) - y[i]
            for j in range(d):
                new_gradient = 2.0 * residual * X[i, j]
                average[j] += (new_gradient - memory[i, j]) / n
                memory[i, j] = new_gradient
                gradient[j] = average[j]
        else:
            for j in range(d):
                gradient[j] = -rhs[j]
                for h in range(d):
                    gradient[j] += gram[j, h] * w[h]
        beta1_power *= .9
        beta2_power *= .999
        for j in range(d):
            if method == 3:
                # Lecture momentum: h[k+1] = .9*h[k] + eta[k]*g[k].
                momentum[j] = .9 * momentum[j] + eta * gradient[j]
                w[j] -= momentum[j]
            elif method == 4:
                first[j] = .9 * first[j] + .1 * gradient[j]
                second[j] = .999 * second[j] + .001 * gradient[j] ** 2
                corrected_first = first[j] / (1. - beta1_power)
                corrected_second = second[j] / (1. - beta2_power)
                w[j] -= eta * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            else:
                w[j] -= eta * gradient[j]
        iterations = k + 1
        if not np.all(np.isfinite(w)) or np.max(np.abs(w)) > divergence_limit:
            status = 2
            break
        if iterations % check_every == 0 or iterations == max_iter:
            full_gradient = gram @ w - rhs
            # Exact MSE quadratic; avoids rescanning all rows at each check.
            loss = .5 * np.dot(w, gram @ w) - np.dot(rhs, w) + initial_loss
            norm = np.sqrt(np.sum(full_gradient ** 2))
            history[used] = (float(iterations), loss, norm)
            used += 1
            if not np.isfinite(loss) or loss > divergence_limit:
                status = 2
                break
            # Tiny learning rate alone cannot trigger convergence.
            if norm <= tolerance:
                status = 0
                break
    return w, iterations, status, history[:used]


@dataclass
class FitResult:
    weights: np.ndarray
    iterations: int
    status: str
    history: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float

    def predict(self, X):
        if self.status == 'diverged':
            raise ValueError('Нельзя использовать разошедшуюся модель.')
        design = np.column_stack([np.ones(len(X)), (np.asarray(X) - self.x_mean) / self.x_scale])
        return (design @ self.weights) * self.y_scale + self.y_mean

    def coefficients(self):
        slopes = self.weights[1:] * self.y_scale / self.x_scale
        intercept = self.y_mean + self.weights[0] * self.y_scale - slopes @ self.x_mean
        return np.r_[intercept, slopes]


def fit_gd(X, y, method='VGD', initial=.1, schedule='constant', max_iter=10000,
           tolerance=1e-7, batch_size=32, seed=42, check_every=50,
           divergence_limit=1e12):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) < 2:
        raise ValueError('X должна быть матрицей; y — вектором той же длины >=2.')
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Пропуски и бесконечные значения не допускаются.')
    if method not in METHODS or max_iter < 1 or check_every < 1 or batch_size < 1:
        raise ValueError('Некорректный метод или бюджет обучения.')
    learning_rate(0, initial, schedule)
    x_mean, x_scale = X.mean(axis=0), X.std(axis=0)
    x_scale[x_scale == 0] = 1.
    y_mean, y_scale = float(y.mean()), float(y.std())
    if y_scale == 0:
        raise ValueError('Целевая переменная постоянна.')
    design = np.ascontiguousarray(np.column_stack([np.ones(len(X)), (X-x_mean)/x_scale]))
    normalized_y = np.ascontiguousarray((y-y_mean)/y_scale)
    w, iterations, status, history = _kernel(
        design, normalized_y, METHODS.index(method), float(initial), schedule == 'time_decay',
        int(max_iter), float(tolerance), int(batch_size), int(seed), int(check_every), float(divergence_limit))
    return FitResult(w, iterations, ('converged', 'max_iter', 'diverged')[status],
                     history, x_mean, x_scale, y_mean, y_scale)


def ols_statistics(X, y, alpha=.05, parameters=None):
    """Lecture 3: intercept, C=s²(X'X)^-1, two-sided t and overall F.

    Solve in standardized coordinates for numeric stability. An independently
    trained GD solution can supply parameters after convergence is verified.
    """
    names = list(X.columns)
    values, y = X.to_numpy(dtype=float), np.asarray(y, dtype=float)
    n, k = values.shape
    mean, scale = values.mean(axis=0), values.std(axis=0)
    if n <= k+1 or np.any(scale == 0):
        raise ValueError('Недостаточно наблюдений или есть постоянный признак.')
    design = np.column_stack([np.ones(n), (values-mean)/scale])
    if np.linalg.matrix_rank(design) < k+1:
        raise ValueError('Матрица признаков имеет неполный ранг.')
    standardized, *_ = np.linalg.lstsq(design, y, rcond=None)
    transform = np.eye(k+1)
    transform[0, 1:] = -mean/scale
    transform[1:, 1:] = np.diag(1/scale)
    reference = transform @ standardized
    params = reference if parameters is None else np.asarray(parameters)
    prediction = params[0] + values @ params[1:]
    degrees = n-k-1
    sse = np.sum((y-prediction)**2)
    # QR avoids forming an ill-conditioned inverse in original units.
    _, triangular = np.linalg.qr(design, mode='reduced')
    inverse_r = np.linalg.solve(triangular, np.eye(k+1))
    covariance = transform @ (inverse_r @ inverse_r.T) @ transform.T * (sse/degrees)
    standard_errors = np.sqrt(np.diag(covariance))
    t_values = params / standard_errors
    p_values = 2 * stats.t.sf(np.abs(t_values), degrees)
    critical_t = stats.t.ppf(1-alpha/2, degrees)
    table = pd.DataFrame({
        'Признак': ['const']+names, 'Коэффициент': params, 'Стандартная ошибка': standard_errors,
        't': t_values, 'p_value': p_values, 'Значим': p_values < alpha,
        'CI_low': params-critical_t*standard_errors, 'CI_high': params+critical_t*standard_errors})
    r2 = r2_score_manual(y, prediction)
    f_value = (r2/k) / ((1-r2)/degrees)
    model = {'n': n, 'k': k, 'df_resid': degrees, 'R2_train': r2,
             'F': f_value, 'F_critical': float(stats.f.ppf(1-alpha, k, degrees)),
             'p_value': float(stats.f.sf(f_value, k, degrees)),
             't_critical': float(critical_t), 'SSE': float(sse),
             'Модель значима': bool(f_value > stats.f.ppf(1-alpha, k, degrees))}
    return table, model, reference


def select_features(X, y, alpha=.05, correlation_threshold=.85):
    """Train-only t screening, correlation pruning, then t rechecking.

    One-hot columns are screened individually, as in the source notebook.
    Selection is repeated separately inside every LOOCV training fold.
    """
    current = list(X.columns)
    log = []
    initial, _, _ = ols_statistics(X[current], y, alpha)
    insignificant = initial.loc[(initial['Признак'] != 'const') & ~initial['Значим']]
    for _, row in insignificant.iterrows():
        log.append({'Признак': row['Признак'], 'Причина': 't: p>=alpha', 'Значение': row['p_value']})
        current.remove(row['Признак'])
    if not current:
        raise ValueError('Не осталось значимых признаков.')
    while len(current) > 1:
        correlation = X[current].corr().abs()
        pairs = [(correlation.iloc[i, j], current[i], current[j])
                 for i in range(len(current)) for j in range(i+1, len(current))
                 if correlation.iloc[i, j] >= correlation_threshold]
        if not pairs:
            break
        strength, a, b = max(pairs)
        target_correlation = X[[a, b]].apply(lambda col: np.corrcoef(col, y)[0, 1]).abs()
        removed = b if target_correlation[a] >= target_correlation[b] else a
        current.remove(removed)
        log.append({'Признак': removed, 'Причина': 'корреляционный дубль', 'Значение': strength})
    while len(current) > 1:
        table, _, _ = ols_statistics(X[current], y, alpha)
        candidate = table[table['Признак'] != 'const'].sort_values('p_value').iloc[-1]
        if candidate['p_value'] < alpha:
            break
        current.remove(candidate['Признак'])
        log.append({'Признак': candidate['Признак'], 'Причина': 'повторная t-проверка', 'Значение': candidate['p_value']})
    return current, pd.DataFrame(log, columns=['Признак', 'Причина', 'Значение'])
