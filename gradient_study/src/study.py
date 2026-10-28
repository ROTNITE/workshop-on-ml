"""Train/validation/LOOCV protocol and export helpers for the notebook."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .methods import METHODS, fit_gd, mse, ols_statistics, r2_score_manual, select_features
except ImportError:
    # This file's definitions are also embedded directly into the notebook.
    pass

TARGET = 'Стоимость_квартиры_тыс_руб'
CATEGORY = 'День_недели_публикации'
IDENTIFIER = 'ID_объекта'


def encode_train(frame, other=None):
    """Categories and output columns are learned on this training frame only."""
    frame = frame.drop(columns=[IDENTIFIER], errors='ignore').copy()
    categories = sorted(frame[CATEGORY].unique())
    frame[CATEGORY] = pd.Categorical(frame[CATEGORY], categories=categories)
    encoded = pd.get_dummies(frame, columns=[CATEGORY], drop_first=True, dtype=float).astype(float)
    if other is None:
        return encoded
    other = other.drop(columns=[IDENTIFIER], errors='ignore').copy()
    if not set(other[CATEGORY]).issubset(categories):
        raise ValueError('В проверочной части есть неизвестные категории.')
    other[CATEGORY] = pd.Categorical(other[CATEGORY], categories=categories)
    transformed = pd.get_dummies(other, columns=[CATEGORY], drop_first=True, dtype=float)
    return encoded, transformed.reindex(columns=encoded.columns, fill_value=0.).astype(float)


def split_dataset(frame, seed=42):
    if TARGET not in frame or frame.isna().any().any():
        raise ValueError('Проверьте целевой столбец и пропуски в данных.')
    if not frame[IDENTIFIER].is_unique:
        raise ValueError('ID объектов должны быть уникальны.')
    # Keep exact original notebook partitions: 60 / 20 / 20.
    train, temporary = train_test_split(frame, test_size=.4, random_state=seed)
    validation, test = train_test_split(temporary, test_size=.5, random_state=seed)
    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(validation.index).isdisjoint(test.index)
    return train, validation, test


def prepare_loocv(X_frame, y, alpha, correlation_threshold):
    """Leave one out, then refit encoder, feature selector and scales inside fit_gd."""
    y = np.asarray(y)
    folds, selections = [], []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        a, b = encode_train(X_frame.iloc[keep], X_frame.iloc[[i]])
        features, _ = select_features(a, y[keep], alpha, correlation_threshold)
        folds.append((a[features].to_numpy(), y[keep], b[features].to_numpy(), float(y[i])))
        selections.append(features)
    counts = pd.Series([name for names in selections for name in names]).value_counts()
    stability = counts.rename_axis('Признак').reset_index(name='Выбран в LOOCV folds')
    stability['Доля folds'] = stability['Выбран в LOOCV folds'] / len(y)
    return folds, stability


def step_description(schedule, initial):
    return f'η = {initial:g}' if schedule == 'constant' else f'η(k) = {initial:g}/√(1+k)'


def experiment(X, y, X_validation, y_validation, folds, config, output):
    """No test argument: candidate selection cannot access test metrics."""
    rows, fold_rows, models = [], [], {}
    for method in config['methods']:
        for schedule in config['schedules']:
            for initial in config['lambda_grid']:
                started = time.perf_counter()
                key = f'{method}_{schedule}_{initial:g}'
                options = dict(method=method, initial=initial, schedule=schedule,
                               max_iter=config['max_iter'], tolerance=config['gradient_tolerance'],
                               batch_size=config['batch_size'], seed=config['seed'],
                               check_every=config['check_every'], divergence_limit=config['divergence_limit'])
                model = fit_gd(X, y, **options)
                models[key] = model
                errors, diverged_folds, capped_folds = [], 0, 0
                for i, (a, ya, b, truth) in enumerate(folds):
                    candidate = fit_gd(a, ya, **options)
                    prediction = None if candidate.status == 'diverged' else float(candidate.predict(b)[0])
                    error = None if prediction is None else prediction-truth
                    diverged_folds += int(candidate.status == 'diverged')
                    capped_folds += int(candidate.status == 'max_iter')
                    if error is not None:
                        errors.append(error)
                    fold_rows.append({'Конфигурация': key, 'fold': i, 'y': truth, 'prediction': prediction,
                                      'error': error, 'Статус': candidate.status, 'Итерации': candidate.iterations})
                cv_rmse = float(np.sqrt(np.mean(np.square(errors)))) if diverged_folds == 0 else np.nan
                row = {'Конфигурация': key, 'Метод': method, 'Тип шага': schedule,
                       'lambda': initial, 'Шаг': step_description(schedule, initial),
                       'Статус': model.status, 'Итерации': model.iterations,
                       'CV_folds': len(folds), 'CV_diverged': diverged_folds, 'CV_max_iter': capped_folds,
                       'RMSE_CV': cv_rmse, 'MSE_CV': cv_rmse**2,
                       'R2_train': np.nan, 'MSE_train': np.nan, 'R2_val': np.nan,
                       'MSE_val': np.nan, 'RMSE_val': np.nan}
                if model.status != 'diverged':
                    prediction_train, prediction_val = model.predict(X), model.predict(X_validation)
                    row.update(R2_train=r2_score_manual(y, prediction_train), MSE_train=mse(y, prediction_train),
                               R2_val=r2_score_manual(y_validation, prediction_val),
                               MSE_val=mse(y_validation, prediction_val), RMSE_val=np.sqrt(mse(y_validation, prediction_val)))
                row['Допущен'] = model.status != 'diverged' and diverged_folds == 0
                row['Секунды'] = time.perf_counter()-started
                rows.append(row)
                pd.DataFrame(rows).to_csv(output / 'search_checkpoint.csv', index=False, encoding='utf-8-sig')
                print(f'{len(rows):02d}/60 {key}: val RMSE={row["RMSE_val"]:.3f}, '
                      f'LOOCV={cv_rmse:.3f}, status={model.status}, {row["Секунды"]:.1f}s', flush=True)
    results = pd.DataFrame(rows)
    eligible = results[results['Допущен']].sort_values(['MSE_val', 'RMSE_CV', 'Конфигурация'], kind='stable')
    if eligible.empty:
        raise RuntimeError('Все конфигурации разошлись. См. search_checkpoint.csv.')
    best_per_method = eligible.drop_duplicates('Метод').reset_index(drop=True)
    best_per_type = eligible.drop_duplicates(['Метод', 'Тип шага']).reset_index(drop=True)
    return results, best_per_method, best_per_type, eligible.iloc[0], models, pd.DataFrame(fold_rows)


def export_table(frame, name, directory):
    frame.to_csv(directory / f'{name}.csv', index=False, encoding='utf-8-sig')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
