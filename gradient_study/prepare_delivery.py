"""Prepare the 5+1 slide outline and portable source/result archive."""
import json
from pathlib import Path
import zipfile

import pandas as pd

root = Path(__file__).resolve().parent
output = root/'outputs'
winner = json.loads((output/'selection_before_test.json').read_text(encoding='utf-8'))['winner']
model = json.loads((output/'best_model.json').read_text(encoding='utf-8'))
statistics = pd.read_csv(output/'tables/model_significance.csv')
best = pd.read_csv(output/'tables/best_by_method.csv')
stability = pd.read_csv(output/'tables/feature_selection_stability.csv')
fields = ['Метод','Шаг','R2_val','RMSE_val','RMSE_CV','Итерации']


def markdown_table(frame):
    headers = list(frame.columns)
    rows = ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for values in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{value:.6f}' if isinstance(value,float) else str(value) for value in values)+' |')
    return '\n'.join(rows)


outline = f'''# Материалы для презентации: пять блоков и вывод

Это содержание слайдов, а не готовая презентация. Код на слайды не переносится.
Каждый из пяти основных блоков notebook соответствует одному слайду.
GIF используются по желанию вместо PNG на тех же слайдах и не добавляют новых слайдов.

## Слайд 1 — данные и обучение на train

800 квартир, ответ — стоимость в тыс. руб. Train: 480, validation: 160, test: 160.
ID исключён, день недели закодирован one-hot. Обучен собственный VGD со свободным членом.
Масштабирование изучает только train. Сходимость предварительного VGD проверена против МНК.

Вставить небольшую таблицу разбиения из `tables/splits.csv`.
Вывод: подготовлена независимая проверка без использования test при выборе.

## Слайд 2 — значимость модели и признаков

Полная модель: R² train = {statistics.iloc[0]['R2_train']:.6f},
F = {statistics.iloc[0]['F']:.3f}, F критическое = {statistics.iloc[0]['F_critical']:.3f}.
При α=0.05 модель значима. t-проверки определяют значимость отдельных коэффициентов.

Показать сокращённую таблицу значимости пяти оставшихся признаков из
`tables/significance_selected.csv`; полную таблицу оставить в приложении к работе.
Ключевое различие: F проверяет модель целиком, t — отдельные коэффициенты.

## Слайд 3 — корреляции и отбор

Вставить `figures/01_correlations.png` или `animations/01_correlations.gif`.
Общая и жилая площадь сильно связаны; расстояние до метро в метрах и минутах почти дублируется.
Незначимые признаки удалены; среди оставшихся нет пары с |r|≥0.85.

Оставлены: {', '.join(model['features'])}.
Состав X изменён, y и наблюдения сохранены. Корреляция не доказывает причинность.

## Слайд 4 — пять методов и лучшие шаги

VGD, SGD, SAG, Momentum, Adam. Два типа шага: постоянный λ и λ/√(1+k).
Шесть значений λ от 10⁻⁵ до 1, всего 60 конфигураций.
Для каждой: прогноз на validation и 480 LOOCV folds внутри train.

{markdown_table(best[fields])}

Победитель по минимальной MSE validation: **{winner['Метод']}, {winner['Шаг']}**.
При использовании `02_methods.png` / `02_methods.gif` уточнить: столбцы показывают **LOOCV**,
а победитель выбирается по **validation**. Поэтому самый низкий столбец может принадлежать другому методу.
Различия между почти сошедшимися методами малы. Победитель SAG исчерпал бюджет 10 000 обновлений.

## Слайд 5 — test

Вставить `figures/03_predictions.png` или `animations/03_predictions.gif`.
Показать три числа: R² = **{model['R2_test']:.6f}**,
RMSE = **{model['RMSE_test']:.2f} тыс. руб.**, MAE = **{model['MAE_test']:.2f} тыс. руб.**
Среднее train даёт RMSE около 6152.47 тыс. руб.
Test использован только после фиксации метода, шага и признаков.

## Слайд 6 — вывод

1. Линейная модель статистически значима и хорошо предсказывает стоимость на данном test.
2. Из 60 вариантов выбран {model['method']} с шагом {model['step']} по validation.
3. Ошибка test — около {model['RMSE_test']:.0f} тыс. руб.; это не нулевая ошибка и не «99% точности».
4. LOOCV = {winner['RMSE_CV']:.2f} тыс. руб. Она выше validation, потому что включает повторный отбор признаков.
5. Время до метро сохранено в 403/480 folds, номер подъезда — в 249/480. Отбор чувствителен к данным.
6. Результат относится к этому разбиению, сетке и бюджету. Универсальное превосходство SAG не доказано.

Для защиты: LOOCV каждый раз учится без одной квартиры. Она не использует test.
R² сравнивает сумму квадратов ошибок с разбросом ответов относительно их среднего.
RMSE измеряется в тыс. руб., MSE — в (тыс. руб.)².
'''
(output/'slide_outline.md').write_text(outline,encoding='utf-8')

archive_path = output/'workshop_ml_solution.zip'
files = []
for p in root.rglob('*'):
    if not p.is_file():
        continue
    relative = p.relative_to(root)
    if any(part in {'.venv','node_modules','.cache','__pycache__','workbook_previews'} for part in relative.parts):
        continue
    if p == archive_path or p.suffix in {'.nbc','.nbi','.pyc'} or p.name.endswith('.inspect.ndjson'):
        continue
    if p.name in {'animation_contact_sheet.png','search_checkpoint.csv'}:
        continue
    files.append(p)
with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for p in sorted(set(files)):
        archive.write(p,p.relative_to(root.parent))
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    assert 'gradient_study/regression (2).ipynb' in archive.namelist()
    assert 'gradient_study/outputs/animations/03_predictions.gif' in archive.namelist()
print(f'Archive: {archive_path}; files: {len(set(files))}; bytes: {archive_path.stat().st_size}')
