"""Build the course notebook from reviewable source files and section cells."""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []


def markdown(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(r'''
# Исследование методов градиентного спуска

**Датасет:** `regression_data.csv`, стоимость квартир в **тыс. руб.**

Решение написано заново по заданию. Приоритет — лекции 1–3; для структуры использованы
семинар 2 и ДЗ-3 из [репозитория](https://github.com/ROTNITE/workshop-on-ml).
Оформление трёх основных графиков основано на `regression (2).ipynb`.

Порядок: train → F/t-проверки → корреляции и отбор → 60 конфигураций и LOOCV →
выбор по validation → единственная итоговая проверка победителя на test → выводы.
Код следует запускать сверху вниз. Командная презентация составляется по итоговым
таблицам, графикам и выводам; состав команды здесь не выдумывается.

**Трактовка задания.** «Обучение на val» означает проверку на val и подбор шага после
обучения на train (лекция 2, с. 5–6). При удалении признаков меняется только X;
целевой столбец y и строки остаются. Test не используется для выбора модели.
''')
code('''
from pathlib import Path
import json
import sys
import platform
import importlib.metadata

import numpy as np
import pandas as pd
from IPython.display import display, Image as NotebookImage, Markdown

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 70)
pd.set_option('display.float_format', lambda value: f'{value:.7g}')

ROOT = Path.cwd()
WORK = ROOT if (ROOT/'config.json').is_file() else ROOT/'gradient_study'
DATA_FILE = WORK/'data/regression_data.csv'
if not DATA_FILE.is_file():
    raise FileNotFoundError('Откройте notebook в gradient_study или в корне репозитория.')
OUTPUT = WORK / 'outputs'
TABLES, FIGURES = OUTPUT / 'tables', OUTPUT / 'figures'
for directory in (OUTPUT, TABLES, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)
CONFIG = json.loads((WORK / 'config.json').read_text(encoding='utf-8'))
ALPHA = CONFIG['alpha']
CORRELATION_THRESHOLD = CONFIG['correlation_threshold']
CREATE_ANIMATIONS = True
print('Python:', platform.python_version())
print('Шаги:', CONFIG['lambda_grid'])
print('Число вариантов:', len(CONFIG['methods'])*len(CONFIG['schedules'])*len(CONFIG['lambda_grid']))
''')
markdown(r'''
## 1. Формулы и реализации

Минимизируем $Q(w)=\frac1n\|Xw-y\|^2$ со свободным членом.
Градиент $g=\frac2nX^T(Xw-y)$.

| Метод | Обновление |
|---|---|
| VGD | $w_{k+1}=w_k-\eta_k g_k$ |
| SGD | тот же шаг по случайному мини-батчу из 32 объектов |
| SAG | пересчитать градиент одного объекта; обновить среднее сохранённых градиентов |
| Momentum | $h_{k+1}=0.9h_k+\eta_k g_k$, $w_{k+1}=w_k-h_{k+1}$ |
| Adam | моменты $\beta_1=0.9$, $\beta_2=0.999$, коррекция смещения, $\varepsilon=10^{-8}$ |

Проверяем **постоянный** шаг $\eta_k=\lambda$ и **функциональный**
$\eta_k=\lambda/\sqrt{1+k}$, $k=0,1,\ldots$ (лекция 2, с. 1–5).
Сетка из семинара 2: $\lambda\in\{10^{-5},10^{-4},10^{-3},10^{-2},10^{-1},1\}$.
«Все шаги» — все значения этой заранее зафиксированной сетки.

Все пять методов реализованы явно. Numba компилирует циклы для ускорения LOOCV,
сами формулы от этого не меняются. И X, и y стандартизуются по обучающей части;
ошибки и коэффициенты возвращаются в исходные единицы.
''')
code((ROOT/'src/methods.py').read_text(encoding='utf-8').replace('@njit(cache=True)', '@njit(cache=False)'))
code((ROOT/'src/study.py').read_text(encoding='utf-8'))
markdown('''
## 2. Данные и независимые выборки

Сохраняем исходное разбиение 60/20/20 и seed=42. ID исключаем как идентификатор.
День недели кодируем one-hot с одной базовой категорией, а не порядковыми числами.
Категории определяются только по обучающей части. Пропуски, повторные ID и
неизвестные категории проверяются явно.
''')
code('''
df = pd.read_csv(DATA_FILE)
train, validation, test = split_dataset(df, CONFIG['seed'])
X_train_raw = train.drop(columns=[TARGET])
X_val_raw = validation.drop(columns=[TARGET])
y_train, y_val = train[TARGET].to_numpy(), validation[TARGET].to_numpy()
X_encoded, X_val_encoded = encode_train(X_train_raw, X_val_raw)

split_table = pd.DataFrame({'Выборка': ['train', 'validation', 'test'],
                            'Объекты': [len(train), len(validation), len(test)],
                            'Доля': [len(train)/len(df), len(validation)/len(df), len(test)/len(df)]})
display(df.head())
display(split_table)
print('Размер датасета:', df.shape, 'Пропуски:', df.isna().sum().sum())
membership = pd.concat([part[[IDENTIFIER]].assign(Выборка=name)
                        for name, part in [('train', train), ('validation', validation), ('test', test)]])
export_table(membership, 'split_membership', TABLES)
export_table(split_table, 'splits', TABLES)
''')
markdown(r'''
## 3. Обучение VGD на train и проверка значимости

Для статистических проверок VGD обучается до сходимости к решению МНК.
Начальный постоянный шаг берётся из спектра матрицы градиента: $0.9/L$, где
$L=\lambda_{max}(2X^TX/n)$ для стандартизованной матрицы со свободным членом.
Это отдельное предварительное обучение, а не вариант в сравнительной сетке.
Независимое решение `lstsq` проверяет, что VGD действительно сошёлся.

**F-проверка:** $H_0:b_1=\dots=b_k=0$, $H_1$ — хотя бы один коэффициент ненулевой.
$F=(R^2/k)/((1-R^2)/(n-k-1))$, сравниваем с $F_{0.95}(k,n-k-1)$.

**t-проверка:** $H_0:b_i=0$ против $H_1:b_i\ne0$.
$s_e^2=SSE/(n-k-1)$, $C=s_e^2(X^TX)^{-1}$, $t_i=b_i/\sqrt{C_{ii}}$.
Двусторонний уровень $\alpha=0.05$ (лекция 3, с. 1–2).
F проверяет модель в целом, t — каждый коэффициент при учёте остальных признаков.
''')
code('''
values = X_encoded.to_numpy()
standardized = (values-values.mean(axis=0))/values.std(axis=0)
design = np.column_stack([np.ones(len(values)), standardized])
lipschitz = np.linalg.eigvalsh(2*design.T@design/len(design)).max()
baseline = fit_gd(values, y_train, method='VGD', initial=.9/lipschitz,
                  max_iter=CONFIG['baseline_max_iter'], tolerance=CONFIG['baseline_tolerance'],
                  seed=CONFIG['seed'])
if baseline.status != 'converged':
    raise RuntimeError('VGD не сошёлся: F/t-проверки нельзя выдавать за точные проверки МНК.')
full_significance, full_model_stats, ols_parameters = ols_statistics(
    X_encoded, y_train, ALPHA, baseline.coefficients())
ols_prediction = ols_parameters[0] + values@ols_parameters[1:]
baseline_difference = float(np.max(np.abs(baseline.predict(values)-ols_prediction)))
assert baseline_difference < .001, 'VGD ещё недостаточно близок к МНК.'
print('Предварительный VGD: шаг =', .9/lipschitz, 'итераций =', baseline.iterations)
print('Максимальное расхождение прогнозов VGD и МНК, тыс. руб.:', baseline_difference)
display(pd.DataFrame([full_model_stats]))
display(full_significance)
export_table(full_significance, 'significance_full', TABLES)
''')
markdown(r'''
## 4. Отбор признаков и парные корреляции

По t-критерию исключаем признаки с $p\ge0.05$, сохраняя свободный член.
Строим полную числовую матрицу корреляций, как в исходном notebook, и таблицу
корреляций оставшихся признаков. Заранее задаём порог избыточности $|r|\ge0.85$.
Если после t-отбора такая пара осталась, сохраняем признак с большей
$|r(x,y)|$ на train, затем повторно проверяем значимость сокращённой модели.

Категориальные индикаторы проверяются по отдельности. Отбор разведочный:
множественные проверки и повторный отбор ограничивают интерпретацию p-value.
Парная корреляция отражает линейную связь, не причинность и не отдельный вклад
признака в многомерную модель. Нельзя удалить полезный признак только за слабую связь с y.
''')
code('''
selected_features, selection_log = select_features(X_encoded, y_train, ALPHA, CORRELATION_THRESHOLD)
X_train_new = X_encoded[selected_features]
X_val_new = X_val_encoded[selected_features]
reduced_significance, reduced_model_stats, _ = ols_statistics(X_train_new, y_train, ALPHA)
corr_matrix = X_train_raw.drop(columns=[IDENTIFIER, CATEGORY]).corr()
corr_selected = X_train_new.corr()
feature_target_corr = X_encoded.apply(lambda col: np.corrcoef(col, y_train)[0,1]).rename('r(X,y)')
correlated_pairs = pd.DataFrame([
    {'Признак 1': a, 'Признак 2': b, 'r': corr_matrix.loc[a,b]}
    for i,a in enumerate(corr_matrix.columns) for b in corr_matrix.columns[i+1:]
    if abs(corr_matrix.loc[a,b]) >= CORRELATION_THRESHOLD])
print('Оставлены:', selected_features)
display(selection_log)
display(corr_matrix)
display(correlated_pairs)
display(corr_selected)
display(pd.DataFrame([reduced_model_stats]))
display(reduced_significance)
assert reduced_significance.loc[reduced_significance['Признак']!='const', 'Значим'].all()
assert np.max(np.abs(corr_selected.to_numpy()-np.eye(len(selected_features)))) < CORRELATION_THRESHOLD
export_table(selection_log, 'removed_features', TABLES)
export_table(reduced_significance, 'significance_selected', TABLES)
export_table(pd.DataFrame([{'Этап':'Полная модель', **full_model_stats},
                           {'Этап':'После отбора', **reduced_model_stats}]), 'model_significance', TABLES)
export_table(corr_matrix.rename_axis('Признак').reset_index(), 'correlations_full', TABLES)
export_table(corr_selected.rename_axis('Признак').reset_index(), 'correlations_selected', TABLES)
export_table(feature_target_corr.rename_axis('Признак').reset_index(), 'correlations_target', TABLES)
export_table(correlated_pairs, 'correlated_pairs', TABLES)
export_table(pd.concat([X_train_new, train[TARGET]],axis=1).assign(Выборка='train'), 'train_selected', TABLES)
''')
markdown(r'''
## 5. Все методы и шаги: validation и LOOCV

Каждая из 60 конфигураций обучается на train и оценивается на validation.
Дополнительно для каждой конфигурации выполняется **настоящая LOOCV внутри train**:
480 обучений на 479 квартирах с прогнозом оставленной квартиры.
В каждом fold заново выполняются кодирование, отбор признаков и масштабирование.
Таким образом, оставленная квартира не влияет даже на выбор признаков.

$LOOCV=\sqrt{\frac1n\sum_i(y_i-\hat y_{-i})^2}$ (лекция 3, с. 3).
Это RMSE в тыс. руб.; MSE имеет единицу $(\text{тыс. руб.})^2$.
$R^2=1-SSE/SST$ — коэффициент детерминации, не «процент точности».

Бюджет — 10 000 обновлений для каждого метода и fold. Остановка раньше возможна
только при норме полного градиента не выше $10^{-7}$. Маленькое изменение потерь
не считается доказательством сходимости. `max_iter` означает исчерпание бюджета,
`diverged` — численную расходимость. Разошедшиеся варианты остаются в таблицах,
но не допускаются к выбору; расходимость хотя бы одного CV-fold исключает вариант.

Число обновлений не равно объёму работы: VGD/Momentum/Adam используют полный
градиент (для квадратичной функции он вычисляется через заранее рассчитанную
матрицу), SGD — 32 случайных наблюдения с повторениями, SAG — одно наблюдение
и память градиентов. По числу итераций нельзя объявлять универсального победителя по скорости.
''')
code('''
folds, selection_stability = prepare_loocv(X_train_raw, y_train, ALPHA, CORRELATION_THRESHOLD)
print('LOOCV folds:', len(folds))
display(selection_stability)
export_table(selection_stability, 'feature_selection_stability', TABLES)
results_df, best_by_method, best_by_type, best, fitted_models, loocv_details = experiment(
    X_train_new.to_numpy(), y_train, X_val_new.to_numpy(), y_val, folds, CONFIG, OUTPUT)
export_table(results_df, 'all_configurations', TABLES)
export_table(best_by_method, 'best_by_method', TABLES)
export_table(best_by_type, 'best_by_method_and_step', TABLES)
export_table(loocv_details, 'loocv_predictions', TABLES)
display(results_df)
''')
markdown(r'''
## 6. Выбор лучшего метода и шага

По лекции 2 (с. 5–6) выбираем минимальную **MSE validation** среди допущенных
конфигураций. LOOCV — дополнительная проверка устойчивости; только при точном
равенстве MSE val она разрешает ничью. Таблица для каждого метода содержит
одну реальную конфигурацию со всеми её метриками, а не минимумы из разных строк.
Победитель фиксируется до обращения к test. На validation не дообучаем.
''')
code('''
columns = ['Метод','Тип шага','lambda','Шаг','R2_train','R2_val','MSE_train',
           'MSE_val','RMSE_val','RMSE_CV','Итерации','Статус','CV_max_iter']
display(best_by_type[columns])
display(best_by_method[columns])
print('Победитель:', best['Метод'], best['Шаг'])
print('RMSE validation:', best['RMSE_val'], 'LOOCV:', best['RMSE_CV'])
winner = {key: (value.item() if isinstance(value,np.generic) else value)
          for key,value in best.to_dict().items()}
save_json(OUTPUT/'selection_before_test.json', {'features':selected_features, 'winner':winner,
          'selection_rule':'min MSE_val, then RMSE_LOOCV, then stable configuration key',
          'dataset_sha256':file_sha256(DATA_FILE)})
''')
markdown('''
## 7. Единственная итоговая проверка на test

Используем уже обученную на train модель выбранной конфигурации. Test не меняет
ни признаки, ни шаг, ни число итераций. Константный контрольный прогноз использует
среднюю стоимость на train. Сравнение с ним помогает понять масштаб улучшения.
''')
code('''
_, X_test_encoded = encode_train(X_train_raw, test.drop(columns=[TARGET]))
X_test_new = X_test_encoded[selected_features]
y_true_test = test[TARGET].to_numpy()
best_model = fitted_models[best['Конфигурация']]
y_pred_test = best_model.predict(X_test_new.to_numpy())
r2_test = r2_score_manual(y_true_test, y_pred_test)
rmse_test = np.sqrt(mse(y_true_test, y_pred_test))
mae_test = float(np.mean(np.abs(y_true_test-y_pred_test)))
constant_prediction = np.full(len(test), y_train.mean())
test_metrics = pd.DataFrame([
    {'Модель':best['Метод'], 'R2_test':r2_test, 'MSE_test':rmse_test**2,
     'RMSE_test':rmse_test, 'MAE_test':mae_test},
    {'Модель':'Среднее train', 'R2_test':r2_score_manual(y_true_test,constant_prediction),
     'MSE_test':mse(y_true_test,constant_prediction),
     'RMSE_test':np.sqrt(mse(y_true_test,constant_prediction)),
     'MAE_test':float(np.mean(np.abs(y_true_test-constant_prediction)))}])
test_predictions = pd.DataFrame({'ID_объекта':test[IDENTIFIER].to_numpy(),
    'Фактическая стоимость':y_true_test, 'Прогноз':y_pred_test,
    'Ошибка':y_true_test-y_pred_test, 'Константный прогноз':constant_prediction})
display(test_metrics)
parameters = best_model.coefficients()
equation = 'ŷ = '+f'{parameters[0]:.6f}'
for name,value in zip(selected_features,parameters[1:]):
    equation += f' {value:+.6f}·[{name}]'
print('Уравнение в исходных единицах:', equation)
export_table(test_metrics, 'test_metrics', TABLES)
export_table(test_predictions, 'test_predictions', TABLES)
save_json(OUTPUT/'best_model.json', {'method':best['Метод'], 'step':best['Шаг'],
    'features':selected_features, 'intercept':float(parameters[0]),
    'coefficients':parameters[1:].tolist(), 'units':'тыс. руб.',
    'R2_test':r2_test,'RMSE_test':float(rmse_test),'MAE_test':mae_test})
''')
markdown('''
## 8. Графики для презентации

Три основных графика сохраняют размеры, цвета, подписи и композицию исходного
notebook. Числа пересчитаны по лекциям. PNG и GIF используют одни и те же данные.
На столбчатом графике показана LOOCV выбранной по validation конфигурации каждого
метода; порядок столбцов определяется этой ошибкой и может отличаться от победителя
по validation. Шкала столбцов начинается с нуля.
''')
code((ROOT/'src/static_charts.py').read_text(encoding='utf-8'))
code('''
main_figures = [
    ('01_correlations.png', correlation_chart(corr_matrix)),
    ('02_methods.png', comparison_chart(best_by_method)),
    ('03_predictions.png', prediction_chart(y_true_test,y_pred_test,r2_test,rmse_test))]
for filename,image in main_figures:
    image.save(FIGURES/filename)
    display(NotebookImage(filename=str(FIGURES/filename)))
''')
code('''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'svg.fonttype':'none'})
fig,ax = plt.subplots(figsize=(10,5.5),layout='constrained')
for _,row in best_by_method.iterrows():
    model = fitted_models[row['Конфигурация']]
    history = model.history
    ax.plot(history[:,0],history[:,1]*model.y_scale**2,label=row['Метод'])
ax.set(xlabel='Число обновлений весов',ylabel='MSE train, (тыс. руб.)²',
       title='Сходимость при лучшем шаге каждого метода',yscale='log')
ax.grid(alpha=.2)
ax.legend()
fig.savefig(FIGURES/'04_convergence.png',dpi=180)
fig.savefig(FIGURES/'04_convergence.svg')
plt.close(fig)
display(NotebookImage(filename=str(FIGURES/'04_convergence.png')))

fig,axes = plt.subplots(1,5,figsize=(17,4),layout='constrained')
for ax,method in zip(axes,CONFIG['methods']):
    for schedule,label in [('constant','Постоянный'),('time_decay','λ/√(1+k)')]:
        rows = results_df[(results_df['Метод']==method)&(results_df['Тип шага']==schedule)].sort_values('lambda')
        valid = rows['Допущен']
        ax.plot(rows.loc[valid,'lambda'],rows.loc[valid,'RMSE_val'],'o-',label=label)
    ax.set(title=method,xscale='log',yscale='log',xlabel='Начальный шаг λ')
    ax.grid(alpha=.2)
axes[0].set_ylabel('RMSE validation, тыс. руб.')
axes[-1].legend(fontsize=8)
fig.suptitle('Подбор шага по validation. Разошедшиеся варианты исключены из кривых')
fig.savefig(FIGURES/'05_step_search.png',dpi=180)
fig.savefig(FIGURES/'05_step_search.svg')
plt.close(fig)
display(NotebookImage(filename=str(FIGURES/'05_step_search.png')))
''')
markdown('''
## 9. Три GIF-анимации

Код анимаций сохранён и переписан для новых результатов. Анимации воспроизводятся
один раз, удерживают последний кадр и сохраняются отдельными файлами в
`gradient_study/outputs/animations`. Последний кадр соответствует статическому
графику с небольшой потерей цвета из-за 256-цветной палитры GIF.
''')
code((ROOT/'src/animations.py').read_text(encoding='utf-8'))
code('''
if CREATE_ANIMATIONS:
    animation_paths = create_animations(corr_matrix,best_by_method,y_true_test,y_pred_test,
                                        r2_test,rmse_test,OUTPUT/'animations')
    for path in animation_paths:
        display(NotebookImage(filename=str(path)))
        print(path.name)
''')
markdown('''
## 10. Выводы и файлы для презентации

Вывод ниже формируется из рассчитанных результатов. Время выполнения и численная
сходимость ограничены данным бюджетом; более низкая validation-ошибка не доказывает
универсального превосходства метода. Все методы решают одну линейную задачу МНК.
При полной сходимости они должны давать близкие прогнозы.
''')
code('''
improvement = 1-rmse_test/test_metrics.iloc[1]['RMSE_test']
conclusions = f"""# Выводы

Обработаны {len(df)} квартир. Train: {len(train)}, validation: {len(validation)}, test: {len(test)}.
F-проверка полной модели: F={full_model_stats['F']:.3f}, критическое значение {full_model_stats['F_critical']:.3f}.
Нулевая гипотеза об отсутствии всех линейных связей отвергается при α=0.05.

После t-проверок и контроля корреляций оставлены: {', '.join(selected_features)}.
При удалении признаков столбец y и строки наблюдений сохранены.
Повторная F-проверка: F={reduced_model_stats['F']:.3f}.
Номер подъезда имеет пограничную значимость на этом разбиении; его устойчивость видна в таблице LOOCV-отбора.

Проверены {len(results_df)} конфигураций: пять методов, два типа шага, шесть значений λ.
Для каждой выполнены {len(folds)} LOOCV folds с повторным отбором признаков внутри обучающей части.
По минимальной MSE validation выбран {best['Метод']}, {best['Шаг']}.
RMSE validation = {best['RMSE_val']:.4f} тыс. руб., R² validation = {best['R2_val']:.6f}.
LOOCV RMSE = {best['RMSE_CV']:.4f} тыс. руб.

На независимом test: R²={r2_test:.6f}, RMSE={rmse_test:.4f} тыс. руб., MAE={mae_test:.4f} тыс. руб.
RMSE ниже ошибки прогноза среднего train на {improvement:.2%}.
R² характеризует долю вариации стоимости, объясняемую прогнозом относительно средней линии.
Это не процент точности и не доказательство причинного влияния признаков.

{equation}

Результат ограничен этим датасетом, разбиением, сеткой шагов, seed и бюджетом.
Значимость после отбора разведочная. Классические F/t-проверки предполагают независимые наблюдения,
корректную линейную спецификацию и одинаковую дисперсию ошибок; точные маловыборочные распределения
дополнительно предполагают нормальность ошибок. На реальных ценах обобщение не проверялось.
Малые различия между почти сошедшимися методами не дают основания объявлять один метод лучшим вообще.
"""
(OUTPUT/'conclusions.md').write_text(conclusions,encoding='utf-8')
display(Markdown(conclusions))

provenance = {
    'dataset_sha256':file_sha256(DATA_FILE),
    'original_notebook_sha256':file_sha256(WORK/'sources/regression.original.ipynb'),
    'upstream_repository':'https://github.com/ROTNITE/workshop-on-ml',
    'upstream_commit':'60d25da62512835765c8ce14597a80c6e0e1aa34',
    'python':sys.version, 'config':CONFIG,
    'libraries':{name:importlib.metadata.version(name) for name in
                 ['numpy','pandas','scipy','scikit-learn','numba','matplotlib','pillow']},
    'lectures':{str(p.relative_to(WORK)):file_sha256(p) for p in (WORK/'sources/lectures').glob('*.pdf')},
    'selection_uses_test':False, 'test_evaluated_for':['winner','train_mean_baseline'],
    'loocv_total_fits':len(results_df)*len(folds)}
save_json(OUTPUT/'provenance.json',provenance)
workbook_data = {p.stem:pd.read_csv(p).replace([np.inf,-np.inf],np.nan).astype(object).where(
                   pd.notna(pd.read_csv(p)),None).to_dict(orient='split')
                 for p in TABLES.glob('*.csv') if p.stem not in ['loocv_predictions','train_selected','split_membership']}
save_json(OUTPUT/'workbook_data.json',workbook_data)
print('Результаты сохранены:', OUTPUT)
''')
# Five main code blocks, one per substantive slide. Conclusions and animations
# are explicitly outside that count. The source sections above stay reviewable.
sections = cells
baseline_source = sections[8].source
training, statistics = baseline_source.split('full_significance, full_model_stats, ols_parameters =', 1)
statistics = 'full_significance, full_model_stats, ols_parameters ='+statistics
cells = [sections[0]]
markdown('''
## Блок 1. Данные и обучение VGD на train

Датасет: 800 квартир, цена в тыс. руб. Разбиение исходного notebook — 480/160/160,
seed=42. ID исключается, день недели кодируется one-hot. X и y масштабируются
только по обучающей части. Функция потерь — MSE со свободным членом.

Предварительный VGD обучается до сходимости с устойчивым постоянным шагом 0.9/L,
где L — максимальное собственное значение матрицы градиента. Независимое решение
МНК служит проверкой сходимости. Сравнительный перебор шагов выполняется в блоке 4.
''')
code('\n\n'.join(sections[i].source for i in [1,3,4,18,6])+'\n\n'+training+
     "\nprint('VGD обучен:', baseline.status, 'итераций:', baseline.iterations)\n")
markdown(r'''
## Блок 2. Значимость модели и признаков

Проверки выполняются только на train для сошедшегося VGD. Предсказания VGD и МНК
сверяются с допуском 0.001 тыс. руб.

**F:** $H_0:b_1=\dots=b_k=0$, $H_1$ — хотя бы один коэффициент ненулевой.
$F=(R^2/k)/((1-R^2)/(n-k-1))$, сравниваем с $F_{0.95}(k,n-k-1)$.

**t:** $H_0:b_i=0$ против $H_1:b_i\ne0$.
$C=s_e^2(X^TX)^{-1}$, $s_e^2=SSE/(n-k-1)$, $t_i=b_i/\sqrt{C_{ii}}$.
Двусторонний уровень α=0.05 (лекция 3, с. 1–2). В следующем блоке незначимые
признаки исключаются программно. Отсутствие значимости не доказывает нулевой эффект.
''')
code(statistics)
markdown(sections[9].source.replace('## 4. Отбор признаков и парные корреляции',
                                    '## Блок 3. Корреляции и окончательный набор признаков'))
code(sections[10].source+'''
image = correlation_chart(corr_matrix)
image.save(FIGURES/'01_correlations.png')
display(NotebookImage(filename=str(FIGURES/'01_correlations.png')))
''')
markdown(sections[11].source.replace('## 5. Все методы и шаги: validation и LOOCV',
                                     '## Блок 4. Сравнение методов, шагов и выбор победителя')+
         '\n\n'+sections[2].source.replace('## 1. Формулы и реализации','### Формулы пяти методов')+
         '\n\n'+sections[13].source.replace('## 6. Выбор лучшего метода и шага','### Правило выбора'))
code(sections[12].source+'\n\n'+sections[14].source+'''
image = comparison_chart(best_by_method)
image.save(FIGURES/'02_methods.png')
display(NotebookImage(filename=str(FIGURES/'02_methods.png')))
'''+sections[20].source)
markdown(sections[15].source.replace('## 7. Единственная итоговая проверка на test',
                                     '## Блок 5. Итоговая проверка победителя на test'))
code(sections[16].source+'''
image = prediction_chart(y_true_test,y_pred_test,r2_test,rmse_test)
image.save(FIGURES/'03_predictions.png')
display(NotebookImage(filename=str(FIGURES/'03_predictions.png')))
''')
markdown(sections[24].source.replace('## 10. Выводы и файлы для презентации',
                                     '## Вывод. Шестой слайд'))
code(sections[25].source)
markdown(sections[21].source.replace('## 9. Три GIF-анимации','## Дополнительно. Три GIF-анимации'))
code(sections[22].source+'\n\n'+sections[23].source)

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {'display_name':'Python 3 (workshop ML)', 'language':'python', 'name':'workshop-ml'}
nb.metadata.language_info = {'name':'python','version':'3.12'}
target = ROOT/'regression (2).ipynb'
nbf.write(nb, target)
print(f'Built {target}: {len(cells)} cells')
