"""Static versions of the three charts in regression (2).ipynb."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=32)
def chart_font(size, bold=False):
    candidates = (
        ['C:/Windows/Fonts/arialbd.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']
        if bold else
        ['C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise FileNotFoundError('Для русских подписей установите Arial или DejaVu Sans.')


def new_chart(width, height, title, subtitle):
    image = Image.new('RGB', (width, height), '#ffffff')
    draw = ImageDraw.Draw(image)
    draw.text((28, 22), title, fill='#17263c', font=chart_font(25, True))
    draw.text((28, 60), subtitle, fill='#6f8096', font=chart_font(14))
    return image, draw


def correlation_chart(correlation):
    labels = [name.replace('_', ' ') for name in correlation.columns]
    count = len(labels)
    left, top, size = 310, 110, 46
    image, draw = new_chart(left + count * size + 35, top + count * size + 60,
                           'Корреляции признаков', 'Train · коэффициент Пирсона')
    for i, label in enumerate(labels):
        draw.text((left-12, top+i*size+size//2), f'{i+1}. {label}',
                  fill='#334155', font=chart_font(12), anchor='rm')
        draw.text((left+i*size+size//2, top-18), str(i+1),
                  fill='#334155', font=chart_font(12, True), anchor='mm')
    draw.text((28, top+count*size+22),
              'Синий — положительная связь   ·   Красный — отрицательная',
              fill='#718096', font=chart_font(12))
    for i in range(count):
        for j in range(count):
            value = correlation.iloc[i, j]
            strength = abs(value)
            color = ((int(245-180*strength), int(248-105*strength), int(252-30*strength))
                     if value >= 0 else
                     (int(252-25*strength), int(246-145*strength), int(245-130*strength)))
            x, y = left+j*size, top+i*size
            draw.rounded_rectangle((x, y, x+size-3, y+size-3), radius=6, fill=color)
            draw.text((x+size//2, y+size//2), f'{value:.2f}',
                      fill='white' if strength > .65 else '#263449',
                      font=chart_font(12), anchor='mm')
    return image


def comparison_chart(best_by_method, metric='RMSE_CV', cv_name='LOOCV'):
    rows = best_by_method.sort_values(metric, kind='stable')
    maximum = rows[metric].max() * 1.18
    image, draw = new_chart(760, 425, 'Сравнение методов',
                           f'Лучшая конфигурация каждого метода · {cv_name} · меньше — лучше')
    left, top, width = 145, 105, 500
    for i, (_, row) in enumerate(rows.iterrows()):
        y = top+i*57
        bar_width = width*row[metric]/maximum
        draw.text((left-15, y+15), row['Метод'], fill='#334155',
                  font=chart_font(15, True), anchor='rm')
        draw.rounded_rectangle((left, y, left+width, y+30), radius=6, fill='#f3f6fa')
        if bar_width >= 1:
            draw.rounded_rectangle((left, y, left+bar_width, y+30),
                                   radius=min(6, bar_width/2),
                                   fill='#2563eb' if i == 0 else '#a8c6ef')
        draw.text((left+bar_width+10, y+15), f'{row[metric]:.2f}',
                  fill='#263449', font=chart_font(13, True), anchor='lm')
    draw.text((left, 395), 'RMSE, тыс. руб. · шкала от нуля',
              fill='#718096', font=chart_font(12))
    return image


def prediction_chart(actual, predicted, r2, rmse, show_points=True):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    pad = max((hi-lo)*.04, 1.)
    lo, hi = lo-pad, hi+pad
    left, top, side = 110, 115, 400
    def coord_x(v):
        return left+(v-lo)/(hi-lo)*side
    def coord_y(v):
        return top+side-(v-lo)/(hi-lo)*side
    image, draw = new_chart(760, 590, 'Прогноз на тестовой выборке',
                           f'R² = {r2:.4f}  ·  RMSE = {rmse:.1f} тыс. руб.')
    for tick in np.linspace(lo, hi, 5):
        x, y = coord_x(tick), coord_y(tick)
        draw.line((x, top, x, top+side), fill='#e8edf4')
        draw.line((left, y, left+side, y), fill='#e8edf4')
        label = f'{tick:,.0f}'.replace(',', ' ')
        draw.text((x, top+side+10), label, fill='#718096', font=chart_font(11), anchor='mt')
        draw.text((left-10, y), label, fill='#718096', font=chart_font(11), anchor='rm')
    draw.line((left, top+side, left+side, top), fill='#ef8d63', width=2)
    draw.text((left, top-25), 'Прогноз, тыс. руб.', fill='#334155', font=chart_font(13))
    draw.text((left+side/2, top+side+42), 'Фактическая стоимость, тыс. руб.',
              fill='#334155', font=chart_font(13), anchor='mt')
    draw.text((left+side+20, top+10), 'Идеальный прогноз', fill='#c76c48', font=chart_font(12))
    layer = Image.new('RGBA', image.size)
    points = ImageDraw.Draw(layer)
    for a, p in zip(actual, predicted) if show_points else []:
        x, y = coord_x(a), coord_y(p)
        radius = 3.4
        points.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(37, 99, 190, 210))
    image.paste(layer, (0, 0), layer)
    return image
