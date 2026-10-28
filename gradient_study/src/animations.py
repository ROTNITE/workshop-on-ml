"""Three optional animations. Static figures and GIFs share the same data."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    from .static_charts import chart_font, comparison_chart, correlation_chart, prediction_chart
except ImportError:
    pass  # Definitions are in the previous cell when embedded in the notebook.


def smooth(t):
    t = float(np.clip(t, 0, 1))
    return t*t*(3-2*t)


def save_animation(frames, path, duration=40):
    """One playback, opaque full-canvas frames, final frame held for 2 seconds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # One palette across all frames avoids flickering colors.
    sample = Image.new('RGB', (frames[0].width, frames[0].height*3))
    for j, i in enumerate([0, len(frames)//2, len(frames)-1]):
        sample.paste(frames[i].convert('RGB'), (0, j*frames[0].height))
    palette = sample.quantize(colors=256)
    encoded = [frame.convert('RGB').quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    encoded[0].save(path, format='GIF', save_all=True, append_images=encoded[1:],
                    duration=[duration]*(len(encoded)-1)+[2000], disposal=1, optimize=False)
    return path


def create_animations(correlation, comparison, actual, predicted, r2, rmse, directory):
    """Same visual composition as the original notebook, recomputed values."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    final = correlation_chart(correlation)
    # Preserve all text; reveal the correlation grid, then its coloring.
    left, top, size, count = 310, 110, 46, len(correlation)
    neutral = final.copy()
    draw = ImageDraw.Draw(neutral)
    for i in range(count):
        for j in range(count):
            x, y = left+j*size, top+i*size
            draw.rounded_rectangle((x, y, x+size-3, y+size-3), radius=6, fill='#f0f3f7')
            draw.text((x+size//2, y+size//2), f'{correlation.iloc[i,j]:.2f}',
                      fill='#526174', font=chart_font(12), anchor='mm')
    base = neutral.copy()
    ImageDraw.Draw(base).rectangle((left, top, left+count*size, top+count*size), fill='white')
    frames = [Image.blend(base, neutral, smooth(t)) for t in np.linspace(0, 1, 24)]
    frames += [neutral.copy() for _ in range(6)]
    frames += [Image.blend(neutral, final, smooth(t)) for t in np.linspace(0, 1, 50)]
    paths = [save_animation(frames, directory/'01_correlations.gif')]

    comparison = comparison.sort_values('RMSE_CV', kind='stable')
    maximum = comparison['RMSE_CV'].max()*1.18
    final = comparison_chart(comparison)
    frames = []
    for step in range(80):
        progress = step/79
        image = final.copy()
        draw = ImageDraw.Draw(image)
        left, top, width = 145, 105, 500
        for i, (_, row) in enumerate(comparison.iterrows()):
            y = top+i*57
            # Clear the old bar and value, keeping names and axes fixed.
            draw.rectangle((left, y-2, 754, y+32), fill='white')
            draw.rounded_rectangle((left, y, left+width, y+30), radius=6, fill='#f3f6fa')
            target_width = width*row['RMSE_CV']/maximum
            current = target_width*smooth((progress-i*.025)/.86)
            if current >= 1:
                draw.rounded_rectangle((left, y, left+current, y+30), radius=min(6, current/2),
                                       fill='#2563eb' if i == 0 else '#a8c6ef')
            fade = smooth((progress-.72)/.28)
            color = tuple(round(255+(c-255)*fade) for c in (38, 52, 73))
            draw.text((left+target_width+10, y+15), f'{row["RMSE_CV"]:.2f}',
                      fill=color, font=chart_font(13, True), anchor='lm')
        frames.append(image)
    frames[-1] = final
    paths.append(save_animation(frames, directory/'02_methods.gif'))

    # Reveal points from left to right. The final canvas is exactly the static plot.
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    final = prediction_chart(actual, predicted, r2, rmse)
    lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    pad = max((hi-lo)*.04, 1.)
    lo, hi = lo-pad, hi+pad
    # Reconstruct the background using the chart renderer with point opacity disabled.
    base = prediction_chart(actual, predicted, r2, rmse, show_points=False)
    rank = np.empty(len(actual))
    rank[np.argsort(actual)] = np.linspace(0, .68, len(actual))
    frames = []
    for step in range(80):
        image = base.copy()
        layer = Image.new('RGBA', image.size)
        draw = ImageDraw.Draw(layer)
        progress = step/79
        for i, (a, p) in enumerate(zip(actual, predicted)):
            opacity = smooth((progress-rank[i])/.32)
            if opacity <= 0:
                continue
            x, y = 110+(a-lo)/(hi-lo)*400, 515-(p-lo)/(hi-lo)*400
            draw.ellipse((x-3.4, y-3.4, x+3.4, y+3.4), fill=(37, 99, 190, round(210*opacity)))
        image.paste(layer, (0, 0), layer)
        frames.append(image)
    frames[-1] = final
    paths.append(save_animation(frames, directory/'03_predictions.gif'))
    return paths
