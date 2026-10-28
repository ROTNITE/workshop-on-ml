"""Independent checks of the executed notebook, exports, charts and animations."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import nbformat
import numpy as np
import pandas as pd
from PIL import Image

root = Path(__file__).resolve().parent
output = root/'outputs'
nb = nbformat.read(root/'regression (2).ipynb',as_version=4)
nbformat.validate(nb)
code = [cell for cell in nb.cells if cell.cell_type=='code']
assert len(code)==7
assert all(cell.execution_count is not None for cell in code)
assert sum(cell.cell_type=='markdown' and cell.source.startswith('## Блок ') for cell in nb.cells)==5
assert not [o for cell in code for o in cell.outputs if o.output_type=='error']
search = pd.read_csv(output/'tables/all_configurations.csv')
assert len(search)==60
assert not search.duplicated(['Метод','Тип шага','lambda']).any()
assert (search.CV_folds==480).all()
folds = pd.read_csv(output/'tables/loocv_predictions.csv')
assert len(folds)==60*480
assert not folds.duplicated(['Конфигурация','fold']).any()
for _,row in search.iterrows():
    group = folds[folds['Конфигурация']==row['Конфигурация']]
    assert (group['Статус']=='diverged').sum()==row.CV_diverged
    if row.CV_diverged==0:
        assert np.isclose(np.sqrt(np.mean(group.error**2)),row.RMSE_CV,rtol=1e-12)
    else:
        assert pd.isna(row.RMSE_CV)
winner = json.loads((output/'selection_before_test.json').read_text(encoding='utf-8'))
eligible = search[search['Допущен']].sort_values(['MSE_val','RMSE_CV','Конфигурация'],kind='stable')
assert winner['winner']['Конфигурация']==eligible.iloc[0]['Конфигурация']
model = json.loads((output/'best_model.json').read_text(encoding='utf-8'))
predictions = pd.read_csv(output/'tables/test_predictions.csv')
truth, prediction = predictions['Фактическая стоимость'],predictions['Прогноз']
rmse = np.sqrt(np.mean((truth-prediction)**2))
assert np.isclose(rmse,model['RMSE_test'],rtol=1e-12)
assert np.isclose(1-np.sum((truth-prediction)**2)/np.sum((truth-truth.mean())**2),model['R2_test'])
original = root/'sources/regression.original.ipynb'
assert hashlib.sha256(original.read_bytes()).hexdigest()=='d1931baad0756bf6b3274c90c78a4f78fd052123f9508739e27a79b3595c13b9'
provenance = json.loads((output/'provenance.json').read_text(encoding='utf-8'))
assert hashlib.sha256((root/'data/regression_data.csv').read_bytes()).hexdigest()==provenance['dataset_sha256']
reference = pd.read_csv(root/'sources/original_figures/correlations.csv',index_col=0)
current = pd.read_csv(output/'tables/correlations_full.csv',index_col=0)
np.testing.assert_array_equal(reference.to_numpy(),current.to_numpy())

animation_checks = []
contact = Image.new('RGB',(805*3,630*3),'#e5e9ef')
for row,name in enumerate(['01_correlations','02_methods','03_predictions']):
    png = Image.open(output/f'figures/{name}.png').convert('RGB')
    with Image.open(output/f'animations/{name}.gif') as animation:
        assert animation.n_frames>2
        assert 'loop' not in animation.info
        for col,frame in enumerate([0,animation.n_frames//2,animation.n_frames-1]):
            animation.seek(frame)
            contact.paste(animation.convert('RGB'),(col*805,row*630))
        last = animation.convert('RGB')
        assert last.size==png.size
        difference = np.abs(np.asarray(last,dtype=float)-np.asarray(png,dtype=float))
        assert difference.mean()<8, (name,difference.mean())
        assert animation.info.get('duration')==2000
        animation_checks.append({'name':name,'frames':animation.n_frames,'size':png.size,
                                 'last_frame_mean_color_error':float(difference.mean())})
contact.save(output/'animation_contact_sheet.png')

with zipfile.ZipFile(output/'results.xlsx') as archive:
    chart_files = [name for name in archive.namelist() if '/charts/' in name and name.endswith('.xml')]
    assert len(chart_files)==1
    xml = ET.fromstring(archive.read(chart_files[0]))
    ns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
    minima = xml.findall('.//c:valAx/c:scaling/c:min',ns)
    assert any(float(node.attrib['val'])==0 for node in minima), 'Bar chart scale must start at zero'
    main_ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    sheet = ET.fromstring(archive.read('xl/worksheets/sheet5.xml'))
    formulas = {cell.attrib['r']:cell.find('s:f',main_ns).text
                for cell in sheet.findall('.//s:c',main_ns) if cell.find('s:f',main_ns) is not None}
    for row in range(6,166):
        assert formulas[f'D{row}']==f'B{row}-C{row}'
        assert formulas[f'F{row}']==f'D{row}^2'
        assert formulas[f'G{row}']==f'ABS(D{row})'
        assert formulas[f'H{row}']==f'(B{row}-E{row})^2'
    assert formulas['K7']=='SQRT(AVERAGE(F6:F165))'
    assert len(ET.fromstring(archive.read('xl/workbook.xml')).find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheets'))==7

report={'notebook':'7 executed code cells: 5 blocks, conclusions, animations; no errors',
        'configurations':60,'loocv_fits':len(folds),'test_rmse':float(rmse),
        'source_and_correlation_unchanged':True,'animations':animation_checks,
        'xlsx':'7 sheets; native chart starts at zero; all 160 residual/square/absolute/baseline formulas verified',
        'not_verified':'Desktop Microsoft Excel and a graphical Jupyter frontend were not launched.'}
(output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
