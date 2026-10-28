import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = path.dirname(fileURLToPath(import.meta.url));
const output = path.join(root, 'outputs');
const records = JSON.parse(await fs.readFile(path.join(output,'workbook_data.json'),'utf8'));
const model = JSON.parse(await fs.readFile(path.join(output,'best_model.json'),'utf8'));
const wb = Workbook.create();
const names = ['Итог','Все варианты','Значимость','Корреляции','Прогноз test','Отбор LOOCV','Данные'];
const sheets = Object.fromEntries(names.map(name=>[name,wb.worksheets.add(name)]));
function column(n){let s='';for(n++;n>0;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;}
function cells(table){return [table.columns,...table.data];}
function objects(table){return table.data.map(row=>Object.fromEntries(table.columns.map((key,i)=>[key,row[i]])));}
function text(sheet,cell,value,size=11){sheet.getRange(cell).values=[[value]];sheet.getRange(cell).format.font={name:'Arial',size,color:'#17263C'};}
function table(sheet,row,matrix,widths){
  const end=column(matrix[0].length-1), last=row+matrix.length-1;
  const range=sheet.getRange(`A${row}:${end}${last}`);
  range.values=matrix;
  range.format.font={name:'Arial',size:11,color:'#17263C'};
  range.format.rowHeight=23;
  range.format.verticalAlignment='center';
  sheet.getRange(`A${row}:${end}${row}`).format={fill:'#20364E',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},
    horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,rowHeight:44};
  for(let i=0;i<matrix[0].length;i++){
    const col=column(i);
    sheet.getRange(`${col}${row}:${col}${last}`).format.columnWidth=widths?.[i]??20;
    if(matrix.slice(1).some(r=>typeof r[i]==='number'))sheet.getRange(`${col}${row+1}:${col}${last}`).setNumberFormat('0.0000');
  }
  for(let i=1;i<matrix.length;i+=2)sheet.getRange(`A${row+i}:${end}${row+i}`).format.fill='#F3F6FA';
  return last;
}
for(const sheet of Object.values(sheets)){sheet.showGridLines=false;sheet.getRange('A1').format.rowHeight=10;}

const summary=sheets['Итог'];summary.tabColor='#20364E';
text(summary,'A2','Градиентный спуск: итог исследования',16);
text(summary,'A3','Выбор по MSE validation. LOOCV — повторный отбор и обучение внутри каждого fold.');
const best=objects(records.best_by_method);
table(summary,5,[['Метод','Шаг','RMSE val, тыс. руб.','LOOCV, тыс. руб.','R² val','Обновления'],
  ...best.map(r=>[r['Метод'],r['Шаг'],r.RMSE_val,r.RMSE_CV,r.R2_val,r['Итерации']])],[18,32,23,22,16,18]);
summary.getRange('F6:F10').setNumberFormat('0');
text(summary,'A13',`Победитель: ${model.method}, ${model.step}`,13);
table(summary,15,[['Test','Значение','Единица'],['R²',null,'доля'],['RMSE',null,'тыс. руб.'],['MAE',null,'тыс. руб.']],[18,32,23]);
summary.getRange('B16:B18').formulas=[["='Прогноз test'!K6"],["='Прогноз test'!K7"],["='Прогноз test'!K8"]];
summary.getRange('B16:B18').setNumberFormat('0.000000');
const chart=summary.charts.add('bar',[summary.getRange('A5:A10'),summary.getRange('D5:D10')]);
chart.title='LOOCV лучших по validation конфигураций';chart.titleTextStyle.typeface='Arial';chart.titleTextStyle.fontSize=14;
chart.hasLegend=false;chart.setPosition('A21','F37');
chart.xAxis={axisType:'textAxis',textStyle:{typeface:'Arial',fontSize:11}};
chart.yAxis={min:0,max:650,numberFormatCode:'0',numberFormatSourceLinked:false,textStyle:{typeface:'Arial'}};
chart.series.items[0].fill='#2563EB';

text(sheets['Все варианты'],'A2','Все 60 конфигураций',16);
text(sheets['Все варианты'],'A3','Статус max_iter — бюджет исчерпан; diverged — численная расходимость. Пустая метрика недоступна.');
const configColumns=['Метод','Тип шага','lambda','Шаг','R2_train','R2_val','RMSE_val','RMSE_CV','MSE_train','MSE_val','Итерации','Статус','CV_diverged','CV_max_iter','Допущен'];
table(sheets['Все варианты'],5,[configColumns,...objects(records.all_configurations).map(r=>configColumns.map(c=>r[c]))],
      [18,18,14,32,16,16,20,20,22,22,18,18,18,18,16]);
sheets['Все варианты'].getRange('C6:C65').setNumberFormat('0.00000');
sheets['Все варианты'].getRange('K6:K65').setNumberFormat('0');
sheets['Все варианты'].getRange('M6:N65').setNumberFormat('0');
sheets['Все варианты'].freezePanes.freezeRows(5);

text(sheets['Значимость'],'A2','Проверки модели и коэффициентов на train',16);
text(sheets['Значимость'],'A3','α=0.05. t — двусторонняя проверка; p после отбора имеют разведочный смысл.');
const statCols=['Признак','Коэффициент','Стандартная ошибка','t','p_value','Значим'];
let last=table(sheets['Значимость'],5,[statCols,...objects(records.significance_full).map(r=>statCols.map(c=>r[c]))],[42,24,26,22,20,15]);
sheets['Значимость'].getRange(`E6:E${last}`).setNumberFormat('0.000E+00');
text(sheets['Значимость'],`A${last+3}`,'После отбора',13);
let start=last+5;
last=table(sheets['Значимость'],start,[statCols,...objects(records.significance_selected).map(r=>statCols.map(c=>r[c]))],[42,24,26,22,20,15]);
sheets['Значимость'].getRange(`E${start+1}:E${last}`).setNumberFormat('0.000E+00');
text(sheets['Значимость'],`A${last+3}`,'Значимость модели в целом',13);
const modelCols=['Этап','n','k','R2_train','F','F_critical','p_value'];
table(sheets['Значимость'],last+5,[modelCols,...objects(records.model_significance).map(r=>modelCols.map(c=>r[c]))],[42,12,12,20,24,22,20]);
for(const [i,width] of [42,25,28,22,24,22,22].entries())
  sheets['Значимость'].getRange(`${column(i)}5:${column(i)}${last+8}`).format.columnWidth=width;

text(sheets['Корреляции'],'A2','Парные корреляции на train',16);
text(sheets['Корреляции'],'A3','До отбора: исходные числовые признаки. Порог избыточности |r| ≥ 0.85.');
const corr=cells(records.correlations_full);const labels=corr[0];
table(sheets['Корреляции'],5,[['Признак',...labels.slice(1).map((_,i)=>String(i+1))],...corr.slice(1).map((row,i)=>[`${i+1}. ${row[0]}`,...row.slice(1)])],[44,...labels.slice(1).map(()=>12)]);
sheets['Корреляции'].getRange('B6:K15').setNumberFormat('0.00');
sheets['Корреляции'].getRange('B6:K15').conditionalFormats.add('colorScale',{colors:['#D77673','#FFFFFF','#458FDE'],thresholds:[{type:'num',value:-1},{type:'num',value:0},{type:'num',value:1}]});
text(sheets['Корреляции'],'A18','После отбора',13);
table(sheets['Корреляции'],20,cells(records.correlations_selected),[44,24,24,24,30,26]);
sheets['Корреляции'].getRange('B21:F25').setNumberFormat('0.000');
for(let i=1;i<=10;i++)sheets['Корреляции'].getRange(`${column(i)}5:${column(i)}25`).format.columnWidth=16;
sheets['Корреляции'].getRange('A20:F20').format.rowHeight=75;

const predictions=sheets['Прогноз test'];
text(predictions,'A2','Прогнозы победителя на test',16);
text(predictions,'A3','Стоимость и ошибки в тыс. руб. Метрики справа вычисляются из этих прогнозов.');
const rows=objects(records.test_predictions),n=rows.length,lastRow=5+n;
table(predictions,5,[['ID','Фактическая стоимость','Прогноз','Ошибка','Среднее train','Ошибка²','|Ошибка|','Ошибка среднего²'],
   ...rows.map(r=>[r['ID_объекта'],r['Фактическая стоимость'],r['Прогноз'],null,r['Константный прогноз'],null,null,null])],[14,25,23,22,24,25,22,28]);
predictions.getRange(`A6:A${lastRow}`).setNumberFormat('0');
for(let r=6;r<=lastRow;r++){
  predictions.getRange(`D${r}`).formulas=[[`=B${r}-C${r}`]];
  predictions.getRange(`F${r}:H${r}`).formulas=[[`=D${r}^2`,`=ABS(D${r})`,`=(B${r}-E${r})^2`]];
}
predictions.getRange('J5:K9').values=[['Метрика','Значение'],['R²',null],['RMSE, тыс. руб.',null],['MAE, тыс. руб.',null],['RMSE среднего train',null]];
predictions.getRange('J5:K9').format.font={name:'Arial',size:11,color:'#17263C'};
predictions.getRange('J5:K9').format.columnWidth=28;
predictions.getRange('K6:K9').formulas=[
  [`=1-SUM(F6:F${lastRow})/(SUMSQ(B6:B${lastRow})-SUM(B6:B${lastRow})^2/COUNT(B6:B${lastRow}))`],
  [`=SQRT(AVERAGE(F6:F${lastRow}))`],[`=AVERAGE(G6:G${lastRow})`],[`=SQRT(AVERAGE(H6:H${lastRow}))`]];
predictions.getRange('K6:K9').setNumberFormat('0.000000');
predictions.getRange(`D6:H${lastRow}`).setNumberFormat('0.0000');
predictions.freezePanes.freezeRows(5);

text(sheets['Отбор LOOCV'],'A2','Устойчивость отбора признаков',16);
text(sheets['Отбор LOOCV'],'A3','Отбор заново выполняется в каждом из 480 LOOCV folds.');
table(sheets['Отбор LOOCV'],5,cells(records.feature_selection_stability),[42,30,20]);
sheets['Отбор LOOCV'].getRange('B6:B10').setNumberFormat('0');
sheets['Отбор LOOCV'].getRange('C6:C10').setNumberFormat('0.0%');
text(sheets['Отбор LOOCV'],'A14','Удалённые признаки на полном train',13);
table(sheets['Отбор LOOCV'],16,cells(records.removed_features),[42,34,22]);

const rawCsv=await fs.readFile(path.join(root,'data','regression_data.csv'),'utf8');
// Original dataset has no quoted commas; numeric values are converted explicitly.
const raw=rawCsv.trim().split(/\r?\n/).map(line=>line.split(','));
const typed=[raw[0],...raw.slice(1).map(row=>row.map(value=>value!==''&&Number.isFinite(Number(value))?Number(value):value))];
text(sheets['Данные'],'A2','Исходные наблюдения',16);
text(sheets['Данные'],'A3','Источник: regression_data.csv, 800 квартир. Лекции 1–3 и ROTNITE/workshop-on-ml, commit 60d25da.');
table(sheets['Данные'],5,typed,raw[0].map((_,i)=>i===10?28:24));
sheets['Данные'].getRange('A6:A805').setNumberFormat('0');
for(const col of ['D','E','F','I','J'])sheets['Данные'].getRange(`${col}6:${col}805`).setNumberFormat('0');
for(const col of ['B','C','G','M'])sheets['Данные'].getRange(`${col}6:${col}805`).setNumberFormat('0.0');
sheets['Данные'].getRange('H6:H805').setNumberFormat('0');
sheets['Данные'].freezePanes.freezeRows(5);

wb.recalculate();
const computed=predictions.getRange('K6:K9').values.map(r=>r[0]);
const expected=[model.R2_test,model.RMSE_test,model.MAE_test];
for(let i=0;i<expected.length;i++)if(Math.abs(computed[i]-expected[i])>1e-7)throw Error(`Metric ${i} mismatch: ${computed[i]} vs ${expected[i]}`);
const saved=predictions.getRange('B6').values[0][0];
predictions.getRange('B6').values=[[saved+100]];wb.recalculate();
if(predictions.getRange('K7').values[0][0]===computed[1])throw Error('RMSE formula did not react to edited observation');
predictions.getRange('B6').values=[[saved]];wb.recalculate();
const errorScan=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!',options:{useRegex:true,maxResults:30},maxChars:2000});
console.log(errorScan.ndjson);
const previewRanges={'Итог':'A1:F38','Все варианты':'A1:H12','Значимость':'A1:F24','Корреляции':'A1:K16','Прогноз test':'A1:K12','Отбор LOOCV':'A1:C12','Данные':'A1:F12'};
await fs.mkdir(path.join(output,'workbook_previews'),{recursive:true});
for(const [sheetName,range] of Object.entries(previewRanges)){
  const preview=await wb.render({sheetName,range,scale:1,format:'png'});
  await fs.writeFile(path.join(output,'workbook_previews',`${sheetName}.png`),new Uint8Array(await preview.arrayBuffer()));
}
for(const [filename,sheetName,range] of [['Значимость_после_отбора','Значимость','A25:G41'],['Корреляции_после_отбора','Корреляции','A18:F26']]){
  const preview=await wb.render({sheetName,range,scale:1,format:'png'});
  await fs.writeFile(path.join(output,'workbook_previews',`${filename}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(wb);
await file.save(path.join(output,'results.xlsx'));
console.log(JSON.stringify({sheets:names,metrics:computed,formulasRecalculate:true,output:'results.xlsx'}));
