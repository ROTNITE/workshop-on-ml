"""Run every notebook cell using this exact Python environment."""
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
os.environ['JUPYTER_PATH'] = str(Path(sys.prefix)/'share/jupyter')
from nbclient import NotebookClient
import nbformat

path = root/'regression (2).ipynb'
notebook = nbformat.read(path,as_version=4)
client = NotebookClient(notebook,timeout=3600,kernel_name='workshop-ml',
                        resources={'metadata':{'path':str(root)}})
def finished(cell, cell_index, **kwargs):
    print(f'Executed cell {cell_index}: {cell.cell_type}',flush=True)
client.on_cell_executed = finished
try:
    client.execute()
finally:
    # Preserve a failed execution too, so errors cannot be hidden.
    nbformat.write(notebook,path)
print('Notebook executed successfully',flush=True)
