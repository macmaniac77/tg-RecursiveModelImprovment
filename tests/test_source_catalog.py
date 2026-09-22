import json
import gzip
import subprocess
from pathlib import Path
from tgaoi.source_catalog import parse_source, build_catalog
from tgaoi.viewer import main


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def test_pinned_complete_inventory_and_no_execution(tmp_path):
    git(tmp_path,'init','-q')
    (tmp_path/'examples').mkdir();(tmp_path/'extra/models').mkdir(parents=True)
    (tmp_path/'examples/demo.py').write_text('from tinygrad import nn\nfrom extra.models.net import Net\nraise RuntimeError("must never execute")\nclass Demo:\n def __init__(self):\n  self.fc=nn.Linear(4,2)\n  self.net=Net()\n')
    (tmp_path/'extra/models/net.py').write_text('class Net: pass\n')
    (tmp_path/'examples/run.sh').write_text('echo helper\n')
    git(tmp_path,'add','.')
    git(tmp_path,'-c','user.name=Test','-c','user.email=test@example.com','commit','-qm','fixture')
    pinned=git(tmp_path,'rev-parse','HEAD')
    first=build_catalog(tmp_path,pinned)
    assert first['summary']['files']==3
    assert first['summary']['python_files']==2
    assert first['summary']['calls']=={'api_mapping':1,'source_reference':1,'opaque':1}
    (tmp_path/'examples/demo.py').write_text('invalid python !!!')
    (tmp_path/'examples/untracked.py').write_text('untracked')
    assert build_catalog(tmp_path,pinned)==first
    output=tmp_path/'catalog.json';output.write_text(json.dumps(first))
    viewer=tmp_path/'view.html';root=Path(__file__).resolve().parents[1]
    main(['--package',str(root/'reference_models/rfdetr'),'--examples',str(output),'--output',str(viewer)])
    assert pinned in viewer.read_text()


def test_no_false_mapping_from_names_or_shadowed_import():
    records=parse_source('examples/fake.py','from other import Linear\nfrom tinygrad import nn\ndef f(nn):\n return nn.Linear(2,3)\nLinear(2,3)\nx.softmax()\n')
    calls=[c for s in records for c in s['calls']]
    assert all(c['status']=='opaque' for c in calls)
    assert any(c['candidate_aoi']=='SOFTMAX' for c in calls)


def test_pinned_catalog_has_every_example_and_yolo_library_links():
    root=Path(__file__).resolve().parents[1]
    data=json.loads(gzip.decompress((root/'reference_models/tinygrad/catalog.json.gz').read_bytes()))
    assert data['summary']['python_files']==86
    assert data['summary']['parse_errors']==0
    files={f['path']:f for f in data['files']}
    for path in ['examples/yolov8.py','examples/llama.py','examples/whisper.py','extra/models/resnet.py']:
        assert files[path]['symbols']
    yolo={s['name']:s for s in files['examples/yolov8.py']['symbols']}
    assert yolo['C2f']['library_aoi']=='YOLOV8_C2F'
    assert yolo['Conv_Block']['library_aoi']=='YOLOV8_CONV'


def test_qwen_source_subgraph_links():
    root=Path(__file__).resolve().parents[1]
    data=json.loads(gzip.decompress((root/'reference_models/tinygrad/catalog.json.gz').read_bytes()))
    f=next(f for f in data['files'] if f['path']=='tinygrad/llm/model.py')
    symbols={s['name']:s for s in f['symbols']}
    assert 'GATED_DELTA_SCAN' in symbols['GatedDeltaNetBlock._attention']['library_aois']
    assert symbols['apply_rope']['library_aois']==['ROPE_HALF_SPLIT']
