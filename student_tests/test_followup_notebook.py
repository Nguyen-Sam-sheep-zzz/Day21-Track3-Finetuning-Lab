import datetime
import hashlib
import json
import pathlib
import sys
import types
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'colab/Lab21_Deadline_Followup.ipynb'


def cells():
    return json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']


def harness(tmp_path, *, status='complete', rc=0, seal=True, payload=True, merge_pass=True):
    downloads = []
    class Process:
        def __init__(self, cmd, **kwargs):
            stage = cmd[cmd.index('--stages') + 1]
            out = tmp_path / 'bonus_results'
            out.mkdir(exist_ok=True)
            record = {'stage':stage,'result':{'status':status},'seal_unchanged':seal}
            (out/f'run_status_{stage}.json').write_text(json.dumps(record),encoding='utf-8')
            if payload:
                full = {'n':50,'rows':[{} for _ in range(50)]}
                if stage == 'regression':
                    data = {'status':'complete','n':15,'rows':[{} for _ in range(15)]}
                    (out/'regression_followup.json').write_text(json.dumps(data),encoding='utf-8')
                elif stage == 'b1':
                    (out/'merge_check.json').write_text(json.dumps({'status':'complete','passed':merge_pass,'before':full,'after':full}),encoding='utf-8')
                    (out/'hot_swap.json').write_text(json.dumps({'status':'complete','same_base_identity':'one-base','adapters':{'correct':{**full,'base_identity':'one-base'},'attn_only':{**full,'base_identity':'one-base'}}}),encoding='utf-8')
                else:
                    (out/'rank_sweep.json').write_text(json.dumps({'status':'complete','ranks':{str(r):full for r in (8,16,64)}}),encoding='utf-8')
            self.stdout = iter(['mock child output\n'])
        def wait(self):return rc
    (tmp_path/'followup_helper_receipt.json').write_text('{}',encoding='utf-8')
    ns={'ROOT':tmp_path,'datetime':datetime,'pathlib':pathlib,'zipfile':zipfile,'hashlib':hashlib,'sys':sys,'json':json,'subprocess':types.SimpleNamespace(Popen=Process,PIPE=-1,STDOUT=-2),'files':types.SimpleNamespace(download=downloads.append)}
    exec(''.join(cells()[5]['source']),ns)
    return ns,downloads


def test_window_is_relative_and_admits_future_execution(tmp_path):
    before=datetime.datetime.now(datetime.timezone.utc)
    ns,_=harness(tmp_path)
    stop=datetime.datetime.fromisoformat(ns['STOP_AT'].replace('Z','+00:00'))
    assert stop > before + datetime.timedelta(minutes=110)
    assert stop < before + datetime.timedelta(minutes=130)


def test_each_setup_uses_a_new_attempt_directory():
    prefix=''.join(cells()[1]['source']).split("SOURCE=")[0]
    first={};second={}
    exec(prefix,first);exec(prefix,second)
    assert first['ROOT'] != second['ROOT']


@pytest.mark.parametrize('stage',['regression','b1','b4'])
def test_complete_real_artifact_contract_downloads_then_returns(tmp_path,stage):
    ns,downloads=harness(tmp_path)
    ns['run_stage'](stage)
    assert len(downloads)==1
    with zipfile.ZipFile(downloads[0]) as z:assert z.testzip() is None


@pytest.mark.parametrize('rc',[0,2])
def test_incomplete_status_is_visible_error_but_evidence_is_downloaded(tmp_path,rc):
    ns,downloads=harness(tmp_path,status='incomplete',rc=rc,payload=False)
    with pytest.raises(RuntimeError):ns['run_stage']('regression')
    assert len(downloads)==1


def test_missing_measurement_cannot_be_called_complete(tmp_path):
    ns,downloads=harness(tmp_path,payload=False)
    with pytest.raises(RuntimeError):ns['run_stage']('regression')
    assert len(downloads)==1


def test_changed_core_seal_is_visible_error(tmp_path):
    ns,downloads=harness(tmp_path,seal=False)
    with pytest.raises(RuntimeError):ns['run_stage']('regression')
    assert len(downloads)==1


def test_merge_gate_failure_cannot_earn_completion_label(tmp_path):
    ns,downloads=harness(tmp_path,merge_pass=False)
    with pytest.raises(RuntimeError):ns['run_stage']('b1')
    assert len(downloads)==1
