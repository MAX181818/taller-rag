"""Pruebas del gold y del pooling sin servicios."""
import copy
import csv
import json
from pathlib import Path

import pytest

from scripts import pool_candidates as pool
from scripts import validate_gold as vg


@pytest.fixture
def gold12():
    rows = []
    for split in ('dev', 'test'):
        for kind in vg.TYPES:
            empty = kind == 'sin_respuesta'
            rows.append(dict(qid=f'Q{len(rows)+1:02}', split=split,
                             question='¿Qué ocurrió?', language='es' if split == 'dev' else 'en',
                             type=kind, collection_target='ambas', relevant=[] if empty else [
                                 dict(document_id='bluesky_posts:a', grade=2,
                                      evidence_quote='Hola @usuario', evidence_span=[0, 11])],
                             reference_url='', reference_answer='', answer_criterion='criterio',
                             expected_abstention=empty, status='borrador'))
    return rows


@pytest.fixture
def gold(gold12):
    rows = copy.deepcopy(gold12)
    for i in range(28):
        row = copy.deepcopy(gold12[i % 12])
        row['qid'] = f'Q{i+13:02}'
        rows.append(row)
    return rows


def test_valid_and_minimum(gold, gold12):
    assert vg.validate(gold, {'bluesky_posts:a'}, set(), {'bluesky_posts:a': 'Hola @x.com!'}) == []
    errors = vg.validate(gold12, {'bluesky_posts:a'}, set())
    assert len(errors) == 1 and '40' in errors[0]


@pytest.mark.parametrize('change,fragment', [
    (lambda g: g[1].update(qid=g[0]['qid']), 'duplicado'),
    (lambda g: [r.update(type='directa') for r in g if r['split'] == 'test'], 'categoría'),
    (lambda g: g[0]['relevant'][0].update(document_id='desconocido'), 'desconocido'),
    (lambda g: g[0]['relevant'][0].update(evidence_quote='x'*301), '300'),
    (lambda g: g[0].update(type='sin_respuesta'), 'sin_respuesta'),
    (lambda g: g[0]['relevant'][0].update(grade=0), 'grade'),
    (lambda g: g[0].pop('question'), 'question'),
    (lambda g: g[0].update(status='otro'), 'status'),
    (lambda g: [r.update(language='es') for r in g], 'idiomas'),
    (lambda g: [r.update(collection_target='bluesky_posts') for r in g], 'colecciones'),
    (lambda g: [r.update(split='dev') for r in g], 'splits'),
])
def test_rules(gold, change, fragment):
    change(gold)
    assert any(fragment in e for e in vg.validate(gold, {'bluesky_posts:a'}, set()))


@pytest.mark.parametrize('span,quote', [([0, 4], 'otro'), ([-1, 3], 'Hola'), ([0, 99], 'Hola')])
def test_spans(gold, span, quote):
    gold[0]['relevant'][0].update(evidence_span=span, evidence_quote=quote)
    assert any('evidence' in e for e in vg.validate(gold, {'bluesky_posts:a'}, set(), {'bluesky_posts:a': 'Hola @x.com!'}))


def test_mask_and_kappa():
    assert vg.mask_handles('Hola @juan.bsky.social y @x.com, @local') == 'Hola @usuario y @usuario, @local'
    assert vg.weighted_kappa([0, 1, 2], [0, 1, 2]) == 1
    assert vg.weighted_kappa([2, 2], [2, 2]) == 1
    # Marginales=(1,1,2); suma ponderada=22/64, observado=2/16.
    expected = 1 - (2/16) / (22/64)
    assert vg.weighted_kappa([0, 1, 2, 2], [0, 2, 2, 1]) == pytest.approx(expected)


@pytest.mark.parametrize('blind', [False, True])
def test_pool(tmp_path, gold12, monkeypatch, blind):
    gold12[0]['relevant'].append(dict(document_id='missing', grade=1, evidence_quote='prueba'))
    path = tmp_path / 'gold.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in gold12), encoding='utf-8')
    calls = []
    def fake(query, **kwargs):
        calls.append(kwargs)
        ids = ['a', 'b'] if kwargs['backend'] == 'solr' else ['b', 'c']
        return [dict(document_id=i, text='Hola\n@x.com', title=i, url='u') for i in ids]
    monkeypatch.setattr(pool, 'retrieve', fake)
    args = ['--gold', str(path), '--out-dir', str(tmp_path), '--qids', 'Q01', '--k', '2']
    assert pool.main(args + (['--blind'] if blind else [])) == 0
    out = tmp_path / ('pool_dev_ciego.csv' if blind else 'pool_dev.csv')
    assert out.read_bytes().startswith(b'\xef\xbb\xbf')
    with out.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        expected = ['qid','question','document_id','rank_solr','rank_milvus','title','snippet','url','grade_gold','grade_humano']
        assert reader.fieldnames == [c for c in expected if not (blind and c == 'grade_gold')]
        rows = {r['document_id']: r for r in reader}
    assert set(rows) == {'a', 'b', 'c', 'bluesky_posts:a', 'missing'}
    assert (rows['b']['rank_solr'], rows['b']['rank_milvus']) == ('2', '1')
    assert rows['missing']['rank_solr'] == rows['missing']['rank_milvus'] == ''
    assert rows['a']['snippet'] == 'Hola @usuario'
    assert all(c['one_per_doc'] and c['k'] == 2 for c in calls)
    if not blind:
        assert rows['missing']['grade_gold'] == '1'


def test_pool_failure(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'g.jsonl'
    path.write_text(json.dumps(dict(qid='Q01', question='q', split='dev', relevant=[])), encoding='utf-8')
    def fail(*args, **kwargs):
        raise RuntimeError('caído')
    monkeypatch.setattr(pool, 'retrieve', fail)
    assert pool.main(['--gold', str(path), '--out-dir', str(tmp_path)]) == 1
    assert 'solr' in capsys.readouterr().err


def test_agreement(tmp_path, monkeypatch, capsys):
    paths = [tmp_path / n for n in ('a.csv', 'b.csv')]
    for path, grades in zip(paths, ([0,1,2,2], [0,2,2,1])):
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['qid','document_id','grade_humano'])
            writer.writerows([['Q01', str(i), g] for i,g in enumerate(grades)])
    monkeypatch.chdir(tmp_path)
    assert vg.main(['--agreement', *map(str, paths)]) == 0
    assert '0.636364' in capsys.readouterr().out
    with (tmp_path / 'eval/annotations/desacuerdos.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and all(r['resolucion'] == '' for r in rows)


def test_validation_cli(tmp_path, gold, capsys):
    paths = {name: tmp_path / name for name in ('gold.jsonl', 'sample.csv', 'synthetic.jsonl', 'texts.jsonl')}
    paths['gold.jsonl'].write_text('\n'.join(json.dumps(r) for r in gold), encoding='utf-8')
    paths['sample.csv'].write_text('document_id\nbluesky_posts:a\n', encoding='utf-8')
    paths['synthetic.jsonl'].write_text('', encoding='utf-8')
    paths['texts.jsonl'].write_text(json.dumps(dict(document_id='bluesky_posts:a',
                                                   text='Hola @x.com!', created_at='2024-04-15T12:00:00Z')), encoding='utf-8')
    args = ['--gold', str(paths['gold.jsonl']), '--sample', str(paths['sample.csv']),
            '--synthetic', str(paths['synthetic.jsonl']), '--texts', str(paths['texts.jsonl'])]
    assert vg.main(args) == 0
    output = capsys.readouterr().out
    assert all(s in output for s in ('40', '2024', '2024-04', 'status', 'collection_target'))
    paths['gold.jsonl'].write_text('{}', encoding='utf-8')
    assert vg.main(args) == 1


@pytest.mark.parametrize('a,b', [([], []), ([0], [0, 1]), ([3], [0])])
def test_kappa_invalid(a, b):
    with pytest.raises(ValueError):
        vg.weighted_kappa(a, b)


def test_invalid_split_returns_errors(gold):
    gold[0]['split'] = []
    assert any('split' in e for e in vg.validate(gold, {'bluesky_posts:a'}, set()))


def test_repository_gold():
    root = Path(__file__).resolve().parents[1]
    path = root / 'eval/gold.jsonl'
    if not path.exists():
        pytest.skip('No existe eval/gold.jsonl')
    sample = vg.load_sample_ids(root / 'data/samples/sample_700.csv')
    synthetic = {r['document_id'] for r in vg.load_gold(root / 'data/synthetic/injection_docs.jsonl')}
    assert vg.validate(vg.load_gold(path), sample, synthetic) == []


def test_mask_handles_covers_bluesky_profile_urls():
    from scripts.validate_gold import mask_handles
    assert mask_handles("https://bsky.app/profile/ana.bsky.social/post/3abc") == "https://bsky.app/profile/usuario/post/3abc"
    assert mask_handles("hola @ana.bsky.social") == "hola @usuario"
    assert mask_handles("https://www.eltiempo.com/x") == "https://www.eltiempo.com/x"
