"""Genera hojas de anotación con candidatos de ambos motores."""
import argparse
import csv
from pathlib import Path
import sys

from scripts.validate_gold import load_gold, mask_handles

COLUMNS = ['qid', 'question', 'document_id', 'rank_solr', 'rank_milvus',
           'title', 'snippet', 'url', 'grade_gold', 'grade_humano']


def retrieve(*args, **kwargs):
    from app.retrieval import retrieve as implementation
    return implementation(*args, **kwargs)


def _pool(question, k, chunker):
    candidates, best_ranks = {}, {}
    for backend in ('solr', 'milvus'):
        try:
            hits = retrieve(question['question'], backend=backend, k=k,
                            chunker=chunker, one_per_doc=True)
            for rank, hit in enumerate(hits[:k], 1):
                doc = hit['document_id']
                if doc not in candidates:
                    candidates[doc] = dict(qid=question['qid'], question=question['question'],
                                           document_id=doc, rank_solr='', rank_milvus='',
                                           title='', snippet='', url='', grade_gold='', grade_humano='')
                row = candidates[doc]
                if row[f'rank_{backend}'] == '':
                    row[f'rank_{backend}'] = rank
                if rank < best_ranks.get(doc, float('inf')):
                    best_ranks[doc] = rank
                    row.update(title=hit.get('title') or '', url=mask_handles(hit.get('url') or ''),
                               snippet=mask_handles(' '.join(str(hit.get('text') or '').split()))[:300])
        except Exception as error:
            raise RuntimeError(f"Error del motor {backend} para {question['qid']}: {error}") from error
    for rel in question['relevant']:
        doc = rel['document_id']
        if doc not in candidates:
            candidates[doc] = dict(qid=question['qid'], question=question['question'],
                                   document_id=doc, rank_solr='', rank_milvus='', title='',
                                   snippet=mask_handles(' '.join(rel.get('evidence_quote', '').split()))[:300],
                                   url=question.get('reference_url', ''), grade_humano='')
        candidates[doc]['grade_gold'] = rel['grade']
    return list(candidates.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=Path('eval/annotations'))
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--chunker', default='tok256')
    parser.add_argument('--qids')
    parser.add_argument('--blind', action='store_true')
    args = parser.parse_args(argv)
    if args.k <= 0:
        parser.error('--k debe ser positivo.')
    try:
        gold = load_gold(args.gold)
        selected = {q.strip() for q in args.qids.split(',')} if args.qids else None
        if selected and selected - {q['qid'] for q in gold}:
            raise ValueError('Hay qids solicitados que no existen en el gold.')
        by_split = {'dev': [], 'test': []}
        for question in gold:
            if selected is None or question['qid'] in selected:
                by_split[question['split']].extend(_pool(question, args.k, args.chunker))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        columns = [c for c in COLUMNS if not (args.blind and c == 'grade_gold')]
        for split, rows in by_split.items():
            out = args.out_dir / f"pool_{split}{'_ciego' if args.blind else ''}.csv"
            with out.open('w', encoding='utf-8-sig', newline='') as target:
                writer = csv.DictWriter(target, fieldnames=columns, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)
            print(f'{out}: {len(rows)} candidatos')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
