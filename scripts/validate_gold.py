"""Valida el gold y compara anotaciones de relevancia."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys

TYPES = ('directa', 'comparativa', 'temporal', 'ambigua', 'sin_respuesta', 'inyeccion')
ENUMS = dict(split=('dev', 'test'), language=('es', 'en', 'otro'), type=TYPES,
             collection_target=('bluesky_posts', 'url_crawls', 'ambas', 'ninguna'),
             status=('borrador', 'verificada'))
REQUIRED = (*ENUMS, 'qid', 'question', 'relevant', 'reference_url',
            'reference_answer', 'answer_criterion', 'expected_abstention')


def mask_handles(s):
    # Cubre @handles y el perfil dentro de los enlaces de Bluesky (bsky.app/profile/<handle>/...).
    s = re.sub(r'@[\w-]+(?:\.[\w-]+)+', '@usuario', s)
    return re.sub(r'(bsky\.app/profile/)[^/\s"]+', r'\1usuario', s)


def load_gold(path):
    rows = []
    with Path(path).open(encoding='utf-8-sig') as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('Se esperaba un objeto')
                rows.append(row)
            except ValueError as error:
                raise ValueError(f'Línea {number}: {error}') from error
    return rows


def load_sample_ids(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as source:
        return {r['document_id'] for r in csv.DictReader(source)}


def validate(gold, sample_ids, synthetic_ids, texts=None):
    errors = []
    if len(gold) < 40:
        errors.append('Se requieren al menos 40 preguntas.')
    seen, languages, collections, splits = set(), set(), set(), set()
    categories = {s: set() for s in ('dev', 'test')}
    for number, row in enumerate(gold, 1):
        if not isinstance(row, dict):
            errors.append(f'Pregunta {number}: se esperaba un objeto.')
            continue
        prefix = f"Pregunta {row.get('qid', number)}"
        def error(message):
            errors.append(f'{prefix}: {message}')
        for field in REQUIRED:
            if field not in row:
                error(f'Falta el campo {field}.')
        for field, options in ENUMS.items():
            if row.get(field) not in options:
                error(f'Valor inválido de {field}.')
        for field in ('qid', 'question', 'reference_url', 'reference_answer', 'answer_criterion'):
            if not isinstance(row.get(field), str):
                error(f'{field} debe ser texto.')
        qid = row.get('qid')
        if isinstance(qid, str):
            if not re.fullmatch(r'Q\d{2,}', qid):
                error('Formato de qid inválido.')
            if qid in seen:
                error('qid duplicado.')
            seen.add(qid)
        split, kind = row.get('split'), row.get('type')
        if isinstance(split, str) and split in categories:
            splits.add(split)
            if kind in TYPES:
                categories[split].add(kind)
        if row.get('language') in ENUMS['language']:
            languages.add(row['language'])
        target = row.get('collection_target')
        if target == 'ambas':
            collections.update(('bluesky_posts', 'url_crawls'))
        elif target in ('bluesky_posts', 'url_crawls'):
            collections.add(target)
        relevant = row.get('relevant')
        if not isinstance(row.get('expected_abstention'), bool):
            error('expected_abstention debe ser booleano.')
        if not isinstance(relevant, list):
            error('relevant debe ser una lista.')
            continue
        if ((kind == 'sin_respuesta') != (relevant == []) or
                (kind == 'sin_respuesta' and row.get('expected_abstention') is not True)):
            error('sin_respuesta requiere relevant vacío y expected_abstention=true; los demás tipos requieren relevantes.')
        for rel in relevant:
            if not isinstance(rel, dict):
                error('Relevante inválido.')
                continue
            doc = rel.get('document_id')
            if not isinstance(doc, str) or doc not in sample_ids | synthetic_ids:
                error(f'document_id desconocido: {doc}.')
            if isinstance(doc, str) and doc.split(':')[0] in ('bluesky_posts', 'url_crawls'):
                collections.add(doc.split(':')[0])
            if type(rel.get('grade')) is not int or rel['grade'] not in (1, 2):
                error('grade debe ser 1 o 2.')
            quote = rel.get('evidence_quote')
            if not isinstance(quote, str):
                error('evidence_quote debe ser texto.')
            elif len(quote) > 300:
                error('evidence_quote supera 300 caracteres.')
            elif mask_handles(quote) != quote:
                error('evidence_quote contiene handles sin enmascarar.')
            span = rel.get('evidence_span')
            valid_span = (isinstance(span, list) and len(span) == 2 and
                          all(type(n) is int for n in span) and 0 <= span[0] < span[1])
            if not valid_span:
                error('evidence_span inválido.')
            if texts is not None:
                text = texts.get(doc) if isinstance(doc, str) else None
                if not isinstance(text, str):
                    error(f'Falta texto para evidence_span de {doc}.')
                elif valid_span and (span[1] > len(text) or mask_handles(text[span[0]:span[1]]) != quote):
                    error('evidence_span no coincide con evidence_quote o excede el texto.')
    if splits != {'dev', 'test'}:
        errors.append('Deben estar presentes ambos splits: dev y test.')
    for split, kinds in categories.items():
        for kind in TYPES:
            if kind not in kinds:
                errors.append(f'Falta categoría {kind} en split {split}.')
    if len(languages) < 2:
        errors.append('Se requieren al menos dos idiomas.')
    if not {'bluesky_posts', 'url_crawls'} <= collections:
        errors.append('Deben cubrirse ambas colecciones: bluesky_posts y url_crawls.')
    return errors


def weighted_kappa(a, b, labels=(0, 1, 2)):
    if len(a) != len(b) or not a:
        raise ValueError('Se requieren listas no vacías de igual longitud.')
    if len(set(labels)) != len(labels) or len(labels) < 2:
        raise ValueError('Se requieren al menos dos etiquetas distintas.')
    if any(x not in labels for x in a + b):
        raise ValueError('Calificación fuera de las etiquetas permitidas.')
    indices = {label: i for i, label in enumerate(labels)}
    weight = lambda x, y: (indices[x] - indices[y]) ** 2 / (len(labels) - 1) ** 2
    observed = sum(weight(x, y) for x, y in zip(a, b)) / len(a)
    ca, cb = Counter(a), Counter(b)
    expected = sum(weight(x, y) * ca[x] * cb[y] for x in labels for y in labels) / len(a) ** 2
    return 1.0 if expected == 0 else 1 - observed / expected


def _annotations(path):
    result = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as source:
        for row in csv.DictReader(source):
            key = row['qid'], row['document_id']
            grade = int(row['grade_humano'])
            if grade not in (0, 1, 2) or key in result:
                raise ValueError(f'Anotación inválida o duplicada: {key}')
            result[key] = grade
    return result


def _agreement(paths):
    a, b = map(_annotations, paths)
    common = sorted(a.keys() & b.keys())
    kappa = weighted_kappa([a[k] for k in common], [b[k] for k in common])
    matrix = [[0] * 3 for _ in range(3)]
    disagreements = []
    for key in common:
        matrix[a[key]][b[key]] += 1
        if a[key] != b[key]:
            disagreements.append((*key, a[key], b[key], ''))
    out = Path('eval/annotations/desacuerdos.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8-sig', newline='') as target:
        writer = csv.writer(target)
        writer.writerow(['qid', 'document_id', 'grade_A', 'grade_B', 'resolucion'])
        writer.writerows(disagreements)
    print(f'Kappa cuadrático de Cohen: {kappa:.6f}; pares comunes: {len(common)}')
    print('Matriz de confusión (filas A, columnas B; 0/1/2):')
    for row in matrix:
        print(row)
    print(f'Desacuerdos ({len(disagreements)}):')
    for row in disagreements:
        print(*row[:4], sep=', ')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', type=Path)
    parser.add_argument('--sample', type=Path)
    parser.add_argument('--synthetic', type=Path)
    parser.add_argument('--texts', type=Path)
    parser.add_argument('--agreement', nargs=2)
    args = parser.parse_args(argv)
    if not args.agreement and not all((args.gold, args.sample, args.synthetic)):
        parser.error('Se requieren --gold, --sample y --synthetic.')
    try:
        if args.agreement:
            _agreement(args.agreement)
            return 0
        gold = load_gold(args.gold)
        docs = load_gold(args.texts) if args.texts else []
        texts = {r['document_id']: r['text'] for r in docs} if args.texts else None
        errors = validate(gold, load_sample_ids(args.sample),
                          {r['document_id'] for r in load_gold(args.synthetic)}, texts)
        print(f'Preguntas: {len(gold)}')
        for field in ('type', 'split', 'language', 'collection_target', 'status'):
            print(f'{field}: {dict(Counter(str(r.get(field)) for r in gold))}')
        relevant = {r.get('document_id') for q in gold for r in q.get('relevant', []) if isinstance(r, dict)}
        dates = [str(d.get('created_at', '')) for d in docs if d['document_id'] in relevant]
        print('Años:', sorted({d[:4] for d in dates if re.match(r'^\d{4}-\d{2}', d)}))
        print('Meses:', sorted({d[:7] for d in dates if re.match(r'^\d{4}-\d{2}', d)}))
        for error in errors:
            print(error, file=sys.stderr)
        return int(bool(errors))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
