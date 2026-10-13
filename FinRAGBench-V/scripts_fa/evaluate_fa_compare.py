"""Compare local Persian ColQwen2/OpenMatch, BM25 and hybrid at k=10.
Run from the original project environment. Only load trusted local pickles.
No corpus encoding or model downloads are performed.
"""
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import argparse
import csv
import gc
import hashlib
import json
import math
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
K = 10


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_questions(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    questions = data if isinstance(data, list) else data['questions']
    if not questions:
        raise ValueError('Empty evaluation dataset')
    seen = set()
    for question in questions:
        qid = question['query_id']
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError('Invalid or duplicate query ID')
        seen.add(qid)
        if not isinstance(question['query'], str) or not question['query'].strip():
            raise ValueError(f'Empty query: {qid}')
        question.setdefault('per_relevant_page_ids', [])
        full = question['relevant_page_ids']
        partial = question['per_relevant_page_ids']
        if not isinstance(full, list) or not full or not isinstance(partial, list):
            raise ValueError(f'Invalid reference lists: {qid}')
        if not all(isinstance(p, str) and p for p in full + partial):
            raise ValueError(f'Invalid page IDs: {qid}')
        if len(set(full)) != len(full) or len(set(partial)) != len(partial):
            raise ValueError(f'Duplicate references: {qid}')
        if set(full) & set(partial):
            raise ValueError(f'Full and partial references overlap: {qid}')
    return questions


def metrics(ranking, references):
    references = set(references)
    if not references:
        raise ValueError('Empty references')
    ids = ranking[:K]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate IDs in ranking')
    ranks = [i for i, p in enumerate(ids, 1) if p in references]
    return {
        'recall_at_10': len(references & set(ids)) / len(references),
        'reciprocal_rank_at_10': 1.0 / min(ranks) if ranks else 0.0,
        'first_relevant_rank_at_10': min(ranks) if ranks else None,
    }


def evaluated_row(question, ranking, details):
    full = metrics(ranking, question['relevant_page_ids'])
    accepted = metrics(ranking, question['relevant_page_ids'] +
                       question['per_relevant_page_ids'])
    return {
        **question, **full,
        'recall_at_10_with_partial': accepted['recall_at_10'],
        'reciprocal_rank_at_10_with_partial': accepted['reciprocal_rank_at_10'],
        'first_accepted_rank_at_10': accepted['first_relevant_rank_at_10'],
        'retrieved_page_ids': ranking[:K],
        'ranking': details[:K],
    }


def summarize(rows):
    count = len(rows)
    return {
        'questions': count, 'k': K,
        'recall_at_10': sum(r['recall_at_10'] for r in rows) / count,
        'mrr_at_10': sum(r['reciprocal_rank_at_10'] for r in rows) / count,
        'recall_at_10_with_partial': sum(
            r['recall_at_10_with_partial'] for r in rows) / count,
        'mrr_at_10_with_partial': sum(
            r['reciprocal_rank_at_10_with_partial'] for r in rows) / count,
    }


def load_pages():
    path = ROOT / 'data/fa/text/pages_text.jsonl'
    pages = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        page = json.loads(line)
        pid = page['corpus-id']
        if pid in pages:
            raise ValueError(f'Duplicate page metadata: {pid}')
        pages[pid] = page
    return path, pages


def load_bm25(text_path, pages):
    from retrieve_fa_bm25 import tokenize, TOKENIZER_VERSION
    path = ROOT / 'retrieval/encode/bm25_fa/index.pkl'
    with path.open('rb') as handle:
        state = pickle.load(handle)
    if state['source_sha256'] != sha256(text_path):
        raise ValueError('BM25 index outdated; run retrieve_fa_bm25.py once')
    if state['tokenizer_version'] != TOKENIZER_VERSION:
        raise ValueError('BM25 tokenizer changed; rebuild the index')
    records = state['records']
    ids = [record['corpus-id'] for record in records]
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(pages):
        raise ValueError('Invalid BM25 page IDs')
    for record in records:
        if record['text_normalized'] != pages[record['corpus-id']]['text_normalized']:
            raise ValueError('BM25 records disagree with page text')
    return path, state, tokenize


def bm25_ranking(query, state, tokenize):
    tokens = tokenize(query)
    records = state['records']
    if not set(tokens) & state['vocabulary']:
        return [], [], []
    scores = [float(s) for s in state['model'].get_scores(tokens)]
    if len(scores) != len(records) or not all(math.isfinite(s) for s in scores):
        raise ValueError('Invalid BM25 scores')
    order = sorted(range(len(records)), key=lambda i: (
        -scores[i], records[i]['corpus-id']))
    ids = [records[i]['corpus-id'] for i in order]
    details = [{'rank': rank, 'page_id': records[i]['corpus-id'],
                'score': scores[i]} for rank, i in enumerate(order, 1)]
    # Preserve the existing hybrid's lexical overlap filter.
    words = set(tokens)
    matching = [records[i]['corpus-id'] for i in order
                if words & set(tokenize(records[i]['text_normalized']))]
    return ids, details, matching


def visual_rankings(questions, embedding_dir, batch_size, pages):
    import torch
    sys.path.insert(0, str(ROOT / 'src'))
    from openmatch.modeling.modeling_colqwen.modeling_colqwen_model import ColQwen2Model
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    manifest_path = embedding_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('image_token_budget') != 2048:
        raise ValueError('Expected existing 2048-token corpus embeddings')
    files = sorted(p for p in embedding_dir.glob('embeddings.corpus.rank.*')
                   if p.is_file() and not p.name.endswith('.tmp'))
    if not files:
        raise FileNotFoundError('Corpus embeddings not found')
    vectors, ids = [], []
    for path in files:
        with path.open('rb') as handle:
            batch, batch_ids = pickle.load(handle)
        if len(batch) != len(batch_ids):
            raise ValueError(f'Embedding/ID mismatch: {path}')
        vectors.extend(batch)
        ids.extend(batch_ids)
    if len(ids) != 122 or len(set(ids)) != len(ids) or set(ids) != set(pages):
        raise ValueError('Expected 122 unique, matching corpus pages')
    shapes = [list(v.shape) for v in vectors]
    if manifest.get('pages') != len(ids) or manifest.get('shapes') != shapes:
        raise ValueError('Manifest disagrees with actual embedding shapes')
    if any(len(s) != 2 or s[1] != 128 for s in shapes):
        raise ValueError('Unexpected embedding dimensions')
    model_path = ROOT / 'retrieval/colqwen2-v1.0-merged'
    print('Loading ColQwen2 / OpenMatch with existing 2048 embeddings...', flush=True)
    model = ColQwen2Model(str(model_path))
    output = {}
    for question in questions:
        with torch.inference_mode():
            inputs = model.processor.process_queries([question['query']]).to('cuda:0')
            query_vectors = model.model(**inputs)
            scores = model.processor.score(query_vectors, vectors, device='cuda',
                                           batch_size=batch_size)[0].detach().float().cpu()
        if scores.numel() != len(ids) or not torch.isfinite(scores).all().item():
            raise ValueError('Invalid visual scores')
        order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
        output[question['query_id']] = (
            [ids[i] for i in order],
            [{'rank': rank, 'page_id': ids[i], 'score': float(scores[i])}
             for rank, i in enumerate(order, 1)],
        )
        print('Visual scored:', question['query_id'], flush=True)
        del inputs, query_vectors, scores
    provenance = {
        'embedding_files': [str(p.relative_to(ROOT)) for p in files],
        'embedding_manifest': manifest,
        'embedding_manifest_sha256': sha256(manifest_path),
        'model_path': str(model_path.relative_to(ROOT)),
        'visual_backend': 'OpenMatch; same query processing and scoring as visual evaluator',
    }
    del model, vectors
    gc.collect()
    torch.cuda.empty_cache()
    return output, provenance


def hybrid_rankings(questions, visual, text_candidates, pages, candidate_count, batch_size):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    manifest_path = ROOT / 'models/reranker_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    options = {'cache_dir': str(ROOT / 'models/hf-cache'), 'local_files_only': True}
    if manifest.get('revision'):
        options['revision'] = manifest['revision']
    print('Loading local text reranker...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(manifest['model_id'], **options)
    reranker = AutoModelForSequenceClassification.from_pretrained(
        manifest['model_id'], torch_dtype=torch.float16, **options).to('cuda:0').eval()
    output = {}
    for question in questions:
        qid, query = question['query_id'], question['query']
        visual_ids = visual[qid][0][:candidate_count]
        text_ids = text_candidates[qid][:candidate_count]
        candidate_ids = sorted(set(visual_ids) | set(text_ids))
        usable = [p for p in candidate_ids if pages[p]['text_normalized'].strip()]
        textless = sorted(set(candidate_ids) - set(usable))
        scores, truncated = {}, []
        for pid in usable:
            length = len(tokenizer(query, pages[pid]['text_normalized'],
                                   truncation=False)['input_ids'])
            if length > 512:
                truncated.append(pid)
        for offset in range(0, len(usable), batch_size):
            batch_ids = usable[offset:offset + batch_size]
            inputs = tokenizer([[query, pages[p]['text_normalized']] for p in batch_ids],
                               padding=True, truncation='only_second', max_length=512,
                               return_tensors='pt').to('cuda:0')
            with torch.inference_mode():
                values = reranker(**inputs).logits.float().cpu().reshape(-1)
            if len(values) != len(batch_ids) or not torch.isfinite(values).all().item():
                raise ValueError('Invalid reranker scores')
            scores.update({p: float(v) for p, v in zip(batch_ids, values)})
            del inputs, values
        ranked = sorted(usable, key=lambda p: (-scores[p], p)) + textless
        vranks = {p: i for i, p in enumerate(visual_ids, 1)}
        tranks = {p: i for i, p in enumerate(text_ids, 1)}
        details = [{'rank': i, 'page_id': p, 'reranker_score': scores.get(p),
                    'visual_rank': vranks.get(p), 'bm25_rank': tranks.get(p)}
                   for i, p in enumerate(ranked, 1)]
        row = evaluated_row(question, ranked, details)
        row.update({'candidate_count': len(candidate_ids), 'candidate_page_ids': candidate_ids,
                    'textless_pages': textless, 'truncated_pages': truncated})
        output[qid] = row
        print('Hybrid scored:', qid, flush=True)
    del reranker, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return output, {'reranker_manifest': manifest,
                    'reranker_manifest_sha256': sha256(manifest_path),
                    'reranker_max_tokens': 512, 'candidates_per_source': candidate_count,
                    'textless_policy': 'after all text-scored candidates; page ID ascending'}


def write_results(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    prefix = output_dir / f'comparison_k10_{stamp}'
    report_path = prefix.with_suffix('.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                           encoding='utf-8')
    summary_path = Path(str(prefix) + '_summary.csv')
    fields = ['method', 'questions', 'k', 'recall_at_10', 'mrr_at_10',
              'recall_at_10_with_partial', 'mrr_at_10_with_partial']
    with summary_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, value in report['methods'].items():
            writer.writerow({'method': method, **value['summary']})
    queries_path = Path(str(prefix) + '_queries.csv')
    fields = ['method', 'query_id', 'query', 'first_relevant_rank_at_10',
              'recall_at_10', 'reciprocal_rank_at_10',
              'recall_at_10_with_partial', 'reciprocal_rank_at_10_with_partial',
              'retrieved_page_ids']
    with queries_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, value in report['methods'].items():
            for result in value['results']:
                row = {key: result.get(key) for key in fields if key != 'method'}
                row['method'] = method
                row['retrieved_page_ids'] = json.dumps(result['retrieved_page_ids'],
                                                       ensure_ascii=False)
                writer.writerow(row)
    print('\nMethod       Recall@10       MRR@10')
    for method, value in report['methods'].items():
        s = value['summary']
        print(f'{method:<12} {s["recall_at_10"]:>9.2%} {s["mrr_at_10"]:>12.6f}')
    for path in [report_path, summary_path, queries_path]:
        print('Saved:', path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', default='data/fa/evaluation/questions_10.json')
    parser.add_argument('--embedding-dir', default='retrieval/encode/colqwen_fa_2048')
    parser.add_argument('--methods', nargs='+', choices=['colqwen2', 'bm25', 'hybrid'],
                        default=['colqwen2', 'bm25', 'hybrid'])
    parser.add_argument('--candidates', type=int, default=20)
    parser.add_argument('--score-batch-size', type=int, default=16)
    parser.add_argument('--rerank-batch-size', type=int, default=4)
    parser.add_argument('--output-dir', default='retrieval/results/fa_comparison')
    args = parser.parse_args()
    if args.candidates < K or min(args.score_batch_size, args.rerank_batch_size) < 1:
        raise ValueError('Candidates must be >=10 and batch sizes must be positive')
    if len(args.methods) != len(set(args.methods)):
        raise ValueError('Duplicate methods')
    dataset_path = ROOT / args.questions
    questions = load_questions(dataset_path)
    text_path, pages = load_pages()
    for question in questions:
        missing = set(question['relevant_page_ids'] + question['per_relevant_page_ids']) - set(pages)
        if missing:
            raise ValueError(f'References missing from corpus metadata: {missing}')
    report = {'dataset': str(dataset_path.relative_to(ROOT)),
              'dataset_sha256': sha256(dataset_path), 'k': K,
              'metric_policy': 'binary full references; separately full plus partial union',
              'tie_breaking': 'score descending, page ID ascending',
              'script_sha256': sha256(Path(__file__)),
              'unique_reference_pages': len({p for q in questions for p in q['relevant_page_ids']}),
              'methods': {}, 'provenance': {}, 'package_versions': {}}
    for package in ['torch', 'transformers', 'rank-bm25']:
        try:
            report['package_versions'][package] = version(package)
        except PackageNotFoundError:
            report['package_versions'][package] = None
    start = time.perf_counter()
    text_candidates = {}
    if 'bm25' in args.methods or 'hybrid' in args.methods:
        index_path, state, tokenize = load_bm25(text_path, pages)
        report['provenance'].update({'bm25_index_sha256': sha256(index_path),
                                     'text_corpus_sha256': sha256(text_path),
                                     'bm25_searchable_pages': len(state['records'])})
        rows = []
        for question in questions:
            ids, details, matching = bm25_ranking(question['query'], state, tokenize)
            rows.append(evaluated_row(question, ids, details))
            text_candidates[question['query_id']] = matching
        if 'bm25' in args.methods:
            report['methods']['bm25'] = {'summary': summarize(rows), 'results': rows}
        del state
    if 'colqwen2' in args.methods or 'hybrid' in args.methods:
        visual, provenance = visual_rankings(questions, ROOT / args.embedding_dir,
                                            args.score_batch_size, pages)
        report['provenance'].update(provenance)
        if 'colqwen2' in args.methods:
            rows = [evaluated_row(q, *visual[q['query_id']]) for q in questions]
            report['methods']['colqwen2'] = {'summary': summarize(rows), 'results': rows}
        if 'hybrid' in args.methods:
            hybrid, provenance = hybrid_rankings(questions, visual, text_candidates, pages,
                                                 args.candidates, args.rerank_batch_size)
            rows = [hybrid[q['query_id']] for q in questions]
            report['methods']['hybrid'] = {'summary': summarize(rows), 'results': rows}
            report['provenance'].update(provenance)
    for method in report['methods'].values():
        for row in method['results']:
            for result in row['ranking']:
                page = pages[result['page_id']]
                for key in ['source_pdf', 'pdf_page_number', 'image_path']:
                    if key in page:
                        result[key] = page[key]
    # This is total batch time with reused visual scores, not comparable per-method latency.
    report['total_batch_seconds'] = time.perf_counter() - start
    write_results(report, ROOT / args.output_dir)


if __name__ == '__main__':
    main()
