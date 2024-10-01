#!/usr/bin/python

import argparse
import sys
import json
import itertools
from collections import Counter

from .main import *
import spanviz


import logging

import math
import webbrowser
import pandas as pd
import os

import io
import base64
import matplotlib.pyplot as plt

import tempfile

OUTPUT_LABELS = 'labels'
OUTPUT_MATCHES = 'matches'
OUTPUT_SCORES_PER_LINE = 'scores'
OUTPUT_AGGREGATE_SCORES = 'score'
OUTPUT_JSONL = 'jsonl'

modes = [OUTPUT_LABELS, OUTPUT_MATCHES, OUTPUT_SCORES_PER_LINE, OUTPUT_AGGREGATE_SCORES, OUTPUT_JSONL]

# TODO: Add characterwise and tokenwise scores as well.
# TODO: Add side-by-side html view with spanviz (requires text as input, which I'll need anyway for tokenwise f1...)
# TODO: Also add sentencewise scores: if spans are expanded to the full sentence; and implement --merge

def main():

    argparser = argparse.ArgumentParser(description='Evaluating multi-disco-spans.')
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="file with text to process (default: stdin)")
    argparser.add_argument('--output', type=str, choices=modes, default=OUTPUT_LABELS, help="whether to output match labels, match orders, per-line scores, aggregate scores, or JSON structures")
    argparser.add_argument('--merge', action='store_true', help="whether to make spans continuous prior to eval")
    argparser.add_argument('--verb', action='store_true', help="verbose, set logging level to debug")
    argparser.add_argument('--ordered', action='store_true', help="if the pairs are in the same order (so don't compare all to all)")
    argparser.add_argument('--html', required=False, nargs='?', type=argparse.FileType('w'), default=tempfile.NamedTemporaryFile('w', delete=False, suffix='.html'), help='output detailed html report to a file; tempfile used if omitted')

    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verb else logging.INFO, format='')

    if args.merge:
        raise NotImplementedError('--merge not yet supported')

    annotator1 = 'left' # TODO determine this from input data
    annotator2 = 'right'
    layers = ['questions', 'answers']

    docs = [parse_line(line) for line in args.file]

    results = do_comparison(docs, annotator1, annotator2, layers)

    if args.html:
        args.html.write(make_report(results, layers))
        args.html.close()
        webbrowser.open('file://' + os.path.abspath(args.html.name))


def do_comparison(docs, annotator1, annotator2, layers):

    bookkeeping = {**{l: {'labels_left': [],    # TODO layer labels
                       'labels_right': [],
                       'predictions': [],
                       'targets': [],
                       'tokenwise_scores': [],
                       'categorical_scores': []} for l in layers},
                    'alignment_mappings': [],
                    'html': [f'<h2>Comparing {annotator1} and {annotator2}</h2>', f'\n<table><tr><td><b>{annotator1}</b></td><td><b>{annotator2}</b></td></tr>']}

    for doc in docs:

        texts, span_layers_left, span_layers_right = doc['text'], doc['spans1'], doc['spans2']

        alignment_mapping = make_alignment_mapping(span_layers_left[0], span_layers_right[0], texts[0])
        bookkeeping['alignment_mappings'].append(alignment_mapping)

        spans2_aligned = [[spans[index] for index in alignment_mapping] for spans in span_layers_right]

        for (layer, text, spans_left, spans_right) in zip(layers, texts, span_layers_left, spans2_aligned):

            if not spans_left or not spans_right:   # TODO necessary?
                return None, None

            predictions, targets = [], []
            labels_left, labels_right = [], []

            for span1, span2 in zip(spans_left, spans_right):
                preds, targs = match_spans_tokenwise(span1, span2, text)

                predictions.extend(preds)
                targets.extend(targs)
                labels_left.append(match_spans_categorical(span1, span2))
                labels_right.append(match_spans_categorical(span2, span1))

            record = bookkeeping[layer]

            record['labels_left'].extend(labels_left)
            record['labels_right'].extend(labels_right)
            record['predictions'].extend(predictions)
            record['targets'].extend(targets)
            record['tokenwise_scores'].append(compute_binary_classification_scores(predictions, targets))
            record['categorical_scores'].append(compute_categorical_scores(labels_left, labels_right))

        bookkeeping['html'].append(make_side_by_side_html(doc['id'], texts, span_layers_left, span_layers_right))

    return bookkeeping


def make_report(bookkeeping, layers):

    html_for_comparison = bookkeeping['html'].copy()

    aggregated_scores = []
    for layer in layers:
        record = bookkeeping[layer]

        scores = {
            ('layer', '', ''): layer,
            **{('tokenwise', 'micro', key): value for key, value in compute_binary_classification_scores(record['predictions'], record['targets']).items()},
            **{('tokenwise', 'macro', key): value for key, value in compute_macro_scores(record['tokenwise_scores']).items()},
            **{('categorical', 'micro', key): value for key, value in compute_categorical_scores(record['labels_left'], record['labels_right']).items()},
            **{('categorical', 'macro', key): value for key, value in compute_macro_scores(record['categorical_scores']).items()},
        }
        aggregated_scores.append(scores)

    df_match_counts = pd.DataFrame.from_dict(Counter(bookkeeping['questions']['labels_left']), orient='index')
    df_match_counts.plot(kind='bar')
    plot_html = plot_to_html()
    html_for_comparison.insert(1, f'<h3>Question spans:</h3>\n{plot_html}\n')

    df_match_counts = pd.DataFrame.from_dict(Counter(bookkeeping['answers']['labels_left']), orient='index')
    df_match_counts.plot(kind='bar')
    plot_html = plot_to_html()
    html_for_comparison.insert(1, f'<h3>Answer spans:</h3>\n{plot_html}\n')

    html_for_comparison.append('</table>\n\n')

    scores_df = pd.DataFrame(aggregated_scores)
    scores_df.columns = pd.MultiIndex.from_tuples(aggregated_scores[0].keys())
    # .to_markdown(floatfmt='.2f')
    logging.info(scores_df.to_string())  # TODO: Maybe also log some plots?
    html_for_comparison.insert(1, scores_df.to_html(float_format='{:.2f}'.format) + '\n\n')

    return ''.join(html_for_comparison)


def plot_to_html():
    """
    https://stackoverflow.com/a/63381737
    """
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    plot_base64 = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return f'<img src="data:image/png;base64,{plot_base64}">'


def make_side_by_side_html(document_id, texts, spans1, spans2):

    questions_text, context = texts
    # question_spans_aligned_left, question_spans_aligned_right, answer_spans_left, answer_spans_right = spans1[0], spans2[0], spans1[1], spans2[2]

    # TODO Avoid need for converting for spanviz
    question_spans_aligned_left = [[{'start': x, 'end': y, 'label': str(n)} for (x, y) in span] for n, span in enumerate(spans1[0])]
    question_spans_aligned_right = [[{'start': x, 'end': y, 'label': str(n)} for (x, y) in span] for n, span in enumerate(spans2[0])]
    answer_spans_left = [[{'start': x, 'end': y, 'label': str(n)} for (x, y) in span] for n, span in enumerate(spans1[1])]
    answer_spans_right = [[{'start': x, 'end': y, 'label': str(n)} for (x, y) in span] for n, span in enumerate(spans1[2])]

    q_left = spanviz.spans_to_html(questions_text, question_spans_aligned_left)
    q_right = spanviz.spans_to_html(questions_text, question_spans_aligned_right)
    a_left = spanviz.spans_to_html(context, answer_spans_left)
    a_right = spanviz.spans_to_html(context, answer_spans_right)

    html = f'''<tr><td></td></tr><tr><td><i>{document_id}</i></td></tr>
    <tr><td><b>Questions:</b></td></tr>
    <tr><td>{q_left}</td>
    <td>{q_right}</td></tr>
    <tr><td></td></tr>
    <tr><td><b>Answers:</b></td></tr>
    <tr><td>{a_left}</td>
    <td>{a_right}</td></tr>
    '''
    return html


def evaluate_all(pairs_to_compare, do_print=False, mode=OUTPUT_AGGREGATE_SCORES, ordered=False):

    # TODO Refactor? with the help of a 'record' dict.
    # TODO: Allow html output

    print_or_log = print if do_print else logging.debug

    all_match_labels_left = []
    all_match_labels_right = []
    scores_per_doc = []

    for item in pairs_to_compare:
        (annotator1, spans1), (annotator2, spans2) = item.items()
        (match_labels_left, match_indices_left), (match_labels_right, match_indices_right), scores_tokenwise = match_span_many_to_many(spans1, spans2, in_order=ordered)

        # TODO: Do something with scores_tokenwise
        scores_categorical = compute_categorical_scores(match_labels_left, match_labels_right)

        if mode == OUTPUT_SCORES_PER_LINE:
            print_or_log('{precision:.2f},{recall:.2f},{f1:.2f}  ({count})'.format(**scores_categorical))
        elif mode == OUTPUT_LABELS:
            print_or_log(','.join(f'{l}' for l in match_labels_left) + '\t' + ','.join(f'{l}' for l in match_labels_right))
        elif mode == OUTPUT_MATCHES:
            print_or_log(','.join(f'{i}' if i is not None else '' for i in match_indices_left) + '\t' +
                         ','.join(f'{i}' if i is not None else '' for i in match_indices_right))
        elif mode == OUTPUT_JSONL:
            print_or_log(json.dumps({
                annotator1: {'labels': match_labels_left, 'indices': match_labels_left},
                annotator2: {'labels': match_labels_right, 'indices': match_indices_right},
                **scores_categorical,
            }))

        all_match_labels_left.extend(match_labels_left)
        all_match_labels_right.extend(match_labels_right)
        scores_per_doc.append(scores_categorical)

    micro_scores = compute_categorical_scores(all_match_labels_left, all_match_labels_right)

    macro_scores = {
        'precision': nanmean(s['precision'] for s in scores_per_doc),
        'recall': nanmean(s['recall'] for s in scores_per_doc),
        'f1': nanmean(s['f1'] for s in scores_per_doc),
        'count': sum(s['count'] for s in scores_per_doc),
    }
    aggregated_scores = {'micro': micro_scores, 'macro': macro_scores, }

    format_scores = '{precision:.2f}, {recall:.2f}, {f1:.2f}  ({count})'.format
    if mode == OUTPUT_AGGREGATE_SCORES:
        print(format_scores(**micro_scores))
        print(format_scores(**macro_scores))
    else:
        logging.info('Aggregated:')
        logging.info('micro: ' + format_scores(**micro_scores))
        logging.info('macro: ' + format_scores(**macro_scores))

    return aggregated_scores


def parse_line(line):
    try:
        d = json.loads(line)    # TODO: type validation (dict with two keys, values [lists of] list of lists of pairs of ints; see test3.txt
        d['spans1'] = [[[(s['start'], s['end']) for s in span] for span in spanlevel] for spanlevel in d['spans1']]
        d['spans2'] = [[[(s['start'], s['end']) for s in span] for span in spanlevel] for spanlevel in d['spans2']]
        return d
    except json.JSONDecodeError:
        left, right = line.strip().split('\t')
        spansets1 = [[[int(i) for i in span.split('-')] for span in s.split(',')] for s in left.split(';')]
        spansets2 = [[[int(i) for i in span.split('-')] for span in s.split(',')] for s in right.split(';')]
        return {'left': spansets1, 'right': spansets2}


if __name__ == '__main__':
    main()