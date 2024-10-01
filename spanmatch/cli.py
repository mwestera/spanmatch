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
import seaborn as sns


OUTPUT_LABELS = 'labels'
OUTPUT_MATCHES = 'matches'
OUTPUT_SCORES_PER_LINE = 'scores'
OUTPUT_AGGREGATE_SCORES = 'score'
OUTPUT_JSONL = 'jsonl'

modes = [OUTPUT_LABELS, OUTPUT_MATCHES, OUTPUT_SCORES_PER_LINE, OUTPUT_AGGREGATE_SCORES, OUTPUT_JSONL]


def main():

    argparser = argparse.ArgumentParser(description='Evaluating multi-disco-spans.')
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="file with text to process (default: stdin)")
    argparser.add_argument('--output', type=str, choices=modes, default=OUTPUT_LABELS, help="whether to output match labels, match orders, per-line scores, aggregate scores, or JSON structures")
    argparser.add_argument('--merge', action='store_true', help="whether to make spans continuous prior to eval")
    argparser.add_argument('--sentence', action='store_true', help="whether to increase all spans to whole-sentence size")
    argparser.add_argument('--aligned', action='store_true', help="if the spans of annotator 1 and annotator 2 are already 'aligned' for comparison")
    argparser.add_argument('--html', required=False, nargs='?', type=argparse.FileType('w'), default=tempfile.NamedTemporaryFile('w', delete=False, suffix='.html'), help='output detailed html report to a file; tempfile used if omitted')
    argparser.add_argument('--debug', action='store_true', help="verbose, set logging level to debug")

    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='')

    if args.sentence:
        raise NotImplementedError('--sentence not yet supported')

    docs = [parse_line(line) for line in args.file]
    peek = docs[0]

    annotators = [key.split('_', maxsplit=1)[-1] for key in peek if key.startswith('spans_')]
    layers = peek['layers']

    results = do_comparison(docs, annotators, layers, already_aligned=args.aligned, merge=args.merge)


    # TODO: Handle output types: args.output

    if args.html:
        args.html.write(make_report(results, layers))
        args.html.close()
        webbrowser.open('file://' + os.path.abspath(args.html.name))


def do_comparison(docs: Iterable[str], annotators, layers: list[str], already_aligned=False, merge=False) -> dict:

    annotator1, annotator2 = annotators

    bookkeeping = {**{l: {'labels_left': [],
                       'labels_right': [],
                       'predictions': [],
                       'targets': [],
                       'tokenwise_scores': [],
                       'categorical_scores': []} for l in layers},
                    'alignment_mappings': [],
                    'html': [f'<h2>Comparing {annotator1} and {annotator2}</h2>', f'\n<table><tr><td><b>{annotator1}</b></td><td><b>{annotator2}</b></td></tr>']}

    for doc in docs:

        texts, span_layers_left, span_layers_right = doc['text'], doc[f'spans_{annotator1}'], doc[f'spans_{annotator2}']

        if not already_aligned:
            alignment_mapping = make_alignment_mapping(span_layers_left[0], span_layers_right[0], texts[0])
            bookkeeping['alignment_mappings'].append(alignment_mapping)
            span_layers_right = [[spans[index] for index in alignment_mapping] for spans in span_layers_right]

        for (layer, text, spans_left, spans_right) in zip(layers, texts, span_layers_left, span_layers_right):

            predictions, targets = [], []
            labels_left, labels_right = [], []

            if merge:
                spans_left = make_spans_continuous(spans_left)
                spans_right = make_spans_continuous(spans_right)

            for span_left, span_right in zip(spans_left, spans_right):
                preds, targs = match_spans_tokenwise(span_left, span_right, text)

                predictions.extend(preds)
                targets.extend(targs)
                labels_left.append(match_spans_categorical(span_left, span_right))
                labels_right.append(match_spans_categorical(span_right, span_left))

            # if mode == OUTPUT_SCORES_PER_LINE:
            #     print('{precision:.2f},{recall:.2f},{f1:.2f}  ({count})'.format(**scores_categorical))
            # elif mode == OUTPUT_LABELS:
            #     print(
            #         ','.join(f'{l}' for l in match_labels_left) + '\t' + ','.join(f'{l}' for l in match_labels_right))
            # elif mode == OUTPUT_MATCHES:
            #     print(','.join(f'{i}' if i is not None else '' for i in match_indices_left) + '\t' +
            #                  ','.join(f'{i}' if i is not None else '' for i in match_indices_right))
            # elif mode == OUTPUT_JSONL:
            #     print(json.dumps({
            #         annotator1: {'labels': match_labels_left, 'indices': match_labels_left},
            #         annotator2: {'labels': match_labels_right, 'indices': match_indices_right},
            #         **scores_categorical,
            #     }))

            record = bookkeeping[layer]

            record['labels_left'].extend(labels_left)
            record['labels_right'].extend(labels_right)
            record['predictions'].extend(predictions)
            record['targets'].extend(targets)
            record['tokenwise_scores'].append(compute_binary_classification_scores(predictions, targets))
            record['categorical_scores'].append(compute_categorical_scores(labels_left, labels_right))

        bookkeeping['html'].append(make_side_by_side_html(doc['id'], texts, span_layers_left, span_layers_right, layers))

    bookkeeping['html'].append('</table>\n\n')
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

    match_counts = (pd.DataFrame({layer: Counter(bookkeeping[layer]['labels_left']) for layer in layers})
                    .melt(ignore_index=False, value_name='count', var_name='layer').reset_index(names='type of (mis)match'))

    sns.barplot(data=match_counts, x='type of (mis)match', y='count', hue='layer')
    plot_html = plot_to_html()
    html_for_comparison.insert(1, f'<h3>Frequencies of types of (mis)match:</h3>\n{plot_html}\n')

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


def make_side_by_side_html(document_id, texts, spans1, spans2, layers):

    html = [f'<tr><td></td></tr><tr><td><i>{document_id}</i></td></tr>']

    for layer, text, spans_left, spans_right in zip(layers, texts, spans1, spans2):

        # TODO Avoid need for converting for spanviz
        spans_left = [[{'start': x, 'end': y, 'label': str(n)} for (x, y) in span] for n, span in enumerate(spans_left)]
        spans_right = [[{'start': x, 'end': y, 'label': str(n)} for (x, y) in span] for n, span in enumerate(spans_right)]

        html_left = spanviz.spans_to_html(text, spans_left)
        html_right = spanviz.spans_to_html(text, spans_right)

        html.append(f'<tr><td><b>{layer}</b></td></tr>\n<tr><td>{html_left}</td>\n<td>{html_right}</td></tr>\n<tr><td></td></tr>')

    return '\n'.join(html)


def parse_line(line):
    try:
        d = json.loads(line)    # TODO: type validation (dict with two keys, values [lists of] list of lists of pairs of ints; see test3.txt
        for span_label in [k for k in d if k.startswith('spans_')]:
            d[span_label] = [[[(s['start'], s['end']) for s in span] for span in spanlevel] for spanlevel in d[span_label]]
        return d
    except json.JSONDecodeError:
        left, right = line.strip().split('\t')
        spansets1 = [[[int(i) for i in span.split('-')] for span in s.split(',')] for s in left.split(';')]
        spansets2 = [[[int(i) for i in span.split('-')] for span in s.split(',')] for s in right.split(';')]
        return {'left': spansets1, 'right': spansets2}


def make_spans_continuous(spans: list[Span]) -> list[Span]:
    return [[(min(s[0] for s in span), max(s[-1] for s in span))] if span else [] for span in spans]


if __name__ == '__main__':
    main()