
import logging
from collections import Counter
import copy

import seaborn as sns
import pandas as pd

import io
import base64
import matplotlib.pyplot as plt

import spanviz
from .scoring import *


class ComparisonAggregator:

    def __init__(self, annotators, layers):
        self.recorded_results = {
            **{l: {'labels_left': [],
            'labels_right': [],
            'predictions': [],
            'targets': [],
            'tokenwise_scores': [],
            'categorical_scores': []} for l in layers},
        }
        self.annotators = annotators
        self.layers = layers
        self.html_comparisons = []

    def process(self, doc):

        annotator1, annotator2 = self.annotators
        texts, span_layers_left, span_layers_right = doc['text'], doc['spans'][annotator1], doc['spans'][annotator2]

        for layer, text in texts.items():
            spans_left, spans_right = span_layers_left[layer], span_layers_right[layer]

            predictions, targets = [], []
            labels_left, labels_right = [], []

            for span_left, span_right in zip(spans_left, spans_right):
                preds, targs = match_spans_tokenwise(span_left, span_right, text)

                predictions.extend(preds)
                targets.extend(targs)
                labels_left.append(match_spans_categorical(span_left, span_right))
                labels_right.append(match_spans_categorical(span_right, span_left))

            record = self.recorded_results[layer]

            record['labels_left'].extend(labels_left)
            record['labels_right'].extend(labels_right)
            record['predictions'].extend(predictions)
            record['targets'].extend(targets)
            record['tokenwise_scores'].append(compute_binary_classification_scores(predictions, targets))
            record['categorical_scores'].append(compute_categorical_scores(labels_left, labels_right))

        self.html_comparisons.append(make_side_by_side_html(doc['id'], texts, span_layers_left, span_layers_right))

    def make_report(self):
        # TODO: Split this function up

        annotator1, annotator2 = self.annotators
        results = self.recorded_results

        aggregated_scores = []
        for layer in self.layers:
            record = results[layer]

            scores = {
                ('layer', '', ''): layer,
                **{('tokenwise', 'micro', key): value for key, value in compute_binary_classification_scores(record['predictions'], record['targets']).items()},
                **{('tokenwise', 'macro', key): value for key, value in compute_macro_scores(record['tokenwise_scores']).items()},
                **{('categorical', 'micro', key): value for key, value in compute_categorical_scores(record['labels_left'], record['labels_right']).items()},
                **{('categorical', 'macro', key): value for key, value in compute_macro_scores(record['categorical_scores']).items()},
            }
            aggregated_scores.append(scores)

        match_counts = (pd.DataFrame({layer: Counter(results[layer]['labels_left']) for layer in self.layers})
                        .melt(ignore_index=False, value_name='count', var_name='layer').reset_index(names='type of (mis)match'))

        sns.barplot(data=match_counts, x='type of (mis)match', y='count', hue='layer')
        plot_html = plot_to_html()

        scores_df = pd.DataFrame(aggregated_scores)
        scores_df.columns = pd.MultiIndex.from_tuples(aggregated_scores[0].keys())
        # .to_markdown(floatfmt='.2f')
        logging.info(scores_df.to_string(float_format='{:.2f}'.format))  # TODO: Maybe also log some plots?

        html_chunks = [
            f'<h2>Comparing {annotator1} and {annotator2}</h2>',
            scores_df.to_html(float_format='{:.2f}'.format) + '\n\n',
            f'<h3>Frequencies of types of (mis)match:</h3>\n{plot_html}\n',
            f'<h3>Side-by-side view:</h3>\n',
            f'\n<table><tr><td><b>{annotator1}</b></td><td><b>{annotator2}</b></td></tr>',
            *self.html_comparisons,
            '</table>\n\n',
        ]

        return ''.join(html_chunks)


def make_side_by_side_html(document_id, texts, spans1, spans2):

    html = [f'<tr><td style="border-bottom:2px solid gray" colspan="2"></td></tr><tr><td><i>{document_id}</i></td></tr>']

    for layer, text in texts.items():
        spans_left, spans_right = spans1[layer], spans2[layer]

        # TODO Avoid need for converting for spanviz
        spans_left = [[{'start': x, 'end': y, 'label': str(n)} for (x, y) in span] for n, span in enumerate(spans_left)]
        spans_right = [[{'start': x, 'end': y, 'label': str(n)} for (x, y) in span] for n, span in enumerate(spans_right)]

        html_left = spanviz.spans_to_html(text, spans_left, rainbow=True)
        html_right = spanviz.spans_to_html(text, spans_right, rainbow=True)

        html.append(f'<tr><td><b>{layer}</b></td></tr>\n<tr><td>{html_left}</td>\n<td>{html_right}</td></tr>\n<tr><td></td></tr>')

    return '\n'.join(html)


def plot_to_html():
    """
    https://stackoverflow.com/a/63381737
    """
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    plot_base64 = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return f'<img src="data:image/png;base64,{plot_base64}">'


def merge_spans_of_doc(doc) -> dict:
    doc = copy.deepcopy(doc)

    for name, span_levels in doc['spans'].items():
        doc['spans'][name] = {layer: [[merge_span(span) for span in spans] for spans in span_level] for layer, span_level in doc['spans'][name]}

    return doc


def merge_span(span: Span) -> Span:
    return [(min(s[0] for s in span), max(s[-1] for s in span))] if span else []


def align_spans_of_doc(doc) -> dict:
    """
    Aligns all annotations of this document, to the final (rightmost) annotation.
    """

    doc = copy.deepcopy(doc)
    spans = doc['spans']

    names = list(spans)
    span_layers_right = spans[names[-1]]
    first_layer_spans_right = next(iter(span_layers_right.values()))
    first_layer_text = next(iter(doc['text'].values()))

    for name in names[:-1]:
        span_layers_left = doc['spans'][name]
        first_layer_spans_left = next(iter(span_layers_left.values()))
        alignment_mapping = make_alignment_mapping(first_layer_spans_right, first_layer_spans_left, first_layer_text)
        spans[name] = {layer: [spans_left[index] for index in alignment_mapping if index is not None] for layer, spans_left in span_layers_left.items()}

    return doc