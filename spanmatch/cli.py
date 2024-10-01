#!/usr/bin/python

import argparse
import sys
import json

from .main import *
import logging

import webbrowser
import os
import tempfile
import copy


OUTPUT_LABELS = 'labels'
OUTPUT_MATCHES = 'matches'
OUTPUT_SCORES_PER_LINE = 'scores'
OUTPUT_AGGREGATE_SCORES = 'score'
OUTPUT_JSONL = 'jsonl'

modes = [OUTPUT_LABELS, OUTPUT_MATCHES, OUTPUT_SCORES_PER_LINE, OUTPUT_AGGREGATE_SCORES, OUTPUT_JSONL]


# TODO: Simplify/generalize input format
# TODO: Allow comparing more than two annotators, pairwise

def main():

    argparser = argparse.ArgumentParser(description='Evaluating multi-disco-spans.')
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="file with json lines to process (default: stdin)")
    argparser.add_argument('--output', type=str, choices=modes, default=None, help="whether to output match labels, match orders, per-line scores, aggregate scores, or JSON structures")
    argparser.add_argument('--merge', action='store_true', help="whether to make spans continuous prior to eval")
    argparser.add_argument('--sentence', action='store_true', help="whether to increase all spans to whole-sentence size")
    argparser.add_argument('--aligned', action='store_true', help="if the spans of annotator 1 and annotator 2 are already 'aligned' for comparison")
    argparser.add_argument('--html', required=False, nargs='?', type=argparse.FileType('w'), default=tempfile.NamedTemporaryFile('w', delete=False, suffix='.html'), help='output detailed html report to a file; tempfile used if omitted')
    argparser.add_argument('--debug', action='store_true', help="verbose, set logging level to debug")

    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='')

    if args.sentence:
        raise NotImplementedError('--sentence not yet supported')
    if args.output:
        raise NotImplementedError('--output format not supported anymore; might reimplement later')

    first_line, lines = peek(args.file)

    peek_doc = parse_line(first_line)
    annotators = [key.split('_', maxsplit=1)[-1] for key in peek_doc if key.startswith('spans_')]
    layers = peek_doc['layers']

    aggregator = ComparisonAggregator(annotators, layers)

    for line in lines:
        doc = parse_line(line)
        if args.merge:
            doc = merge_spans_of_doc(doc, annotators)
        if not args.aligned:
            doc = align_spans_of_doc(doc, annotators)
        aggregator.process(doc)

    html_report = aggregator.make_report()

    if args.html:
        args.html.write(html_report)
        args.html.close()
        webbrowser.open('file://' + os.path.abspath(args.html.name))


def peek(iterator):
    first = next(iterator)
    restored_it = itertools.chain([first], iterator)
    return first, restored_it


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


if __name__ == '__main__':
    main()