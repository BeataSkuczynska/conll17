#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Compatible with Python 2.7 and 3.2+, can be used either as a module
# or a standalone executable.
#
# Copyright 2017, 2018 Institute of Formal and Applied Linguistics (UFAL),
# Faculty of Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Authors: Milan Straka, Martin Popel <surname@ufal.mff.cuni.cz>
#
# Changelog:
# - [12 Apr 2018] Version 0.9: Initial release.
# - [19 Apr 2018] Version 1.0: Fix bug in MLAS (duplicate entries in functional_children).
#                              Add --counts option.
# - [02 May 2018] Version 1.1: When removing spaces to match gold and system characters,
#                              consider all Unicode characters of category Zs instead of
#                              just ASCII space.

# This is a modified and extension of the original conll18_ud_eval.py script
# prepared for PolEval 2018 task by Katarzyna Krasnowska-KieraĹ, ICS PAS Warsaw.

# Command line usage
# ------------------
# poleval2018.py [-v] gold_conllu_file system_conllu_file
#
# - if no -v is given, only the official CoNLL18 UD Shared Task evaluation metrics
#   are printed
# - if -v is given, more metrics are printed (as precision, recall, F1 score,
#   and in case the metric is computed on aligned words also accuracy on these):
#   - Tokens: how well do the gold tokens match system tokens
#   - Sentences: how well do the gold sentences match system sentences
#   - Words: how well can the gold words be aligned to system words
#   - UPOS: using aligned words, how well does UPOS match
#   - XPOS: using aligned words, how well does XPOS match
#   - UFeats: using aligned words, how well does universal FEATS match
#   - AllTags: using aligned words, how well does UPOS+XPOS+FEATS match
#   - Lemmas: using aligned words, how well does LEMMA match
#   - UAS: using aligned words, how well does HEAD match
#   - LAS: using aligned words, how well does HEAD+DEPREL(ignoring subtypes) match
#   - ELAS: using aligned words, how well does HEAD+DEPREL+DEPS match
#   - SLAS: using aligned words, how well does HEAD+DEPREL+SEM match
#   - CLAS: using aligned words with content DEPREL, how well does
#       HEAD+DEPREL(ignoring subtypes) match
#   - MLAS: using aligned words with content DEPREL, how well does
#       HEAD+DEPREL(ignoring subtypes)+UPOS+UFEATS+FunctionalChildren(DEPREL+UPOS+UFEATS) match
#   - BLEX: using aligned words with content DEPREL, how well does
#       HEAD+DEPREL(ignoring subtypes)+LEMMAS match
# - if -c is given, raw counts of correct/gold_total/system_total/aligned words are printed
#   instead of precision/recall/F1/AlignedAccuracy for all metrics.

# API usage
# ---------
# - load_dependency(file, file_format='conllu')
#   - loads CoNLL(-U) file from given file object to an internal representation
#   - the file object should return str in both Python 2 and Python 3
#   - raises UDError exception if the given file cannot be loaded
# - evaluate(gold_ud, system_ud)
#   - evaluate the given gold and system CoNLL(-U) files (loaded with load_dependency)
#   - raises UDError if the concatenated tokens of gold and system file do not match
#   - returns a dictionary with the metrics described above, each metric having
#     three fields: precision, recall and f1

# Description of token matching
# -----------------------------
# In order to match tokens of gold file and system file, we consider the text
# resulting from concatenation of gold tokens and text resulting from
# concatenation of system tokens. These texts should match -- if they do not,
# the evaluation fails.
#
# If the texts do match, every token is represented as a range in this original
# text, and tokens are equal only if their range is the same.

# Description of word matching
# ----------------------------
# When matching words of gold file and system file, we first match the tokens.
# The words which are also tokens are matched as tokens, but words in multi-word
# tokens have to be handled differently.
#
# To handle multi-word tokens, we start by finding "multi-word spans".
# Multi-word span is a span in the original text such that
# - it contains at least one multi-word token
# - all multi-word tokens in the span (considering both gold and system ones)
#   are completely inside the span (i.e., they do not "stick out")
# - the multi-word span is as small as possible
#
# For every multi-word span, we align the gold and system words completely
# inside this span using LCS on their FORMs. The words not intersecting
# (even partially) any multi-word span are then aligned as tokens.


from __future__ import division
from __future__ import print_function

import argparse
import io
import sys
import unicodedata
import unittest

# CoNLL-U column names
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC, SEM = range(11)


# CoNLL-U specifics
class conllu(object):
    # Content and functional relations
    CONTENT_DEPRELS = {
        "nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "obl", "vocative",
        "expl", "dislocated", "advcl", "advmod", "discourse", "nmod", "appos",
        "nummod", "acl", "amod", "conj", "fixed", "flat", "compound", "list",
        "parataxis", "orphan", "root"
    }

    FUNCTIONAL_DEPRELS = {
        "aux", "cop", "mark", "det", "clf", "case", "cc"
    }

    FEATURES = None


# CoNLL specifics
class conll(object):
    # Content and functional relations
    CONTENT_DEPRELS = {}

    FUNCTIONAL_DEPRELS = {"refl", "aux", "aglt", "cond", "imp", "neg", "pre_coord"}

    FEATURES = None


# UD Error is used when raising exceptions in this module
class UDError(Exception):
    pass


num_cols = None


# Load given CoNLL(-U) file into internal representation
def load_dependency(file, file_format=conllu):
    global num_cols
    num_cols = None

    # Parse a single CoNNL(-U) line into 11 CoNNL-U-style columns
    def line2columns(line):
        global num_cols
        values = line.split("\t")
        if num_cols is None:
            num_cols = len(values)
        if file_format == conllu:
            if len(values) not in (10, 11):
                raise UDError("The CoNLL-U line does not contain 10 or 11 tab-separated columns: '{}'".format(line))
            if len(values) != num_cols:
                raise UDError("Inconsistent number of lines ({}) in the CoNLL-U line: '{}'".format(len(values), line))
            num_cols = len(values)
            if len(values) == 10:
                # Append an empty semantic label if not given in the file.
                values.append("_")
        elif file_format == conll:
            if len(values) != (10):
                raise UDError("The CoNLL line does not contain 10 tab-separated columns: '{}'".format(line))
            # CoNLL:   ID, FORM, LEMMA, CPOS,  POS, FEATS, HEAD, DEPREL, MISC, SEM
            # CoNLL-U: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS,  MISC,  SEM
            values = values[:3] + [values[4], values[4]] + values[5:8] + ["_"] * 2 + [values[9]]
        else:
            raise UDError("Unsupported file format: '{}'".format(file_format))
        return values

        # Internal representation classes

    class UDRepresentation:
        def __init__(self):
            # Characters of all the tokens in the whole file.
            # Whitespace between tokens is not included.
            self.characters = []
            # List of UDSpan instances with start&end indices into `characters`.
            self.tokens = []
            # List of UDWord instances.
            self.words = []
            # List of UDSpan instances with start&end indices into `characters`.
            self.sentences = []

    class UDSpan:
        def __init__(self, start, end):
            self.start = start
            # Note that self.end marks the first position **after the end** of span,
            # so we can use characters[start:end] or range(start, end).
            self.end = end

    class UDWord:
        def __init__(self, span, columns, is_multiword):
            # Span of this word (or MWT, see below) within ud_representation.characters.
            self.span = span
            # 10/11 columns of the CoNLL-U file: ID, FORM, LEMMA,...
            self.columns = columns
            # is_multiword==True means that this word is part of a multi-word token.
            # In that case, self.span marks the span of the whole multi-word token.
            self.is_multiword = is_multiword
            # Reference to the UDWord instance representing the HEAD (or None if root).
            self.parent = None
            # References to the UDWord instances representing the enhanced HEADs (or None if root).
            self.eparents = []
            # Enhanced deprels (same order as in self.eparents).
            self.edeprels = []
            # List of references to UDWord instances representing functional-deprel children.
            self.functional_children = []
            # Only consider required FEATS if provided, else take all.
            feats = sorted(columns[FEATS].split("|"))
            if file_format.FEATURES is not None:
                feats = filter(lambda f: f.split("=", 1)[0] in file_format.FEATURES, feats)
            self.columns[FEATS] = "|".join(sorted(feats))
            # *Donât* ignore language-specific deprel subtypes.
            self.columns[DEPREL] = columns[DEPREL]
            # Collect enhanced dependencies from HEAD, DEPREL, DEPS, remove duplicates, ignore deprel subtypes.
            enh_deps = set([(int(self.columns[HEAD]), self.columns[DEPREL])])
            if self.columns[DEPS] != '_':
                for d in self.columns[DEPS].split('|'):
                    n, rel = d.split(':')[:2]
                    enh_deps.add((int(n), rel))
            self.columns[DEPS] = list(sorted(enh_deps))

            # Precompute which deprels are CONTENT_DEPRELS and which FUNCTIONAL_DEPRELS
            universal_deprel = self.columns[DEPREL].split(":")[0]
            self.is_content_deprel = universal_deprel in file_format.CONTENT_DEPRELS
            self.is_functional_deprel = universal_deprel in file_format.FUNCTIONAL_DEPRELS
            if (self.is_content_deprel and self.is_functional_deprel):
                print("WARNING: relation <{}> is listed as both content and functional".format(self.columns[DEPREL]))
            if (universal_deprel not in ('punct', '???') and not (self.is_content_deprel or self.is_functional_deprel)):
                print("WARNING: relation <{}> is listed as neither content nor functional".format(self.columns[DEPREL]))
            # Semantic role (PBDUD-specific, optional column).
            self.sem = self.columns[SEM]

    ud = UDRepresentation()

    # Load the CoNLL-U file
    index, sentence_start = 0, None
    while True:
        line = file.readline()
        if not line:
            break
        line = line.rstrip("\r\n")

        # Handle sentence start boundaries
        if sentence_start is None:
            # Skip comments
            if line.startswith("#"):
                continue
            # Start a new sentence
            ud.sentences.append(UDSpan(index, 0))
            sentence_start = len(ud.words)
        if not line:
            try:
                # Add parent and children UDWord links and check there are no cycles
                def process_word(word):
                    if word.parent == "remapping":
                        raise UDError("There is a cycle in a sentence")
                    if word.parent is None:
                        head = int(word.columns[HEAD])
                        if head < 0 or head > len(ud.words) - sentence_start:
                            raise UDError("HEAD '{}' points outside of the sentence".format(word.columns[HEAD]))
                        if head:
                            parent = ud.words[sentence_start + head - 1]
                            word.parent = "remapping"
                            process_word(parent)
                            word.parent = parent

                for word in ud.words[sentence_start:]:
                    process_word(word)
                # func_children cannot be assigned within process_word
                # because it is called recursively and may result in adding one child twice.
                for word in ud.words[sentence_start:]:
                    if word.parent and word.is_functional_deprel:
                        word.parent.functional_children.append(word)

                # enhanced parents
                for word in ud.words[sentence_start:]:
                    # if len(word.columns[DEPS]) > 1:
                    #    print(word.columns)
                    for head, deprel in word.columns[DEPS]:
                        assert (head == 0 or int(ud.words[sentence_start + head - 1].columns[ID]) == head)
                        word.eparents.append(ud.words[sentence_start + head - 1])
                        word.edeprels.append(deprel)

                # TODO enhanced children?

                # Check there is a single root node
                if len([word for word in ud.words[sentence_start:] if word.parent is None]) != 1:
                    raise UDError("There are multiple roots in a sentence")

            except UDError as error:
                print("WARNING: encoutered invalid tree: {}".format(error))
                print("    all its tokens will receive 0 scores in all metrics!")

                for i, word in enumerate(ud.words[sentence_start:]):
                    ud.words[sentence_start + i] = UDWord(word.span,
                                                          [word.columns[0], word.columns[1], '???', '???', '???', '???',
                                                           '-1', '???', '_', '_', '_'], word.is_multiword)

            # End the sentence
            ud.sentences[-1].end = index
            sentence_start = None
            continue

        # Read next token/word
        columns = line2columns(line)

        # Skip empty nodes
        if "." in columns[ID]:
            continue

        # Delete spaces from FORM, so gold.characters == system.characters
        # even if one of them tokenizes the space. Use any Unicode character
        # with category Zs.
        if sys.version_info < (3, 0) and isinstance(line, str):
            columns[FORM] = columns[FORM].decode("utf-8")
        columns[FORM] = "".join(filter(lambda c: unicodedata.category(c) != "Zs", columns[FORM]))
        if sys.version_info < (3, 0) and isinstance(line, str):
            columns[FORM] = columns[FORM].encode("utf-8")
        if not columns[FORM]:
            raise UDError("There is an empty FORM in the CoNLL(-U) file")

        # Save token
        ud.characters.extend(columns[FORM])
        ud.tokens.append(UDSpan(index, index + len(columns[FORM])))
        index += len(columns[FORM])

        # Handle multi-word tokens to save word(s)
        if "-" in columns[ID]:
            # TODO shouldnât ID, HEAD etc. also be checked here?
            try:
                start, end = map(int, columns[ID].split("-"))
            except:
                raise UDError("Cannot parse multi-word token ID '{}'".format(columns[ID]))

            for _ in range(start, end + 1):
                word_line = file.readline().rstrip("\r\n")
                word_columns = line2columns(word_line)
                ud.words.append(UDWord(ud.tokens[-1], word_columns, is_multiword=True))
        # Basic tokens/words
        else:
            try:
                word_id = int(columns[ID])
            except:
                raise UDError("Cannot parse word ID '{}'".format(columns[ID]))
            if word_id != len(ud.words) - sentence_start + 1:
                raise UDError("Incorrect word ID '{}' for word '{}', expected '{}'".format(columns[ID], columns[FORM],
                                                                                           len(
                                                                                               ud.words) - sentence_start + 1))

            try:
                head_id = int(columns[HEAD])
            except:
                raise UDError("Cannot parse HEAD '{}'".format(columns[HEAD]))
            if head_id < 0:
                raise UDError("HEAD cannot be negative")

            ud.words.append(UDWord(ud.tokens[-1], columns, is_multiword=False))

    if sentence_start is not None:
        raise UDError("The CoNLL(-U) file does not end with empty line")

    return ud


# Evaluate the gold and system treebanks (loaded using load_dependency).
def evaluate(gold_ud, system_ud):
    class Score:
        def __init__(self, gold_total, system_total, correct, aligned_total=None):
            self.correct = correct
            self.gold_total = gold_total
            self.system_total = system_total
            self.aligned_total = aligned_total
            self.precision = correct / system_total if system_total else 0.0
            self.recall = correct / gold_total if gold_total else 0.0
            self.f1 = 2 * correct / (system_total + gold_total) if system_total + gold_total else 0.0
            self.aligned_accuracy = correct / aligned_total if aligned_total else aligned_total

    class AlignmentWord:
        def __init__(self, gold_word, system_word):
            self.gold_word = gold_word
            self.system_word = system_word

    class Alignment:
        def __init__(self, gold_words, system_words):
            self.gold_words = gold_words
            self.system_words = system_words
            self.matched_words = []
            self.matched_words_map = {}

        def append_aligned_words(self, gold_word, system_word):
            self.matched_words.append(AlignmentWord(gold_word, system_word))
            self.matched_words_map[system_word] = gold_word

    def lower(text):
        if sys.version_info < (3, 0) and isinstance(text, str):
            return text.decode("utf-8").lower()
        return text.lower()

    def spans_score(gold_spans, system_spans):
        correct, gi, si = 0, 0, 0
        while gi < len(gold_spans) and si < len(system_spans):
            if system_spans[si].start < gold_spans[gi].start:
                si += 1
            elif gold_spans[gi].start < system_spans[si].start:
                gi += 1
            else:
                correct += gold_spans[gi].end == system_spans[si].end
                si += 1
                gi += 1

        return Score(len(gold_spans), len(system_spans), correct)

    def alignment_score(alignment, key_fn=None, filter_fn=None):
        if filter_fn is not None:
            gold = sum(1 for gold in alignment.gold_words if filter_fn(gold))
            system = sum(1 for system in alignment.system_words if filter_fn(system))
            aligned = sum(1 for word in alignment.matched_words if filter_fn(word.gold_word))
        else:
            gold = len(alignment.gold_words)
            system = len(alignment.system_words)
            aligned = len(alignment.matched_words)

        if key_fn is None:
            # Return score for whole aligned words
            return Score(gold, system, aligned)

        def gold_aligned_gold(word):
            return word

        def gold_aligned_system(word):
            return alignment.matched_words_map.get(word, "NotAligned") if word is not None else None

        correct = 0
        for words in alignment.matched_words:
            if filter_fn is None or filter_fn(words.gold_word):
                if key_fn(words.gold_word, gold_aligned_gold) == key_fn(words.system_word, gold_aligned_system):
                    correct += 1

        return Score(gold, system, correct, aligned)

    def beyond_end(words, i, multiword_span_end):
        if i >= len(words):
            return True
        if words[i].is_multiword:
            return words[i].span.start >= multiword_span_end
        return words[i].span.end > multiword_span_end

    def extend_end(word, multiword_span_end):
        if word.is_multiword and word.span.end > multiword_span_end:
            return word.span.end
        return multiword_span_end

    def find_multiword_span(gold_words, system_words, gi, si):
        # We know gold_words[gi].is_multiword or system_words[si].is_multiword.
        # Find the start of the multiword span (gs, ss), so the multiword span is minimal.
        # Initialize multiword_span_end characters index.
        if gold_words[gi].is_multiword:
            multiword_span_end = gold_words[gi].span.end
            if not system_words[si].is_multiword and system_words[si].span.start < gold_words[gi].span.start:
                si += 1
        else:  # if system_words[si].is_multiword
            multiword_span_end = system_words[si].span.end
            if not gold_words[gi].is_multiword and gold_words[gi].span.start < system_words[si].span.start:
                gi += 1
        gs, ss = gi, si

        # Find the end of the multiword span
        # (so both gi and si are pointing to the word following the multiword span end).
        while not beyond_end(gold_words, gi, multiword_span_end) or \
                not beyond_end(system_words, si, multiword_span_end):
            if gi < len(gold_words) and (si >= len(system_words) or
                                         gold_words[gi].span.start <= system_words[si].span.start):
                multiword_span_end = extend_end(gold_words[gi], multiword_span_end)
                gi += 1
            else:
                multiword_span_end = extend_end(system_words[si], multiword_span_end)
                si += 1
        return gs, ss, gi, si

    def compute_lcs(gold_words, system_words, gi, si, gs, ss):
        lcs = [[0] * (si - ss) for i in range(gi - gs)]
        for g in reversed(range(gi - gs)):
            for s in reversed(range(si - ss)):
                if lower(gold_words[gs + g].columns[FORM]) == lower(system_words[ss + s].columns[FORM]):
                    lcs[g][s] = 1 + (lcs[g + 1][s + 1] if g + 1 < gi - gs and s + 1 < si - ss else 0)
                lcs[g][s] = max(lcs[g][s], lcs[g + 1][s] if g + 1 < gi - gs else 0)
                lcs[g][s] = max(lcs[g][s], lcs[g][s + 1] if s + 1 < si - ss else 0)
        return lcs

    def align_words(gold_words, system_words):
        alignment = Alignment(gold_words, system_words)

        gi, si = 0, 0
        while gi < len(gold_words) and si < len(system_words):
            if gold_words[gi].is_multiword or system_words[si].is_multiword:
                # A: Multi-word tokens => align via LCS within the whole "multiword span".
                gs, ss, gi, si = find_multiword_span(gold_words, system_words, gi, si)

                if si > ss and gi > gs:
                    lcs = compute_lcs(gold_words, system_words, gi, si, gs, ss)

                    # Store aligned words
                    s, g = 0, 0
                    while g < gi - gs and s < si - ss:
                        if lower(gold_words[gs + g].columns[FORM]) == lower(system_words[ss + s].columns[FORM]):
                            alignment.append_aligned_words(gold_words[gs + g], system_words[ss + s])
                            g += 1
                            s += 1
                        elif lcs[g][s] == (lcs[g + 1][s] if g + 1 < gi - gs else 0):
                            g += 1
                        else:
                            s += 1
            else:
                # B: No multi-word token => align according to spans.
                if (gold_words[gi].span.start, gold_words[gi].span.end) == (
                system_words[si].span.start, system_words[si].span.end):
                    alignment.append_aligned_words(gold_words[gi], system_words[si])
                    gi += 1
                    si += 1
                elif gold_words[gi].span.start <= system_words[si].span.start:
                    gi += 1
                else:
                    si += 1

        return alignment

    # Check that the underlying character sequences do match.
    if gold_ud.characters != system_ud.characters:
        index = 0
        while index < len(gold_ud.characters) and index < len(system_ud.characters) and \
                gold_ud.characters[index] == system_ud.characters[index]:
            index += 1

        raise UDError(
            "The concatenation of tokens in gold file and in system file differ!\n" +
            "First 20 differing characters in gold file: '{}' and system file: '{}'".format(
                "".join(gold_ud.characters[index:index + 20]),
                "".join(system_ud.characters[index:index + 20])
            )
        )

    # Align words
    alignment = align_words(gold_ud.words, system_ud.words)

    # Compute the F1-scores
    return {
        "Tokens": spans_score(gold_ud.tokens, system_ud.tokens),
        "Sentences": spans_score(gold_ud.sentences, system_ud.sentences),
        "Words": alignment_score(alignment),
        "UPOS": alignment_score(alignment, lambda w, _: w.columns[UPOS]),
        "XPOS": alignment_score(alignment, lambda w, _: w.columns[XPOS]),
        "UFeats": alignment_score(alignment, lambda w, _: w.columns[FEATS]),
        "AllTags": alignment_score(alignment, lambda w, _: (w.columns[UPOS], w.columns[XPOS], w.columns[FEATS])),
        "Lemmas": alignment_score(alignment, lambda w, ga: w.columns[LEMMA] if ga(w).columns[LEMMA] != "_" else "_"),
        "UAS": alignment_score(alignment, lambda w, ga: ga(w.parent)),
        "LAS": alignment_score(alignment, lambda w, ga: (ga(w.parent), w.columns[DEPREL])),
        # ELAS is an extension of LAS: all enhanced heads must also be correct
        "ELAS": alignment_score(alignment,
                                lambda w, ga: (ga(w.parent), w.columns[DEPREL], list(map(ga, w.eparents)), w.edeprels)),
        # SLAS is another extension of LAS: the semantic label must also be correct, only content nodes counted
        "SLAS": alignment_score(alignment, lambda w, ga: (ga(w.parent), w.columns[DEPREL], w.columns[SEM]),
                                filter_fn=lambda w: w.is_content_deprel),
        "CLAS": alignment_score(alignment, lambda w, ga: (ga(w.parent), w.columns[DEPREL]),
                                filter_fn=lambda w: w.is_content_deprel),
        "MLAS": alignment_score(alignment,
                                lambda w, ga: (ga(w.parent), w.columns[DEPREL], w.columns[UPOS], w.columns[FEATS],
                                               [(ga(c), c.columns[DEPREL], c.columns[UPOS], c.columns[FEATS])
                                                for c in w.functional_children]),
                                filter_fn=lambda w: w.is_content_deprel),
        "BLEX": alignment_score(alignment, lambda w, ga: (ga(w.parent), w.columns[DEPREL],
                                                          w.columns[LEMMA] if ga(w).columns[LEMMA] != "_" else "_"),
                                filter_fn=lambda w: w.is_content_deprel),
    }


def load_dependency_file(path, file_format=conllu):
    _file = open(path, mode="r", **({"encoding": "utf-8"} if sys.version_info >= (3, 0) else {}))
    return load_dependency(_file, file_format=file_format)


def evaluate_wrapper(args):
    # Load CoNLL(-U) files
    print("Using *{}* file format".format(args.format))
    file_format = eval(args.format)
    gold_ud = load_dependency_file(args.gold_file, file_format=file_format)
    system_ud = load_dependency_file(args.system_file, file_format=file_format)
    return evaluate(gold_ud, system_ud)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file", type=str,
                        help="Name of the CoNLL(-U) file with the gold data.")
    parser.add_argument("system_file", type=str,
                        help="Name of the CoNLL(-U) file with the predicted data.")
    parser.add_argument("--format", type=str, default="conllu", choices=["conllu", "conll"],
                        help="Dependency file format.")
    parser.add_argument("--verbose", "-v", default=False, action="store_true",
                        help="Print all metrics.")
    parser.add_argument("--counts", "-c", default=False, action="store_true",
                        help="Print raw counts of correct/gold/system/aligned words instead of prec/rec/F1 for all metrics.")
    args = parser.parse_args()

    # Evaluate
    evaluation = evaluate_wrapper(args)
    assert (evaluation["LAS"].f1 >= evaluation["ELAS"].f1)

    # Print the evaluation
    if not args.verbose and not args.counts:
        print("LAS F1 Score: {:.2f}".format(100 * evaluation["LAS"].f1))
        print("ELAS F1 Score: {:.2f}".format(100 * evaluation["ELAS"].f1))
        print("SLAS F1 Score: {:.2f}".format(100 * evaluation["SLAS"].f1))
        print("MLAS Score: {:.2f}".format(100 * evaluation["MLAS"].f1))
        print("BLEX Score: {:.2f}".format(100 * evaluation["BLEX"].f1))
    else:
        if args.counts:
            print("Metric     \t Correct   \t      Gold \t Predicted \t Aligned")
        else:
            print("Metric     \t Precision \t    Recall \t  F1 Score \t AligndAcc")
        # print("-----------+-----------+-----------+-----------+-----------")
        for metric in ["Tokens", "Sentences", "Words", "UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS",
                       "ELAS", "SLAS", "CLAS", "MLAS", "BLEX"]:
            if args.counts:
                print("{:11}\t{:10} \t{:10} \t{:10} \t{:10}".format(
                    metric,
                    evaluation[metric].correct,
                    evaluation[metric].gold_total,
                    evaluation[metric].system_total,
                    evaluation[metric].aligned_total or (evaluation[metric].correct if metric == "Words" else "")
                ))
            else:
                print("{:11}\t{:10.2f} \t{:10.2f} \t{:10.2f} \t{}".format(
                    metric,
                    100 * evaluation[metric].precision,
                    100 * evaluation[metric].recall,
                    100 * evaluation[metric].f1,
                    "{:10.2f}".format(100 * evaluation[metric].aligned_accuracy) if evaluation[
                                                                                        metric].aligned_accuracy is not None else ""
                ))


if __name__ == "__main__":
    main()


# Tests, which can be executed with `python -m unittest conll18_ud_eval`.
class TestAlignment(unittest.TestCase):
    @staticmethod
    def _load_words(words):
        """Prepare fake CoNLL-U files with fake HEAD to prevent multiple roots errors."""
        lines, num_words = [], 0
        for w in words:
            parts = w.split(" ")
            if len(parts) == 1:
                num_words += 1
                lines.append("{}\t{}\t_\t_\t_\t_\t{}\t_\t_\t_".format(num_words, parts[0], int(num_words > 1)))
            else:
                lines.append(
                    "{}-{}\t{}\t_\t_\t_\t_\t_\t_\t_\t_".format(num_words + 1, num_words + len(parts) - 1, parts[0]))
                for part in parts[1:]:
                    num_words += 1
                    lines.append("{}\t{}\t_\t_\t_\t_\t{}\t_\t_\t_".format(num_words, part, int(num_words > 1)))
        return load_dependency((io.StringIO if sys.version_info >= (3, 0) else io.BytesIO)("\n".join(lines + ["\n"])))

    def _test_exception(self, gold, system):
        self.assertRaises(UDError, evaluate, self._load_words(gold), self._load_words(system))

    def _test_ok(self, gold, system, correct):
        metrics = evaluate(self._load_words(gold), self._load_words(system))
        gold_words = sum((max(1, len(word.split(" ")) - 1) for word in gold))
        system_words = sum((max(1, len(word.split(" ")) - 1) for word in system))
        self.assertEqual((metrics["Words"].precision, metrics["Words"].recall, metrics["Words"].f1),
                         (correct / system_words, correct / gold_words, 2 * correct / (gold_words + system_words)))

    def test_exception(self):
        self._test_exception(["a"], ["b"])

    def test_equal(self):
        self._test_ok(["a"], ["a"], 1)
        self._test_ok(["a", "b", "c"], ["a", "b", "c"], 3)

    def test_equal_with_multiword(self):
        self._test_ok(["abc a b c"], ["a", "b", "c"], 3)
        self._test_ok(["a", "bc b c", "d"], ["a", "b", "c", "d"], 4)
        self._test_ok(["abcd a b c d"], ["ab a b", "cd c d"], 4)
        self._test_ok(["abc a b c", "de d e"], ["a", "bcd b c d", "e"], 5)

    def test_alignment(self):
        self._test_ok(["abcd"], ["a", "b", "c", "d"], 0)
        self._test_ok(["abc", "d"], ["a", "b", "c", "d"], 1)
        self._test_ok(["a", "bc", "d"], ["a", "b", "c", "d"], 2)
        self._test_ok(["a", "bc b c", "d"], ["a", "b", "cd"], 2)
        self._test_ok(["abc a BX c", "def d EX f"], ["ab a b", "cd c d", "ef e f"], 4)
        self._test_ok(["ab a b", "cd bc d"], ["a", "bc", "d"], 2)
        self._test_ok(["a", "bc b c", "d"], ["ab AX BX", "cd CX a"], 1)