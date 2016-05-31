#!/usr/bin/env python

"""Script for scoring timit"""

__author__ = 'Tomas Novacik'

import sys
import h5py
import editdistance
import argparse


parser = argparse.ArgumentParser(description='Scoring TIMIT results.')

parser.add_argument('--net-output', type=str, help='NN output in hdf5 format', default="output.hdf5")
parser.add_argument('--phoneme-map', type=str, help='File with integer mapping for phonemes', default="data/local/dict/units.txt")
parser.add_argument('--expectation', type=str, help='File with reference', default="data/dev/text")
parser.add_argument('--reduction', type=str, help='File with reduced mapping', default="./conf/phones.60-48-39.map")

args = parser.parse_args()

i_filename = args.net_output
m_filename = args.phoneme_map
e_filename = args.expectation
r_filename = args.reduction

i_file = h5py.File(i_filename)
with open(m_filename, "r") as m_file:
    mapping = {}
    for line in m_file.read().split("\n"):
        if line:
            key, val = line.split(" ")
            mapping[key] = int(val)

    i_map = {}
    with open(r_filename, "r") as r_file:
        for line in r_file.read().split("\n"):
            if line:
                l_split = line.split("\t")
                mapping[l_split[1]] = mapping[l_split[2]]
                i_map[mapping[l_split[1]]] = mapping[l_split[2]]

    e_output = {}
    with open(e_filename, "r") as e_file:
        for line in e_file.read().split("\n"):
            if line:
                l_split = line.split(" ")
                utt = l_split[0]
                e_output[utt] = [mapping[i] for i in l_split[1:] if i != 1]

    dist = 0
    items = 0
    for utt in i_file:
        out = i_file[utt]["data"].value
        classes = out.argmax(1).tolist()
        classes = [i_map[i] for i in classes if i in i_map]
        # remove blanks and repeating symbols and silence
        for i in xrange(len(classes) - 1, -1, -1):
            if classes[i] == 0 or classes[i] == 1 or (i < 0 and classes[i] == classes[i - 1]):
                del classes[i]
        # remove sils
        e_output[utt] = [i for i in e_output[utt] if i != 1]
        # remove repetitions after phoneme set reduction
        for i in xrange(len(classes) - 1, 0, -1):
            if classes[i] == classes[i-1]:
                del classes[i]

        for i in xrange(len(e_output[utt]) - 1, 0, -1):
            if e_output[utt][i] == e_output[utt][i-1]:
                del e_output[utt][i]

        dist += editdistance.eval(classes, e_output[utt])
        items += len(e_output[utt])

    print(str(dist / float(items) * 100) + " %")

# eof
