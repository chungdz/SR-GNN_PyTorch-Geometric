#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import numpy as np
import json

dataset = 'events.csv'

origin_session = {}
print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    
    reader = csv.DictReader(f, delimiter=',')
    for data in reader:
        if data['event'] != 'view':
            continue
        vid = data['visitorid']
        if vid not in origin_session:
            origin_session[vid] = []
        origin_session[vid].append((int(data['timestamp']) / 1000, int(data['itemid'])))
print("-- Reading data @ %ss" % datetime.datetime.now())

session_id = 0
sess_clicks = {}
sess_date = {}
interval = datetime.timedelta(minutes=30)
for k, v in origin_session.items():
    cur_len = len(v)
    if cur_len < 2:
        continue
    clicks = sorted(v, key=lambda t: t[0])

    new_session = [clicks[0][1]]
    new_date = clicks[0][0]
    for i in range(1, cur_len):
        if (datetime.datetime.fromtimestamp(clicks[i][0]) - datetime.datetime.fromtimestamp(clicks[i - 1][0])) <= interval:
            new_session.append(clicks[i][1])
        else:
            if len(new_session) > 1:
                sess_clicks[session_id] = new_session
                sess_date[session_id] = new_date
                session_id += 1
            new_session = [clicks[i][1]]
            new_date = clicks[i][0]
    
    if len(new_session) > 1:
        sess_clicks[session_id] = new_session
        sess_date[session_id] = new_date
        session_id += 1

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

# 只保留出现次数大于5的item，并且如果session删改后小于2，删除
length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# Split out test set based on dates
dates = list(sess_date.items())

# 7 days for test
t_str = '2015-09-01 00:00:00'
splitdate = datetime.datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S')

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate.timestamp(), dates)
tes_sess = filter(lambda x: x[1] > splitdate.timestamp(), dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print(len(tra_sess))    # 186670    # 7966257
print(len(tes_sess))    # 15979     # 15324
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
item_dict['<pad>'] = 0
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print('item count', item_ctr)     # 43098, 37484
    return train_ids, train_dates, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs

# session id, session dates, session sequence
tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()


def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids

# 最后一天作为标签，并且生成大量子session
tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all_length = []

for seq in tra_seqs:
    all_length.append(len(seq))
for seq in tes_seqs:
    all_length.append(len(seq))

seq_unique, seq_counts = np.unique(all_length, return_counts=True)
print('avg length: ', seq_unique, seq_counts)

if not os.path.exists('retailrocket'):
    os.makedirs('retailrocket')
    os.makedirs('retailrocket/raw')
    os.makedirs('retailrocket/processed')
    os.makedirs('retailrocket/result')
    os.makedirs('retailrocket/checkpoint')
pickle.dump(tra, open('retailrocket/raw/train.txt', 'wb'))
pickle.dump(tes, open('retailrocket/raw/test.txt', 'wb'))
pickle.dump(tra_seqs, open('retailrocket/raw/all_train_seq.txt', 'wb'))
json.dump(item_dict, open('retailrocket/item_dict.json', 'w', encoding='utf-8'))


print('Done.')
