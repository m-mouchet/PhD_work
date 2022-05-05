import csv
import ast
import numpy as np
import sys
csv.field_size_limit(sys.maxsize)


def WriteMomentsFile(FileName, idx, res):
    with open(FileName, 'w') as csvfile:
        fieldnames = ['idx', 'm0', 'm1', 'var0', 'var1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(idx)):
            writer.writerow({'idx': idx[i], 'm0': list(res[i][0]), 'm1': list(res[i][1]), 'var0': res[i][2], 'var1': res[i][3]})
    return 0


def ReadMomentsFile(FileName):
    idx = []
    mom0 = []
    mom1 = []
    norm0 = []
    norm1 = []
    with open(FileName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pair = row['idx']
            idx.append(int(pair))
            pair = row['m0']
            pair = ast.literal_eval(pair)
            mom0.append(np.array(pair))
            pair = row['m1']
            pair = ast.literal_eval(pair)
            mom1.append(np.array(pair))
            pair = row['n0']
            pair = ast.literal_eval(pair)
            norm0.append(np.array(pair))
            pair = row['n1']
            pair = ast.literal_eval(pair)
            norm1.append(np.array(pair))
    return np.array(idx), [mom0, mom1, np.array(var0), np.array(var1)]


def WriteAllPairsErrorsFile(FileName, res):
    with open(FileName, 'w') as csvfile:
        fieldnames = ['ref', 'indeces', 'mom0', 'mom1', 'var0', 'var1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(res)):
            writer.writerow({"ref": res[i][0], "idxes": list(res[i][1]), "mom0": list(res[i][2]), "mom1": list(res[i][3]), "var0": res[i][4], "var1": res[i][5]})
    return 0


def ReadAllPairsErrorsFile(FileName):
    refs = []
    indexes = []
    mom0 = []
    mom1 = []
    var0 = []
    var1 = []
    with open(FileName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ref = row["ref"]
            refs.append(int(ref))
            pair = row['idxes']
            pair = ast.literal_eval(pair)
            indexes.append(np.array(pair))
            pair = row['mom0']
            pair = ast.literal_eval(pair)
            mom0.append(np.array(pair))
            pair = row['mom1']
            pair = ast.literal_eval(pair)
            mom1.append(np.array(pair))
            pair = row['var0']
            pair = ast.literal_eval(pair)
            indexes.append(np.array(pair))
            pair = row['var1']
            pair = ast.literal_eval(pair)
            indexes.append(np.array(pair))
    return np.array(refs), indexes, mom0, mom1, var0, var1

