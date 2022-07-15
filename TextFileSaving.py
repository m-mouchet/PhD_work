import csv
import ast
import numpy as np
import sys
csv.field_size_limit(sys.maxsize)


def WriteEdgesGraph(FileName, pairs):
    with open(FileName, 'w') as csvfile:
        fieldnames = ['pairs']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(pairs)):
            writer.writerow({'pairs': pairs[i]})
    return 0


def WriteMomentsFile(FileName, idx, res):
    with open(FileName, 'w') as csvfile:
        fieldnames = ['idx', 'm0', 'm1', 'var0', 'var1', 'tot_var0', 'tot_var1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(idx)):
            writer.writerow({'idx': idx[i], 'm0': list(res[i][0]), 'm1': list(res[i][1]), 'var0': list(res[i][2]), 'var1': list(res[i][3]),'tot_var0': res[i][4], 'tot_var1': res[i][5]})
    return 0


def WriteALLMomentsFile(FileName, idx, res):
    with open(FileName, 'w') as csvfile:
        fieldnames = ['idx', 'mhr0', 'mhr1', 'mhh0', 'mhh1', 'mk0', 'mk1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(idx)):
            writer.writerow({'idx': idx[i], 'mhr0': list(res[i][0]), 'mhr1': list(res[i][1]), 'mhh0': list(res[i][2]), 'mhh1': list(res[i][3]),'mk0': list(res[i][4]), 'mk1': list(res[i][5])})
    return 0


def WriteBPMomentsFile(FileName, idx, res):
    with open(FileName, 'w') as csvfile:
        fieldnames = ['idx', 'm0', 'm1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(idx)):
            writer.writerow({'idx': idx[i], 'm0': list(res[i][0]), 'm1': list(res[i][1])})
    return 0


def ReadALLMomentsFile(FileName):
    idx = []
    mhr0 = []
    mhr1 = []
    mhh0 = []
    mhh1 = []
    mk0 = []
    mk1 = []
    with open(FileName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pair = row['idx']
            idx.append(int(pair))
            pair = row['mhr0']
            pair = ast.literal_eval(pair)
            mhr0.append(np.array(pair))
            pair = row['mhr1']
            pair = ast.literal_eval(pair)
            mhr1.append(np.array(pair))
            pair = row['mhh0']
            pair = ast.literal_eval(pair)
            mhh0.append(np.array(pair))
            pair = row['mhh1']
            pair = ast.literal_eval(pair)
            mhh1.append(np.array(pair))
            pair = row['mk0']
            pair = ast.literal_eval(pair)
            mk0.append(np.array(pair))
            pair = row['mk1']
            pair = ast.literal_eval(pair)
            mk1.append(np.array(pair))
    return [np.array(idx), mhr0, mhr1, mhh0, mhh1, mk0, mk1]


def ReadMomentsFile(FileName):
    idx = []
    mom0 = []
    mom1 = []
    var0 = []
    var1 = []
    tot_var0 = []
    tot_var1 = []
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
            pair = row['var0']
            pair = ast.literal_eval(pair)
            var0.append(np.array(pair))
            pair = row['var1']
            pair = ast.literal_eval(pair)
            var1.append(np.array(pair))
            pair = row['tot_var0']
            pair = ast.literal_eval(pair)
            tot_var0.append(np.array(pair))
            pair = row['tot_var1']
            pair = ast.literal_eval(pair)
            tot_var1.append(np.array(pair))
    return np.array(idx), [mom0, mom1, var0, var1, np.array(tot_var0), np.array(tot_var1)]


def ReadBPMomentsFile(FileName):
    idx = []
    mom0 = []
    mom1 = []
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
    return [np.array(idx), mom0, mom1]


def WriteAllPairsErrorsFile(FileName, res):
    with open(FileName, 'w') as csvfile:
        fieldnames = ['ref', 'indeces', 'var0', 'var1', 'tot_var0', 'tot_var1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(res)):
            writer.writerow({"ref": res[i][0], "indeces": list(res[i][1]), 'm0': list(res[i][2]), 'm1': list(res[i][3]), 'var0': list(res[i][4]), 'var1': list(res[i][5]),'tot_var0': res[i][6], 'tot_var1': res[i][7]})
    return 0


def ReadAllPairsErrorsFile(FileName):
    refs = []
    indexes = []
    mom0 = []
    mom1 = []
    var0 = []
    var1 = []
    tot_var0 = []
    tot_var1 = []
    with open(FileName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ref = row["ref"]
            refs.append(int(ref))
            pair = row['indeces']
            pair = ast.literal_eval(pair)
            indexes.append(np.array(pair))
            pair = row['m0']
            pair = ast.literal_eval(pair)
            mom0.append(np.array(pair))
            pair = row['m1']
            pair = ast.literal_eval(pair)
            mom1.append(np.array(pair))
            pair = row['var0']
            pair = ast.literal_eval(pair)
            var0.append(np.array(pair))
            pair = row['var1']
            pair = ast.literal_eval(pair)
            var1.append(np.array(pair))
            pair = row['tot_var0']
            pair = ast.literal_eval(pair)
            tot_var0.append(np.array(pair))
            pair = row['tot_var1']
            pair = ast.literal_eval(pair)
            tot_var1.append(np.array(pair))
    return np.array(refs), indexes, mom0, mom1, var0, var1, np.array(tot_var0), np.array(tot_var1)
