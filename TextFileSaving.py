import csv
import ast
import numpy as np

def WriteMomentsFile(FileName, idx, res, N):
    with open(FileName, 'w') as csvfile:
        fieldnames = ['idx','m0','m1','n0','n1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(N):
            writer.writerow({'idx': idx[i],'m0': list(res[i][0]),'m1': list(res[i][1]), 'n0': list(res[i][2]),'n1': list(res[i][3])})
    return 0


def ReadMoments(FileName):
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
    return np.array(idx), [mom0, mom1, norm0, norm1]



def WritePairsFile(FileName, pairs):
    with open(FileName, 'w') as csvfile:
        fieldnames = ["pairs"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(pairs)):
            writer.writerow({"pairs": list(pairs[i])})
    return 0


def ReadPairs(FileName):
    pairs = []
    with open(FileName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            term = row['pairs']
            term = ast.literal_eval(term)
            pairs.append(tuple(term))
    return pairs


def WriteMediansFile(FileName, indexes, res):
    with open(FileName, 'w') as csvfile:
        fieldnames = ["idx", "err"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(indexes)):
            writer.writerow({"idx": indexes[i], "err": res[i]})
    return 0


def ReadMediansFile(FileName):
    indexes = []
    medians = []
    with open(FileName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            idx = row["abs"]
            indexes.append(int(idx))
            err = row["err"]
            medians.append(np.float32(err))
    return np.array(indexes), np.array(medians)


def WriteAllPairsErrorsFile(FileName, ref, res):
    with open(FileName, 'w') as csvfile:
        fieldnames = ['ref','errors_list']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(ref)):
            writer.writerow({"ref": ref[i], "errors_list": list(res[i])})
    return 0


def ReadAllPairsErrorsFile(FileName):
    indexes = []
    res = []
    with open(FileName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            idx = row["ref"]
            indexes.append(int(idx))
            pair = row['errors_list']
            pair = ast.literal_eval(pair)
            res.append(np.array(pair))
    return np.array(indexes), res