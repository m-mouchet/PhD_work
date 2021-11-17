import csv


def WriteMomentsFile(FileName, res, N):
    with open(FileName, 'w') as csvfile:
        fieldnames = ['m0','m1','n0','n1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(N):
            writer.writerow({'m0': list(res[i][0]),'m1': list(res[i][1]), 'n0': list(res[i][2]),'n1': list(res[i][3])})
    return 0