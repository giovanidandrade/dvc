from os import environ, listdir
from os.path import isfile, join

from librosa import load, lpc
from librosa.sequence import dtw
from json import dump

import numpy as np

path = environ["SAMPLE_PATH"]
sample_paths = [
    file for file in listdir(path)
    if isfile(join(path, file))
]

results = {}

for sample in sample_paths:
    amps, sample_rate = load(join(path, sample), sr = None)

    window_time = 20e-3
    window_samples = int(sample_rate * window_time)

    amp_chunks = [
        amps[n : n + window_samples]
        for n in range(0, len(amps), window_samples)  
    ]

    lpc_chunks = [
        lpc(chunk, order = 16).tolist()
        for chunk in amp_chunks
    ]

    results[sample] = lpc_chunks
    #print(f"{sample}: {len(lpc_chunks)} janelas")

def find_all_prefix(dic, prefix):
    """Given a dictionary, find all keys that start with the same prefix,
    and returns an array of (suffix, value)."""

    prefix_length = len(prefix)
    results = []
    for k, v in dic.items():
        if k.startswith(prefix):
            suffix = k[prefix_length:]
            results += [(suffix, v)]
    
    return results

def find_most_representative(coeffs):
    """Given a vector of coefficients in the form (name, coeffs), applies
    DTW to the coeffs and returns (name, dist) of the most representative."""

    dists = [[name, 0] for (name, _) in coeffs]
    for index, elem1 in enumerate(coeffs):
        for elem2 in coeffs:
            # transposing to meet shape criterion
            costs, path = dtw(
                np.transpose(elem1[1]),
                np.transpose(elem2[1])
            )

            dist = costs[
                path[-1, 0], path[-1, 1]
            ]

            dists[index][1] += dist
    
    return tuple(min(dists, key = lambda tp: tp[1]))

codebook = {}
for prefix in ["baixo", "cima", "esquerda", "direita"]:
    coeffs = find_all_prefix(results, prefix)
    rep = find_most_representative(coeffs)

    print(f"{prefix + rep[0]}: {rep[1]}")
    codebook[prefix + rep[0]] = results[prefix + rep[0]]

with open('samples.json', 'w') as output:
    dump(codebook, output)