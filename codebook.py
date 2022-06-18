from os import environ, listdir
from os.path import isfile, join

from librosa import load, lpc
from json import dump

path = environ["SAMPLE_PATH"]
sample_paths = [
    file for file in listdir(path)
    if isfile(join(path, file))
]

results = {}

for sample in sample_paths:
    amps, sample_rate = load(join(path, sample))

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

with open('samples.json', 'w') as output:
    dump(results, output)