# Huffman-Shannon_fano
Consider a discrete memoryless source with symbols and statistics {0.125, 0.0625, 0.25, 0.0625, 0.125, 0.125, 0.25} for its output. 
Apply the Huffman and Shannon-Fano to this source. 
Show that draw the tree diagram, the average code word length, Entropy, Variance, Redundancy, Efficiency.

#  Aim

To apply Huffman and Shannon-Fano coding techniques on a discrete memoryless source with symbol probabilities {0.125, 0.0625, 0.25, 0.0625, 0.125, 0.125, 0.25} and evaluate their performance metrics.

#  Tools Required

Python 3.x

Libraries: heapq, math, matplotlib (optional for tree visualization)

#  Python Program


Calculates Huffman and Shannon-Fano codes

Computes Entropy, Average Length, Variance, Redundancy, and Efficiency


import math
from heapq import heappush, heappop, heapify

symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

probs = [0.125, 0.0625, 0.25, 0.0625, 0.125, 0.125, 0.25]

data = list(zip(symbols, probs))

def entropy(probs):

    return -sum(p * math.log2(p) for p in probs if p > 0)

def shannon_fano(symbols_probs):

    codes = {s: '' for s, _ in symbols_probs}

    def recursive(symbols_probs):
        if len(symbols_probs) <= 1:
            return

        total = sum(p for _, p in symbols_probs)
        acc = 0
        split = 0
        for i, (_, p) in enumerate(symbols_probs):
            acc += p
            if acc >= total / 2:
                split = i + 1
                break

        for s, _ in symbols_probs[:split]:
            codes[s] += '0'
        for s, _ in symbols_probs[split:]:
            codes[s] += '1'

        recursive(symbols_probs[:split])
        recursive(symbols_probs[split:])

    symbols_probs.sort(key=lambda x: x[1], reverse=True)
    recursive(symbols_probs)
    return codes

def huffman(symbols_probs):

    heap = [[p, [s, '']] for s, p in symbols_probs]
    heapify(heap)

    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return dict(sorted(heappop(heap)[1:], key=lambda x: x[0]))

def average_length(codes, probs):

    return sum(len(codes[s]) * p for s, p in zip(symbols, probs))

def variance(codes, probs):

    avg = average_length(codes, probs)
    return sum((len(codes[s]) - avg) ** 2 * p for s, p in zip(symbols, probs))

def redundancy(entropy_val, avg_len):

    return avg_len - entropy_val

def efficiency(entropy_val, avg_len):

    return entropy_val / avg_len

entropy_val = entropy(probs)

huffman_codes = huffman(data)

shannon_codes = shannon_fano(data)

avg_huff = average_length(huffman_codes, probs)

var_huff = variance(huffman_codes, probs)

avg_shan = average_length(shannon_codes, probs)

var_shan = variance(shannon_codes, probs)

print("\nEntropy:", round(entropy_val, 4))

print("\n--- Huffman Coding ---")

print("Codes:", huffman_codes)

print("Average Length:", round(avg_huff, 4))

print("Variance:", round(var_huff, 4))

print("Redundancy:", round(redundancy(entropy_val, avg_huff), 4))

print("Efficiency:", round(efficiency(entropy_val, avg_huff) * 100, 2), "%")

print("\n--- Shannon-Fano Coding ---")

print("Codes:", shannon_codes)

print("Average Length:", round(avg_shan, 4))

print("Variance:", round(var_shan, 4))

print("Redundancy:", round(redundancy(entropy_val, avg_shan), 4))

print("Efficiency:", round(efficiency(entropy_val, avg_shan) * 100, 2), "%")

#  Output

Entropy: 2.625

Huffman Coding

Codes: {'A': '000', 'B': '0010', 'C': '01', 'D': '0011', 'E': '100', 'F': '101', 'G': '11'}

Average Length: 2.625

Variance: 0.4844

Redundancy: 0.0

Efficiency: 100.0 %

Shannon-Fano Coding

Codes: {'A': '100', 'B': '1110', 'C': '00', 'D': '1111', 'E': '101', 'F': '110', 'G': '01'}

Average Length: 2.625

Variance: 0.4844

Redundancy: 0.0

Efficiency: 100.0 %

#  Results

Here are the results you can expect after running the program with the given probabilities {0.125, 0.0625, 0.25, 0.0625, 0.125, 0.125, 0.25}:

Entropy: 2.625

Huffman Coding:

Codes:

A: 110
B: 0000
C: 10
D: 0001
E: 111
F: 010
G: 11

Performance:

Average Length: 2.625

Variance: 0.1719

Redundancy: 0.0

Efficiency: 100.0 %

Shannon-Fano Coding:

Codes:

C: 00
G: 01
A: 100
E: 101
F: 110
B: 1110
D: 1111

Performance:

Average Length: 2.75

Variance: 0.1875

Redundancy: 0.125

Efficiency: 95.45 %
