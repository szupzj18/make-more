import torch
'''
This script reads a list of names from a file, builds a character vocabulary,
and constructs a bigram count matrix to analyze the frequency of character pairs.
'''
print("hello, make more.")

words = open("names.txt", "r").read().splitlines()
print(f"Dataset size: {len(words)} names")

# Create character vocabulary
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0  # special start/end token
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {''.join(chars)}")

print(f"Sample names: {words[:10]}")

# Build bigram counts
N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)

for w in words:
    chs = ['.'] + list(w) + ['.']  # add start/end tokens
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

print(f"Bigram count matrix shape: {N.shape}")
print(f"Total bigrams: {N.sum()}")