from typing import List, Tuple, Union
import numpy.typing as npt
import numpy as np

def enumerate_pairs(fastafile: str) -> List[Tuple[int, int]]:
    # 課題 2-1
    seq = ""
    with open(fastafile) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == ">":
                continue
            seq += line.upper()
    seq = seq.replace("T", "U")

    pair_set = {("A", "U"), ("U", "A"), ("C", "G"), ("G", "C")}
    results = []

    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if (seq[i], seq[j]) in pair_set:
                results.append((i + 1, j + 1))
    return results

def enumerate_possible_pairs(fastafile: str, min_distance: int=4) -> List[Tuple[int, int]]:
    # 課題 2-2
    seq = ""
    with open(fastafile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq += line.upper()

    seq = seq.replace("T", "U")
    pair_set = {("A", "U"), ("U", "A"), ("C", "G"), ("G", "C")}
    results = []

    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if (seq[i], seq[j]) in pair_set:
                if (j + 1) - (i + 1) >= min_distance:  # 1-origin差值
                    results.append((i + 1, j + 1))

    return results

def enumerate_continuous_pairs(fastafile: str, min_distance: int=4, min_length: int=2) -> List[Tuple[int, int, int]]:
    # 課題 2-3
    possible_pairs = set(enumerate_possible_pairs(fastafile, min_distance))
    results = []

    for i, j in sorted(possible_pairs):
        length = 0
        while (i + length, j - length) in possible_pairs:
            length += 1
        if length >= min_length:
            results.append((i, j, length))
    return results

def create_dotbracket_notation(fastafile: str, min_distance: int=4, min_length: int=2) -> str:
    # 課題 2-4
    seq = ""
    with open(fastafile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq += line.upper()
    n = len(seq)
    stems = ['.'] * n

    continuous_pairs = enumerate_continuous_pairs(fastafile, min_distance, min_length)
    for i, j, l in continuous_pairs:
        for k in range(l):
            # iとｊは１-origin.だから、1を引くと、添え字になる
            stems[i + k - 1] = "("
            stems[j - k - 1] = ")"
    return "".join(stems)

if __name__ == "__main__":
    filepath = "data/AUCGCCAU.fasta"
    # 課題 2-1
    print(enumerate_pairs(filepath))
    # 課題 2-2
    print(enumerate_possible_pairs(filepath))
    # 課題 2-3
    print(enumerate_continuous_pairs(filepath, 2))
    # 課題 2-4
    print(create_dotbracket_notation(filepath, 2))


