from typing import List, Union
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.units import current


def base_count(fastafile: str) -> List[int]:
    # 課題 1-1
    seq = ""
    with open(fastafile) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == ">":
                continue
            seq += line.upper()
    return [seq.count("A"), seq.count("T"), seq.count("G"), seq.count("C")] # A, T, G, C

def gen_rev_comp_seq(fastafile: str) -> str:
    # 課題 1-2
    seq = ""
    with open(fastafile) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == ">":
                continue
            seq += line.upper()

    comp_map = {"A":"T", "T":"A", "G":"C", "C":"G"}
    comp_seq = "".join(comp_map[base] for base in seq)
    return comp_seq[::-1]

def calc_gc_content(fastafile: str, window: int=1000, step: int=300) -> Union[npt.NDArray[np.float64], List[float]]:
    # 課題 1-3
    seq = ""
    with open(fastafile) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == ">":
                continue
            seq += line.upper()

    gc_contents = []
    for i in range(0, len(seq) - window + 1, step):
        sub_seq = seq[i:i + window]
        gc = (sub_seq.count("G") + sub_seq.count("C")) / len(sub_seq) * 100
        gc_contents.append(gc)
    # 値を出力するところまで。matplotlibを使う部分は別途実装してください。
    return gc_contents

def search_motif(fastafile: str, motif: str) -> List[str]:
    # 課題 1-4
    seq = ""
    with open(fastafile) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == ">":
                continue
            seq += line.upper()

    motif_seq = motif.upper()
    comp_map = {"A": "T", "T": "A", "G": "C", "C": "G"}
    rev_motif_seq = "".join(comp_map[base] for base in motif_seq)[::-1]

    results = []
    m_len = len(motif_seq)

    # 逆相補鎖の場合
    for i in range(0, len(seq) - m_len + 1):
        if seq[i:i + m_len] == rev_motif_seq:
            results.append(f"R{i + m_len}")
    # 順方向の場合
    for i in range(0, len(seq) - m_len + 1):
        if seq[i:i + m_len] == motif_seq:
            results.append(f"F{i+1}")

    return results

def translate(fastafile: str) -> List[str]:
    # 課題 1-5
    seq = ""
    with open(fastafile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq += line.upper()

    comp_map = {"A": "T", "T": "A", "G": "C", "C": "G"}
    rev_motif_seq = "".join(comp_map[base] for base in seq)[::-1]

    codon_table = {
        "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
        "TAT": "Y", "TAC": "Y", "TAA": "_", "TAG": "_",
        "TGT": "C", "TGC": "C", "TGA": "_", "TGG": "W",
        "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
        "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G"
    }

    proteins = []

    for base_seq in [seq, rev_motif_seq]:
        for frame in [0, 1, 2]:
            aa_seq = ""
            for i in range(frame, len(base_seq) - 2, 3):
                codon = base_seq[i:i + 3]
                aa_seq += codon_table.get(codon, "")

            current = ""
            recording = False
            for aa in aa_seq:
                if not recording:
                    if aa == "M":
                        current = "M"
                        recording = True
                else:
                    current += aa
                    if aa == "_":
                        proteins.append(current)
                        current = ""
                        recording = False
            if recording and current:
                proteins.append(current)

    return proteins

if __name__ == "__main__":
    filepath = "data/NT_113952.1.fasta"
    # 課題 1-1
    print(base_count(filepath))
    # 課題 1-2
    print(gen_rev_comp_seq(filepath))
    # 課題 1-3
    print(calc_gc_content(filepath))
    gc_val = calc_gc_content(filepath)
    plt.figure(figsize=(10, 4))
    plt.plot(gc_val)
    plt.xlabel("Window index")
    plt.ylabel("GC content")
    plt.title("GC content of sequence")
    plt.show()
    # 課題 1-4
    print(search_motif(filepath, "ATG"))
    # 課題 1-5
    print(translate(filepath))
