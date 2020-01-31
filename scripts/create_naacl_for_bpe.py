import sys

def naacl_line(sen_no, l1_pos, l2_pos, alignment_type="S"):
    """
        :param sen_no: 0-indexed sentence number
        :param l1_pos: 0-indexed position in the L1
        :param l2_pos: 0-indexed position in the L2
        :param alignment_type: S or P for sure or probable
    """
    return f"{sen_no+1} {l1_pos+1} {l2_pos+1} {alignment_type}"

def main(split_file, naacl_file):
    sf = open(split_file)
    nf = open(naacl_file, "w+")

    for sen_idx, line in enumerate(sf):
        line = line.strip()
        tokens = line.split()
        l2_pos = 0
        for l1_pos, token in enumerate(tokens):
            nf.write(f"{naacl_line(sen_idx, l1_pos, l2_pos)}\n")
            if token[-2:] != "@@":
                l2_pos += 1

    sf.close()
    nf.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python create_naacl_for_bpe.py <split_file> <naacl_file>")
        sys.exit(1)
    split_file = sys.argv[1]
    naacl_file = sys.argv[2]
    main(split_file, naacl_file)
