import sys

import random

def naacl_line(sen_no, l1_pos, l2_pos, alignment_type="S"):
    """
        :param sen_no: 0-indexed sentence number
        :param l1_pos: 0-indexed position in the L1
        :param l2_pos: 0-indexed position in the L2
        :param alignment_type: S or P for sure or probable
    """
    return f"{sen_no+1} {l1_pos+1} {l2_pos+1} {alignment_type}"

def main(input_file, output_file, naacl_file):

    # Open files.
    fi = open(input_file)
    fo = open(output_file, "w+")
    if naacl_file:
        fn = open(naacl_file, "w+")

    for sen_idx, line in enumerate(fi):
        line = line.strip()
        tokens = line.split()
        split_tokens = []
        l1_idx = 0
        for l2_idx, token in enumerate(tokens):
            if random.random() < 0.5:
                split_tokens.append(token)
                if naacl_file:
                    fn.write(f"{naacl_line(sen_idx, l1_idx, l2_idx)}\n")
                    l1_idx += 1
            elif len(token) > 1:
                split_point = len(token) // 2
                split_tokens.append(token[:split_point])
                split_tokens.append(token[split_point:])
                if naacl_file:
                    fn.write(f"{naacl_line(sen_idx, l1_idx, l2_idx)}\n")
                    l1_idx += 1
                    fn.write(f"{naacl_line(sen_idx, l1_idx, l2_idx)}\n")
                    l1_idx += 1
        fo.write(f"{' '.join(split_tokens)}\n")

    # Close files.
    fi.close()
    fo.close()
    if naacl_file:
        fn.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_toy_data.py <input_file> <output_file> <naacl_file>")
        sys.exit(0)
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    naacl_file = None
    if len(sys.argv) > 3:
        naacl_file = sys.argv[3]

    main(input_file, output_file, naacl_file)
