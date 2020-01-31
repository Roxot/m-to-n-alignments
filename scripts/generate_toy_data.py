import random
import sys

max_sentence_length = 10

prefixes = ["bee", "buu", "boo", "baa", "bii", "dee", "duu", "doo", "daa", "dii"] + [""]*5
roots = ["dem", "dum", "dim", "dom", "dam", "zem", "zum", "zim", "zom", "zem"]
suffixes = ["", "be", "bu", "bi", "bo", "be", "ze", "zu", "zi", "zo", "ze"] + [""]*5
eos = "."

def construct_word(root, prefix="", suffix=""):
    return f"{prefix}{root}{suffix}"

def generate_data(size, output_file_prefix):
    split_dataset = open(f"{output_file_prefix}.split", "w+")
    merged_dataset = open(f"{output_file_prefix}.merged", "w+")

    for _ in range(size):
        random_sentence_length = random.randint(1, max_sentence_length-1)
        tokens = []
        split_tokens = []
        for _ in range(random_sentence_length):

            random_prefix = prefixes[random.randint(0, len(prefixes)-1)]
            random_root = roots[random.randint(0, len(roots)-1)]
            random_suffix = suffixes[random.randint(0, len(suffixes)-1)]
            random_word = construct_word(random_root, prefix=random_prefix, suffix=random_suffix)

            tokens.append(random_word)
            if len(random_prefix) > 0:
                split_tokens.append(random_prefix+"@@")

            if len(random_suffix) > 0:
                split_tokens.append(random_root+"@@")
            else:
                split_tokens.append(random_root)

            split_tokens.append(random_suffix)
        tokens.append(eos)
        split_tokens.append(eos)

        merged_sentence = " ".join(tokens)
        split_sentence = " ".join(split_tokens)

        split_dataset.write(f"{split_sentence}\n")
        merged_dataset.write(f"{merged_sentence}\n")

    split_dataset.close()
    merged_dataset.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python create_artificial_dataset <dataset_size> <output_file_prefix>")
        sys.exit(1)

    size = int(sys.argv[1])
    output_file_prefix = sys.argv[2]
    generate_data(size, output_file_prefix)
