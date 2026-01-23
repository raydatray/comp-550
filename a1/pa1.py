import numpy as np


def load_dataset(file_class0: str, file_class1: str) -> tuple[list[str], np.ndarray]:
    sentences = []
    labels = []

    with open(file_class0, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
                labels.append(0)

    with open(file_class1, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
                labels.append(1)

    return sentences, np.array(labels)


def main():
    print("loading dataset 1: sentiment classification (semantics")
    sentences_synsem, labels_synsem = load_dataset("synsem0.txt", "synsem1.txt")

    print("loading dataset 2: double letters classification (spelling)")
    sentences_morph, label_morph = load_dataset("morphphon0.txt", "morphphon1.txt")


if __name__ == "__main__":
    main()
