import os
import sys

def load_triples(file_path):
    triples = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            triple = tuple(line.strip().split('\t'))
            triples.add(triple)
    return triples

def main(data_dir):
    train_triples = load_triples(os.path.join(data_dir, 'train.txt'))
    test_triples = load_triples(os.path.join(data_dir, 'test.txt'))
    valid_triples = load_triples(os.path.join(data_dir, 'valid.txt'))

    test_valid_triples = test_triples | valid_triples

    only_in_train = train_triples - test_valid_triples
    print("Trainのみのtriple:")
    for triple in only_in_train:
        print('\t'.join(triple))

    only_in_test_valid = test_valid_triples - train_triples
    print("\nTest/Validのみのtriple:")
    for triple in only_in_test_valid:
        print('\t'.join(triple))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_and_correct_triples.py <data_dir>")
    else:
        main(sys.argv[1])
