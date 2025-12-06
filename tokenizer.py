from collections import defaultdict

class BPETokenizer:
    def __init__(self):
        # initialise the tokenizer
        # vocab: list of all tokens
        # merges: Dictionary mapping a pair of tokens to a new merged token.
        self.vocab = []
        self.merges = {}

    def _pre_tokenize(self, text_corpus):
        # pretokenize the corpus
        # spltits text into words, then breaks into characters
        # text_corpus: the training data
        word_splits = defaultdict(int)
        for text in text_corpus:
            words = text.split(" ")
            for word in words:
                if not word:
                    continue
                # Convert the word to characters
                # Examples: "Cat" -> ['C', 'a', 't' '</w>']
                char_list = list(word) + ["</w>"]
                # convert the tuple to a dictionary key
                word_tuple = tuple(char_list)

                word_splits[word_tuple] += 1
        return word_splits


    def _get_pair_stats(self, splits):
        pair_counts = defaultdict(int)
        for word_tuple, frequency in splits.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i+1])
                pair_counts[pair] += frequency
        return pair_counts


# Test to see if it works
if __name__== "__main__":
    tokenizer = BPETokenizer()
    sample_corpus = ["The cat is fast", "The car is red"]
    splits = tokenizer._pre_tokenize(sample_corpus)
    pair_stats = tokenizer._get_pair_stats(splits)
    print("Step 2 output:")
    for pair in pair_stats:
        count = pair_stats[pair]
        print(f"Pair {pair} appears {count} times")