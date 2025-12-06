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

# Test to see if it works
if __name__== "__main__":
    tokenizer = BPETokenizer()
    sample_corpus = ["The cat is fast", "The car is red"]
    splits = tokenizer._pre_tokenize(sample_corpus)
    print("Stpe 1 output:", dict(splits))