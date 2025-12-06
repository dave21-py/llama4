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

    def _merge_pair(self, pair_to_merge, splits):
        new_splits = defaultdict(int)
        first_part = pair_to_merge[0]
        second_part = pair_to_merge[1]
        merged_token = first_part + second_part
        for word_tuple, frequency in splits.items():
            new_word_list = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == first_part and word_tuple[i+1] == second_part:
                    new_word_list.append(merged_token)
                    i += 2 # skip the next character because we just merged it
                else:
                    new_word_list.append(word_tuple[i])
                    i += 1
            new_splits[tuple(new_word_list)] = frequency
        return new_splits

    def train(self, text_corpus, num_merges = 20):
        # num of tokens is num_merges
        self.vocab = 0
        self.merges = {}
        # pretokenize
        splits = tokenizer._pre_tokenize(text_corpus)
        # inintalise the vocab
        unique_chars = set()
        for word_tuple in splits:
            for char in word_tuple:
                unique_chars.add(char)
        self.vocab = sorted(list(unique_chars))

        print(f"Base Vocabulary size: {len(self.vocab)}")

        # Main loop
        for i in range(num_merges):
            pair_stats = self._get_pair_stats(splits)

            if not pair_stats:
                break
            # Get the best pair with highest count
            best_pair = max(pair_stats, key=pair_stats.get)
            best_count = pair_stats[best_pair]

            # Merge the pair
            splits = self._merge_pair(best_pair, splits)

            # Add new token to vocabulary and record the merge rule
            new_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = new_token
            self.vocab.append(new_token)
            print(f"Merge {i+1}/{num_merges}: Merged {best_pair} -> '{new_token}' (Count: {best_count})")
        
        print("Training Complete!")
        print(f"Final Vocabulary Size: {len(self.vocab)}")






# Test to see if it works
if __name__== "__main__":
    tokenizer = BPETokenizer()
    # A slightly larger corpus to make it interesting
    corpus = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]
    # train for 15 merges
    tokenizer.train(corpus, num_merges=15)


    print("\nTop 5 Learned Merges:")
    # Print the first 5 rules we learned
    for i, (pair, token) in enumerate(tokenizer.merges.items()):
        if i >= 5: break
        print(f"{pair} -> {token}")