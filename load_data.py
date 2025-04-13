import os
from typing import Optional
import pandas as pd
import os
import re
from collections import Counter
import pandas as pd
import numpy as np
import torch
import nltk
from nltk.metrics import edit_distance
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from collections import Counter
from torch.utils.data import DataLoader

#nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


from torch.utils.data import Dataset, DataLoader

def lemmatize_word(word):
   return lemmatizer.lemmatize(word.lower())

def find_wordnet_synonym(word, vocab):
    """
    Find the closest synonym in vocab using WordNet, ensuring the correct POS match.
    """
    word_synsets = wordnet.synsets(word)
    if not word_synsets:
        return None  # No synsets found

    word_pos = word_synsets[0].pos()  # Get POS of original word
    for syn in word_synsets:
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")
            if synonym in vocab and any(s.pos() == word_pos for s in wordnet.synsets(synonym)):
                print("synonym-----")
                print(word)
                print(synonym)
                print()
                return synonym  # ✅ Correct POS match

    return None  # No valid synonym found


def find_closest_word(word, vocab):
 
    word = word.lower()
    if word in vocab:
        return word  

    lemma = lemmatize_word(word)
    if lemma in vocab:
        return lemma  


    result = find_wordnet_synonym(word, vocab)
    if result: 
        return result 

    # # ✅ Step 2: Use `get_close_matches()` for fast fuzzy matching
    # close_matches = difflib.get_close_matches(word, vocab.keys(), n=1, cutoff=0.85)
    # if close_matches:
    #     print(word)
    #     print(close_matches[0])
    #     print()
    #     return close_matches[0]

    # ✅ Step 3: Use Edit Distance *only if no other match found*
    closest_word = min(vocab.keys(), key=lambda w: edit_distance(word, w) if abs(len(w) - len(word)) <= 2 else float('inf'))
    if edit_distance(word, closest_word) <= 1:  # Set a reasonable threshold (2 edits max)
        print('edit-----')
        print(word)
        print(closest_word)
        print()
        return closest_word

    return "<UNK>"

# =========================
# TEXT PREPROCESSING
# =========================
def preprocess_text(text: str):
    """Convert to lowercase, remove punctuation, and lemmatize words."""
    text = text.lower().strip()  # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    words = text.split()  # Tokenize
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]  # Apply lemmatization
    return lemmatized_words

# =========================
# SKIP-GRAM DATASET
# =========================

class SkipGramDataset(Dataset):
    def __init__(self, sentences, vocab, window_size=5):
        """
        Args:
            sentences: List of tokenized sentences.
            vocab: Dictionary mapping words to indices.
            window_size: Context window size.
        """
        self.sentences = sentences
        self.vocab = vocab
        self.window_size = window_size

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        """Returns a tokenized sentence instead of word-context pairs."""
        return self.sentences[idx]  # ✅ Returns full sentence


# =========================
# COLLATE FUNCTION
# =========================

def collate_fn(batch_sentences, vocab, window_size, num_neg_samples, word_freqs):
    """
    Custom collate function to dynamically extract word-context pairs from a batch of sentences.
    """
    word_inputs, context_inputs = [], []  # Lists to store extracted word-context pairs

    # 🔹 Extract positive word-context pairs dynamically
    for sentence in batch_sentences:
        indices = [vocab.get(word, vocab["<UNK>"]) for word in sentence]  # Convert words to indices
        for i, word_idx in enumerate(indices):
            start, end = max(0, i - window_size), min(len(indices), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    word_inputs.append(word_idx)
                    context_inputs.append(indices[j])

    # Convert to PyTorch tensors
    word_inputs = torch.tensor(word_inputs, dtype=torch.long)
    context_inputs = torch.tensor(context_inputs, dtype=torch.long)

    # 🔹 Negative Sampling (Efficiently Sample Negative Words)
    word_freq_tensor = torch.tensor(list(word_freqs.values()), dtype=torch.float32) ** 0.75
    word_freq_tensor /= word_freq_tensor.sum()  # Normalize probabilities

    negative_samples = torch.multinomial(word_freq_tensor, len(word_inputs) * num_neg_samples, replacement=True)
    negative_samples = negative_samples.view(len(word_inputs), num_neg_samples)

    return word_inputs, context_inputs, negative_samples




def load_csv_data(data_folder, filename):
    """Loading csv (dev/test data).
    """
    data_path = os.path.join(data_folder, filename)
    data = pd.read_csv(data_path)
    return data

def load_data_word2vec(use_additional_data: Optional[bool] = False):
  
    batch_size = 2048
    window_size = 5
    num_neg_samples = 5

    def load_training_data(file_path, limit=None):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if limit:
                lines = lines[:limit]
        return [preprocess_text(line) for line in lines]
    corpus = load_training_data("data/training/training-data.1m", 1000)
    # corpus = load_training_data("data/training/training-data.1m")

    word_counts = Counter()
    for sentence in corpus:
        word_counts.update(sentence)

    vocab = {"<UNK>": 0}
    vocab.update({word: i+1 for i, word in enumerate(word_counts.keys())})
    word_freqs = dict(word_counts)

    train_dataset = SkipGramDataset(corpus, vocab, window_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, vocab, window_size, num_neg_samples, word_freqs),
        pin_memory=True, drop_last=True
    )

    def load_word_pairs_with_context(file_path, vocab):
        df = pd.read_csv(file_path)
        unk_id = vocab["<UNK>"]
        word1_ids = [vocab.get(lemmatize_word(w), unk_id) for w in df["word1"]]
        word2_ids = [vocab.get(lemmatize_word(w), unk_id) for w in df["word2"]]
        context = df["context"].tolist()
        return torch.tensor(word1_ids), torch.tensor(word2_ids), df, context

    def load_word_pairs(file_path, vocab):
        df = pd.read_csv(file_path)
        unk_id = vocab["<UNK>"]
        word1_ids = [vocab.get(lemmatize_word(w), unk_id) for w in df["word1"]]
        word2_ids = [vocab.get(lemmatize_word(w), unk_id) for w in df["word2"]]
        return torch.tensor(word1_ids), torch.tensor(word2_ids), df

    # Isolated dev/test
    dev_word1, dev_word2, dev_df = load_word_pairs("data/isolated_similarity/isolated_dev_x.csv", vocab)
    dev_labels = torch.tensor(pd.read_csv("data/isolated_similarity/isolated_dev_y.csv")["sim"].tolist(), dtype=torch.float32)
    test_word1, test_word2, test_df = load_word_pairs("data/isolated_similarity/isolated_test_x.csv", vocab)

    isol_dev_data = type("DatasetObject", (object,), {
        "word1_ids": dev_word1, "word2_ids": dev_word2,
        "labels": dev_labels, "original_word1": dev_df["word1"],
        "original_word2": dev_df["word2"]
    })
    isol_test_data = type("DatasetObject", (object,), {
        "word1_ids": test_word1, "word2_ids": test_word2,
        "labels": None, "original_word1": test_df["word1"],
        "original_word2": test_df["word2"]
    })

    # Contextual dev/test
    cont_dev_word1, cont_dev_word2, cont_dev_df, cont_dev_context = load_word_pairs_with_context(
        "data/contextual_similarity/contextual_dev_x.csv", vocab)
    cont_dev_labels = torch.tensor(pd.read_csv("data/contextual_similarity/contextual_dev_y.csv")["sim"].tolist(), dtype=torch.float32)
    cont_test_word1, cont_test_word2, cont_test_df, cont_test_context = load_word_pairs_with_context(
        "data/contextual_similarity/contextual_test_x.csv", vocab)

    cont_dev_data = type("DatasetObject", (object,), {
        "word1_ids": cont_dev_word1, "word2_ids": cont_dev_word2,
        "labels": cont_dev_labels, "original_word1": cont_dev_df["word1"],
        "original_word2": cont_dev_df["word2"], "context": cont_dev_context
    })
    cont_test_data = type("DatasetObject", (object,), {
        "word1_ids": cont_test_word1, "word2_ids": cont_test_word2,
        "labels": None, "original_word1": cont_test_df["word1"],
        "original_word2": cont_test_df["word2"], "context": cont_test_context
    })

    return train_dataset, train_loader, cont_dev_data, cont_test_data, isol_dev_data, isol_test_data, vocab




class ModelContextualSimilarityDataset(Dataset):
    def __init__(self, model_type, x_csv, y_csv):
        # Get tokenizer
        if model_type == "gpt2":
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Compute maximum length in either case
        self.max_len = self.get_max_len(x_csv, tokenizer)

        # Process each input
        self.data = []
        self.process_data(model_type, tokenizer, x_csv, y_csv)

    def __len__(self):
        # the length of all the lists should be the same
        return len(self.data)
    
    
    def get_max_len(self, x_csv, tokenizer):
        # TODO
        df = pd.read_csv(x_csv)
        max_len = 0
        for text in df["context"]:
            clean_text = re.sub(r"</?strong>", "", text)
            tokens = tokenizer(clean_text, add_special_tokens=True)
            max_len = max(max_len, len(tokens["input_ids"]))

        return max_len

    def process_data(self, model_type, tokenizer, x_csv, y_csv):
        #process and save all the data, so that every getitem call is just indexing into the data and returning one data point.
        # you can define any number of helper functions to do this.
        # make sure the data are processed and added to self.data in the order they appear. 
        ## TODO
        df_x = pd.read_csv(x_csv)
        word1_list = df_x["word1"].tolist()
        word2_list = df_x["word2"].tolist()
        context_list = df_x["context"].tolist()

        for word1, word2, context in zip(word1_list, word2_list, context_list):
            # Find <strong> spans and remove tags
            clean_context = re.sub(r"</?strong>", "", context)

            # Tokenize for model input
            encoded = tokenizer(clean_context, padding='max_length', max_length=self.max_len,
                                return_attention_mask=True, return_offsets_mapping=True,
                                truncation=True, return_tensors="pt")

            input_ids = encoded["input_ids"].squeeze()
            attention_mask = encoded["attention_mask"].squeeze()
            offsets = encoded["offset_mapping"].squeeze().tolist()

            # Find character positions of strong-tagged words in original context
            def get_char_span(tagged_context, target_idx):
                # Match using regex
                matches = list(re.finditer(r"<strong>(.*?)</strong>", tagged_context))
                match = matches[target_idx]
                return match.start(1), match.end(1)

            span1_chars = get_char_span(context, 0)  # First <strong>
            span2_chars = get_char_span(context, 1)  # Second <strong>

            # Map character spans to token indices
            def get_token_span(char_span, offsets):
                return [i for i, (s, e) in enumerate(offsets) if s >= char_span[0] and e <= char_span[1]]

            span1 = get_token_span(span1_chars, offsets)
            span2 = get_token_span(span2_chars, offsets)

            # Save
            self.data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "span1": span1,
                "span2": span2,
                "word1": word1,
                "word2": word2
            })

    def __getitem__(self, idx):
        curr_dict = self.data[idx]

        #curr_dict['input_ids'] is the ids of the context sequence, including anything special tokens automatically added by the tokenizer (e.g., [CLS] and [SEP] tokens for bert) while excluding the <strong> tokens in the input.   
        return (
            torch.tensor(curr_dict['input_ids']),
            torch.tensor(curr_dict['attention_mask']),
            curr_dict['span1'],
            curr_dict['span2'],
            curr_dict['word1'],
            curr_dict['word2']
        )
            

class ModelIsolatedSimilarityDataset(Dataset):

    def __init__(self, model_type, x_csv, y_csv):
        # Get tokenizer
        if model_type == "gpt2":
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Compute maximum length in either case
        self.max_len = self.get_max_len(x_csv, tokenizer)
        
        # Process each input
        self.data = []
        self.process_data(model_type, tokenizer, x_csv, y_csv)
        self.tokenizer=tokenizer

    def __len__(self):
        return len(self.data)

    def get_max_len(self, x_csv, tokenizer):
        # TODO
        df = pd.read_csv(x_csv)
        word1_list = df["word1"].tolist()
        word2_list = df["word2"].tolist()

        max_len = 0
        for word in word1_list + word2_list:
            tokens = tokenizer(word, add_special_tokens=True)
            max_len = max(max_len, len(tokens["input_ids"]))
        return max_len

    def process_data(self, model_type, tokenizer, x_csv, y_csv):
        # you can define any number of helper functions to do this.
        # make sure the data are processed and added to self.data in the order they appear. 
        df = pd.read_csv(x_csv)
        
        word1_list = df["word1"].tolist()
        word2_list = df["word2"].tolist()

        for w1, w2 in zip(word1_list, word2_list):
            data_point = {}

            for word, label in zip([w1, w2], ["word1", "word2"]):
                encoded = tokenizer(word,
                                    padding="max_length",
                                    max_length=self.max_len,
                                    return_attention_mask=True,
                                    return_offsets_mapping=True,
                                    truncation=True)

                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
                offsets = encoded["offset_mapping"]

                # Token span: skip special tokens if model adds them (assume special tokens are at ends)
                token_span = [i for i, (start, end) in enumerate(offsets)
                            if start != 0 or end != 0]  # Ignore special tokens with (0, 0)

                data_point[f"{label}_ids"] = input_ids
                data_point[f"{label}_mask"] = attention_mask
                data_point[f"span{1 if label == 'word1' else 2}"] = token_span
                data_point[label] = word

            self.data.append(data_point)

    def __getitem__(self, idx):
        curr_dict = self.data[idx]
        #curr_dict['word1_ids'] and curr_dict['word2_ids'] are the ids corresponding to word1 and word2 respectively, including anything special tokens automatically added by the tokenizer (e.g., [CLS] and [SEP] tokens for bert). 
        return (
            torch.tensor(curr_dict['word1_ids']),
            torch.tensor(curr_dict['word1_mask']),
            curr_dict['span1'],
            torch.tensor(curr_dict['word2_ids']),
            torch.tensor(curr_dict['word2_mask']),
            curr_dict['span2'],
            curr_dict['word1'],
            curr_dict['word2']
        )



def load_data_pretrained_models(model_type: str):
    """
    Function for loading and processing the evaluation datasets to be used
    by BERT or GPT-2. You may modify the function header and outputs as necessary.

    As general tips:
        - For the contextual data, you will need to take into account the "<strong>" and "</strong>" tags
          in the context string. Your input to the model should not have these tags.
        - Unlike word2vec, you will need to reason about subword tokenization. Your data processing should
          take into account what span of tokens each target word will map to, so that you may extract the
          embeddings from the correct positions in the input sequence. 

          As a sanity check, you should ensure that the tokens in the spans you extract do in fact minimally match
          your target words at the position they occur. Some things to keep in mind:
              - Your tokenizer's .decode() function takes in a sequence of tokens and outputs the corresponding string.
              - The return_offsets_mapping keyword in the tokenizer call can help in debugging.
              - Your target word may be a substring of another word in the sequence. For a given target, your span needs 
                correspond to the word originally in between by the <strong> and </strong> tags.
        
        - For pretrained models, please complete the two customized dataset classes, ModelContextualSimilarityDataset and ModelIsolatedSimilarityDataset, to process the contextual and isolated data for pretrained models. You can also define more helper functions. For Word2Vec, you can define any customized datasets in anyway you want. 
    Inputs:
        model_type (str): One of BERT or GPT-2.
    
    Outputs: 
        ont_dev_data, cont_test_data, isol_dev_data, isol_test_data
        (we recommend these to be in dataloader format)
    """
    if model_type.lower() == "bert":
        model_name = "bert-base-uncased"
    elif model_type.lower() == "gpt2":
        model_name = "gpt2"
    else:
        raise ValueError("Invalid model_type. Use 'bert' or 'gpt2'.")

    # Paths to CSV files
    CONTEXT_DEV_X = "data/contextual_similarity/contextual_dev_x.csv"
    CONTEXT_DEV_Y = "data/contextual_similarity/contextual_dev_y.csv"
    CONTEXT_TEST_X = "data/contextual_similarity/contextual_test_x.csv"
    CONTEXT_TEST_Y = "data/contextual_similarity/contextual_test_y.csv"  # if exists

    ISOLATED_DEV_X = "data/isolated_similarity/isolated_dev_x.csv"
    ISOLATED_DEV_Y = "data/isolated_similarity/isolated_dev_y.csv"
    ISOLATED_TEST_X = "data/isolated_similarity/isolated_test_x.csv"
    ISOLATED_TEST_Y = "data/isolated_similarity/isolated_test_y.csv"  # if exists

    # Instantiate datasets
    cont_dev_data = ModelContextualSimilarityDataset(model_name, CONTEXT_DEV_X, CONTEXT_DEV_Y)
    cont_test_data = ModelContextualSimilarityDataset(model_name, CONTEXT_TEST_X, CONTEXT_TEST_Y)
    isol_dev_data = ModelIsolatedSimilarityDataset(model_name, ISOLATED_DEV_X, ISOLATED_DEV_Y)
    isol_test_data = ModelIsolatedSimilarityDataset(model_name, ISOLATED_TEST_X, ISOLATED_TEST_Y)

    # Optionally wrap in DataLoader (batch_size=1 to keep control over individual inputs)
    # You can skip this step if your model works directly with the dataset objects.
    cont_dev_loader = DataLoader(cont_dev_data, batch_size=1, shuffle=False)
    cont_test_loader = DataLoader(cont_test_data, batch_size=1, shuffle=False)
    isol_dev_loader = DataLoader(isol_dev_data, batch_size=1, shuffle=False)
    isol_test_loader = DataLoader(isol_test_data, batch_size=1, shuffle=False)

    return cont_dev_loader, cont_test_loader, isol_dev_loader, isol_test_loader
    
