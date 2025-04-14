import os, argparse
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

from load_data import load_data_pretrained_models
from utils import get_similarity_scores, compute_spearman_correlation
from transformers import AutoModel, AutoTokenizer


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PretrainedEmbeddingModel(nn.Module):
    """
    Pretrained model to extract embeddings from.
    """
    def __init__(self, model: str, layers: str, merge_strategy: str, layer_merging: str):
        """
        Initializes GPT-2 or BERT as an instance variable. You may alter the function header or
        add new keyword arguments as needed. The header and variable descriptions are only intended 
        to guide you.

        You should be loading GPT-2 and BERT from Huggingface transformers. You are restricted to the base
        models for this assignment, meaning that you should use the following strings:
            For GPT-2: "gpt2"
            For BERT: "bert-base-uncased"

        Inputs:
            model (str): Which model to load (GPT-2 or BERT)
            layers (str): The hidden states of which layers should we use? 
            merge_stragegy (str): How do we merge subwords when a word is split into multiple subwords?
            layer_strategy (str): If we use multiple layers, how do we combine them?
        """
        # TODO
        super().__init__()
        self.model_name = model
        if model == 'bert':
            model = 'bert-base-uncased'

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, output_hidden_states=True)
        self.model.eval().to(DEVICE)


        self.layers = [int(layer) for layer in layers.split(",")]  
        self.merge_strategy = merge_strategy  
        self.layer_merging = layer_merging 

    def _combine_layers(self, hidden_states):
        selected = [hidden_states[i] for i in self.layers]
        if self.layer_merging == "mean":
            return torch.mean(torch.stack(selected, dim=0), dim=0)
        elif self.layer_merging == "sum":
            return torch.sum(torch.stack(selected, dim=0), dim=0)
        elif self.layer_merging == "last":
            return selected[-1]
        else:
            raise ValueError("Unsupported layer_merging strategy")
        
    def _merge_subwords(self, token_embeddings):
        if self.merge_strategy == "first":
            return token_embeddings[0]
        elif self.merge_strategy == "mean":
            return torch.mean(token_embeddings, dim=0)
        else:
            raise ValueError("Unsupported merge strategy")

    def extract_embedding_from_outputs(self, model_output, word_span):
        """
        Extracts the embedding corresponding to the input word span according to whatever
        strategy the model has been initialized with.

        This function is only here to guide you. You may choose to use or modify this
        function if you wish.

        As a general tip, looking carefully at what models you load from transformers output
        and how what they output align with different portions of the model source code (accessible
        from the model pages on Huggingface) will be very helpful.

        Input:
            model_output (Any): The output of the model following a forward pass. 
            word_span (torch.LongTensor): A minibatch of word spans, where each span contains
                                          the start and end position of the word in the tokenized model input.
        """
        hidden_states = model_output.hidden_states
        combined_hidden = self._combine_layers(hidden_states)

        word_embeddings = []
        for i, token_ids in enumerate(word_span):  # token_ids: List[int]
            # token_embed = combined_hidden[i, token_ids, :]  # shape: [num_subwords, dim]
            token_embed = combined_hidden[i, token_ids.view(-1), :]

            merged_embed = self._merge_subwords(token_embed)
            word_embeddings.append(merged_embed)
        return word_embeddings

    def extract_isolated(self, isolated_data: Any):
        """
        Extracts word embeddings for isolated word pairs. You are free to approach this function however
        you would like.

        Inputs:
            isolated_data (Any): A dataset containing processed data for isolated word pairs. Recommended
                                 to be in a Dataloader format.

        Returns:
            word1_embeds (List): A list of embeddings for the first words of the word pairs in the dataset, 
                                 in the order they appear.
            word2_embeds (List): A list of embeddings for the second words of the word pairs in the dataset, 
                                 in the order they appear.
        """
        # TODO
        word1_embeds, word2_embeds = [], []

        for w1_ids, w1_mask, span1, w2_ids, w2_mask, span2, word1, word2 in isolated_data:
            for input_ids, mask, span, collector in zip(
                [w1_ids, w2_ids],
                [w1_mask, w2_mask],
                [span1, span2],
                [word1_embeds, word2_embeds]
            ):  
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                input_ids = input_ids.to(DEVICE)
                
                outputs = self.model(input_ids)
                embed = self.extract_embedding_from_outputs(outputs, [span])[0]
                collector.append(embed.cpu().detach())

        return word1_embeds, word2_embeds

    def extract_contextual(self, contextual_data: Any):
        """
        Extracts word embeddings for contextual word pairs. You are free to approach this function however
        you would like.

        Inputs:
            contextual_data (Any): A dataset containing processed data for contextual word pairs. Recommended
                                 to be in a Dataloader format.

        Returns:
            word1_embeds (List): A list of embeddings for the first words of the word pairs in the dataset, 
                                 in the order they appear.
            word2_embeds (List): A list of embeddings for the second words of the word pairs in the dataset, 
                                 in the order they appear.
        """
        # TODO
        word1_embeds, word2_embeds = [], []

        for input_ids, attention_mask, span1, span2, word1, word2 in contextual_data:

            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            # input_ids = input_ids.clone().detach().unsqueeze(0).to(DEVICE)
            # attention_mask = attention_mask.clone().detach().unsqueeze(0).to(DEVICE)

            # input_ids = input_ids.unsqueeze(0).to(DEVICE) if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids).unsqueeze(0).to(DEVICE)
            # attention_mask = attention_mask.unsqueeze(0).to(DEVICE) if isinstance(attention_mask, torch.Tensor) else torch.tensor(attention_mask).unsqueeze(0).to(DEVICE)


            outputs = self.model(input_ids, attention_mask=attention_mask)

            for span, collector in zip([span1, span2], [word1_embeds, word2_embeds]):
                # if not span: continue
                embed = self.extract_embedding_from_outputs(outputs, [span])[0]
                collector.append(embed.cpu().detach())

        return word1_embeds, word2_embeds

def get_args():
    """
    You may freely add new command line arguments to this function, or change them.
    """
    parser = argparse.ArgumentParser(description='word2vec model')
    parser.add_argument('-m', '--model_type', type=str, choices=['gpt2', 'bert'],
                        help='Which pretrained model will we use?')
    
    parser.add_argument('-l', '--layers', type=str, default='12',
                        help="The hidden dimension outputs of which layers will we use?")
    parser.add_argument('-sm', '--subword_merging', type=str, 
                        help="How do we merge subwords?")
    parser.add_argument('-lm', '--layer_merging', type=str, 
                        help="How do we merge layers, if we do this at all?")

    parser.add_argument('-e', '--experiment_name', type=str, default='testing',
                        help="What should we name our experiment?")
    args = parser.parse_args()
    return args

def save_embeddings(filepath, words, embeddings):
    with open(filepath, "w") as f:
        for word, vec in zip(words, embeddings):
            vec_str = " ".join(f"{x:.6f}" for x in vec)
            f.write(f"{word} {vec_str}\n")


def main():
    args = get_args()
    model_type = args.model_type
    # if model_type == 'bert':
    #     model_type = "bert-base-uncased"
    layers = args.layers
    merge_strategy = args.subword_merging
    layer_merging = args.layer_merging
    experiment_name = args.experiment_name

    # Load data
    cont_dev_data, cont_test_data, isol_dev_data, isol_test_data, isol_dev_labels, cont_dev_labels = load_data_pretrained_models(model_type)


   
    # Load model
    model = PretrainedEmbeddingModel(model_type, layers, merge_strategy, layer_merging)
    model.to(DEVICE)

    # Note: The following code is a template, you can choose to use it or create your 
    # own evaluation pipeline. You are free to change the code as you see fit.

    isol_dev_embeds_word1, isol_dev_embeds_word2 = model.extract_isolated(isol_dev_data)
    isol_test_embeds_word1, isol_test_embeds_word2 = model.extract_isolated(isol_test_data) 
    cont_dev_embeds_word1, cont_dev_embeds_word2 = model.extract_contextual(cont_dev_data)
    cont_test_embeds_word1, cont_test_embeds_word2 = model.extract_contextual(cont_test_data) 

    # Save the embeddings to text file
    save_embeddings(
        f"results/{model_type}_isol_test_words1_embeddings.txt",
        [w1[0] for _, _, _, _, _, _, w1, _ in isol_test_data],
        isol_test_embeds_word1
    )
    save_embeddings(
        f"results/{model_type}_isol_test_words2_embeddings.txt",
        [w2[0] for _, _, _, _, _, _, _, w2 in isol_test_data],
        isol_test_embeds_word2
    )

    save_embeddings(
        f"results/{model_type}_cont_test_words1_embeddings.txt",
        [w1[0] for _, _, _, _, w1, _ in cont_test_data],
        cont_test_embeds_word1
    )
    save_embeddings(
        f"results/{model_type}_cont_test_words2_embeddings.txt",
        [w2[0] for _, _, _, _, _, w2 in cont_test_data],
        cont_test_embeds_word2
    )

    # Compute word pair similarity scores using your embedding
    isol_dev_sim_scores = get_similarity_scores(isol_dev_embeds_word1, isol_dev_embeds_word2)
    isol_test_sim_scores = get_similarity_scores(isol_test_embeds_word1, isol_test_embeds_word2)
    cont_dev_sim_scores = get_similarity_scores(cont_dev_embeds_word1, cont_dev_embeds_word2)
    cont_test_sim_scores = get_similarity_scores(cont_test_embeds_word1, cont_test_embeds_word2)

    # Evaluate your similarity scores against human ratings
    isol_dev_corr = compute_spearman_correlation(isol_dev_sim_scores, isol_dev_labels)
    # isol_test_corr = compute_spearman_correlation(isol_test_sim_scores, isol_test_data.labels)
    cont_dev_corr = compute_spearman_correlation(cont_dev_sim_scores, cont_dev_labels)
    # cont_test_corr = compute_spearman_correlation(cont_test_sim_scores, cont_test_data.labels)

    print("\n\n\nEvaluating on: isolated word pairs")
    print("Correlation score on dev set:", isol_dev_corr)
    # print("Correlation score on test set:", isol_test_corr)

    print("\n\n\nEvaluating on: contextual word pairs")
    print("Correlation score on dev set:", cont_dev_corr)
    # print("Correlation score on test set:", cont_test_corr)


if __name__ == "__main__":
    main()
