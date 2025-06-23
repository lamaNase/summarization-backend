import os
import json
import math
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from nltk.util import ngrams

def load_json_file(file_path):
    """Load and parse a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_sentences(data, skip_title=True):
    """Extract all sentences from the nested JSON structure.
    
    Args:
        data: The JSON data containing paragraphs and sentences
        skip_title: If True, skip paragraph 0 (title)
    
    Returns:
        A list of tuples: (sentence_text, (paragraph_idx, sentence_idx))
    """
    sentences = []
    
    # Go through all paragraphs
    for para_idx in data:
        # Skip paragraph 0 (title) if skip_title is True
        if skip_title and para_idx == '0':
            continue
            
        # Go through all sentences in the paragraph
        for sent_idx in data[para_idx]:
            if isinstance(data[para_idx][sent_idx], str):
                # Store sentence text and its indices
                sentences.append((data[para_idx][sent_idx], (para_idx, sent_idx)))
    
    return sentences

def get_ngrams(text, n_grams=1):
    """Extract n-grams from text.
    
    Args:
        text: Input text
        n_grams: Size of n-grams (1 for unigrams, 2 for bigrams, etc.)
    
    Returns:
        List of n-grams
    """
    tokens = text.split()
    if n_grams == 1:
        return tokens
    else:
        return [' '.join(gram) for gram in ngrams(tokens, n_grams)]
    

def compute_tf(sentence):
    """Compute term frequency for a sentence using n-grams."""
    # Extract n-grams from the sentence
    terms = get_ngrams(sentence)
    
    # Count the frequency of each term
    term_count = Counter(terms)
    
    # Calculate term frequency
    sentence_length = len(terms)
    
    # Avoid division by zero
    if sentence_length == 0:
        return {}
        
    tf = {term: count / sentence_length for term, count in term_count.items()}
    
    return tf

def compute_isf(sentence_texts):
    """Compute inverse sentence frequency for all terms in the corpus using n-grams."""
    # Count the number of sentences containing each term
    term_sentence_count = defaultdict(int)
    
    for sentence in sentence_texts:
        terms = set(get_ngrams(sentence))  # Use set to count each term only once per sentence
        for term in terms:
            term_sentence_count[term] += 1
    
    # Calculate ISF: log(total number of sentences / number of sentences containing the term)
    num_sentences = len(sentence_texts)
    isf = {term: math.log(num_sentences / count) for term, count in term_sentence_count.items()}
    
    return isf

def compute_tf_isf(sentence_data):
    """Compute TF-ISF vectors for each sentence using n-grams."""
    # Extract just the text from sentence data for ISF calculation
    sentence_texts = [s[0] for s in sentence_data]
    
    # Get ISF values for all terms
    isf = compute_isf(sentence_texts)
    
    # Compute TF for each sentence
    tf_per_sentence = [compute_tf(sentence) for sentence in sentence_texts]
    
    # Create a list of all unique terms
    all_terms = sorted(list(isf.keys()))
    term_to_idx = {term: idx for idx, term in enumerate(all_terms)}
    
    # Compute TF-ISF for each sentence
    tf_isf_vectors = []
    
    for tf in tf_per_sentence:
        # Initialize vector with zeros
        vector = np.zeros(len(all_terms))
        
        # Fill in TF-ISF values
        for term, tf_value in tf.items():
            if term in term_to_idx:  # Check if term exists (should always be true)
                idx = term_to_idx[term]
                vector[idx] = tf_value * isf[term]
        
        tf_isf_vectors.append(vector)
    
    return tf_isf_vectors, all_terms

def compute_title_tf_isf(title_text, sentence_data, n=1):
    """Compute TF-ISF vector for the title using the ISF values from the document's sentences."""
    # Extract all sentence texts (excluding the title) for ISF calculation
    sentence_texts = [s[0] for s in sentence_data]
    
    # Compute ISF values
    isf = compute_isf(sentence_texts)
    
    # Get all unique terms
    all_terms = sorted(list(isf.keys()))
    term_to_idx = {term: idx for idx, term in enumerate(all_terms)}
    
    # Compute TF for the title
    tf_title = compute_tf(title_text)
    
    # Initialize the title vector with zeros
    title_vector = np.zeros(len(all_terms))
    
    # Compute TF-ISF for the title
    for term, tf_value in tf_title.items():
        if term in term_to_idx:  # Ensure the term exists in ISF
            idx = term_to_idx[term]
            title_vector[idx] = tf_value * isf[term]
    
    return title_vector

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    return dot_product / (norm_v1 * norm_v2)

def modified_similarity(v1, v2, title, keyphrase_score_v2):
    #print(keyphrase_score_v2)
    return cosine_similarity(v1, v2) * (1 + keyphrase_score_v2 + cosine_similarity(v2, title))

def build_graph(sentence_data, title_text, keyphrase_scores):
    """
    Build a graph where nodes are sentences and edges represent modified similarity above a threshold.
    Each node also includes its similarity with the title.
    
    Args:
        sentence_data: List of tuples (sentence_text, (para_idx, sent_idx))
        title_text: The text of the document title
        keyphrase_scores: Dictionary of keyphrase scores by paragraph and sentence indices
        threshold: Minimum similarity threshold for creating an edge
        ngram_size: Size of n-grams to use for TF-ISF calculation
    
    Returns:
        Tuple of (graph, tf_isf_vectors, terms)
    """
    # settings can be changed later
    similarity_measure = 1
    with_edge_thresholding = False
    edge_weight_threshold = 0.5

    # Compute TF-ISF vectors
    tf_isf_vectors, terms = compute_tf_isf(sentence_data)
    
    # Create graph
    G = nx.Graph()
    
    # Compute the title vector
    title_vector = compute_title_tf_isf(title_text, sentence_data)
    
    # Add nodes (sentences) with their indices
    for i, (sentence_text, (para_idx, sent_idx)) in enumerate(sentence_data):
        node_id = f"{para_idx}_{sent_idx}"
        
        # Get keyphrase score for this sentence (default to 0 if not found)
        keyphrase_score = 0
        statistical_score = 0
        if para_idx in keyphrase_scores and sent_idx in keyphrase_scores[para_idx]:
            keyphrase_score = keyphrase_scores[para_idx][sent_idx].get('keyphrase_score', 0)
            statistical_score = keyphrase_scores[para_idx][sent_idx].get('statistical_score', 0)
        
        # Calculate similarity with the title
        title_similarity = cosine_similarity(tf_isf_vectors[i], title_vector)
        
        G.add_node(
            node_id, 
            text=sentence_text, 
            para_idx=para_idx, 
            sent_idx=sent_idx, 
            vector_idx=i,
            keyphrase_score=keyphrase_score,
            statistical_score=statistical_score,
            title_similarity=title_similarity,  # Add title similarity to node attributes
        )
    
    # Calculate similarities for all pairs
    similarities = []
    sentence_pairs = []
    
    for i, (_, (para_i, sent_i)) in enumerate(sentence_data):
        node_i = f"{para_i}_{sent_i}"
        keyphrase_score_i = G.nodes[node_i]['keyphrase_score']
        
        for j, (_, (para_j, sent_j)) in enumerate(sentence_data[i+1:], i+1):
            node_j = f"{para_j}_{sent_j}"
            keyphrase_score_j = G.nodes[node_j]['keyphrase_score']
            
            # Use modified similarity instead of cosine similarity
            if (similarity_measure == 2):
                similarity = modified_similarity(
                    tf_isf_vectors[i], 
                    tf_isf_vectors[j], 
                    title_vector,
                    keyphrase_score_j
                )
            else:
                similarity = cosine_similarity(
                    tf_isf_vectors[i], 
                    tf_isf_vectors[j]
                )
            
            similarities.append(similarity)
            sentence_pairs.append((node_i, node_j))
            
    # Normalize similarities if there are any
    if similarities:
        min_sim = min(similarities)
        max_sim = max(similarities)
        
        # Avoid division by zero if all similarities are the same
        if max_sim > min_sim:
            normalized_similarities = [(sim - min_sim) / (max_sim - min_sim) for sim in similarities]
        else:
            normalized_similarities = [1.0 for _ in similarities]
        
        
        # Add edges based on normalized similarities
        for (node_i, node_j), norm_sim in zip(sentence_pairs, normalized_similarities):
            if with_edge_thresholding:
                if norm_sim > edge_weight_threshold:
                    G.add_edge(node_i, node_j, weight=norm_sim)
            else:
                G.add_edge(node_i, node_j, weight=norm_sim)
    
    return G, tf_isf_vectors, terms

def process_file(processed_classical, combined_scores):
    """
    Process a single document with enhanced similarity measures using in-memory data.

    Args:
        processed_classical (dict): Preprocessed classical sentence dictionary.
        combined_scores (dict): Dictionary containing keyphrase_score and statistical_score per sentence.

    Returns:
        dict: Dictionary with processing results or None if processing failed.
    """
    # Extract the title
    title_text = ""
    if '0' in processed_classical and '0' in processed_classical['0']:
        title_text = processed_classical['0']['0']

    # Extract sentences excluding title
    sentence_data = extract_sentences(processed_classical, skip_title=True)

    if not sentence_data:
        print("⚠️ No valid sentences found after skipping title.")
        return None

    # Build graph using the combined scores
    graph, vectors, terms = build_graph(
        sentence_data,
        title_text,
        combined_scores  # Contains both keyphrase_score and statistical_score
    )

    return {
        'graph': graph,
        'vectors': vectors,
        'terms': terms,
        'sentences': sentence_data,
        'title': title_text,
        'keyphrase_scores': combined_scores,  # Keeping this name for compatibility
        'filename': None  # Optional: could pass doc ID or None since there's no file
    }