import numpy as np
from graph_construction.build_graph import load_json_file

def formula_3_16_1N(result, original_sentences, damping=0.85, max_iterations=100, alpha=0.89):
    """
    TextRank implementation combining graph-based scores with statistical scores.
    
    Args:
        result (dict): The dictionary returned by process_document, includes graph and sentences.
        original_sentences (dict): Original sentence text dictionary {para_idx: {sent_idx: text}}
        damping (float): Damping factor for TextRank
        max_iterations (int): Maximum iterations for convergence
        alpha (float): Weight for combining TextRank and statistical scores
    
    Returns:
        List of tuples: [(node, final_score, original_sentence_text), ...] sorted descending by score
    """
    graph = result['graph']
    sentences = result['sentences']
        
    statistical_scores = {}
    # If you have statistical scores in combined keyphrase/stat dict:
    if 'keyphrase_scores' in result:
        statistical_scores = result['keyphrase_scores']
    
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
           
    # Initialize TextRank scores: 1/N
    scores = np.ones(n) / n 
          
    # Precompute outgoing weights
    out_weights_sum = {}
    for node in nodes:
        out_sum = sum(graph[node][nbr]['weight'] for nbr in graph.neighbors(node))
        out_weights_sum[node] = max(out_sum, 1e-10)
        
    # Run TextRank iterations
    converged = False
    for iteration in range(1, max_iterations + 1):
        new_scores = np.ones(n) * (damping)
        for i, node_i in enumerate(nodes):
            score_sum = 0
            for neighbor in graph.neighbors(node_i):
                j = node_to_idx[neighbor]
                weight = graph[node_i][neighbor]['weight']
                norm_factor = np.sqrt(out_weights_sum[node_i] * out_weights_sum[neighbor])
                score_sum += (weight * scores[j]) / norm_factor
            new_scores[i] += (1 - damping) * score_sum
            
        if np.allclose(scores, new_scores, atol=1e-6):
            converged = True
            break
        scores = new_scores
        
    # Collect raw scores (TextRank and statistical)
    raw_textrank_scores = scores.copy()
    raw_stat_scores = np.zeros(n)
    for i, node in enumerate(nodes):
        node_data = graph.nodes[node]
        para_idx = node_data['para_idx']
        sent_idx = node_data['sent_idx']
        stat_score = statistical_scores.get(para_idx, {}).get(sent_idx, {}).get('statistical_score', 0)
        raw_stat_scores[i] = stat_score 

    # Normalize both using same min and max
    combined_min = min(np.min(raw_textrank_scores), np.min(raw_stat_scores))
    combined_max = max(np.max(raw_textrank_scores), np.max(raw_stat_scores))
    combined_range = max(combined_max - combined_min, 1e-10)
    normalized_textrank = (raw_textrank_scores - combined_min) / combined_range
    normalized_stat = (raw_stat_scores - combined_min) / combined_range

    # Combine scores
    final_scores = alpha * normalized_textrank + (1 - alpha) * normalized_stat
        
    # Use original text from original_sentences dict instead of graph text
    sentence_scores = []
    for i, node in enumerate(nodes):
        node_data = graph.nodes[node]
        para_idx = node_data['para_idx']
        sent_idx = node_data['sent_idx']
        original_text = original_sentences.get(para_idx, {}).get(sent_idx, "")
        sentence_scores.append((node, float(final_scores[i]), original_text))
        
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    return sentence_scores