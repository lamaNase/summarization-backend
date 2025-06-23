import os
import re
import numpy as np
import math
import logging
import warnings
from tqdm import tqdm
from preprocessing.preprocess import normalize_arabic

def extract_keyphrases(preprocessed_dl_sentences, kw_extractor=None):
    """
    Extracts and normalizes keyphrase scores from preprocessed deep learning sentences.

    Args:
        preprocessed_dl_sentences (dict): Dictionary of normalized sentences (DL version), indexed by paragraph/sentence.
        kw_extractor: An instance of a keyphrase extractor (e.g., KeyBERT)

    Returns:
        dict: Nested dictionary {paragraph_index: {sentence_index: normalized_score}}
    """
    import os
    import warnings
    import logging
    from tqdm import tqdm

    # Suppress verbose output
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    warnings.filterwarnings('ignore')
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Flatten sentences (skip title paragraph 0)
    flattened_sentences = {}
    for para_index, sentences in preprocessed_dl_sentences.items():
        if para_index == "0":
            continue
        for sent_index, sentence in sentences.items():
            sentence_id = f"P{para_index}-S{sent_index}"
            flattened_sentences[sentence_id] = sentence

    N = len(flattened_sentences)
    if N == 0:
        print("⚠️ No valid sentences found for keyphrase extraction.")
        return {}

    # If kw_extractor is None, return zeros for all sentences
    if kw_extractor is None:
        print("⚠️ kw_extractor is None. Returning zero scores for all sentences.")
        zero_results = {}
        for para_index, sentences in preprocessed_dl_sentences.items():
            if para_index == "0":
                continue
            zero_results[para_index] = {}
            for sent_index in sentences:
                zero_results[para_index][sent_index] = 0
        return zero_results

    sentence_scores = {}
    for sent_id, sentence in tqdm(flattened_sentences.items(), desc="Extracting keyphrases", leave=False):
        if not sentence.strip():
            sentence_scores[sent_id] = 0
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            keywords = kw_extractor.extract_keywords(
                sentence,
                keyphrase_ngram_range=(1, 3),
                top_n=5,
                use_mmr=True,
                diversity=0.7
            )

        filtered_keywords = [kw for kw in keywords if kw[1] >= 0.5]
        score = sum(kw[1] for kw in filtered_keywords)
        sentence_scores[sent_id] = score

    # Normalize scores
    if sentence_scores:
        max_score = max(sentence_scores.values())
        min_score = min(sentence_scores.values())
        score_range = max_score - min_score
        normalized_scores = {}

        if score_range > 0:
            for sent_id, score in sentence_scores.items():
                normalized_scores[sent_id] = (score - min_score) / score_range
        else:
            for sent_id in sentence_scores:
                normalized_scores[sent_id] = 0.5 if max_score > 0 else 0
    else:
        normalized_scores = {}

    # Restructure into paragraph/sentence form
    results = {}
    for para_index, sentences in preprocessed_dl_sentences.items():
        if para_index == "0":
            continue
        results[para_index] = {}
        for sent_index in sentences:
            sent_id = f"P{para_index}-S{sent_index}"
            results[para_index][sent_index] = normalized_scores.get(sent_id, 0)

    return results

def calculate_entropy(text):
    """Calculate Shannon entropy of a text string"""
    # Count the frequency of each character
    chars = {}
    for char in text:
        if char in chars:
            chars[char] += 1
        else:
            chars[char] = 1
            
    # Calculate entropy
    length = len(text)
    entropy = 0
    for count in chars.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    
    return entropy

def calculate_sentence_length_scores(processed_classical_sentences):
    """
    Calculate and normalize sentence length scores from preprocessed classical sentence dictionary.

    Args:
        processed_classical_sentences (dict): Preprocessed sentences for classical representation.

    Returns:
        dict: Nested dictionary {paragraph_index: {sentence_index: normalized_score}}
    """
    # Find the longest sentence length (excluding title)
    max_word_count = 0
    for para_index, sentences in processed_classical_sentences.items():
        if para_index == "0":
            continue
        for _, sentence in sentences.items():
            word_count = len(sentence.split())
            max_word_count = max(max_word_count, word_count)

    paragraph_scores = {}
    all_scores = []

    for para_index, sentences in processed_classical_sentences.items():
        if para_index == "0":
            continue

        sentence_scores = {}
        for sent_index, sentence in sentences.items():
            if not sentence or len(sentence.strip()) == 0:
                sentence_scores[sent_index] = 0
                continue

            word_count = len(sentence.split())
            entropy = calculate_entropy(sentence)

            # Sentence length score = (word_count / max_word_count) * entropy
            score = (word_count / max_word_count) * entropy if max_word_count > 0 else 0
            sentence_scores[sent_index] = score
            all_scores.append(score)

        paragraph_scores[para_index] = sentence_scores

    # Normalize scores
    if all_scores:
        min_score = min(all_scores)
        max_score = max(all_scores)
        score_range = max_score - min_score

        if score_range > 0:
            for para_index in paragraph_scores:
                for sent_index in paragraph_scores[para_index]:
                    old_score = paragraph_scores[para_index][sent_index]
                    normalized_score = (old_score - min_score) / score_range
                    paragraph_scores[para_index][sent_index] = normalized_score
        else:
            default_value = 0.5 if max_score > 0 else 0
            for para_index in paragraph_scores:
                for sent_index in paragraph_scores[para_index]:
                    paragraph_scores[para_index][sent_index] = default_value

    return paragraph_scores

def calculate_sentence_location_scores(processed_classical_sentences):
    """
    Calculate and normalize sentence location scores from preprocessed classical sentence dictionary.

    Args:
        processed_classical_sentences (dict): Preprocessed sentences for classical representation.

    Returns:
        dict: Nested dictionary {paragraph_index: {sentence_index: normalized_location_score}}
    """
    paragraph_scores = {}
    all_scores = []

    for para_index, sentences in processed_classical_sentences.items():
        if para_index == "0":
            continue

        p_idx = int(para_index)
        sentence_scores = {}

        for sent_index, _ in sentences.items():
            s_idx = int(sent_index) + 1

            if s_idx == 1:
                location_score = 1 / p_idx
            else:
                location_score = 1 / (p_idx * s_idx)

            sentence_scores[sent_index] = location_score
            all_scores.append(location_score)

        paragraph_scores[para_index] = sentence_scores

    # Normalize scores
    if all_scores:
        min_score = min(all_scores)
        max_score = max(all_scores)
        score_range = max_score - min_score

        if score_range > 0:
            for para_index in paragraph_scores:
                for sent_index in paragraph_scores[para_index]:
                    old_score = paragraph_scores[para_index][sent_index]
                    normalized_score = (old_score - min_score) / score_range
                    paragraph_scores[para_index][sent_index] = normalized_score
        else:
            default_value = 0.5 if max_score > 0 else 0
            for para_index in paragraph_scores:
                for sent_index in paragraph_scores[para_index]:
                    paragraph_scores[para_index][sent_index] = default_value

    return paragraph_scores

def load_cue_words(file_path):
    """Load cue words from file and normalize them"""
    with open(file_path, "r", encoding="utf-8") as f:
        cue_words = [normalize_arabic(line.strip()) for line in f.readlines()]
    return cue_words

def count_cue_words(text, cue_words):
    """Count cue words in normalized text"""
    count = 0
    for cue_word in cue_words:
        # Count occurrences of the cue word in the normalized text
        count += text.count(cue_word)
    return count

def calculate_cue_word_scores(cue_words, processed_dl_sentences):
    """
    Calculate and normalize cue word scores from preprocessed DL sentence dictionary.

    Args:
        cue_words (set or list): Set of cue words to look for in sentences.
        processed_dl_sentences (dict): Dictionary of DL-normalized sentences.

    Returns:
        dict: Nested dictionary {paragraph_index: {sentence_index: normalized_score}}
    """
    paragraph_scores = {}
    all_scores = []

    for para_index, sentences in processed_dl_sentences.items():
        if para_index == "0":
            continue

        # Count total cue words in paragraph
        paragraph_cue_count = sum(
            count_cue_words(sentence, cue_words)
            for sentence in sentences.values()
        )

        sentence_scores = {}
        for sent_index, sentence in sentences.items():
            sentence_cue_count = count_cue_words(sentence, cue_words)
            cue_score = sentence_cue_count / paragraph_cue_count if paragraph_cue_count > 0 else 0
            sentence_scores[sent_index] = cue_score
            all_scores.append(cue_score)

        paragraph_scores[para_index] = sentence_scores

    # Normalize scores
    if all_scores:
        min_score = min(all_scores)
        max_score = max(all_scores)
        score_range = max_score - min_score

        if score_range > 0:
            for para_index in paragraph_scores:
                for sent_index in paragraph_scores[para_index]:
                    old_score = paragraph_scores[para_index][sent_index]
                    normalized_score = (old_score - min_score) / score_range
                    paragraph_scores[para_index][sent_index] = normalized_score
        else:
            default_value = 0.5 if max_score > 0 else 0
            for para_index in paragraph_scores:
                for sent_index in paragraph_scores[para_index]:
                    paragraph_scores[para_index][sent_index] = default_value

    return paragraph_scores

def count_numerical_data(text):
    """Count numerical data in text (both Arabic and English digits)"""
    # Match both Arabic and English digits
    pattern = r'[\u0660-\u0669\d]+'
    matches = re.findall(pattern, text)
    return len(matches)

def calculate_numerical_data_scores(original_sentences):
    """
    Calculate and normalize numerical data scores from original sentence dictionary.

    Args:
        original_sentences (dict): Original unprocessed sentence dictionary.

    Returns:
        dict: Nested dictionary {paragraph_index: {sentence_index: normalized_score}}
    """
    paragraph_scores = {}
    all_scores = []

    for para_index, sentences in original_sentences.items():
        if para_index == "0":
            continue

        # Count total numerical data in the paragraph
        paragraph_numerical_count = sum(
            count_numerical_data(sentence)
            for sentence in sentences.values()
        )

        sentence_scores = {}
        for sent_index, sentence in sentences.items():
            sentence_numerical_count = count_numerical_data(sentence)

            numerical_score = (
                sentence_numerical_count / paragraph_numerical_count
                if paragraph_numerical_count > 0 else 0
            )

            sentence_scores[sent_index] = numerical_score
            all_scores.append(numerical_score)

        paragraph_scores[para_index] = sentence_scores

    # Normalize scores
    if all_scores:
        min_score = min(all_scores)
        max_score = max(all_scores)
        score_range = max_score - min_score

        if score_range > 0:
            for para_index in paragraph_scores:
                for sent_index in paragraph_scores[para_index]:
                    old_score = paragraph_scores[para_index][sent_index]
                    normalized_score = (old_score - min_score) / score_range
                    paragraph_scores[para_index][sent_index] = normalized_score
        else:
            default_value = 0.5 if max_score > 0 else 0
            for para_index in paragraph_scores:
                for sent_index in paragraph_scores[para_index]:
                    paragraph_scores[para_index][sent_index] = default_value

    return paragraph_scores


def calculate_final_statistical_scores(
    cue_words_file_path,
    keyphrase_scores,
    original_sentences,
    processed_classical_sentences,
    processed_dl_sentences
):
    """
    Calculate final statistical scores and combine with keyphrase scores.

    Args:
        cue_words_file_path (str): Path to cue words file.
        keyphrase_scores (dict): Dictionary of keyphrase scores.
        original_sentences (dict): Original sentences dictionary.
        processed_classical_sentences (dict): Classical preprocessed sentences.
        processed_dl_sentences (dict): Deep learning preprocessed sentences.

    Returns:
        dict: Combined dictionary with both keyphrase_score and statistical_score.
    """
    # Load cue words
    cue_words = load_cue_words(cue_words_file_path)

    # Calculate all statistical feature scores
    length_scores = calculate_sentence_length_scores(processed_classical_sentences)
    location_scores = calculate_sentence_location_scores(processed_classical_sentences)
    cue_word_scores = calculate_cue_word_scores(cue_words, processed_dl_sentences)
    numerical_scores = calculate_numerical_data_scores(original_sentences)

    # Combine statistical scores
    final_scores = {}

    for para_index in length_scores:
        final_scores[para_index] = {}
        for sent_index in length_scores[para_index]:
            stat_score = (
                length_scores[para_index].get(sent_index, 0) +
                location_scores[para_index].get(sent_index, 0) +
                cue_word_scores[para_index].get(sent_index, 0) +
                numerical_scores[para_index].get(sent_index, 0)
            )
            final_scores[para_index][sent_index] = stat_score

    # Combine statistical scores with keyphrase scores
    combined_results = {}
    for para_index in final_scores:
        combined_results[para_index] = {}
        for sent_index in final_scores[para_index]:
            combined_results[para_index][sent_index] = {
                "keyphrase_score": keyphrase_scores.get(para_index, {}).get(sent_index, 0),
                "statistical_score": final_scores[para_index][sent_index]
            }

    #print(str(combined_results) + "\n\n")
    return combined_results