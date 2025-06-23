from flask import Flask, request, jsonify
from flask_cors import CORS
from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from preprocessing.preprocess import preprocess_text
from preprocessing.feature_extraction import extract_keyphrases
from preprocessing.feature_extraction import calculate_final_statistical_scores
from graph_construction.build_graph import process_file
from ranking.rank_sentences import formula_3_16_1N

app = Flask(__name__)
CORS(app)  # Allow all origins for simplicity; restrict in production

# Load AraBERT model silently
with tqdm(total=1, desc="Loading AraBERT model", leave=False) as pbar:
    model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv02", 
                                         return_dict=True)
    kw_extractor = KeyBERT(model)
    pbar.update(1)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text', '')

    # Step 1: Preprocessing and Feature Extraction
    original, preprocessed_dl, preprocessed_classical = preprocess_text(text)
    keyphrase_scores = extract_keyphrases(preprocessed_dl, kw_extractor)
    combined_results = calculate_final_statistical_scores(
        "assetes/arabic_cue_words.txt",
        keyphrase_scores,
        original,
        preprocessed_classical,
        preprocessed_dl)

    # Step 2: Graph and Ranking
    result = process_file(preprocessed_classical, combined_results)
    ranked_sentences = formula_3_16_1N(result, original)

    # Step 3: Return all scored sentences (sorted descending by score)
    response_data = [
        {
            "para_idx": node[0],
            "sent_idx": node[1],
            "text": text,
            "score": score
        }
        for node, score, text in ranked_sentences
    ]

    return jsonify({"sentences": response_data})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
