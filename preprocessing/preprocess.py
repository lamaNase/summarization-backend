import re
import nltk
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
import pyarabic.araby as araby

def ISRI_Stemmer(text):
    #making an object
    stemmer = ISRIStemmer()
    
    #stemming each word
    text = stemmer.stem(text)
    text = stemmer.pre32(text)
    text = stemmer.suf32(text)
    
    return text

# Arabic normalization function
def normalize_arabic(text):
    text = text.strip()
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    
    #remove repetetions
    text = re.sub("[إأٱآا]", "ا", text)
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ى')
    text = text.replace('ييي', 'ى')
    text = text.replace('اا', 'ا')

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    
    # Remove longation
    text = re.sub(r'(.)\1+', r"\1\1", text) 
    
    #Strip vowels from a text, include Shadda.
    text = araby.strip_tashkeel(text)
    
    #Strip diacritics from a text, include harakats and small lettres The striped marks are
    text = araby.strip_diacritics(text)
    return text


def preprocess_text(text):
    """
    Preprocesses a single text string and returns three dictionaries:
    - original_sentences
    - preprocessed_dl_sentences
    - preprocessed_classical_sentences

    Args:
        text (str): The input text to preprocess

    Returns:
        tuple: (original_sentences, preprocessed_dl_sentences, preprocessed_classical_sentences)
    """
    try:
        arabic_stopwords = set(stopwords.words("arabic"))
    except:
        nltk.download("stopwords")
        arabic_stopwords = set(stopwords.words("arabic"))

    paragraphs = text.split("\n")  # Paragraph segmentation
    paragraphs = [p.strip() for p in paragraphs if p.strip()]  # Remove empty paragraphs

    original_sentences = {}
    preprocessed_dl_sentences = {}
    preprocessed_classical_sentences = {}

    for p_idx, paragraph in enumerate(paragraphs):
        sentences = re.split(r"[.?!]", paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            continue

        original_sentences[str(p_idx)] = {str(s_idx): s for s_idx, s in enumerate(sentences)}

        # Normalize for DL
        normalized_sentences = {str(s_idx): normalize_arabic(s) for s_idx, s in enumerate(sentences)}
        preprocessed_dl_sentences[str(p_idx)] = normalized_sentences

        # Classical preprocessing
        classical_sentences = {
            str(s_idx): " ".join(
                [ISRI_Stemmer(word) for word in s.split() if word not in arabic_stopwords]
            )
            for s_idx, s in normalized_sentences.items()
        }
        preprocessed_classical_sentences[str(p_idx)] = classical_sentences

    if not original_sentences:
        print("⚠️ No valid content found in the provided text")
        return None, None, None

    return original_sentences, preprocessed_dl_sentences, preprocessed_classical_sentences