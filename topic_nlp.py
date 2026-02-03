import json
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from langdetect import detect, LangDetectException
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

# paths
SCRIPT_DIR = Path(__file__).parent.absolute()

DATA_DIR = SCRIPT_DIR / 'data'
OUTPUT_DIR = SCRIPT_DIR / 'output'

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / 'eu_consultation_responses.json'
PREPROCESSED_FILE = OUTPUT_DIR / 'preprocessed_responses.json'
RESULTS_FILE = OUTPUT_DIR / 'complete_topic_modeling_results.json'
PLOT_FILE = OUTPUT_DIR / 'all_approaches_comparison.png'


# ============ STEP 1: TEXT PREPROCESSING ============

def detect_language(text):
    """Detect if text is English"""
    try:
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        return True

def remove_punctuation(text):
    """Remove punctuation"""
    return text.translate(str.maketrans('', '', string.punctuation))

def spellcheck(text):
    """Fix spelling errors"""
    blob = TextBlob(text)
    return str(blob.correct())

def remove_stopwords(tokens):
    """Remove stopwords"""
    return [token for token in tokens if token.lower() not in stop_words]

def lemmatise_words(tokens):
    """Convert all words to base form (removes plurals)"""
    lemmatiser = WordNetLemmatizer()
    lemmatised = [lemmatiser.lemmatize(token) for token in tokens]
    return lemmatised

def tokenize(text):
    """Tokenize text"""
    return word_tokenize(text.lower())

def preprocess_text(text):
    """Full preprocessing pipeline"""
    # Language detection
    if not detect_language(text):
        return None
    
    # Remove punctuation
    text = remove_punctuation(text)
    
    # Spellcheck
    text = spellcheck(text)
    
    # Tokenize
    tokens = tokenize(text)
    
    # Remove stopwords
    tokens = remove_stopwords(tokens)

    # Lemmatise words
    tokens = lemmatise_words(tokens)
    
    return tokens

# ============ STEP 2: N-GRAM & NOUN EXTRACTION ============

def extract_nouns(tokens):
    """Extract only nouns using POS tagging"""
    pos_tags = pos_tag(tokens, tagset='universal')
    nouns = [token for token, pos in pos_tags if pos == 'NOUN']
    return nouns

def create_ngrams(tokens, n=2):
    """Create n-grams"""
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def prepare_documents(processed_responses):
    """
    Prepare full text and noun-only versions
    """
    full_texts = []
    noun_texts = []
    metadata = []
    
    for response in processed_responses:
        tokens = response['tokens']
        
        # Full text version
        full_text = ' '.join(tokens)
        full_texts.append(full_text)
        
        # Noun-only version
        nouns = extract_nouns(tokens)
        if nouns:  # Only include if has nouns
            noun_text = ' '.join(nouns)
            noun_texts.append(noun_text)
        else:
            noun_texts.append(full_text)  # Fallback to full text if no nouns
        
        metadata.append({
            'page': response['page'],
            'original_text': response['original_text'],
            'n_tokens': len(tokens),
            'n_nouns': len(nouns)
        })
    
    return full_texts, noun_texts, metadata

# ============ STEP 3: VECTORISATION ============

def create_bow_vectors(texts, max_features=1000):
    """Create BoW vectors"""
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=0.01,
        max_df=0.5,
        ngram_range=(1, 2)  # Include bigrams
    )
    bow_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return bow_matrix, vectorizer, feature_names

def create_tfidf_vectors(texts, max_features=1000):
    """Create TF-IDF vectors"""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=0.01,
        max_df=0.5,
        ngram_range=(1, 2)  # Include bigrams
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return tfidf_matrix, vectorizer, feature_names

# ============ STEP 4: TOPIC MODELING ============

def apply_lda(bow_matrix, n_topics=5, max_iter=20):
    """Apply LDA"""
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=max_iter,
        learning_method='online'
    )
    lda_output = lda.fit_transform(bow_matrix)
    
    return lda, lda_output

def apply_lsa(tfidf_matrix, n_topics=5):
    """Apply LSA"""
    lsa = TruncatedSVD(
        n_components=n_topics,
        random_state=42,
        n_iter=100
    )
    lsa_output = lsa.fit_transform(tfidf_matrix)
    
    return lsa, lsa_output

# ============ STEP 5: EVALUATION ============
# top words parameter
n_words = 10

def get_top_words_per_topic(model, feature_names, n_words):
    """Extract top words for each topic"""
    topics = {}
    for topic_id, topic in enumerate(model.components_):
        top_word_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_word_indices]
        topics[f'Topic {topic_id}'] = top_words
    
    return topics

def calculate_perplexity(model, data_matrix):
    """Calculate perplexity for LDA"""
    if hasattr(model, 'perplexity'):
        return model.perplexity(data_matrix)
    return None

def calculate_silhouette_score(representations):
    """Calculate average internal similarity"""
    similarities = []
    for i in range(len(representations)):
        for j in range(i+1, min(i+10, len(representations))):  # Sample pairs
            sim = cosine_similarity(
                representations[i].reshape(1, -1),
                representations[j].reshape(1, -1)
            )[0][0]
            similarities.append(sim)
    return np.mean(similarities) if similarities else 0

def evaluate_all_models(results_dict):
    """Create comparison table"""
    comparison = []
    
    for approach in ['Full Text', 'Nouns Only']:
        for algo in ['LDA', 'LSA']:
            key = f"{approach}_{algo}"
            if key in results_dict:
                data = results_dict[key]
                comparison.append({
                    'Approach': approach,
                    'Algorithm': algo,
                    'Matrix Size': str(data['matrix_shape']),
                    'Avg Coherence': f"{data['coherence_score']:.4f}",
                    'Top Doc-Topic Variance': f"{data['topic_variance']:.4f}"
                })
    
    return pd.DataFrame(comparison)

# ============ MAIN PIPELINE ============

print("=" * 70)
print("COMPLETE TEXT PREPROCESSING & TOPIC MODELING PIPELINE")
print("=" * 70)

# Load raw responses
print("\n[STEP 1] Loading raw consultation responses...")
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    raw_responses = json.load(f)
print(f"Loaded {len(raw_responses)} responses")

# Preprocess
print("\n[STEP 2] Preprocessing texts (language detection, spell-check, tokenisation)...")
processed_responses = []
skipped = 0

for i, response in enumerate(raw_responses):

    tokens = preprocess_text(response['text'])
    
    if tokens:
        processed_responses.append({
            'page': response['page'],
            'original_text': response['text'],
            'tokens': tokens
        })
    else:
        skipped += 1

print(f"Processed {len(processed_responses)} English responses (skipped {skipped} non-English)")

# Save preprocessed data
with open(PREPROCESSED_FILE, 'w', encoding='utf-8') as f:
    json.dump(processed_responses, f, indent=2, ensure_ascii=False)

# Prepare full text and noun-only versions~
print("\n[STEP 3] Preparing full text and noun-only documents...")
full_texts, noun_texts, metadata = prepare_documents(processed_responses)
print(f"Full texts: {len(full_texts)} documents")
print(f"Noun texts: {len(noun_texts)} documents")

# Training configuration
n_topics = 3
results = {}

# ============ TRAIN ON FULL TEXT ============
print("\n" + "=" * 70)
print("TRAINING ON FULL TEXT")
print("=" * 70)

print("\n[BoW Vectorization (Full Text)]")
bow_full, bow_vec_full, bow_feat_full = create_bow_vectors(full_texts, max_features=500)
print(f"Matrix shape: {bow_full.shape}")

print("\n[LDA on Full Text]")
lda_full, lda_out_full = apply_lda(bow_full, n_topics=n_topics)
lda_topics_full = get_top_words_per_topic(lda_full, bow_feat_full, n_words)
print("LDA Topics:")
for topic, words in lda_topics_full.items():
    print(f"  {topic}: {', '.join(words)}")

print("\n[TF-IDF Vectorization (Full Text)]")
tfidf_full, tfidf_vec_full, tfidf_feat_full = create_tfidf_vectors(full_texts, max_features=500)
print(f"Matrix shape: {tfidf_full.shape}")

print("\n[LSA on Full Text]")
lsa_full, lsa_out_full = apply_lsa(tfidf_full, n_topics=n_topics)
lsa_topics_full = get_top_words_per_topic(lsa_full, tfidf_feat_full, n_words)
print("LSA Topics:")
for topic, words in lsa_topics_full.items():
    print(f"  {topic}: {', '.join(words)}")

results['Full Text_LDA'] = {
    'model': lda_full,
    'output': lda_out_full,
    'topics': lda_topics_full,
    'matrix_shape': bow_full.shape,
    'coherence_score': calculate_silhouette_score(lda_out_full),
    'topic_variance': np.var(np.argmax(lda_out_full, axis=1))
}

results['Full Text_LSA'] = {
    'model': lsa_full,
    'output': lsa_out_full,
    'topics': lsa_topics_full,
    'matrix_shape': tfidf_full.shape,
    'coherence_score': calculate_silhouette_score(lsa_out_full),
    'topic_variance': np.var(np.argmax(np.abs(lsa_out_full), axis=1))
}

# ============ TRAIN ON NOUNS ONLY ============
print("\n" + "=" * 70)
print("TRAINING ON NOUNS ONLY")
print("=" * 70)

print("\n[BoW Vectorization (Nouns)]")
bow_nouns, bow_vec_nouns, bow_feat_nouns = create_bow_vectors(noun_texts, max_features=500)
print(f"Matrix shape: {bow_nouns.shape}")

print("\n[LDA on Nouns]")
lda_nouns, lda_out_nouns = apply_lda(bow_nouns, n_topics=n_topics)
lda_topics_nouns = get_top_words_per_topic(lda_nouns, bow_feat_nouns, n_words)
print("LDA Topics:")
for topic, words in lda_topics_nouns.items():
    print(f"  {topic}: {', '.join(words)}")

print("\n[TF-IDF Vectorization (Nouns)]")
tfidf_nouns, tfidf_vec_nouns, tfidf_feat_nouns = create_tfidf_vectors(noun_texts, max_features=500)
print(f"Matrix shape: {tfidf_nouns.shape}")

print("\n[LSA on Nouns]")
lsa_nouns, lsa_out_nouns = apply_lsa(tfidf_nouns, n_topics=n_topics)
lsa_topics_nouns = get_top_words_per_topic(lsa_nouns, tfidf_feat_nouns, n_words)
print("LSA Topics:")
for topic, words in lsa_topics_nouns.items():
    print(f"  {topic}: {', '.join(words)}")

results['Nouns Only_LDA'] = {
    'model': lda_nouns,
    'output': lda_out_nouns,
    'topics': lda_topics_nouns,
    'matrix_shape': bow_nouns.shape,
    'coherence_score': calculate_silhouette_score(lda_out_nouns),
    'topic_variance': np.var(np.argmax(lda_out_nouns, axis=1))
}

results['Nouns Only_LSA'] = {
    'model': lsa_nouns,
    'output': lsa_out_nouns,
    'topics': lsa_topics_nouns,
    'matrix_shape': tfidf_nouns.shape,
    'coherence_score': calculate_silhouette_score(lsa_out_nouns),
    'topic_variance': np.var(np.argmax(np.abs(lsa_out_nouns), axis=1))
}

# ============ COMPARISON & RESULTS ============
print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)

comparison_df = evaluate_all_models(results)
print("\n", comparison_df.to_string(index=False))

# Save comprehensive results
save_results = {
    'n_documents': len(processed_responses),
    'n_topics': n_topics,
    'Full_Text_LDA_topics': results['Full Text_LDA']['topics'],
    'Full_Text_LSA_topics': results['Full Text_LSA']['topics'],
    'Nouns_Only_LDA_topics': results['Nouns Only_LDA']['topics'],
    'Nouns_Only_LSA_topics': results['Nouns Only_LSA']['topics'],
    'comparison_metrics': comparison_df.to_dict(),
    'Full_Text_LDA_doc_topics': results['Full Text_LDA']['output'].tolist(),
    'Full_Text_LSA_doc_topics': results['Full Text_LSA']['output'].tolist(),
    'Nouns_Only_LDA_doc_topics': results['Nouns Only_LDA']['output'].tolist(),
    'Nouns_Only_LSA_doc_topics': results['Nouns Only_LSA']['output'].tolist(),
}

with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
    json.dump(save_results, f, indent=2, ensure_ascii=False)

# Visualization: Compare all four approaches
print("\n[Creating visualizations]...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Full Text LDA
axes[0, 0].hist(np.argmax(lda_out_full, axis=1), bins=n_topics, edgecolor='black', color='skyblue')
axes[0, 0].set_title('Full Text - LDA', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Topic ID')
axes[0, 0].set_ylabel('# Documents')

# Full Text LSA
axes[0, 1].hist(np.argmax(np.abs(lsa_out_full), axis=1), bins=n_topics, edgecolor='black', color='lightcoral')
axes[0, 1].set_title('Full Text - LSA', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Topic ID')
axes[0, 1].set_ylabel('# Documents')

# Nouns LDA
axes[1, 0].hist(np.argmax(lda_out_nouns, axis=1), bins=n_topics, edgecolor='black', color='lightgreen')
axes[1, 0].set_title('Nouns Only - LDA', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Topic ID')
axes[1, 0].set_ylabel('# Documents')

# Nouns LSA
axes[1, 1].hist(np.argmax(np.abs(lsa_out_nouns), axis=1), bins=n_topics, edgecolor='black', color='plum')
axes[1, 1].set_title('Nouns Only - LSA', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Topic ID')
axes[1, 1].set_ylabel('# Documents')

plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
print("Saved: all_approaches_comparison.png")

plt.show()

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print("\nGenerated files:")
print("  1. preprocessed_responses.json - Cleaned and tokenized texts")
print("  2. complete_topic_modeling_results.json - All topics and results")
print("  3. all_approaches_comparison.png - Visual comparison of 4 approaches")