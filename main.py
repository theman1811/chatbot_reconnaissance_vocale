import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from striprtf.striprtf import rtf_to_text
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import io
import os

if os.environ.get("STREAMLIT_CLOUD"):
    st.warning("Le microphone est désactivé en mode cloud")
    pyaudio = None
else:
    import pyaudio


# Chemin personnalisé pour les données NLTK (optionnel)
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Téléchargements garantis
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)

"""try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')"""

nltk.data.path.append('/home/adminuser/nltk_data')

# Charger le modèle français de spaCy
nlp = spacy.load("fr_core_news_sm")

# Téléchargements NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Chargement du fichier RTF
with open('essaie.rtf', 'r', encoding='utf-8') as f:
    rtf_text = f.read()
    data = rtf_to_text(rtf_text).replace('\n', ' ')

# Tokeniser les phrases et garder les originales
original_sentences = sent_tokenize(data)


# Prétraitement avec spaCy
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
           and not token.is_punct
           and not token.is_space
    ]
    return ' '.join(tokens)


# Préparer les données pour TF-IDF
processed_sentences = [preprocess(sent) for sent in original_sentences]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_sentences)


def get_most_relevant_sentence(query):
    query_processed = preprocess(query)
    query_vec = vectorizer.transform([query_processed])

    similarities = cosine_similarity(query_vec, X)
    max_index = similarities.argmax()

    if similarities[0, max_index] > 0.1:  # Seuil minimal
        return original_sentences[max_index]
    else:
        return None


def chatbot(question):
    relevant_sentence = get_most_relevant_sentence(question)
    if relevant_sentence:
        return relevant_sentence.strip()
    else:
        return "Désolé, je n'ai pas d'information sur ce sujet."


def transcribe_speech():
    """Convertit la parole en texte"""
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Parlez maintenant (5 secondes max)...")
            audio = r.listen(source, timeout=5)
        text = r.recognize_google(audio, language='fr-FR')
        return text
    except sr.UnknownValueError:
        st.error("Impossible de comprendre l'audio")
        return None
    except sr.RequestError as e:
        st.error(f"Erreur du service : {e}")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue : {e}")
        return None


def text_to_speech(text, lang='fr'):
    """Convertit le texte en parole"""
    if not text:
        return None
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        st.error(f"Erreur TTS : {e}")
        return None


# Interface Streamlit
def main():
    st.title("Chatbot Intelligent avec Reconnaissance Vocale")
    st.write("Posez-moi une question par texte ou par voix !")

    # Choix de la méthode d'entrée
    input_method = st.radio("Méthode d'entrée :",
                            ["Texte", "Voix"],
                            horizontal=True)

    question = ""

    # Gestion des entrées
    if input_method == "Texte":
        question = st.text_input("Entrez votre question :", key="text_input")
    else:
        if st.button("Enregistrer la question audio"):
            question = transcribe_speech()
            if question:
                st.session_state.text_input = question  # Met à jour l'entrée texte

    # Bouton de soumission
    if st.button("Soumettre"):
        if question:
            response = chatbot(question)
            st.session_state.response = response

            # Conversion texte vers parole
            if response:
                audio_data = text_to_speech(response)
                if audio_data:
                    st.session_state.audio = audio_data

    # Affichage des résultats
    if 'response' in st.session_state:
        st.subheader("Réponse du Chatbot :")
        st.text_area("", value=st.session_state.response, height=100, disabled=True)

        if 'audio' in st.session_state:
            st.audio(st.session_state.audio, format='audio/mp3')


if __name__ == "__main__":
    main()