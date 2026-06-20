"""Preparación del entorno (NLTK, spaCy)."""
import nltk
import spacy

from nlp.preprocessing import ensure_nltk_resources


def prepare_environment() -> None:
    """Descarga recursos NLTK y verifica el modelo spaCy."""
    ensure_nltk_resources()
    for resource in ("stopwords", "wordnet", "averaged_perceptron_tagger_eng"):
        nltk.download(resource, quiet=True)
    spacy.load("en_core_web_sm")
