"""Enmascarado común de identificadores personales."""
import re


def mask_handles(text: str) -> str:
    """Oculta handles con dominio y perfiles de Bluesky."""
    text = re.sub(r'(?<![\w@])@[\w-]+(?:\.[\w-]+)+', '@usuario', text)
    return re.sub(r'(bsky\.app/profile/)[^/\s"<>?#]+', r'\1usuario', text)
