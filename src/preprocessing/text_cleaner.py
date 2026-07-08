import re
import unicodedata

class TextCleaner:
    """Industrial-grade text cleaner for multilingual feedback.
    
    Handles Unicode normalization, quotes, punctuation repetition, whitespace,
    and currency normalization while preserving emojis and semantic markers.
    """
    
    def __init__(self) -> None:
        # Quote normalization patterns
        self.quote_replacements = [
            (re.compile(r"[‘’'`′]"), "'"),
            (re.compile(r"[“”«»\"]"), '"'),
            (re.compile(r"…"), "..."),
        ]
        
        # Currency symbol replacement
        self.currency_pattern = re.compile(r"₼")
        
        # Zero-width joiner & variation selectors (emoji formatting noise)
        self.invisible_formatting_pattern = re.compile(r"[\u200d\ufe0f]")
        
        # Repeated letters: collapse 4+ same letters to 3.
        # e.g. 'работаааает' -> 'работааает', 'cooool' -> 'coool'
        # We keep up to 3 for natural emphasis; beyond that is pure noise.
        self.repeated_letters_pattern = re.compile(
            r"([a-zA-Zа-яА-ЯёЁəışçğöüƏİŞÇĞÖÜ])\1{3,}"
        )
        
        # Repeated punctuation patterns (collapse 4+ same punctuation to 3)
        self.repeated_punct_pattern = re.compile(r"([.!?\-])\1{3,}")
        
        # Whitespace normalizer
        self.whitespace_pattern = re.compile(r"\s+")
        
    def clean(self, text: str) -> str:
        """Cleans and normalizes the input text.
        
        Args:
            text: Raw input string.
            
        Returns:
            Normalized and cleaned string.
        """
        if not isinstance(text, str):
            return ""
            
        # 1. Unicode NFKC Normalization
        text = unicodedata.normalize("NFKC", text)
        
        # 2. Quotation and ellipsis normalization
        for pattern, replacement in self.quote_replacements:
            text = pattern.sub(replacement, text)
            
        # 3. Currency symbol normalization (preserves value representation)
        text = self.currency_pattern.sub(" manat ", text)
        
        # 4. Remove invisible formatting characters
        text = self.invisible_formatting_pattern.sub("", text)
        
        # 5. Limit letter repetition: collapse 4+ to 3 (e.g., 'работаааает' -> 'работааает')
        text = self.repeated_letters_pattern.sub(r"\1\1\1", text)
        
        # 6. Limit punctuation repetition (e.g., !!!!! -> !!!)
        text = self.repeated_punct_pattern.sub(r"\1\1\1", text)
        
        # 7. Whitespace collapse
        text = self.whitespace_pattern.sub(" ", text)
        
        return text.strip()
