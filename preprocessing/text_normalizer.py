"""
Text Normalizer for Indian English and Kannada TTS/STT
Handles numbers, abbreviations, symbols, and special characters.
"""

import re
import unicodedata
from num2words import num2words


class TextNormalizer:
    """Normalize raw text for TTS and STT preprocessing."""

    # Common Indian English abbreviations
    ABBREVIATIONS = {
        "dr.": "doctor",
        "mr.": "mister",
        "mrs.": "missus",
        "ms.": "miss",
        "prof.": "professor",
        "sr.": "senior",
        "jr.": "junior",
        "st.": "saint",
        "govt.": "government",
        "dept.": "department",
        "univ.": "university",
        "avg.": "average",
        "approx.": "approximately",
        "inc.": "incorporated",
        "ltd.": "limited",
        "pvt.": "private",
        "rs.": "rupees",
        "no.": "number",
        "nos.": "numbers",
        "tel.": "telephone",
        "vs.": "versus",
        "etc.": "etcetera",
        "i.e.": "that is",
        "e.g.": "for example",
    }

    # Indian currency and unit symbols
    SYMBOLS = {
        "₹": "rupees",
        "$": "dollars",
        "€": "euros",
        "£": "pounds",
        "%": "percent",
        "&": "and",
        "@": "at",
        "#": "number",
        "°": "degrees",
        "km": "kilometers",
        "kg": "kilograms",
        "cm": "centimeters",
        "mm": "millimeters",
        "ml": "milliliters",
        "hr": "hour",
        "hrs": "hours",
        "min": "minutes",
        "sec": "seconds",
    }

    def __init__(self, language: str = "en-in"):
        self.language = language

    def normalize(self, text: str) -> str:
        """Full normalization pipeline."""
        text = self._clean_whitespace(text)
        text = self._expand_abbreviations(text)
        text = self._expand_currency(text)
        text = self._expand_numbers(text)
        text = self._expand_symbols(text)
        text = self._clean_special_chars(text)
        text = self._clean_whitespace(text)
        return text.strip()

    def _clean_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n+", ". ", text)
        return text.strip()

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        for abbr, expansion in self.ABBREVIATIONS.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            text = pattern.sub(expansion, text)
        return text

    def _expand_currency(self, text: str) -> str:
        """Expand currency notations like ₹500, $1000."""
        # ₹ followed by number
        def _rupee_replace(match):
            num = match.group(1).replace(",", "")
            try:
                words = num2words(int(num), lang="en_IN")
                return f"{words} rupees"
            except (ValueError, OverflowError):
                return f"{num} rupees"

        text = re.sub(r"₹\s?([\d,]+)", _rupee_replace, text)

        # $ followed by number
        def _dollar_replace(match):
            num = match.group(1).replace(",", "")
            try:
                words = num2words(int(num), lang="en")
                return f"{words} dollars"
            except (ValueError, OverflowError):
                return f"{num} dollars"

        text = re.sub(r"\$\s?([\d,]+)", _dollar_replace, text)
        return text

    def _expand_numbers(self, text: str) -> str:
        """Convert numbers to words using Indian numbering system."""
        def _number_to_words(match):
            num_str = match.group(0).replace(",", "")
            try:
                # Handle decimals
                if "." in num_str:
                    parts = num_str.split(".")
                    integer_part = num2words(int(parts[0]), lang="en_IN")
                    decimal_part = " ".join(
                        num2words(int(d), lang="en_IN") for d in parts[1]
                    )
                    return f"{integer_part} point {decimal_part}"
                else:
                    return num2words(int(num_str), lang="en_IN")
            except (ValueError, OverflowError):
                return num_str

        # Match numbers (with optional commas and decimals)
        text = re.sub(r"\d[\d,]*\.?\d*", _number_to_words, text)
        return text

    def _expand_symbols(self, text: str) -> str:
        """Replace symbols with their spoken forms."""
        for symbol, word in self.SYMBOLS.items():
            text = text.replace(symbol, f" {word} ")
        return text

    def _clean_special_chars(self, text: str) -> str:
        """Remove or replace special characters."""
        # Keep basic punctuation for prosody
        text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text)
        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)
        return text


class KannadaTextNormalizer(TextNormalizer):
    """Extended normalizer for Kannada text."""

    # Kannada digits to words mapping
    KANNADA_DIGITS = {
        "೦": "0", "೧": "1", "೨": "2", "೩": "3", "೪": "4",
        "೫": "5", "೬": "6", "೭": "7", "೮": "8", "೯": "9",
    }

    def __init__(self):
        super().__init__(language="kn")

    def normalize(self, text: str) -> str:
        """Kannada-specific normalization."""
        text = self._normalize_kannada_digits(text)
        text = self._clean_whitespace(text)
        text = self._normalize_kannada_unicode(text)
        text = self._clean_whitespace(text)
        return text.strip()

    def _normalize_kannada_digits(self, text: str) -> str:
        """Convert Kannada digits to Arabic numerals for processing."""
        for kn_digit, ar_digit in self.KANNADA_DIGITS.items():
            text = text.replace(kn_digit, ar_digit)
        return text

    def _normalize_kannada_unicode(self, text: str) -> str:
        """Normalize Kannada Unicode: NFC normalization, handle nukta, etc."""
        text = unicodedata.normalize("NFC", text)
        # Remove zero-width joiners/non-joiners if problematic
        text = text.replace("\u200d", "")  # ZWJ
        text = text.replace("\u200c", "")  # ZWNJ
        return text
