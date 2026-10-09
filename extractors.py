"""Feature extractors for model-agnostic monitoring in AERO.

Extractors convert arbitrary raw input data (tabular rows, raw text strings, etc.)
into a standard pandas DataFrame where each column is a numeric feature and each
row corresponds to one sample. This allows the core drift monitoring engine
to remain completely model-agnostic and data-type-agnostic.
"""
from abc import ABC, abstractmethod
import string
import numpy as np
import pandas as pd


class FeatureExtractor(ABC):
    """Abstract base class for all feature extractors."""

    @abstractmethod
    def extract(self, raw_data) -> pd.DataFrame:
        """Extract numeric features from raw_data.

        Parameters
        ----------
        raw_data : Any
            Raw input data (e.g., pandas DataFrame, list of strings, etc.).

        Returns
        -------
        pd.DataFrame
            DataFrame of numeric features, with one row per sample.
        """
        pass


class TabularExtractor(FeatureExtractor):
    """Extractor for tabular data.

    Passes numeric columns through unchanged to preserve existing behavior.
    """

    def extract(self, raw_data) -> pd.DataFrame:
        """Extract numeric columns from tabular data.

        If a DataFrame is provided, numeric columns are passed through unchanged.
        If a NumPy array is provided, it is wrapped in a DataFrame.
        """
        if isinstance(raw_data, pd.DataFrame):
            # Pass numeric columns through; if all columns are numeric, keep original
            numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return raw_data[numeric_cols].copy()
            return raw_data.copy()
        elif isinstance(raw_data, np.ndarray):
            return pd.DataFrame(raw_data)
        return pd.DataFrame(raw_data)


class TextExtractor(FeatureExtractor):
    """Extractor for text data.

    Takes a list of strings (or a pandas Series of strings) and extracts
    7 interpretable numeric text statistics per sample:
    1. word_count: total words in the text
    2. char_count: total characters in the text
    3. avg_word_len: average character length per word
    4. uppercase_ratio: proportion of characters that are uppercase
    5. digit_ratio: proportion of characters that are digits
    6. punct_ratio: proportion of characters that are punctuation
    7. unique_word_ratio: ratio of unique words to total words (vocabulary richness)

    Uses only Python standard library and pandas/numpy.
    """

    PUNCTUATION_SET = set(string.punctuation)

    def extract(self, raw_data) -> pd.DataFrame:
        """Extract numeric features from a collection of text strings."""
        if isinstance(raw_data, (pd.Series, np.ndarray)):
            texts = raw_data.tolist()
        elif isinstance(raw_data, str):
            texts = [raw_data]
        elif hasattr(raw_data, "__iter__"):
            texts = list(raw_data)
        else:
            texts = [str(raw_data)]

        rows = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text) if text is not None else ""

            char_count = len(text)
            words = text.split()
            word_count = len(words)

            # 1. Word count & character count
            # 2. Average word length (avoid division by zero)
            avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0

            # 3. Character-level ratios
            if char_count > 0:
                uppercase_ratio = sum(1 for c in text if c.isupper()) / char_count
                digit_ratio = sum(1 for c in text if c.isdigit()) / char_count
                punct_ratio = sum(1 for c in text if c in self.PUNCTUATION_SET) / char_count
            else:
                uppercase_ratio = 0.0
                digit_ratio = 0.0
                punct_ratio = 0.0

            # 4. Vocabulary uniqueness ratio
            if word_count > 0:
                unique_words = len(set(w.lower() for w in words))
                unique_word_ratio = unique_words / word_count
            else:
                unique_word_ratio = 0.0

            rows.append({
                "word_count": float(word_count),
                "char_count": float(char_count),
                "avg_word_len": float(avg_word_len),
                "uppercase_ratio": float(uppercase_ratio),
                "digit_ratio": float(digit_ratio),
                "punct_ratio": float(punct_ratio),
                "unique_word_ratio": float(unique_word_ratio),
            })

        return pd.DataFrame(rows)


class ImageExtractor(FeatureExtractor):
    """Extractor for grayscale image data.

    Takes a NumPy array of grayscale images with shape (n_samples, height, width)
    (or a collection of 2D images).
    Normalizes images using a fixed global scale (never per-image min-max scaling),
    then extracts 6 interpretable numeric visual quality and distribution features per image:
    1. brightness: mean pixel value
    2. contrast: standard deviation of pixel values
    3. sharpness: mean absolute difference between neighbouring pixels (horizontal & vertical)
    4. dark_ratio: share of pixels below 0.2
    5. bright_ratio: share of pixels above 0.8
    6. noise_level: standard deviation of the difference between the image and a
       simple 3x3 average-blurred version of it

    Parameters
    ----------
    max_value : float, optional
        Maximum possible pixel intensity for normalization to [0, 1].
        If None, inferred ONCE from reference data passed to build_baseline
        (e.g. 16 for sklearn digits, or 255 for standard 8-bit images) and stored.

    Uses only NumPy and pandas.
    """

    def __init__(self, max_value: float = None):
        self.max_value = float(max_value) if max_value is not None else None

    def extract(self, raw_data) -> pd.DataFrame:
        """Extract 6 numeric features from grayscale images using a fixed global scale."""
        images = np.asarray(raw_data, dtype=float)
        if images.ndim == 2:
            images = images[np.newaxis, ...]
        elif images.ndim == 4 and images.shape[-1] == 1:
            images = images.squeeze(-1)

        if images.ndim != 3:
            raise ValueError(
                f"ImageExtractor expects 3D array of shape (n_samples, height, width), "
                f"got array with shape {images.shape}"
            )

        # Infer fixed global max_value ONCE if not already set
        if self.max_value is None:
            raw_max = float(np.max(images)) if len(images) > 0 else 1.0
            if raw_max <= 1.0:
                self.max_value = 1.0
            elif raw_max <= 16.0:
                self.max_value = 16.0
            elif raw_max <= 255.0:
                self.max_value = 255.0
            else:
                self.max_value = raw_max

        scale = self.max_value if self.max_value > 0 else 1.0

        rows = []
        for img in images:
            # Fixed global normalization: maps [0, scale] to [0, 1]
            norm = np.clip(img / scale, 0.0, 1.0)

            # 1. Brightness: mean pixel value
            brightness = float(np.mean(norm))

            # 2. Contrast: standard deviation of pixel values
            contrast = float(np.std(norm))

            # 3. Sharpness: mean absolute difference between neighbouring pixels (h & v)
            diff_h = np.abs(norm[:, 1:] - norm[:, :-1])
            diff_v = np.abs(norm[1:, :] - norm[:-1, :])
            sharpness = float((np.mean(diff_h) + np.mean(diff_v)) / 2.0)

            # 4. Dark ratio: share of pixels below 0.2
            dark_ratio = float(np.mean(norm < 0.2))

            # 5. Bright ratio: share of pixels above 0.8
            bright_ratio = float(np.mean(norm > 0.8))

            # 6. Noise level: std of difference between image and 3x3 average-blurred version
            pad = np.pad(norm, pad_width=1, mode="edge")
            blurred = (
                pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
                pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:] +
                pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]
            ) / 9.0
            noise_level = float(np.std(norm - blurred))

            rows.append({
                "brightness": brightness,
                "contrast": contrast,
                "sharpness": sharpness,
                "dark_ratio": dark_ratio,
                "bright_ratio": bright_ratio,
                "noise_level": noise_level,
            })

        return pd.DataFrame(rows)


def get_extractor(data_type: str = "tabular", **kwargs) -> FeatureExtractor:
    """Factory function returning the appropriate feature extractor.

    Parameters
    ----------
    data_type : str
        Type of data: 'tabular', 'text', or 'image'.
    **kwargs : dict
        Additional parameters passed to extractor constructor (e.g. max_value).

    Returns
    -------
    FeatureExtractor
        Instantiated feature extractor.
    """
    if not isinstance(data_type, str):
        raise TypeError(f"data_type must be a string, got {type(data_type).__name__}")

    dt = data_type.strip().lower()
    if dt == "tabular":
        return TabularExtractor()
    elif dt == "text":
        return TextExtractor()
    elif dt == "image":
        return ImageExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown data_type: {data_type!r}. Supported types: 'tabular', 'text', 'image'.")
