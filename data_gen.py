"""Synthetic credit-risk data generator.

`drift` goes from 0.0 (same world as training) to 1.0 (very different world).
`missing_rate` = fraction of 'income' values that go missing (pipeline break).
`unit_error_rate` = fraction of rows from a broken upstream feed (a feed outage zero-fills debt ratio, utilisation and late payments).
"""
import numpy as np
import pandas as pd

FEATURES = ["age", "income_k", "debt_ratio", "credit_utilization",
            "late_payments", "account_age_years", "loan_amount_k"]


def make_data(n, seed, drift=0.0, missing_rate=0.0, unit_error_rate=0.0):
    rng = np.random.default_rng(seed)
    d = float(drift)

    age = np.clip(rng.normal(42 - 14 * d, 10, n), 18, 80)
    income_k = np.clip(rng.lognormal(np.log(55 - 20 * d), 0.35, n), 8, 400)
    debt_ratio = np.clip(rng.normal(0.33 + 0.22 * d, 0.12, n), 0.0, 1.5)
    util = np.clip(rng.normal(0.40 + 0.20 * d, 0.18, n), 0.0, 1.2)
    late = rng.poisson(0.6 + 1.2 * d, n)
    acct_age = np.clip(rng.normal(8 - 3 * d, 4, n), 0.2, 40)
    loan_k = np.clip(rng.lognormal(np.log(15 + 10 * d), 0.5, n), 1, 200)

    # The "true rule" for who defaults. A new pattern appears as drift grows
    # (young borrowers become riskier) = concept drift.
    logit = 1.5 * (-1.4 - 2.3 * d
                   + 3.2 * (debt_ratio - 0.33)
                   + 1.8 * (util - 0.40)
                   + 0.45 * late
                   - 0.035 * (income_k - 55)
                   - 0.04 * (acct_age - 8)
                   + 0.015 * (loan_k - 15)
                   + (d ** 2) * 2.4 * (age < 30)
                   + (d ** 2) * 1.5 * (debt_ratio > 0.5))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({"age": age, "income_k": income_k, "debt_ratio": debt_ratio,
                       "credit_utilization": util, "late_payments": late,
                       "account_age_years": acct_age, "loan_amount_k": loan_k})
    if unit_error_rate > 0:
        bad = rng.random(n) < unit_error_rate
        df.loc[bad, ["debt_ratio", "credit_utilization", "late_payments"]] = 0.0  # feed outage -> zero-filled
    if missing_rate > 0:
        mask = rng.random(n) < missing_rate
        df.loc[mask, "income_k"] = np.nan
    return df, y


def make_text_data(n=500, seed=42, drifted=False):
    """Generate synthetic text dataset for sentiment classification / NLP monitoring.

    Parameters
    ----------
    n : int
        Number of text samples to generate (default: 500).
    seed : int
        Random seed for reproducibility.
    drifted : bool
        If False, generates short, standard customer product reviews (reference world).
        If True, generates severely drifted inputs (much longer texts, heavy uppercase,
        error codes, lots of numbers, excessive punctuation, or gibberish).

    Returns
    -------
    texts : list of str
        Generated text samples.
    labels : np.ndarray
        Binary labels (1 = positive, 0 = negative).
    """
    rng = np.random.default_rng(seed)

    # Vocabulary components for normal product reviews
    pos_subjects = [
        "This product", "The sound quality", "The build quality",
        "Battery life", "Customer service", "The device", "This item"
    ]
    pos_adverbs = [
        "is remarkably", "is truly", "is genuinely",
        "is surprisingly", "has been absolutely"
    ]
    pos_adjectives = [
        "fantastic", "reliable", "easy to use", "well designed",
        "impressive", "top notch", "great value"
    ]
    pos_conclusions = [
        "and I highly recommend it.", "and exceeded my expectations.",
        "and works like a charm.", "for daily use.", "and arrived very fast."
    ]

    neg_subjects = [
        "The product", "The battery", "The packaging",
        "Customer service", "The hardware", "This device", "The material"
    ]
    neg_adverbs = [
        "is completely", "is terribly", "is extremely",
        "has been very", "turned out quite"
    ]
    neg_adjectives = [
        "disappointing", "flawed", "difficult to set up",
        "unreliable", "overpriced", "cheaply made", "defective"
    ]
    neg_conclusions = [
        "and broke within a week.", "and I requested a refund.",
        "and does not match the description.", "would not recommend.", "very frustrating."
    ]

    texts = []
    labels = []

    if not drifted:
        for _ in range(n):
            is_pos = rng.random() > 0.5
            labels.append(1 if is_pos else 0)
            if is_pos:
                s = rng.choice(pos_subjects)
                adv = rng.choice(pos_adverbs)
                adj = rng.choice(pos_adjectives)
                c = rng.choice(pos_conclusions)
                text = f"{s} {adv} {adj} {c}"
            else:
                s = rng.choice(neg_subjects)
                adv = rng.choice(neg_adverbs)
                adj = rng.choice(neg_adjectives)
                c = rng.choice(neg_conclusions)
                text = f"{s} {adv} {adj} {c}"
            texts.append(text)
    else:
        drift_templates = [
            "URGENT CRITICAL SYSTEM ALERT #{id} DETECTED ON PORT {port} WITH ERROR CODE 0x{hex}: {gibberish} TRANSACTION FAILED $$$ {amount} USD !!!???",
            "SPAM PROMO WARNING 100% OFF ORDER {id} NOW AT HTTP://DEAL-{id}.XYZ/VERIFY?CODE={hex} CALL +1-800-{id} $$$$$ LIMITED TIME OFFER !!!",
            "CRITICAL EXCEPTION IN BATCH PROCESSOR PID={id} THREAD_{id} MEMORY DUMP: {gibberish} STATUS CODE 500 INTERNAL FAILURE !!!",
            "ATTENTION CUSTOMER ACCOUNT #{id}: REFUND REJECTED FOR TRANSACTION {id} AMOUNT ${amount}. CONTACT ADMIN IMMEDIATELY AT 9999-8888-7777 !!!???",
            "DATASTREAM CORRUPTION IN FEED #{id}: {gibberish} CHECKSUM MISMATCH 0x{hex} COUNT={id} LATENCY={amount}ms OVERFLOW ERROR !!!",
        ]
        gibberish_tokens = ["XZQWPV", "KLMN987", "ASDFG", "QWERTY99", "ZZZZZ", "101010", "XYZ999", "ERR_SYS"]

        for _ in range(n):
            labels.append(int(rng.random() > 0.5))
            tpl = rng.choice(drift_templates)
            ident = rng.integers(100000, 999999)
            port = rng.integers(1000, 9999)
            hex_val = f"{rng.integers(0x1000, 0xFFFF):X}"
            amount = f"{rng.uniform(10.0, 999.0):.2f}"
            gib = " ".join(rng.choice(gibberish_tokens, size=rng.integers(4, 9)))
            raw = tpl.format(id=ident, port=port, hex=hex_val, gibberish=gib, amount=amount)
            text = f"{raw} EXTRA_LOG: {gib} STATUS=FLAGGED_OUTLIER_RUN_{ident} !!!"
            texts.append(text)

    return texts, np.array(labels)


def make_image_data(drifted=False, seed=42, corruption="mixed"):
    """Load and optionally corrupt 8x8 grayscale digit images from sklearn.datasets.load_digits.

    Parameters
    ----------
    drifted : bool, default=False
        If False, returns clean/reference digit images.
        If True, returns images corrupted according to `corruption`.
    seed : int, default=42
        Random seed for shuffling and synthetic corruption generation.
    corruption : str, default='mixed'
        Corruption type applied when drifted=True:
        - 'dark'  : darker images (pixel values scaled by ~0.4)
        - 'noise' : added Gaussian sensor noise
        - 'blur'  : 3x3 uniform spatial averaging blur
        - 'mixed' : combination of darker, slight blur, and added noise

    Returns
    -------
    images : np.ndarray
        Array of shape (1797, 8, 8) with float pixel values.
    labels : np.ndarray
        Ground-truth digit labels (0-9).
    """
    from sklearn.datasets import load_digits
    digits = load_digits()
    imgs = digits.images.astype(float)
    y = digits.target.copy()

    # Shuffle dataset deterministically by seed
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(imgs))
    imgs = imgs[perm]
    y = y[perm]

    if not drifted:
        return imgs, y

    corr = str(corruption).lower().strip()
    corrupted = imgs.copy()

    # 1. Darker: scale pixel intensities down
    if corr in ("dark", "mixed"):
        corrupted = corrupted * 0.4

    # 2. Slight blur: 3x3 convolution averaging
    if corr in ("blur", "mixed"):
        pad = np.pad(corrupted, ((0, 0), (1, 1), (1, 1)), mode="edge")
        blurred = (
            pad[:, :-2, :-2] + pad[:, :-2, 1:-1] + pad[:, :-2, 2:] +
            pad[:, 1:-1, :-2] + pad[:, 1:-1, 1:-1] + pad[:, 1:-1, 2:] +
            pad[:, 2:, :-2] + pad[:, 2:, 1:-1] + pad[:, 2:, 2:]
        ) / 9.0
        corrupted = 0.4 * corrupted + 0.6 * blurred

    # 3. Add random noise: Gaussian sensor perturbation
    if corr in ("noise", "mixed"):
        noise_scale = 2.5 if corr == "mixed" else 3.5
        noise = rng.normal(0, noise_scale, size=corrupted.shape)
        corrupted = corrupted + noise

    corrupted = np.clip(corrupted, 0.0, 16.0)
    return corrupted, y
