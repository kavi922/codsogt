import pandas as pd
from typing import Optional

# expected columns: text, label
# labels: real, fake, bank_fake (for bank statement fake news)

def merge_general_and_bank(general_csv: str, bank_csv: str, label_bank: str = "bank_fake") -> pd.DataFrame:
    general = pd.read_csv(general_csv)
    bank = pd.read_csv(bank_csv)
    if "text" not in general.columns or "label" not in general.columns:
        raise ValueError("General dataset must include 'text' and 'label'.")
    if "text" not in bank.columns:
        raise ValueError("Bank dataset must include 'text'.")

    bank = bank.copy()
    if "label" not in bank.columns:
        bank["label"] = label_bank
    else:
        bank["label"] = bank["label"].fillna(label_bank)

    merged = pd.concat([general[["text", "label"]], bank[["text", "label"]]], ignore_index=True)
    merged = merged.dropna(subset=["text", "label"]).reset_index(drop=True)
    return merged