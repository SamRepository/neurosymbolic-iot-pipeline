from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class CasasWindow:
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    label: str
    token_ids: List[int]
    time_deltas: List[float]  # seconds from window start


def build_casas_windows_from_raw(
    cfg: Dict,
    *,
    window_minutes: int,
    stride_minutes: int,
    min_events: int,
) -> pd.DataFrame:
    """
    Build window boundaries from raw CASAS events.
    Output DataFrame columns:
      window_id, start_time, end_time, label
    """
    from neurosymbolic_iot.data_processing.preprocess_casas import load_casas_events

    events = load_casas_events(cfg)
    if events.empty:
        raise RuntimeError("CASAS: no events loaded. Check datasets.casas.raw_dir in config.")

    events = events.dropna(subset=["timestamp", "sensor", "value"]).copy()
    events = events.sort_values("timestamp").reset_index(drop=True)

    win = pd.Timedelta(minutes=int(window_minutes))
    stride = pd.Timedelta(minutes=int(stride_minutes))

    start = events["timestamp"].min()
    end = events["timestamp"].max()

    rows = []
    window_id = 0
    t = start

    while t + win <= end:
        w = events[(events["timestamp"] >= t) & (events["timestamp"] < t + win)]
        if len(w) >= int(min_events):
            # Majority activity label within the window (if available)
            labs = w["activity"].dropna()
            if not labs.empty:
                label = str(labs.value_counts().idxmax())
                rows.append(
                    {
                        "window_id": window_id,
                        "start_time": t,
                        "end_time": t + win,
                        "label": label,
                    }
                )
                window_id += 1
        t = t + stride

    dfw = pd.DataFrame(rows)
    if dfw.empty:
        raise RuntimeError("CASAS: no windows built. Try lowering min_events or adjusting window/stride.")
    return dfw


def build_casas_sequences(
    cfg: Dict,
    windows: pd.DataFrame,
    *,
    max_seq_len: int,
    vocab: Dict[str, int] | None = None,
) -> Tuple[List[CasasWindow], Dict[str, int]]:
    """
    Create token sequences per window.

    Token definition:
      token = f"{sensor}:{value}"

    Vocab:
      0 = <PAD>
      1 = <UNK>
      2.. = observed tokens
    """
    from neurosymbolic_iot.data_processing.preprocess_casas import load_casas_events

    events = load_casas_events(cfg)
    if events.empty:
        raise RuntimeError("CASAS: no events loaded. Check datasets.casas.raw_dir in config.")

    events = events.dropna(subset=["timestamp", "sensor", "value"]).copy()
    events = events.sort_values("timestamp").reset_index(drop=True)

    events["token"] = events["sensor"].astype(str) + ":" + events["value"].astype(str)

    # Build vocab (preferably from the passed windows, typically TRAIN windows)
    if vocab is None:
        from collections import Counter

        all_tokens: List[str] = []
        for _, r in windows.iterrows():
            t0 = r["start_time"]
            t1 = r["end_time"]
            w = events[(events["timestamp"] >= t0) & (events["timestamp"] < t1)]
            if not w.empty:
                all_tokens.extend(w["token"].astype(str).tolist())

        cnt = Counter(all_tokens)
        tokens_sorted = [t for t, _ in cnt.most_common()]

        vocab = {"<PAD>": 0, "<UNK>": 1}
        for tok in tokens_sorted:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    seqs: List[CasasWindow] = []
    for _, r in windows.iterrows():
        t0 = r["start_time"]
        t1 = r["end_time"]
        label = str(r["label"])

        w = events[(events["timestamp"] >= t0) & (events["timestamp"] < t1)]
        if w.empty:
            continue

        toks = w["token"].astype(str).tolist()
        ts = w["timestamp"]
        deltas = (ts - t0).dt.total_seconds().to_list()

        # truncate
        if len(toks) > int(max_seq_len):
            toks = toks[: int(max_seq_len)]
            deltas = deltas[: int(max_seq_len)]

        token_ids = [vocab.get(tok, 1) for tok in toks]
        seqs.append(
            CasasWindow(
                start_time=t0,
                end_time=t1,
                label=label,
                token_ids=token_ids,
                time_deltas=[float(x) for x in deltas],
            )
        )

    if not seqs:
        raise RuntimeError("CASAS: no sequences created. Check windowing and raw data.")
    return seqs, vocab
