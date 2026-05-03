"""CIC-IoT-2023 multi-class network flow dataset loader.

The dataset is published as many CSV files (~46 GB total) where each file
holds flows for a particular attack. We:

1. Glob every ``*.csv`` under ``data_dir``.
2. Read each file (optionally truncated for fast testing).
3. Standardise label values via :data:`LABEL_MAP` so that 30+ verbose attack
   names collapse into ten unified categories used in the paper.
4. Concatenate and emit a single ``(df, class_names)`` tuple suitable for
   the rest of the PrivFed-TCN pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical attack-class mapping for CIC-IoT-2023.
# Keys are lowercased / stripped raw labels; values are unified categories.
# Anything not in the map is *kept* under its raw (lowercased) name so we
# can audit it later.
# ---------------------------------------------------------------------------
LABEL_MAP = {
    # ---------------- DDoS (includes DoS — merging eliminates confusion) --
    "ddos": "DDoS",
    "ddos-udpflood": "DDoS",
    "ddos-tcpflood": "DDoS",
    "ddos-icmpflood": "DDoS",
    "ddos-http": "DDoS",
    "ddos-slowloris": "DDoS",
    "ddos-synflood": "DDoS",
    "distributed denial of service": "DDoS",
    "ddos-rstfinflood": "DDoS",
    "ddos-pshackflood": "DDoS",
    "ddos-synonymousip_flood": "DDoS",
    "ddos-icmp_fragmentation": "DDoS",
    "ddos-udp_fragmentation": "DDoS",
    "ddos-ack_fragmentation": "DDoS",
    "ddos-http_flood": "DDoS",
    # DoS merged into DDoS — DoS had only 12.3% recall due to confusion
    # with DDoS subtypes (TCP_Flood, UDP_Flood, SYN_Flood).
    "dos": "DDoS",
    "dos-synflood": "DDoS",
    "dos-http": "DDoS",
    "dos-tcp_flood": "DDoS",
    "dos-udp_flood": "DDoS",
    "dos-syn_flood": "DDoS",
    "dos-http_flood": "DDoS",
    # ---------------- Reconnaissance -----------------------------------
    "scanning": "Reconnaissance",
    "recon": "Reconnaissance",
    "reconnaissance": "Reconnaissance",
    "portscan": "Reconnaissance",
    "port scan": "Reconnaissance",
    "os fingerprinting": "Reconnaissance",
    "vulnerability scan": "Reconnaissance",
    "recon-portscan": "Reconnaissance",
    "recon-osscan": "Reconnaissance",
    "recon-pingsweep": "Reconnaissance",
    "recon-hostdiscovery": "Reconnaissance",
    "vulnerabilityscan": "Reconnaissance",
    # ---------------- Botnet -------------------------------------------
    "botnet": "Botnet",
    "bot": "Botnet",
    "mirai": "Botnet",
    "mirai-greeth_flood": "Botnet",
    "mirai-greip_flood": "Botnet",
    "mirai-udpplain": "Botnet",
    # ---------------- Other_Attack (rare classes merged) ----------------
    # BruteForce (9 samples), Injection (11), XSS (2) are too rare to
    # learn under DP-SGD noise. Merging preserves their signal.
    "injection": "Other_Attack",
    "sql injection": "Other_Attack",
    "command injection": "Other_Attack",
    "sqlinjection": "Other_Attack",
    "commandinjection": "Other_Attack",
    "browserhijacking": "Other_Attack",
    "backdoor_malware": "Other_Attack",
    "backdoor": "Other_Attack",
    "uploading_attack": "Other_Attack",
    "xss": "Other_Attack",
    "cross-site scripting": "Other_Attack",
    "dictionarybruteforce": "Other_Attack",
    "brute_force": "Other_Attack",
    "bruteforce": "Other_Attack",
    # ---------------- IoT/MQTT specific --------------------------------
    "mqtt": "MQTT_Attack",
    "mqtt-publish-flood": "MQTT_Attack",
    "mqtt_publish_flood": "MQTT_Attack",
    # ---------------- Ransomware ---------------------------------------
    "ransomware": "Ransomware",
    # ---------------- MitM ---------------------------------------------
    "mitm": "MitM",
    "man-in-the-middle": "MitM",
    "dns spoofing": "MitM",
    "arp spoofing": "MitM",
    "mitm-arpspoofing": "MitM",
    "dns_spoofing": "MitM",
    # ---------------- Normal --------------------------------------------
    "normal": "Normal",
    "benign": "Normal",
    "begnin": "Normal",
    "benigntraffic": "Normal",
    "benign_traffic": "Normal",
}

# Columns we always exclude from the feature set (identifiers, timestamps,
# raw IP/MAC). Matching is case-insensitive and uses a substring rule.
_FEATURE_BLOCKLIST_SUBSTRINGS = (
    "label", "timestamp", "flow_id", "flow id",
    "src_ip", "src ip", "dst_ip", "dst ip",
    "src_mac", "src mac", "dst_mac", "dst mac",
    "source ip", "destination ip",
)

_LABEL_CANDIDATES = ("label", "Label", "attack_type", "Attack_type", "attack")


# ---------------------------------------------------------------------------
class CICIoTLoader:
    """Loads and unifies the CIC-IoT-2023 dataset from a directory of CSVs."""

    def __init__(self,
                 data_dir: str | Path,
                 limit_files: Optional[int] = None,
                 limit_rows_per_file: Optional[int] = None) -> None:
        self.data_dir = Path(data_dir)
        self.limit_files = limit_files
        self.limit_rows_per_file = limit_rows_per_file

    # ------------------------------------------------------------------
    def _discover_csvs(self) -> List[Path]:
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"CIC-IoT-2023 directory does not exist: {self.data_dir}")
        files = sorted(p for p in self.data_dir.rglob("*.csv") if p.is_file())
        if not files:
            raise FileNotFoundError(
                f"No CSV files found under {self.data_dir}")
        if self.limit_files is not None:
            files = files[: self.limit_files]
        return files

    # ------------------------------------------------------------------
    @staticmethod
    def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    @staticmethod
    def _find_label_column(df: pd.DataFrame) -> str:
        # Try the canonical names first (already lower-cased above).
        for c in (lc.lower() for lc in _LABEL_CANDIDATES):
            if c in df.columns:
                return c
        # Heuristic fallback: any column whose name contains "label".
        for c in df.columns:
            if "label" in c:
                return c
        raise KeyError(
            "Could not find a label column in CSV. Tried: "
            f"{_LABEL_CANDIDATES}; available: {list(df.columns)[:10]}...")

    # ------------------------------------------------------------------
    def load(self) -> Tuple[pd.DataFrame, List[str]]:
        """Load and concatenate all CSVs, returning ``(df, class_names)``."""
        files = self._discover_csvs()
        n_files = len(files)
        frames: List[pd.DataFrame] = []
        for i, path in enumerate(files, start=1):
            df = pd.read_csv(path, low_memory=False, nrows=self.limit_rows_per_file)
            print(f"Loading file {i}/{n_files}: {path.name} (rows: {len(df):,})")
            df = self._standardise_columns(df)
            try:
                lbl_col = self._find_label_column(df)
            except KeyError as e:
                print(f"  -> SKIP ({e})")
                continue
            if lbl_col != "label":
                df = df.rename(columns={lbl_col: "label"})
            df["label_raw"] = df["label"].astype(str)
            df = df.dropna(subset=["label"])
            frames.append(df)

        if not frames:
            raise RuntimeError("No usable CSV files (no label column found anywhere).")

        big = pd.concat(frames, ignore_index=True)
        # Map raw labels → unified categories (keep raw if unknown).
        raw_lower = big["label_raw"].str.strip().str.lower()
        # First pass: direct lookup in LABEL_MAP.
        big["label"] = raw_lower.map(LABEL_MAP)
        # Second pass: try again with underscores stripped (catches
        # "ddos-icmp_flood" → "ddos-icmpflood" etc.)
        unmapped = big["label"].isna()
        if unmapped.any():
            no_under = raw_lower[unmapped].str.replace("_", "", regex=False)
            big.loc[unmapped, "label"] = no_under.map(LABEL_MAP)
        # Third pass: prefix-based fallback for any remaining ddos-*/dos-*
        still_unmapped = big["label"].isna()
        if still_unmapped.any():
            raw_still = raw_lower[still_unmapped]
            big.loc[still_unmapped & raw_still.str.startswith("ddos"), "label"] = "DDoS"
            big.loc[still_unmapped & raw_still.str.startswith("dos"), "label"] = "DDoS"
            big.loc[still_unmapped & raw_still.str.startswith("mirai"), "label"] = "Botnet"
            big.loc[still_unmapped & raw_still.str.startswith("recon"), "label"] = "Reconnaissance"
        # Anything still unmapped keeps its raw label for auditing.
        still_na = big["label"].isna()
        if still_na.any():
            big.loc[still_na, "label"] = big.loc[still_na, "label_raw"]

        counts = big["label"].value_counts()
        print("\nLabel distribution after mapping:")
        for name, cnt in counts.items():
            print(f"  {name:<22s} {cnt:>12,}")

        class_names = sorted(big["label"].unique().tolist())
        return big, class_names

    # ------------------------------------------------------------------
    def get_feature_columns(self, df: pd.DataFrame, target: int = 41) -> List[str]:
        """Return up to ``target`` numeric feature columns (variance-ranked)."""
        cols: List[str] = []
        for c in df.columns:
            cl = c.lower()
            if any(b in cl for b in _FEATURE_BLOCKLIST_SUBSTRINGS):
                continue
            if c == "label_raw":
                continue
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            if df[c].isna().all():
                continue
            cols.append(c)
        if not cols:
            raise RuntimeError(
                "No numeric feature columns found after filtering. Check the CSV schema.")
        if len(cols) > target:
            # Rank by variance and keep top-target.
            variances = df[cols].var(numeric_only=True).fillna(0.0)
            cols = variances.sort_values(ascending=False).head(target).index.tolist()
        return cols


# ---------------------------------------------------------------------------
# Self-test (synthetic) so this file is runnable standalone.
# ---------------------------------------------------------------------------
def _synthetic_test() -> None:
    """Generate two tiny CSVs in a temp dir and run the loader on them."""
    import tempfile
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for name, lbl in [("file_a.csv", "DDoS-UDPFlood"), ("file_b.csv", "BenignTraffic")]:
            df = pd.DataFrame(rng.normal(size=(200, 5)),
                              columns=[f"feat_{i}" for i in range(5)])
            df["timestamp"] = np.arange(200)
            df["src_ip"] = "10.0.0.1"
            df["Label"] = lbl
            df.to_csv(tmp_path / name, index=False)

        loader = CICIoTLoader(tmp_path, limit_files=None, limit_rows_per_file=None)
        df, names = loader.load()
        feats = loader.get_feature_columns(df, target=41)
        assert "DDoS" in names, names
        assert "Normal" in names, names
        assert all("ip" not in f and "timestamp" not in f for f in feats), feats
        print(f"\nSelf-test OK: {df.shape}, classes={names}, "
              f"#features={len(feats)}")


if __name__ == "__main__":
    _synthetic_test()
