from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Awaitable

import time  # ✅ ADD THIS

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf


DETAIL_COLS = [
    "Header_Length", "Protocol Type", "Duration", "Rate",
    "TCP", "UDP", "ARP", "ICMP", "HTTP", "HTTPS", "DNS",
    "syn_flag_number", "ack_flag_number", "rst_flag_number", "fin_flag_number",
]


def normalize_label(s: str) -> str:
    s = str(s)

    while s.lower().endswith(".csv"):
        s = s[:-4]

    if s.lower().endswith("_train"):
        s = s[:-6]
    if s.lower().endswith("_test"):
        s = s[:-5]

    if s.lower().startswith("benign"):
        return "Benign"

    return s


class IDSEngine:
    """
    OLD simulation behavior:
    - event.is_alert = (pred_label != Benign) and confidence >= alert_threshold
    - streams events using sliding window on CSV traffic stream
    """

    def __init__(
        self,
        window_size: int = 20,
        stride: int = 1,
        alert_threshold: float = 0.50,
        sleep_sec: float = 0.0,
        loop: bool = True,
    ):
        self.window_size = window_size
        self.stride = stride
        self.alert_threshold = alert_threshold
        self.sleep_sec = sleep_sec
        self.loop = loop
        self._stop = False

        self.root = Path(__file__).resolve().parents[1]

        # Paths (match your old script)
        self.model_path = Path(r"D:\Grad\model_output\cnn_lstm_final.keras")
        self.scaler_path = Path(r"D:\Grad\derived_dataset\scaler.joblib")
        self.label_map_path = Path(r"D:\Grad\derived_dataset\label_mapping.csv")
        self.feature_cols_path = Path(r"D:\Grad\derived_dataset\feature_columns.txt")
        self.live_traffic_path = Path(r"D:\Grad\deployment\traffic_stream_labeled_small_unmixed.csv")

        self.model: Optional[tf.keras.Model] = None
        self.scaler = None
        self.labels: Optional[List[str]] = None
        self.feature_cols: Optional[List[str]] = None

    def stop(self) -> None:
        self._stop = True

    def reset(self) -> None:
        self._stop = False

    def _load_assets_sync(self) -> tuple[tf.keras.Model, Any, List[str], List[str]]:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model: {self.model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler: {self.scaler_path}")
        if not self.label_map_path.exists():
            raise FileNotFoundError(f"Missing label map: {self.label_map_path}")
        if not self.feature_cols_path.exists():
            raise FileNotFoundError(f"Missing feature columns: {self.feature_cols_path}")
        if not self.live_traffic_path.exists():
            raise FileNotFoundError(f"Missing live traffic: {self.live_traffic_path}")

        model = tf.keras.models.load_model(self.model_path)
        scaler = joblib.load(self.scaler_path)

        label_map = pd.read_csv(self.label_map_path)
        labels = label_map.sort_values("Encoded")["Label"].tolist()

        with open(self.feature_cols_path, "r", encoding="utf-8") as f:
            feature_cols = [line.strip() for line in f if line.strip()]

        return model, scaler, labels, feature_cols

    async def load_assets(self, publish_event: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        await publish_event({"type": "status", "message": "⏳ Loading model + scaler..."})
        self.model, self.scaler, self.labels, self.feature_cols = await asyncio.to_thread(self._load_assets_sync)
        await publish_event({"type": "status", "message": "✅ Model + scaler loaded. Streaming events..."})

    async def run(self, publish_event: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        try:
            if self.model is None:
                await self.load_assets(publish_event)

            assert self.model is not None
            assert self.scaler is not None
            assert self.labels is not None
            assert self.feature_cols is not None

            while not self._stop:
                df = pd.read_csv(self.live_traffic_path)

                missing = [c for c in self.feature_cols if c not in df.columns]
                if missing:
                    raise ValueError(f"Missing required features: {missing[:10]} ... total={len(missing)}")

                X = df[self.feature_cols]
                X_scaled = self.scaler.transform(X)

                available_detail_cols = [c for c in DETAIL_COLS if c in df.columns]
                buffer: List[np.ndarray] = []

                await publish_event({"type": "status", "message": "🚦 IDS simulation started"})

                for i in range(0, len(X_scaled), self.stride):
                    if self._stop:
                        break

                    buffer.append(X_scaled[i])
                    if len(buffer) < self.window_size:
                        continue
                    if len(buffer) > self.window_size:
                        buffer.pop(0)

                    window = np.array(buffer, dtype=np.float32).reshape(1, self.window_size, -1)

                    # ✅ 1) inference timing (real)
                    t0 = time.perf_counter()
                    probs = await asyncio.to_thread(lambda: self.model.predict(window, verbose=0)[0])
                    infer_ms = (time.perf_counter() - t0) * 1000.0

                    pred_id = int(np.argmax(probs))
                    pred_label_raw = self.labels[pred_id]
                    pred_label = normalize_label(pred_label_raw)
                    confidence = float(np.max(probs))

                    is_alert = (pred_label != "Benign") and (confidence >= self.alert_threshold)

                    row = df.iloc[i]
                    details = {c: row[c] for c in available_detail_cols}

                    # ✅ 2) detect timestamp (right after decision is made)
                    detect_ts_ms = int(time.time() * 1000)

                    await publish_event({
                        "type": "event",
                        "flow": int(i),
                        "is_alert": bool(is_alert),
                        "pred_label": pred_label,
                        "confidence": confidence,
                        "details": details,

                        # ✅ NEW fields
                        "detect_ts_ms": detect_ts_ms,
                        "infer_ms": infer_ms,
                    })

                    if self.sleep_sec > 0:
                        await asyncio.sleep(self.sleep_sec)

                if self._stop:
                    break

                await publish_event({"type": "summary", "message": "✅ Simulation finished (restarting replay)"})

                if not self.loop:
                    break

                await asyncio.sleep(1.0)

            await publish_event({"type": "status", "message": "⛔ Simulation stopped"})

        except Exception as e:
            await publish_event({"type": "status", "message": f"❌ Simulation error: {e}"})
            raise
