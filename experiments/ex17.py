import gzip
import json
import struct

import brotlicffi
import numpy as np
import pandas as pd
import zstandard as zstd
from sklearn.metrics import f1_score

from tools import (
    model_to_dict,
    round_model,
    smart_round,
    shrink_model,
)
from train_model import train_model

# Binary formats are little-endian:
# - bin_f32: uint16 n_classes, uint16 n_features, then float32 coef (row-major),
#   then float32 intercept.
# - bin_f16: same header, then float16 arrays.
# - bin_int8_sym: uint16 n_classes, uint16 n_features, float32 scale,
#   then int8 data for [coef.flatten(), intercept].


def gzip_size(data: bytes) -> int:
    return len(gzip.compress(data))


def brotli_size(data: bytes) -> int:
    return len(brotlicffi.compress(data))


_ZSTD = zstd.ZstdCompressor()


def zstd_size(data: bytes) -> int:
    return len(_ZSTD.compress(data))


def json_payload(
    coef: np.ndarray,
    intercept: np.ndarray,
    flat: bool,
    decimals: int = 1,
) -> bytes:
    coef_json = smart_round(coef, decimals)
    intercept_json = smart_round(intercept, decimals)
    if flat:
        payload = {
            "coef": [value for row in coef_json for value in row],
            "coef_shape": [coef.shape[0], coef.shape[1]],
            "intercept": intercept_json,
        }
    else:
        payload = {"coef": coef_json, "intercept": intercept_json}
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def tsv_payload(
    coef: np.ndarray,
    intercept: np.ndarray,
    decimals: int = 1,
) -> bytes:
    coef_rows = smart_round(coef, decimals)
    intercept_row = smart_round(intercept, decimals)
    lines = ["\t".join(str(value) for value in intercept_row)]
    lines.extend("\t".join(str(value) for value in row) for row in coef_rows)
    return "\n".join(lines).encode("utf-8")


def tsv_payload_int10(
    coef: np.ndarray,
    intercept: np.ndarray,
    decimals: int = 1,
) -> bytes:
    coef_rows = smart_round(coef, decimals)
    intercept_row = smart_round(intercept, decimals)
    lines = ["\t".join(str(int(round(value * 10))) for value in intercept_row)]
    lines.extend(
        "\t".join(str(int(round(value * 10))) for value in row) for row in coef_rows
    )
    return "\n".join(lines).encode("utf-8")


def pack_header(n_classes: int, n_features: int) -> bytes:
    return struct.pack("<HH", n_classes, n_features)


def pack_float32(coef: np.ndarray, intercept: np.ndarray) -> bytes:
    header = pack_header(coef.shape[0], coef.shape[1])
    return (
        header
        + coef.astype(np.float32).tobytes()
        + intercept.astype(np.float32).tobytes()
    )


def pack_float16(coef: np.ndarray, intercept: np.ndarray) -> bytes:
    header = pack_header(coef.shape[0], coef.shape[1])
    return (
        header
        + coef.astype(np.float16).tobytes()
        + intercept.astype(np.float16).tobytes()
    )


def pack_int8_symmetric(coef: np.ndarray, intercept: np.ndarray) -> bytes:
    n_classes, n_features = coef.shape
    values = np.concatenate([coef.reshape(-1), intercept]).astype(np.float32)
    max_abs = float(np.max(np.abs(values)))
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = np.round(values / scale).astype(np.int8)
    header = struct.pack("<HHf", n_classes, n_features, scale)
    return header + q.tobytes()


def pack_int8_fixed_scale(
    coef: np.ndarray,
    intercept: np.ndarray,
    scale: float = 0.1,
) -> tuple[bytes, bool]:
    n_classes, n_features = coef.shape
    values = np.concatenate([coef.reshape(-1), intercept]).astype(np.float32)
    q = np.round(values / scale).astype(np.int32)
    clipped = np.any(q < -128) or np.any(q > 127)
    q = np.clip(q, -128, 127).astype(np.int8)
    header = struct.pack("<HHf", n_classes, n_features, float(scale))
    return header + q.tobytes(), clipped


def decode_json(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(payload.decode("utf-8"))
    if "coef_shape" in data:
        coef = np.array(data["coef"], dtype=np.float32).reshape(data["coef_shape"])
    else:
        coef = np.array(data["coef"], dtype=np.float32)
    intercept = np.array(data["intercept"], dtype=np.float32)
    return coef, intercept


def decode_f32(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    n_classes, n_features = struct.unpack("<HH", payload[:4])
    offset = 4
    coef_size = n_classes * n_features
    coef = np.frombuffer(payload, dtype=np.float32, count=coef_size, offset=offset)
    coef = coef.reshape((n_classes, n_features))
    offset += coef_size * 4
    intercept = np.frombuffer(payload, dtype=np.float32, offset=offset)
    return coef, intercept


def decode_f16(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    n_classes, n_features = struct.unpack("<HH", payload[:4])
    offset = 4
    coef_size = n_classes * n_features
    coef = np.frombuffer(payload, dtype=np.float16, count=coef_size, offset=offset)
    coef = coef.reshape((n_classes, n_features)).astype(np.float32)
    offset += coef_size * 2
    intercept = np.frombuffer(payload, dtype=np.float16, offset=offset).astype(
        np.float32
    )
    return coef, intercept


def decode_int8(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    n_classes, n_features, scale = struct.unpack("<HHf", payload[:8])
    data = np.frombuffer(payload, dtype=np.int8, offset=8).astype(np.float32)
    coef_size = n_classes * n_features
    coef = (data[:coef_size] * scale).reshape((n_classes, n_features))
    intercept = data[coef_size:] * scale
    return coef, intercept


def decode_tsv(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    text = payload.decode("utf-8")
    rows = [row.split("\t") for row in text.splitlines() if row]
    intercept = np.array([float(value) for value in rows[0]], dtype=np.float32)
    coef_rows = [[float(value) for value in row] for row in rows[1:]]
    coef = np.array(coef_rows, dtype=np.float32)
    return coef, intercept


def decode_tsv_int10(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    text = payload.decode("utf-8")
    rows = [row.split("\t") for row in text.splitlines() if row]
    intercept = np.array([int(value) for value in rows[0]], dtype=np.float32) / 10.0
    coef_rows = [[int(value) for value in row] for row in rows[1:]]
    coef = np.array(coef_rows, dtype=np.float32) / 10.0
    return coef, intercept


def score_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    coef: np.ndarray | None = None,
    intercept: np.ndarray | None = None,
) -> float:
    coef = coef if coef is not None else model.coef_
    intercept = intercept if intercept is not None else model.intercept_
    scores = X @ coef.T + intercept
    preds = np.take(model.classes_, np.argmax(scores, axis=1))
    return f1_score(y, preds, average="macro")


def format_rows(rows: list[tuple[str, int, int, float]]) -> str:
    name_width = max(len(row[0]) for row in rows)
    lines = [f"{'Format'.ljust(name_width)}  Raw (KB)  Gzip (KB)    F1"]
    for name, raw, gz, f1 in rows:
        raw_kb = raw / 1024
        gz_kb = gz / 1024
        lines.append(
            f"{name.ljust(name_width)}  {raw_kb:8.2f}  {gz_kb:8.2f}  {f1:5.2%}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    df = pd.read_parquet("E:/Datasets/the_stack_10_line_snippets.parquet")
    results = train_model(df)
    model = round_model(results.model)
    X = results.X_val
    y = results.y_val

    # Reduce features
    n = 400
    model = shrink_model(model, n)
    X = X.iloc[:, :n]

    coef = model.coef_.copy()
    intercept = model.intercept_.copy()
    n_classes, n_features = coef.shape

    baseline_f1 = score_model(model, X, y, coef=coef, intercept=intercept)
    rows: list[tuple[str, int, int, float]] = []

    model_dict = model_to_dict(model)
    payload = json.dumps(model_dict, separators=(",", ":")).encode("utf-8")

    decoded_coef, decoded_intercept = decode_json(payload)
    f1 = score_model(model, X, y, coef=decoded_coef, intercept=decoded_intercept)
    rows.append(("json", len(payload), gzip_size(payload), f1))
    rows.append(("json_br", len(payload), brotli_size(payload), f1))
    rows.append(("json_zst", len(payload), zstd_size(payload), f1))

    meta_payload = json.dumps(
        {
            "features": model_dict["features"],
            "classes": model_dict["classes"],
            "coef_shape": [n_classes, n_features],
            "intercept_shape": [n_classes],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    param_payload = pack_int8_symmetric(coef, intercept)
    decoded_coef, decoded_intercept = decode_int8(param_payload)
    f1 = score_model(model, X, y, coef=decoded_coef, intercept=decoded_intercept)
    rows.append(
        (
            "bin_int8_sym",
            len(meta_payload) + len(param_payload),
            gzip_size(meta_payload) + gzip_size(param_payload),
            f1,
        )
    )
    rows.append(
        (
            "bin_int8_sym_br",
            len(meta_payload) + len(param_payload),
            brotli_size(meta_payload) + brotli_size(param_payload),
            f1,
        )
    )
    rows.append(
        (
            "bin_int8_sym_zst",
            len(meta_payload) + len(param_payload),
            zstd_size(meta_payload) + zstd_size(param_payload),
            f1,
        )
    )

    # payload = json_payload(coef, intercept, flat=False)
    # decoded_coef, decoded_intercept = decode_json(payload)
    # f1 = score_model(model, X, y, coef=decoded_coef, intercept=decoded_intercept)
    # rows.append(("json_nested_f32", len(payload), gzip_size(payload), f1))
    #
    # payload = json_payload(coef, intercept, flat=True)
    # decoded_coef, decoded_intercept = decode_json(payload)
    # f1 = score_model(model, X, y, coef=decoded_coef, intercept=decoded_intercept)
    # rows.append(("json_flat_f32", len(payload), gzip_size(payload), f1))
    #
    # payload = pack_float32(coef, intercept)
    # decoded_coef, decoded_intercept = decode_f32(payload)
    # f1 = score_model(model, X, y, coef=decoded_coef, intercept=decoded_intercept)
    # rows.append(("bin_f32", len(payload), gzip_size(payload), f1))
    #
    # payload = pack_float16(coef, intercept)
    # decoded_coef, decoded_intercept = decode_f16(payload)
    # f1 = score_model(model, X, y, coef=decoded_coef, intercept=decoded_intercept)
    # rows.append(("bin_f16", len(payload), gzip_size(payload), f1))
    #
    # payload, clipped = pack_int8_fixed_scale(coef, intercept, scale=0.1)
    # decoded_coef, decoded_intercept = decode_int8(payload)
    # f1 = score_model(model, X, y, coef=decoded_coef, intercept=decoded_intercept)
    # rows.append(("bin_int8_0p1", len(payload), gzip_size(payload), f1))
    #
    # payload = tsv_payload(coef, intercept)
    # decoded_coef, decoded_intercept = decode_tsv(payload)
    # f1 = score_model(model, X, y, coef=decoded_coef, intercept=decoded_intercept)
    # rows.append(("tsv_f32_1dp", len(payload), gzip_size(payload), f1))
    #
    # payload = tsv_payload_int10(coef, intercept)
    # decoded_coef, decoded_intercept = decode_tsv_int10(payload)
    # f1 = score_model(model, X, y, coef=decoded_coef, intercept=decoded_intercept)
    # rows.append(("tsv_int10", len(payload), gzip_size(payload), f1))

    print(f"Classes: {n_classes}")
    print(f"Features: {n_features}")
    print(f"Total params: {coef.size + intercept.size}")
    print(f"Baseline F1: {baseline_f1:.2%}")
    print()
    print(format_rows(rows))
