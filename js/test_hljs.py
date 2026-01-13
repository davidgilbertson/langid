import json
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from data.utils import get_stack_data
from tools import stopwatch


class HljsWorker:
    def __init__(self) -> None:
        self._script_path = Path("hljs_worker.js")  # Root relative
        self._proc: subprocess.Popen[str] | None = None

    def __enter__(self) -> "HljsWorker":
        self.start()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            ["node", str(self._script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        finally:
            self._proc = None

    def predict(self, snippet: str) -> str | None:
        if self._proc is None or self._proc.poll() is not None:
            self.start()
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("HLJS worker failed to start.")

        payload = {"snippet": snippet}
        self._proc.stdin.write(json.dumps(payload))
        self._proc.stdin.write("\n")
        self._proc.stdin.flush()

        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("HLJS worker returned no output.")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response.get("predicted")


def get_hljs_predictions(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    with HljsWorker() as worker:
        for row in df.to_dict("records"):
            snippet = row["Snippet"]
            results.append(
                dict(
                    Snippet=snippet,
                    Language=row.get("Language"),
                    Prediction=worker.predict(snippet),
                )
            )
    return pd.DataFrame(results)


if __name__ == "__main__":
    df = get_stack_data(snippet_limit=10, subset=0.01)

    with stopwatch("Predicting HLJS predictions"):
        results_df = get_hljs_predictions(df)

    f1 = f1_score(
        results_df.Language, results_df.Prediction.fillna(""), average="macro"
    )
    print(f"F1 (macro): {f1:.1%}")
