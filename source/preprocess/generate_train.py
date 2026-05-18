"""
generate_train.py
Tạo dữ liệu huấn luyện từ các file trong folder concat.

Split:
  - 10% file  →  dedicated test  (1 easy_positive + 1 hard_positive / segment)
  - 90% file  →  8 train + 1 val + 1 test queries / segment

Output giống cấu trúc concat:
  source/data/train/K01_V001.json       → [{id, query}, ...]
  source/data/val/K01_V001.json         → [{id, query}, ...]
  source/data/test/K01_V001.json        → [{id, query}, ...]
  source/data/test_dedicated/K01_V001.json → [{id, easy_positive, hard_positive}, ...]
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from urllib import error, request as urllib_request

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "gemma-4-26b-a4b-it"
GEMMA_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

BASE_DIR = Path(r"D:\aic")
CONCAT_DIR = BASE_DIR / "dataset" / "concat"
OUTPUT_BASE = BASE_DIR / "data"

SEED = 42
DEDICATED_TEST_RATIO = 0.10
TRAIN_PER_SEG = 8
VAL_PER_SEG = 1
TEST_PER_SEG = 1
TOTAL_QUERIES = TRAIN_PER_SEG + VAL_PER_SEG + TEST_PER_SEG  # 10
MAX_TEXT_CHARS = 1500   # giới hạn mỗi segment trong batch (4 × 1500 = 6000 chars input)
BATCH_SIZE = 4          # số segment xử lý trong 1 request

SYSTEM_PROMPT = (
    "Bạn là chuyên gia tạo dữ liệu huấn luyện cho mô hình truy xuất thông tin bản tin thời sự. "
    "TUYỆT ĐỐI không đề cập đến: trường quay, đài truyền hình, kênh truyền hình, "
    "biên tập viên, phát thanh viên, người dẫn chương trình, logo, đồ họa chương trình, "
    "phát sóng, bản tin truyền hình. "
    "Chỉ tạo câu hỏi về NỘI DUNG TIN TỨC: sự kiện thực tế, địa điểm cụ thể, "
    "nhân vật (chính khách, nạn nhân, cơ quan), con số, thời gian xảy ra sự kiện."
)

API_KEYS = [
    # # angoc1
    # "AIzaSyD2P8zQESf6fIm33VccqGzMqHmkRbsny_g",
    # "AIzaSyDjTDWmM-mJsTiewc5_55zg-oh6zos5WDU",
    # "AIzaSyCeJj53gsgqF6J0fj219rmR5YoP4pzTRts",
    # "AIzaSyC0WupVzrWdp0V5XASite1BIpt3H0noBig",
    # "AIzaSyBa78cT8fFoe23_dtoWWnA1989RLu7l3Hg",
    # # angoc2
    # "AIzaSyDQkSZ5scPQVYocgiV5vHujo5KgW4ZVyVM",
    # "AIzaSyDTyB7vEv7CHn1NQZKU38nTDBhhoHjjjLA",
    # "AIzaSyBcntaozpkN_HlazZsZ6uVSUZ1IjQOce0E",
    # "AIzaSyApI5pmnsqxQM4dCHqqzlh4wFEFxeOi3aM",
    # "AIzaSyA1wRumYa8eTN2fFdPBKccOdEgeF60sp0k",
    # # angoc3
    # "AIzaSyBQ7ShiXEbw4igWkb2LCszgLyVHBCE_cA8",
    # "AIzaSyCyg04vEpSq8-27eAP4xfp0o-Klo7St8Eg",
    # "AIzaSyCdaAPr8eczxyvtXh36ZY44yml1jL1WgeI",
    # "AIzaSyBPt-das-rLPd2plxYxUlOv8Jt70rpCEjg",
    # "AIzaSyAkjQAECdFWgv0qiyJKwRkjv9edg4ihoig",
    # vbui1
    "AIzaSyCNuw2RQhY1G3PvZffZUaX3K72zqfXErcQ",
    "AIzaSyCsJkH6ceCNHO_YNiw-E2g2A2dhC8bxoxM",
    "AIzaSyC7ARnHzQX6gzxgQM9P4Lm6edptEgJqEig",
    "AIzaSyDP7Ue69xCg2sL-CtX6EPdUWr1vX91B170",
    "AIzaSyAP2CmKevO29RSwK9-RmXU-zRjsMKn0swk",
    # vbui2
    "AIzaSyBaaZSEkSo6Hgt_qFUSomOep8GcCM01INk",
    "AIzaSyC0HR9_n48ZZHf2HCSf2yWCyKCAMbCy5b4",
    "AIzaSyDETSSd-SqJzPpmU-IUsXP-_u-enrk7JOg",
    "AIzaSyC-jKDPazrqnit_vTdpWvUNHbvGhAE_KCk",
    "AIzaSyBSnhEKrqrxucUHFnFkaM0ClrObu0NUr2I",
    # vbui3
    "AIzaSyCI8ovOwS6ib8qgZ7KQXzESGU9MzvSoYcY",
    "AIzaSyD435IkcFF4c4ZAcC0XXfphBeaD_clgap8",
    "AIzaSyBT5Y4slOMbyLPCo1yY3R2DwzP8_khq-OQ",
    "AIzaSyCAOgyE8OdjiiIE7PyneYa93VC6YByAGvA",
    "AIzaSyC0ADQDb8Fq9_q8hMt2CPvV1-4cNb_Jxug",
    
]

# ── Logging ───────────────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


# ── API call ──────────────────────────────────────────────────────────────────
_DAILY_QUOTA_SIGNALS = (
    "per day", "per_day", "daily", "free tier", "billing", "GenerateRequestsPerDay"
)


def _call_api(api_key: str, user_prompt: str, max_tokens: int = 1024, temp: float = 0.7) -> str:
    key = api_key.strip()
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temp,
    }
    endpoint = f"{GEMMA_BASE_URL.rstrip('/')}/chat/completions"
    req = urllib_request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"No choices in response: {data}")

    raw = choices[0].get("message", {}).get("content", "")
    if isinstance(raw, list):
        raw = "\n".join(p.get("text", "") for p in raw if isinstance(p, dict))
    return re.sub(r"<thought>.*?</thought>", "", raw, flags=re.DOTALL).strip()


def _call_with_retry(
    api_key: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temp: float = 0.7,
    max_retries: int = 2,
) -> str:
    for attempt in range(max_retries + 1):
        try:
            result = _call_api(api_key, user_prompt, max_tokens, temp)
            time.sleep(20)  # sleep 20s sau mỗi request thành công
            return result
        except Exception as exc:
            msg = str(exc)
            is_rate = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()
            if is_rate:
                if any(s.lower() in msg.lower() for s in _DAILY_QUOTA_SIGNALS):
                    _log(f"  [QUOTA_DAILY] key ...{api_key[-4:]}: {msg[:80]}")
                    raise RuntimeError("quota_exhausted") from exc
                if attempt >= max_retries:
                    raise RuntimeError("quota_exhausted") from exc
                _log(f"  [RATE] key ...{api_key[-4:]}, sleep 60s (attempt {attempt + 1}/{max_retries})")
                time.sleep(60)
                continue
            if attempt >= max_retries:
                raise
            time.sleep(2.0 * (2 ** attempt))
    raise RuntimeError("Max retries exceeded")


# ── Prompts (batch) ───────────────────────────────────────────────────────────
def _trunc(text: str) -> str:
    return text[:MAX_TEXT_CHARS] + ("…" if len(text) > MAX_TEXT_CHARS else "")


def _prompt_regular_batch(texts: list[str]) -> str:
    n = len(texts)
    segments = "\n\n".join(
        f"[Đoạn {i + 1}]\n<content>\n{_trunc(t)}\n</content>"
        for i, t in enumerate(texts)
    )
    return (
        f"Dưới đây là {n} đoạn bản tin thời sự. "
        f"Tạo dữ liệu huấn luyện mô hình truy xuất video cho từng đoạn.\n\n"
        f"{segments}\n\n"
        f"Yêu cầu cho mỗi đoạn:\n"
        f"1. queries: {TOTAL_QUERIES} truy vấn tìm kiếm tiếng Việt — KHÔNG phải câu hỏi, "
        f"viết dạng mô tả sự kiện (10–20 từ), phản ánh ít nhất 1/3 nội dung đoạn "
        f"(kết hợp sự kiện + địa điểm + số liệu/thời gian/nhân vật), đa dạng góc độ.\n"
        f"   Đúng: \"lũ ống bất ngờ xã Lượng Minh huyện Tương Dương Nghệ An khiến bản làng ngập nặng đêm 30 tháng 9\"\n"
        f"   Sai: \"lũ ống Nghệ An\" / \"Trận lũ ống diễn ra như thế nào?\"\n"
        f"2. negatives: 4 truy vấn cùng cấu trúc và độ dài nhưng thay entity "
        f"(địa điểm/nhân vật/số liệu/sự kiện) bằng entity khác — KHÔNG khớp với đoạn đó.\n\n"
        f"Trả về JSON array với đúng {n} phần tử theo thứ tự Đoạn 1 → Đoạn {n}:\n"
        f'[{{"queries":["q1",...,"{TOTAL_QUERIES} items"],"negatives":["neg1","neg2","neg3","neg4"]}},'
        f'...{n} phần tử]'
    )


def _prompt_dedicated_batch(texts: list[str]) -> str:
    n = len(texts)
    segments = "\n\n".join(
        f"[Đoạn {i + 1}]\n<content>\n{_trunc(t)}\n</content>"
        for i, t in enumerate(texts)
    )
    return (
        f"Dưới đây là {n} đoạn bản tin thời sự. "
        f"Tạo dữ liệu đánh giá mô hình truy xuất cho từng đoạn.\n\n"
        f"{segments}\n\n"
        f"Yêu cầu cho mỗi đoạn (KHÔNG phải câu hỏi — mô tả sự kiện 10–20 từ, "
        f"phản ánh ít nhất 1/3 nội dung, không đề cập trường quay/đài truyền hình/biên tập viên/logo):\n"
        f"- easy_positive: truy vấn rõ ràng về thông tin nổi bật nhất, model dễ khớp đúng.\n"
        f"- hard_positive: truy vấn về chi tiết đặc trưng (số liệu chính xác, địa danh cụ thể, "
        f"diễn biến/hậu quả), dễ nhầm với bản tin tương tự.\n\n"
        f"Trả về JSON array với đúng {n} phần tử:\n"
        f'[{{"easy_positive":"...","hard_positive":"..."}},...{n} phần tử]'
    )


# ── Parsing (batch) ───────────────────────────────────────────────────────────
def _parse_regular_batch(text: str, n: int) -> list[tuple[list[str], list[str]]]:
    s, e = text.find("["), text.rfind("]") + 1
    if s < 0:
        raise ValueError(f"No JSON array in: {text[:300]}")
    items = json.loads(text[s:e])
    if len(items) < n:
        raise ValueError(f"Expected {n} items, got {len(items)}")
    results = []
    for i, obj in enumerate(items[:n]):
        queries = [str(q).strip() for q in obj.get("queries", []) if str(q).strip()]
        negatives = [str(ng).strip() for ng in obj.get("negatives", []) if str(ng).strip()]
        if len(queries) < TOTAL_QUERIES:
            raise ValueError(f"Item {i}: expected {TOTAL_QUERIES} queries, got {len(queries)}")
        if len(negatives) < 4:
            raise ValueError(f"Item {i}: expected 4 negatives, got {len(negatives)}")
        results.append((queries[:TOTAL_QUERIES], negatives[:4]))
    return results


def _parse_dedicated_batch(text: str, n: int) -> list[dict]:
    s, e = text.find("["), text.rfind("]") + 1
    if s < 0:
        raise ValueError(f"No JSON array in: {text[:300]}")
    items = json.loads(text[s:e])
    if len(items) < n:
        raise ValueError(f"Expected {n} items, got {len(items)}")
    results = []
    for i, obj in enumerate(items[:n]):
        if "easy_positive" not in obj or "hard_positive" not in obj:
            raise ValueError(f"Item {i}: missing keys in {obj}")
        results.append({
            "easy_positive": str(obj["easy_positive"]).strip(),
            "hard_positive": str(obj["hard_positive"]).strip(),
        })
    return results


# ── Task ──────────────────────────────────────────────────────────────────────
class QueryTask:
    __slots__ = ("concat_path", "seg_index", "seg_id", "text", "task_type")

    def __init__(
        self,
        concat_path: Path,
        seg_index: int,
        seg_id: str,
        text: str,
        task_type: str,  # "regular" | "dedicated"
    ):
        self.concat_path = concat_path
        self.seg_index = seg_index
        self.seg_id = seg_id
        self.text = text
        self.task_type = task_type


# ── File splitting ────────────────────────────────────────────────────────────
def _split_files(seed: int = SEED) -> tuple[list[Path], list[Path]]:
    files = sorted(CONCAT_DIR.glob("*.json"))
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    n_ded = max(1, int(len(shuffled) * DEDICATED_TEST_RATIO))
    return shuffled[n_ded:], shuffled[:n_ded]  # regular, dedicated


# ── Core ──────────────────────────────────────────────────────────────────────
def _run(api_keys: list[str], test_mode: bool = False) -> None:
    valid_keys = [k.strip() for k in api_keys if k.strip()]
    if not valid_keys:
        raise ValueError("No valid API keys")

    dirs = {
        "train": OUTPUT_BASE / "train",
        "val": OUTPUT_BASE / "val",
        "test": OUTPUT_BASE / "test",
        "test_dedicated": OUTPUT_BASE / "test_dedicated",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    _log(f"[INFO] Output dir: {OUTPUT_BASE}")

    regular, dedicated = _split_files()

    # Test mode: chỉ xử lý 1 file regular + 1 file dedicated
    if test_mode:
        regular = regular[:1]
        dedicated = dedicated[:1]
        _log("[INFO] *** TEST MODE: chỉ xử lý 1 regular + 1 dedicated file ***")

    # Save manifest
    manifest = {
        "seed": SEED,
        "dedicated_test_ratio": DEDICATED_TEST_RATIO,
        "queries_per_segment": {
            "train": TRAIN_PER_SEG,
            "val": VAL_PER_SEG,
            "test": TEST_PER_SEG,
        },
        "split": {
            "regular": sorted(f.name for f in regular),
            "dedicated_test": sorted(f.name for f in dedicated),
        },
    }
    (OUTPUT_BASE / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _log(f"[INFO] Files total   : {len(regular) + len(dedicated)}")
    _log(f"[INFO] Regular       : {len(regular)}  →  {TRAIN_PER_SEG}/{VAL_PER_SEG}/{TEST_PER_SEG} queries/seg")
    _log(f"[INFO] Dedicated test: {len(dedicated)}  →  easy+hard positive/seg")
    _log(f"[INFO] API keys      : {len(valid_keys)}\n")

    # Collect tasks — skip từng segment đã có, chỉ thêm task cho segment còn thiếu
    tasks: list[QueryTask] = []
    file_seg_count: dict[Path, int] = {}
    file_task_type: dict[Path, str] = {}
    results: dict[Path, dict[int, dict]] = {}

    def _load_done_regular(f: Path) -> dict[str, dict]:
        """Load các segment đã xong từ output files, trả về {seg_id -> result dict}."""
        train_p = dirs["train"] / f.name
        val_p   = dirs["val"]   / f.name
        test_p  = dirs["test"]  / f.name
        if not train_p.exists():
            return {}
        try:
            train_data = json.loads(train_p.read_text(encoding="utf-8"))
            val_data   = json.loads(val_p.read_text(encoding="utf-8")) if val_p.exists() else []
            test_data  = json.loads(test_p.read_text(encoding="utf-8")) if test_p.exists() else []
            val_by_id  = {e["id"]: e["positive"] for e in val_data}
            test_by_id = {e["id"]: e["positive"] for e in test_data}
            by_id: dict[str, dict] = {}
            for e in train_data:
                sid = e["id"]
                if sid not in by_id:
                    by_id[sid] = {"seg_id": sid, "queries": [], "negatives": e.get("negatives", [])}
                by_id[sid]["queries"].append(e["positive"])
            for sid, r in by_id.items():
                if sid in val_by_id:
                    r["queries"].append(val_by_id[sid])
                if sid in test_by_id:
                    r["queries"].append(test_by_id[sid])
            return by_id
        except Exception as exc:
            _log(f"  [WARN] load existing {f.name}: {exc}")
            return {}

    def _load_done_dedicated(f: Path) -> dict[str, dict]:
        """Load các segment đã xong từ test_dedicated file."""
        out = dirs["test_dedicated"] / f.name
        if not out.exists():
            return {}
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
            return {e["id"]: {"seg_id": e["id"],
                               "easy_positive": e["easy_positive"],
                               "hard_positive": e["hard_positive"]} for e in data}
        except Exception as exc:
            _log(f"  [WARN] load existing dedicated/{f.name}: {exc}")
            return {}

    for f in regular:
        segs = json.loads(f.read_text(encoding="utf-8"))
        if not segs:
            continue
        done = _load_done_regular(f)
        pending = [s for s in segs if s["id"] not in done]
        if not pending:
            _log(f"  skip  {f.name}  (all {len(segs)} segments done)")
            continue
        if done:
            _log(f"  resume {f.name}  ({len(done)}/{len(segs)} done, {len(pending)} remaining)")
        file_seg_count[f] = len(segs)
        file_task_type[f] = "regular"
        results[f] = {i: done[s["id"]] for i, s in enumerate(segs) if s["id"] in done}
        for i, seg in enumerate(segs):
            if seg["id"] not in done:
                tasks.append(QueryTask(f, i, seg["id"], seg["text"], "regular"))

    for f in dedicated:
        segs = json.loads(f.read_text(encoding="utf-8"))
        if not segs:
            continue
        done = _load_done_dedicated(f)
        pending = [s for s in segs if s["id"] not in done]
        if not pending:
            _log(f"  skip  dedicated/{f.name}  (all {len(segs)} segments done)")
            continue
        if done:
            _log(f"  resume dedicated/{f.name}  ({len(done)}/{len(segs)} done, {len(pending)} remaining)")
        file_seg_count[f] = len(segs)
        file_task_type[f] = "dedicated"
        results[f] = {i: done[s["id"]] for i, s in enumerate(segs) if s["id"] in done}
        for i, seg in enumerate(segs):
            if seg["id"] not in done:
                tasks.append(QueryTask(f, i, seg["id"], seg["text"], "dedicated"))

    if not tasks:
        _log("\n[INFO] All segments already done.")
        return

    _log(f"[INFO] Segments to process: {len(tasks)}\n")
    results_lock = Lock()  # bảo vệ cả results dict lẫn file write
    done_count = [0]
    fail_count = [0]
    global_lock = Lock()

    def _write_file(concat_path: Path) -> None:
        # Ghi ngay sau mỗi segment hoàn thành (ghi đè file với kết quả hiện có)
        # Phải được gọi bên trong results_lock
        file_res = results[concat_path]
        n_done = len(file_res)
        n_total = file_seg_count[concat_path]
        is_dedicated = file_task_type[concat_path] == "dedicated"

        if is_dedicated:
            entries = [
                {
                    "id": file_res[i]["seg_id"],
                    "easy_positive": file_res[i]["easy_positive"],
                    "hard_positive": file_res[i]["hard_positive"],
                }
                for i in sorted(file_res)
            ]
            out = dirs["test_dedicated"] / concat_path.name
            out.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            train_entries, val_entries, test_entries = [], [], []
            for i in sorted(file_res):
                r = file_res[i]
                sid = r["seg_id"]
                qs = r["queries"]
                negs = r["negatives"]
                for q in qs[:TRAIN_PER_SEG]:
                    train_entries.append({"id": sid, "positive": q, "negatives": negs})
                val_entries.append({"id": sid, "positive": qs[TRAIN_PER_SEG]})
                test_entries.append({"id": sid, "positive": qs[TRAIN_PER_SEG + VAL_PER_SEG]})

            for split, entries in [
                ("train", train_entries),
                ("val", val_entries),
                ("test", test_entries),
            ]:
                out = dirs[split] / concat_path.name
                out.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

        status = "COMPLETE" if n_done == n_total else f"partial {n_done}/{n_total}"
        _log(f"  [write-{status}] {concat_path.name}")

    queue: Queue[QueryTask] = Queue()
    for t in tasks:
        queue.put(t)

    def worker(api_key: str) -> None:
        while True:
            # Thu gom batch BATCH_SIZE task cùng loại
            try:
                first = queue.get_nowait()
            except Empty:
                break

            task_type = first.task_type
            batch: list[QueryTask] = [first]
            requeue: list[QueryTask] = []

            while len(batch) < BATCH_SIZE:
                try:
                    t = queue.get_nowait()
                    if t.task_type == task_type:
                        batch.append(t)
                    else:
                        requeue.append(t)
                except Empty:
                    break

            for t in requeue:
                queue.put(t)

            labels = ", ".join(f"{t.concat_path.name}[{t.seg_id}]" for t in batch)
            _log(f"  [batch×{len(batch)}] ...{api_key[-4:]} | {labels}")

            try:
                if task_type == "regular":
                    raw = _call_with_retry(
                        api_key,
                        _prompt_regular_batch([t.text for t in batch]),
                        max_tokens=len(batch) * 600,
                    )
                    parsed = _parse_regular_batch(raw, len(batch))
                    batch_results = [
                        {"seg_id": t.seg_id, "queries": q, "negatives": n}
                        for t, (q, n) in zip(batch, parsed)
                    ]
                else:
                    raw = _call_with_retry(
                        api_key,
                        _prompt_dedicated_batch([t.text for t in batch]),
                        max_tokens=len(batch) * 256,
                    )
                    parsed = _parse_dedicated_batch(raw, len(batch))
                    batch_results = [
                        {"seg_id": t.seg_id, **obj}
                        for t, obj in zip(batch, parsed)
                    ]

                with results_lock:
                    for t, result in zip(batch, batch_results):
                        results[t.concat_path][t.seg_index] = result
                        _write_file(t.concat_path)

                with global_lock:
                    done_count[0] += len(batch)
                _log(f"  [ok×{len(batch)}] {labels} ({done_count[0]}/{len(tasks)})")

            except RuntimeError as exc:
                if "quota_exhausted" in str(exc):
                    _log(f"  [QUOTA] ...{api_key[-4:]} exhausted, returning {len(batch)} tasks")
                    for t in batch:
                        queue.put(t)
                    break
                _log(f"  [FAIL] {labels}: {exc}")
                with global_lock:
                    fail_count[0] += len(batch)
            except Exception as exc:
                _log(f"  [FAIL] {labels}: {exc}")
                with global_lock:
                    fail_count[0] += len(batch)
            finally:
                for _ in batch:
                    queue.task_done()

    with ThreadPoolExecutor(max_workers=len(valid_keys)) as executor:
        futures = [executor.submit(worker, k) for k in valid_keys]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as exc:
                _log(f"[ERROR] Worker exception: {exc}")

    _log(f"\n[DONE] {done_count[0]}/{len(tasks)} ok | {fail_count[0]} failed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tạo dữ liệu huấn luyện từ concat folder")
    parser.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Override API keys (space-separated). Mặc định dùng API_KEYS trong file.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Chỉ xử lý 1 file regular + 1 file dedicated để kiểm tra.",
    )
    args = parser.parse_args()

    _run(args.keys if args.keys else API_KEYS, test_mode=args.test)
