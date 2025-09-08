# pip install requests pandas pyarrow tqdm
import os, json, time, random, math
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional
import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from tqdm import tqdm

DATA_DIR = Path("data")            # 数据输出目录
DATA_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = Path("ckpt")      # 断点目录
CHECKPOINT_DIR.mkdir(exist_ok=True)

BASE_URL = "https://store.steampowered.com/appreviews/{appid}"
PER_PAGE = 100                     # 上限 100
DEFAULT_LANG = "all"

def build_session(total_retries=5, backoff_factor=0.6) -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    sess.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/121.0.0.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://store.steampowered.com/",
        "Connection": "keep-alive",
        "Accept-Encoding": "gzip, deflate, br",
    })
    return sess

def load_ckpt(appid: int, suffix: str = "") -> dict:
    p = CHECKPOINT_DIR / f"{appid}{suffix}.json"
    if p.exists():
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            pass
    return {"cursor": "*", "pages_done": 0}

def save_ckpt(appid: int, cursor: str, pages_done: int, suffix: str = ""):
    p = CHECKPOINT_DIR / f"{appid}{suffix}.json"
    p.write_text(json.dumps({"cursor": cursor, "pages_done": pages_done, "ts": int(time.time())}, indent=2, ensure_ascii=False), "utf-8")

def reviews_to_df(reviews: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rv in reviews:
        a = rv.get("author", {}) or {}
        rows.append({
            "recommendationid": rv.get("recommendationid"),
            "language": rv.get("language"),
            "review": rv.get("review"),
            "voted_up": bool(rv.get("voted_up")),
            "votes_up": rv.get("votes_up", 0),
            "timestamp": pd.to_datetime(rv.get("timestamp_created", 0), unit="s"),
            "playtime_at_review_min": a.get("playtime_at_review", 0),
            "steam_purchase": a.get("steam_purchase", None),
            "received_for_free": a.get("received_for_free", None),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["playtime_hours"] = (df["playtime_at_review_min"].fillna(0) / 60).round(2)
        df["date"] = df["timestamp"].dt.date
    return df

def append_parquet(df: pd.DataFrame, path: Path):
    """
    以数据集目录追加写入，使用“自增分片计数器”确保永不覆盖。
    - 输出目录： data/reviews_all_<appid>.dset/
    - 分片文件： part-000000.parquet, part-000001.parquet, ...
    - 原子写入：先写 .tmp 再 rename
    """
    if df.empty:
        return

    import pyarrow as pa
    import pyarrow.parquet as pq
    import uuid, json

    dset_dir = Path(str(path).replace(".parquet", ".dset"))
    dset_dir.mkdir(parents=True, exist_ok=True)

    # 将计数器放到 ckpt，同一个 appid 持久维护
    # e.g. ckpt/1030300.parts.json
    appid = int("".join([c for c in dset_dir.name if c.isdigit()]) or "0")
    parts_ckpt = Path("ckpt") / f"{appid}.parts.json"
    if parts_ckpt.exists():
        try:
            counter = json.loads(parts_ckpt.read_text("utf-8")).get("next_idx", 0)
        except Exception:
            counter = 0
    else:
        # 如果目录中已有分片（比如之前就写过），从现有数量继续
        existing = sorted(dset_dir.glob("part-*.parquet"))
        counter = len(existing)

    # 生成下一个稳定唯一文件名
    out_file = dset_dir / f"part-{counter:06d}.parquet"
    tmp_file = dset_dir / f".part-{counter:06d}.{uuid.uuid4().hex}.tmp"

    # 写 tmp 再原子重命名，确保不会半截文件
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp_file)
    tmp_file.replace(out_file)

    # 计数器 +1
    parts_ckpt.write_text(json.dumps({"next_idx": counter + 1}, ensure_ascii=False), "utf-8")



def fetch_segment(
    appid: int,
    max_pages: int,
    language: str = DEFAULT_LANG,
    purchase_type: str = "all",
    filter_val: str = "all",
    delay_base: float = 0.7,
    delay_jitter: float = 0.5,
    long_sleep_after_fail: float = 30.0,
    suffix: str = "",
) -> Iterable[List[Dict[str, Any]]]:
    """
    逐页生成器：返回 reviews 列表（每页一次），带断点续抓、应用层退避。
    """
    sess = build_session()
    url = BASE_URL.format(appid=appid)
    ck = load_ckpt(appid, suffix)
    cursor = ck.get("cursor", "*")
    pages_done = ck.get("pages_done", 0)
    consec_fail = 0

    for page_idx in range(pages_done, max_pages):
        params = {
            "json": 1,
            "language": language,
            "purchase_type": purchase_type,
            "filter": filter_val,
            "num_per_page": PER_PAGE,
            "cursor": cursor,
        }

        for attempt in range(6):
            try:
                resp = sess.get(url, params=params, timeout=(8, 20))
                resp.raise_for_status()
                data = resp.json()
                reviews = data.get("reviews", [])
                if not reviews:
                    save_ckpt(appid, cursor, page_idx, suffix)
                    return
                cursor = data.get("cursor", cursor)
                consec_fail = 0
                save_ckpt(appid, cursor, page_idx + 1, suffix)

                yield reviews
                time.sleep(delay_base + random.random() * delay_jitter)
                break
            except (requests.exceptions.SSLError,
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectTimeout):
                time.sleep(min(8.0, 0.8 * (2 ** attempt)) + random.random() * 0.6)
                if attempt == 5:
                    consec_fail += 1
                    time.sleep(long_sleep_after_fail)
            except requests.RequestException:
                time.sleep(1.0 + random.random())
                if attempt == 5:
                    consec_fail += 1
                    time.sleep(long_sleep_after_fail)

        if consec_fail >= 4:
            save_ckpt(appid, cursor, page_idx, suffix)
            return

def run_fetch(
    appids: List[int],
    max_pages_per_app: int = 1200,
    out_prefix: str = "reviews",
    segment_hint: Optional[str] = None,
):
    """
    appids 可一次传多个；每个 APP 抓满 max_pages_per_app。
    segment_hint 仅用于区分不同过滤配置（如 recent 等），会反映在输出文件名与 ckpt 后缀。
    """
    suf = f"_{segment_hint}" if segment_hint else ""
    for appid in appids:
        out_path = DATA_DIR / f"{out_prefix}_{appid}{suf}.parquet"
        pbar = tqdm(total=max_pages_per_app, desc=f"APP {appid}")
        got_pages = 0
        for page_reviews in fetch_segment(appid, max_pages=max_pages_per_app, suffix=suf):
            df = reviews_to_df(page_reviews)
            append_parquet(df, out_path)
            # 在抓取循环里打印每个分片路径，便于确认
            print(f"[APP {appid}] write {len(df):4d} rows -> {str(out_path).replace('.parquet', '.dset')}")
            got_pages += 1
            pbar.update(1)
        pbar.close()
        print(f"[APP {appid}] wrote to: {out_path} | pages: {got_pages}")

if __name__ == "__main__":
    # 同时抓（可换成你自己的），在此处输入对应steam游戏的appid
    APPIDS = [1030300]
    run_fetch(APPIDS, max_pages_per_app=600, out_prefix="reviews_all")  # 先小跑 600 页/APP
    # 想抓“最近”窗口可以再跑一遍：
    # run_fetch(APPIDS, max_pages_per_app=300, out_prefix="reviews_recent", segment_hint="recent")
