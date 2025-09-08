# pip install streamlit pandas pyarrow plotly
import streamlit as st
import pandas as pd
from pathlib import Path


DATA_DIR = Path("data")

def load_data(paths: list[Path]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if not p.exists():
            continue
        if p.is_dir():
            dfs.append(pd.read_parquet(p))  # 读取数据集目录
        else:
            dfs.append(pd.read_parquet(p))  # 读取单文件
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True).drop_duplicates("recommendationid")


@st.cache_data
def load_all(prefix: str) -> pd.DataFrame:
    # 同时兼容 目录数据集(.dset) 和 单文件(.parquet)
    files = sorted(DATA_DIR.glob(f"{prefix}_*.parquet")) \
          + sorted(DATA_DIR.glob(f"{prefix}_*.dset"))
    return load_data(files)


def language_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("language").agg(
        total=("recommendationid","count"),
        pos=("voted_up","sum"),
        neg=("voted_up", lambda s: int((~s).sum())),
        avg_helpful=("votes_up","mean")
    ).sort_values("total", ascending=False)
    g["pos_rate"] = g["pos"] / g["total"]
    return g

def bin_playtime(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0,10,20,50,100,10**9]
    labels = ["0-10h","10-20h","20-50h","50-100h","100h+"]
    dff = df.copy()
    dff["playtime_bin"] = pd.cut(dff["playtime_hours"].fillna(0), bins=bins, labels=labels, right=False)
    g = dff.groupby("playtime_bin").agg(
        total=("recommendationid","count"),
        pos=("voted_up","sum"),
        neg=("voted_up", lambda s: int((~s).sum())),
        avg_helpful=("votes_up","mean")
    )
    g["pos_rate"] = g["pos"] / g["total"]
    return g.reindex(labels)

def trend(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    dff = df.set_index("timestamp").sort_index()
    by = dff.resample(freq).agg(total=("recommendationid","count"), pos=("voted_up","sum"))
    by["pos_rate"] = by["pos"] / by["total"].where(by["total"]>0, 1)
    by = by.dropna()
    return by

st.set_page_config(page_title="Steam 评论分析", layout="wide")
st.title("Steam 评论分析 / Steam Review Analytics")

with st.sidebar:
    st.header("数据源")
    prefix = st.selectbox("选择数据前缀", ["reviews_all", "reviews_recent"], index=0)
    df = load_all(prefix)

    if df.empty:
        st.warning("没有数据。先运行 fetch_and_dump.py 抓取评论。")
        st.stop()

    # 过滤器
    appids = sorted(df["recommendationid"].astype(str))  # 这里只为 UI；真实过滤用不到
    langs = ["(All)"] + sorted(df["language"].dropna().unique().tolist())
    lang_pick = st.selectbox("语言", langs, index=0)

    bins = ["(All)","0-10h","10-20h","20-50h","50-100h","100h+"]
    bin_pick = st.selectbox("游玩时长分箱", bins, index=0)

    date_min, date_max = df["timestamp"].min(), df["timestamp"].max()
    dr = st.date_input("日期范围", (date_min.date(), date_max.date()))

    topn = st.slider("高赞 Top N", min_value=5, max_value=50, value=10, step=5)
    freq = st.selectbox("趋势频率", ["D","W","M"], index=0)

# 应用过滤
d = df.copy()
if lang_pick != "(All)":
    d = d[d["language"] == lang_pick]
if bin_pick != "(All)":
    bins_map = ["0-10h","10-20h","20-50h","50-100h","100h+"]
    pd_bins = pd.cut(d["playtime_hours"].fillna(0), bins=[0,10,20,50,100,10**9], labels=bins_map, right=False)
    d = d[pd_bins == bin_pick]
if isinstance(dr, tuple) and len(dr) == 2:
    start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1]) + pd.Timedelta(days=1)
    d = d[(d["timestamp"] >= start) & (d["timestamp"] < end)]

# 四块：汇总、语言分布、分箱、趋势 & 高赞
c1, c2, c3 = st.columns(3)
c1.metric("评论总数", f"{len(d):,}")
c2.metric("好评率", f"{d['voted_up'].mean():.1%}")
c3.metric("平均有用票", f"{d['votes_up'].mean():.2f}")

st.subheader("语言维度")
st.dataframe(language_summary(d).head(30))

st.subheader("按游玩时长分箱")
st.dataframe(bin_playtime(d))

st.subheader("时间趋势")
tr = trend(d, freq=freq)
st.line_chart(tr[["total"]])
st.line_chart(tr[["pos_rate"]])

st.subheader("高赞评论")
cols = ["timestamp","language","playtime_hours","voted_up","votes_up","review"]
st.dataframe(d.sort_values("votes_up", ascending=False)[cols].head(topn))
