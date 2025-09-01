#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

=============
1. **Parse Report Publication Date**: Identify the "report publication date" from the prompt text (if unable to extract from context with "publication" semantics, fall back to the first date appearing in the prompt).
2. **Insert Prediction Condition Update** — Prediction records must satisfy either:
   A) forecast.ann_date is in the same month as the prompt report publication date;
   B) The interval forecast.[ann_date, end_date] intersects with the month of the prompt report publication date
      (i.e., ann_date <= end of month and end_date >= beginning of month).
   — Otherwise, consider no new prediction reports.
3. Continue outputting in the prediction summary: prediction report time / prediction type / prediction signal (Chinese label), and retain
   the"【括号内为新增预测报告，只作为参考，请实际分析】",
    "【In brackets are new prediction reports, for reference only, please analyze actually】" prompt.
4. Stock Name Cleaning: Remove "This is with".

Command Line
------
python gen_financial_reports_from_tushare.py \
    --prompts stage1_testdata.json \
    --token YOURTOKEN

Dependencies
----
pip install tushare pandas python-dateutil tqdm opencc-python-reimplemented hanziconv

Environment Variables
--------
export TUSHARE_TOKEN="xxxxxxxxxxxx"   # If not using --token

-------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import logging
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
from dateutil import parser as dtparser

# tqdm progress bar
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **k):  # minimal fallback, disables bar
        return x

# ---------------------------------------------------------------------------
# TuShare Token (can be hard-coded; recommended to use environment variable or --token to override)
# ---------------------------------------------------------------------------
with open("/content/main_notebook/tushare_token.txt", "r") as f:
    tushare_token = f.read().strip()
    
TUSHARE_TOKEN = os.getenv(
    "TUSHARE_TOKEN",
    {tushare_token}  #
)

try:
    import tushare as ts  # type: ignore
except Exception as _e:  # pragma: no cover
    print("[WARN] tushare not installed: pip install tushare --upgrade", file=sys.stderr)
    ts = None  # Will degrade to no data mode

_PRO_CLIENT = None  # type: ignore


def _get_pro() -> Optional["ts.pro_api"]:
    """'Lazy' get TuShare pro_api client; return None on failure."""
    global _PRO_CLIENT
    if _PRO_CLIENT is not None:
        return _PRO_CLIENT
    if ts is None:
        logging.warning("TuShare package unavailable, cannot fetch financial data.")
        return None
    token = getattr(_get_pro, "_override_token", None) or TUSHARE_TOKEN or ""
    if token:
        try:
            ts.set_token(token)
        except Exception as e:  # pragma: no cover
            logging.warning("ts.set_token failed: %s", e)
    try:
        _PRO_CLIENT = ts.pro_api(token or None)
    except Exception as e:  # pragma: no cover
        logging.warning("Get TuShare pro_api failed: %s", e)
        _PRO_CLIENT = None
    return _PRO_CLIENT


# # ---------------- Global Output Paths ---------------- #
# ASSET_ROOT = "assets"
# FIN_DIR    = os.path.join(ASSET_ROOT, "finance")
# os.makedirs(FIN_DIR, exist_ok=True)
#
# # Retain old version
# FIN_FILE_OLD   = os.path.join(FIN_DIR, "finance_all_reports-TuShare.json")
# # New simplified output
# FIN_FILE_NEW   = os.path.join(FIN_DIR, "finance_reports-TuShare.json")


# ---------------- Global Output Paths (Colab) ---------------- #
ASSET_ROOT = "/content/rag_enhanced_data"   # Base directory in Colab
os.makedirs(ASSET_ROOT, exist_ok=True)

# Old test data (original source)
FIN_FILE_OLD = os.path.join(ASSET_ROOT, "stage1_testdata.json")

# Retain old version output
FIN_FILE_OLD_OUT = os.path.join(ASSET_ROOT, "finance_all_reports-TuShare.json")

# New simplified output
FIN_FILE_NEW = os.path.join(ASSET_ROOT, "finance_reports-TuShare.json")



# ---------------- Logging ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)


# ============================================================================
# Chinese Traditional-Simplified (fault-tolerant)
# ============================================================================
try:
    from opencc import OpenCC
    _t2s = OpenCC("t2s").convert
except Exception:
    try:
        from hanziconv import HanziConv
        _t2s = HanziConv.toSimplified
    except Exception:
        _t2s = lambda x: x


# ============================================================================
# Name Cleaning
# ============================================================================
def clean_name(raw: Optional[str]) -> str:
    """
    Clean stock name:
    - Remove "This is with" segments occasionally mixed in scripts/data sources
    - Remove leading/trailing whitespace
    """
    if not raw:
        return ""
    name = str(raw).replace("这是以", "")
    return name.strip()


# ============================================================================
# Prompt Parsing & Monthly Window
# ============================================================================
_code_pat = re.compile(r"[（(](\d{6})[)）]")
_date_pat = re.compile(r"\d{4}-\d{2}-\d{2}")
# Capture dates with "publication" "publication date" semantics (loose; look ahead 10 characters)
_pub_context_pat = re.compile(
    r"(?:发布日期|发布日|日期发布|发布|披露|公告)[^\n]{0,20}?(?P<date>\d{4}-\d{2}-\d{2})"
)

def month_window(d: datetime) -> Tuple[datetime, datetime]:
    """Return [beginning of month, end of month 23:59:59] for d."""
    first = datetime(d.year, d.month, 1)
    if d.month == 12:
        next_month = datetime(d.year + 1, 1, 1)
    else:
        next_month = datetime(d.year, d.month + 1, 1)
    end = next_month - timedelta(seconds=1)
    return first, end

def parse_prompt(prompt_text: str) -> Tuple[Optional[str], Optional[datetime], Optional[str]]:
    """
    Return (6-digit stock code, first date detected in this prompt, stock Chinese name or code).

    *First date* is not necessarily the "report publication date", only as a fallback.
    """
    code_match = _code_pat.search(prompt_text)
    code = code_match.group(1) if code_match else None

    date_match = _date_pat.search(prompt_text)
    dtv = None
    if date_match:
        try:
            dtv = dtparser.parse(date_match.group(0))
        except Exception:
            dtv = None

    # Name: last text before bracket
    name = None
    if code_match:
        prefix = prompt_text[:code_match.start()].strip()
        if prefix:
            name = prefix.split()[-1]
    if not name:
        name = code or ""

    # Cleaning
    name = clean_name(name)

    return code, dtv, name

def parse_prompt_pub_date(prompt_text: str) -> Optional[datetime]:
    """
    Attempt to identify the *report publication date* from the prompt text:
    Prioritize searching for dates with contexts like "publication/publication date/announcement/disclosure/date publication";
    If not found, fall back to the first YYYY-MM-DD date in the text.
    """
    # First match with semantic context
    for m in _pub_context_pat.finditer(prompt_text):
        ds = m.group("date")
        try:
            return dtparser.parse(ds)
        except Exception:
            continue

    # Fallback: first date
    m2 = _date_pat.search(prompt_text)
    if m2:
        try:
            return dtparser.parse(m2.group(0))
        except Exception:
            return None
    return None


# ============================================================================
# Stock Mapping (symbol -> ts_code, name)
# ============================================================================
_STOCK_CACHE: Optional[pd.DataFrame] = None

def _load_stock_cache(force=False) -> Optional[pd.DataFrame]:
    global _STOCK_CACHE
    if (not force) and _STOCK_CACHE is not None:
        return _STOCK_CACHE
    pro = _get_pro()
    if pro is None:
        return None
    try:
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,market,list_date')
        _STOCK_CACHE = df
        return df
    except Exception as e:
        logging.warning("Load TuShare stock_basic failed: %s", e)
        return None

def _map_symbol_to_tscode(symbol: str) -> Tuple[str,str]:
    symbol = symbol.strip()
    cache = _load_stock_cache()
    if cache is not None and not cache.empty:
        hit = cache.loc[cache['symbol'] == symbol]
        if not hit.empty:
            row = hit.iloc[0]
            return row['ts_code'], str(row['name'])
    # heuristic fallback
    exch = 'SZ' if symbol and symbol[0] in '023' else 'SH'
    return f"{symbol}.{exch}", symbol


# ============================================================================
# Financial Fetching (forecast + fina_audit)
#   — Unified external filtering to apply A/B rules
# ============================================================================
_FORECAST_POS_TYPES = {'预增','略增','续盈','扭亏','扭亏为盈','增','增长','大幅上升','盈利','增加'}
_FORECAST_NEG_TYPES = {'预减','略减','续亏','首亏','减','下降','亏','亏损','大幅下降'}

def _quarter_date(d: date) -> str:
    if d.month <= 3:  return f"{d.year}0331"
    if d.month <= 6:  return f"{d.year}0630"
    if d.month <= 9:  return f"{d.year}0930"
    return f"{d.year}1231"

def _signal_from_forecast_type(t: str) -> str:
    if t is None:
        return 'neutral'
    s = _t2s(str(t)).lower()
    if any(k in s for k in _FORECAST_POS_TYPES): return 'positive'
    if any(k in s for k in _FORECAST_NEG_TYPES): return 'negative'
    if '不确定' in s or '变动' in s or '无法' in s: return 'uncertain'
    return 'neutral'

def _parse_any_date(d: Any) -> Optional[date]:
    """
    Parse ann_date / end_date / any string into date; return None on failure.
    Supports YYYYMMDD / YYYY-MM-DD / datetime objects / pandas etc.
    """
    if d is None or (isinstance(d, float) and pd.isna(d)):
        return None
    if isinstance(d, (datetime, pd.Timestamp)):
        return d.date()
    s = str(d).strip()
    if not s or s.lower() == "nan":
        return None
    if re.fullmatch(r"\d{8}", s):
        try:
            return datetime.strptime(s, "%Y%m%d").date()
        except Exception:
            return None
    try:
        return dtparser.parse(s).date()
    except Exception:
        return None

def fetch_financials(symbol: str,
                     stock_name: str,
                     ref_date: Optional[datetime]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch forecast & fina_audit (fetch based on the quarter of ref_date).
    *No window filtering*, external logic applies A/B rules for filtering.
    """
    pro = _get_pro()
    if pro is None:
        return {}

    if ref_date is None:
        ref_date = datetime.today()

    ts_code, _ = _map_symbol_to_tscode(symbol)
    qd = _quarter_date(ref_date.date())

    out: Dict[str, List[Dict[str, Any]]] = {}

    # forecast
    try:
        df = pro.forecast(ts_code=ts_code, period=qd)
    except Exception as e:
        logging.warning("forecast %s: %s", ts_code, e)
    else:
        if df is not None and not df.empty:
            rec = df.to_dict('records')
            for r in rec:
                r['stock_name'] = stock_name
            out['forecast'] = rec

    # fina_audit (not available for all accounts; missing is fine)
    try:
        df2 = pro.fina_audit(ts_code=ts_code, period=qd)  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        logging.debug("fina_audit %s: %s", ts_code, e)
    else:
        if df2 is not None and not df2.empty:
            rec2 = df2.to_dict('records')
            for r in rec2:
                r['stock_name'] = stock_name
            out['fina_audit'] = rec2

    return out


# ============================================================================
# Prediction Filtering Rules (A/B)
# ============================================================================
def _forecast_matches_prompt(rec: Dict[str, Any],
                             pub_month_start: date,
                             pub_month_end: date) -> bool:
    """
    Apply "new prediction report insertion" rules.

    Parameters:
        rec: Single forecast record (dict)
        pub_month_start, pub_month_end: Start and end dates of the month for the prompt report publication date (date)

    Return True if either:
      A) ann_date falls within [pub_month_start, pub_month_end];
      B) [ann_date, end_date] intersects with [pub_month_start, pub_month_end].
    """
    ann = _parse_any_date(rec.get('ann_date'))
    end = _parse_any_date(rec.get('end_date'))

    if ann is None and end is None:
        return False

    # Condition A: ann in month
    if ann and (pub_month_start <= ann <= pub_month_end):
        return True

    # Condition B: interval intersection (ann <= end of month && end >= beginning of month)
    # If ann missing, use end; if end missing, use ann; both missing already excluded above
    lo = ann or end
    hi = end or ann
    if lo and hi:
        if lo <= pub_month_end and hi >= pub_month_start:
            return True

    return False

def _audit_matches_prompt(rec: Dict[str, Any],
                          pub_month_start: date,
                          pub_month_end: date) -> bool:
    """
    Audit record filtering: Use the same A/B rules as forecast.
    """
    return _forecast_matches_prompt(rec, pub_month_start, pub_month_end)


# ============================================================================
# Simplify Fields and Build Report Entry
# ============================================================================
def _extract_forecast(lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = {'ann_date','end_date','type','p_change_min','p_change_max',
            'net_profit_min','net_profit_max','summary','change_reason','stock_name'}
    out = []
    for r in lst:
        out.append({k: r.get(k) for k in keep if k in r})
    return out

def _extract_audit(lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = {'ann_date','end_date','audit_result','audit_fees',
            'audit_agency','audit_sign','stock_name'}
    out = []
    for r in lst:
        out.append({k: r.get(k) for k in r if k in keep})
    return out

def _normalize_ann_date(d: Optional[str]) -> Optional[str]:
    """
    Uniform ann_date / end_date various formats (YYYYMMDD, YYYY-MM-DD, etc.) to YYYY-MM-DD.
    Return None if unable to parse.
    """
    if not d:
        return None
    s = str(d).strip()
    # Common 8-digit pure number
    if re.fullmatch(r"\d{8}", s):
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    # Allow 10-digit with separators
    try:
        dtv = dtparser.parse(s)
        return dtv.strftime("%Y-%m-%d")
    except Exception:
        return None

def build_report_entry(symbol: str, stock_name: str,
                       raw_data: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, Dict[str, Any]]:
    """
    Build unfiltered version (external already filtered); fields:
        name
        forecast (list)
        fina_audit (list)
        forecast_latest_type
        forecast_latest_date
        forecast_signal
    """
    ts_code, _ = _map_symbol_to_tscode(symbol)
    stock_name = clean_name(stock_name)
    entry: Dict[str, Any] = {"name": stock_name}

    if raw_data.get('forecast'):
        entry['forecast'] = _extract_forecast(raw_data['forecast'])
    if raw_data.get('fina_audit'):
        entry['fina_audit'] = _extract_audit(raw_data['fina_audit'])

    # derive forecast_latest_type & signal & date
    if entry.get('forecast'):
        try:
            latest = max(entry['forecast'], key=lambda x: x.get('ann_date') or '')
            ftype = latest.get('type')
            fdate = _normalize_ann_date(latest.get('ann_date') or latest.get('end_date'))
            entry['forecast_latest_type'] = ftype
            entry['forecast_latest_date'] = fdate
            entry['forecast_signal'] = _signal_from_forecast_type(ftype)
        except Exception:
            pass

    return ts_code, entry


# ============================================================================
# Prediction Summary Text Generation (Main Sentence + Structured Description + Time + Bracket Note)
# ============================================================================
MAX_SNIPPET_LEN = 180  # Relaxed to accommodate bracket notes

def gen_prediction_snippet(name: str,
                           ftype: Optional[str],
                           signal: Optional[str],
                           audit_result: Optional[str] = None,
                           forecast_date: Optional[str] = None) -> str:
    """
    Return Chinese prediction summary, wrapped in full-width brackets, with bracket note explanation.

    Format:
    （【预测】{name}{...}。预测报告时间：YYYY-MM-DD；预测类型为：X；预测信号为：Y。【括号内为新增预测报告，只作为参考，请实际分析】
    【Prediction】{name}{...}. Prediction report time: YYYY-MM-DD; Prediction type: X; Prediction signal: Y. [In brackets are new prediction reports, for reference only, please analyze actually]）
    """
    name = clean_name(name or "").strip() or "该股"
    type_part = (ftype or "").strip()

    sig_map = {
        "positive": "盈利前景偏正面",
        "negative": "业绩压力较大",
        "neutral":  "业绩变化有限",
        "uncertain":"业绩不确定性高",
    }
    sig_part = sig_map.get(signal or "neutral", "业绩变化有限")

    audit_part = ""
    if audit_result:
        ar = str(audit_result)
        if any(k in ar for k in ("无保留","标准无保留")):
            audit_part = "，审计标准无保留"
        elif any(k in ar for k in ("保留","否定","无法表示")):
            audit_part = "，审计意见存保留"

    # Chinese signal label (for structured part)
    sig_cn_map = {
        "positive": "正面",
        "negative": "负面",
        "neutral":  "中性",
        "uncertain":"不确定",
    }
    sig_cn = sig_cn_map.get(signal or "neutral", "中性")

    # Main sentence
    if type_part:
        main = f"【预测】{name}{type_part}，{sig_part}{audit_part}。"
    else:
        main = f"【预测】{name}{sig_part}{audit_part}。"

    # Time format
    fdate_show = forecast_date or "未知"

    # Structured description
    type_show = type_part or "无"
    sig_show  = sig_cn  # If English needed, append f"{sig_cn}({signal})"
    extra = f"预测报告时间：{fdate_show}；预测类型为：{type_show}；预测信号为：{sig_show}。"

    note = "【括号内为新增预测报告，只作为参考，请实际分析】"

    text_inner = main + extra + note
    text = f"（{text_inner}）"

    # Length limit
    if len(text) < 20:
        text = text.replace("）", "，后续关注。）")
    if len(text) > MAX_SNIPPET_LEN:
        # Retain bracket note ending
        clip = MAX_SNIPPET_LEN - 1
        if clip < 0: clip = 0
        text = text[:clip] + "…"

    return text

def gen_no_forecast_snippet(name: str) -> str:
    """
    Prompt when no new prediction report (wrapped in full-width brackets + reference note).
    """
    name = clean_name(name or "").strip() or "该股"
    note = "【括号内为新增预测报告，只作为参考，请实际分析】"
    text_inner = f"【预测】{name}：本期无新增预测报告。{note}"
    return f"（{text_inner}）"


# ============================================================================
# Insert Prediction Text into prompt.input
# ============================================================================
_INSERT_ANCHOR = "研报内容如下:"

def insert_prediction_into_input(input_text: str, prediction_text: str) -> Tuple[str, str]:
    """
    Insert prediction_text + newline before "研报内容如下:" in input_text.
    If anchor not found, append to the end of the text (preceded by newline).

    Return (new text, insertion method description: 'anchor' | 'append' | 'none')
    """
    if not prediction_text:
        return input_text, 'none'
    idx = input_text.find(_INSERT_ANCHOR)
    if idx >= 0:
        before = input_text[:idx].rstrip("\n")
        after  = input_text[idx:]
        return before + "\n" + prediction_text + "\n" + after, 'anchor'
    else:
        return input_text.rstrip("\n") + "\n" + prediction_text + "\n", 'append'


# ============================================================================
# Main Processing: Single Prompt
# ============================================================================
def process_prompt_record(rec: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Tuple[str, Dict[str, Any]]], Dict[str, Any]]:
    """
    Input a {"input": "..."} record;
    Return (modified rec, report tuple or None, meta info meta)

    meta: {
        'symbol': str|None,
        'name': str|None,
        'insert_mode': 'anchor'|'append'|'none',
        'prediction': str,
        'inserted_real_forecast': bool,
        'signal': str|None,
        'forecast_type': str|None,
        'forecast_date': str|None,
        'has_forecast': bool,
        'has_audit': bool,
        'pub_date': str|None,         # Prompt report publication date
    }
    """
    raw_text = rec.get("input", "")
    symbol, dt_any, name = parse_prompt(raw_text)
    pub_dt = parse_prompt_pub_date(raw_text)  # Report publication date (fallback to parse_prompt's date if empty)
    if pub_dt is None:
        pub_dt = dt_any

    meta = {
        'symbol': symbol,
        'name': name,
        'insert_mode': 'none',
        'prediction': '',
        'inserted_real_forecast': False,
        'signal': None,
        'forecast_type': None,
        'forecast_date': None,
        'has_forecast': False,
        'has_audit': False,
        'pub_date': pub_dt.strftime("%Y-%m-%d") if pub_dt else None,
    }

    if not symbol:
        logging.info("Skip: Unable to recognize stock code (input=%r)", raw_text[:60])
        return rec, None, meta

    pro_name = name or symbol

    # Fetch financials (no window filtering)
    fin_raw_all = fetch_financials(symbol, pro_name, pub_dt)

    # ---- Apply A/B rules filtering ----
    if pub_dt is not None:
        m_start, m_end = month_window(pub_dt)
        m_start_d, m_end_d = m_start.date(), m_end.date()
        # forecast
        flist = fin_raw_all.get('forecast') or []
        flist_filt = [r for r in flist if _forecast_matches_prompt(r, m_start_d, m_end_d)]
        # audit
        alist = fin_raw_all.get('fina_audit') or []
        alist_filt = [r for r in alist if _audit_matches_prompt(r, m_start_d, m_end_d)]
    else:
        flist_filt = []
        alist_filt = []

    fin_raw = {}
    if flist_filt:
        fin_raw['forecast'] = flist_filt
    if alist_filt:
        fin_raw['fina_audit'] = alist_filt

    ts_code, entry = build_report_entry(symbol, pro_name, fin_raw)

    meta['has_forecast']   = bool(entry.get("forecast"))
    meta['has_audit']      = bool(entry.get("fina_audit"))
    meta['forecast_type']  = entry.get("forecast_latest_type")
    meta['forecast_date']  = entry.get("forecast_latest_date")
    meta['signal']         = entry.get("forecast_signal")

    # Get audit_result (latest)
    audit_result = None
    if entry.get("fina_audit"):
        try:
            latest_a = max(entry["fina_audit"], key=lambda x: x.get("ann_date") or "")
            audit_result = latest_a.get("audit_result")
        except Exception:
            pass

    # --- Determine whether to insert real prediction ---
    if entry.get("forecast") and entry.get("forecast_latest_date") and pub_dt is not None:
        # Re-check once (redundant guarantee)
        m_start, m_end = month_window(pub_dt)
        if _forecast_matches_prompt(
            {'ann_date': entry.get("forecast_latest_date"), 'end_date': entry.get("forecast")[0].get("end_date") if entry.get("forecast") else None},
            m_start.date(), m_end.date()
        ):
            pred_text = gen_prediction_snippet(
                name=entry.get("name", pro_name),
                ftype=entry.get("forecast_latest_type"),
                signal=entry.get("forecast_signal"),
                audit_result=audit_result,
                forecast_date=entry.get("forecast_latest_date"),
            )
            meta['inserted_real_forecast'] = True
        else:
            pred_text = gen_no_forecast_snippet(entry.get("name", pro_name))
            meta['inserted_real_forecast'] = False
    else:
        pred_text = gen_no_forecast_snippet(entry.get("name", pro_name))
        meta['inserted_real_forecast'] = False

    meta['prediction'] = pred_text

    new_input, insert_mode = insert_prediction_into_input(raw_text, pred_text)
    meta['insert_mode'] = insert_mode

    new_rec = dict(rec)
    new_rec["input"] = new_input

    return new_rec, (ts_code, entry), meta


# ============================================================================
# ETA Formatting
# ============================================================================
def _fmt_dur(sec: float) -> str:
    if sec < 0: sec = 0
    m, s = divmod(int(sec + 0.5), 60)
    h, m = divmod(m, 60)
    if h:   return f"{h:d}h{m:02d}m{s:02d}s"
    if m:   return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


# ============================================================================
# Batch Process All Prompts (with Progress Bar/ETA)
# ============================================================================
def process_prompts_file(prompts_path: str) -> None:
    if not os.path.isfile(prompts_path):
        logging.error("Prompts file not found: %s", prompts_path)
        return
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    if not isinstance(prompts, list):
        logging.error("prompts JSON must be a list.")
        return

    total = len(prompts)
    logging.info("Read prompts: total %d entries.", total)

    # Backup original file
    ts_now = datetime.now().strftime("%Y%m%d%H%M%S")
    bak_path = f"{prompts_path}.orig-{ts_now}.json"
    with open(bak_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    logging.info("Backed up original prompts → %s", bak_path)

    # Processing
    new_prompts: List[Dict[str, Any]] = []
    report_bundle: Dict[str, Dict[str, Any]] = {}

    # Statistics
    cnt_no_code = 0
    cnt_ok      = 0
    cnt_fail    = 0
    cnt_real_pred = 0  # Number of real prediction insertions

    # Time consumption
    t_start = time.perf_counter()
    recent_secs: List[float] = []

    # tqdm outer layer
    pbar = tqdm(range(total), desc="Processing prompts", unit="entry", dynamic_ncols=True)

    for i in pbar:
        rec = prompts[i]
        idx1 = i + 1  # 1-based

        # Current name for pbar desc
        txt = rec.get("input", "")
        sym, _, nm = parse_prompt(txt)
        pbar.set_description_str(f"{idx1}/{total} {nm or ''}({sym or '??'})")

        t0 = time.perf_counter()
        new_rec, rep, meta = process_prompt_record(rec)
        dt = time.perf_counter() - t0
        recent_secs.append(dt)
        if len(recent_secs) > 20:  # Rolling window
            recent_secs.pop(0)

        new_prompts.append(new_rec)

        if not meta['symbol']:
            cnt_no_code += 1
        else:
            cnt_ok += 1  # Count as success even without forecast

        if meta['inserted_real_forecast']:
            cnt_real_pred += 1

        # Log: Insert summary
        logging.info(
            "Prompt #%d | %s(%s) | pub=%s | ins_mode=%s | ins_pred=%s | type=%s | fdate=%s | signal=%s | Time %.2fs",
            idx1,
            meta.get('name') or '',
            meta.get('symbol') or '',
            meta.get('pub_date'),
            meta.get('insert_mode'),
            meta.get('inserted_real_forecast'),
            meta.get('forecast_type'),
            meta.get('forecast_date'),
            meta.get('signal'),
            dt,
        )

        # Accumulate report
        if rep:
            ts_code, entry = rep
            if ts_code in report_bundle:
                old = report_bundle[ts_code]
                # name
                if not old.get("name") and entry.get("name"):
                    old["name"] = entry["name"]
                # forecast
                if entry.get("forecast"):
                    old.setdefault("forecast", [])
                    old["forecast"].extend(entry["forecast"])
                # fina_audit
                if entry.get("fina_audit"):
                    old.setdefault("fina_audit", [])
                    old["fina_audit"].extend(entry["fina_audit"])
                # Update core fields (if missing)
                for k in ("forecast_latest_date","forecast_latest_type","forecast_signal"):
                    if entry.get(k) and not old.get(k):
                        old[k] = entry[k]
            else:
                report_bundle[ts_code] = entry
        else:
            cnt_fail += 1

        # ETA update
        avg_sec = sum(recent_secs) / len(recent_secs)
        remain  = (total - idx1) * avg_sec
        pbar.set_postfix_str(f"avg={avg_sec:.2f}s ETA={_fmt_dur(remain)}")

    pbar.close()

    # Deduplicate after merging & recalculate forecast_latest_type/signal/date
    for ts_code, entry in report_bundle.items():
        # Deduplicate forecast
        if entry.get("forecast"):
            seen = set()
            uniq = []
            for r in entry["forecast"]:
                sig = (r.get("ann_date"), r.get("type"))
                if sig in seen: continue
                seen.add(sig)
                uniq.append(r)
            entry["forecast"] = uniq

        # Deduplicate audit
        if entry.get("fina_audit"):
            seen = set()
            uniq = []
            for r in entry["fina_audit"]:
                sig = (r.get("ann_date"), r.get("audit_result"))
                if sig in seen: continue
                seen.add(sig)
                uniq.append(r)
            entry["fina_audit"] = uniq

        # Latest forecast
        if entry.get("forecast"):
            try:
                latest = max(entry["forecast"], key=lambda x: x.get("ann_date") or "")
                ftype = latest.get("type")
                fdate = _normalize_ann_date(latest.get("ann_date") or latest.get("end_date"))
                entry["forecast_latest_type"] = ftype
                entry["forecast_latest_date"] = fdate
                entry["forecast_signal"] = _signal_from_forecast_type(ftype)
            except Exception:
                pass

    # Write report JSON
    with open(FIN_FILE_NEW, "w", encoding="utf-8") as f:
        json.dump(report_bundle, f, ensure_ascii=False, indent=2, default=str)
    logging.info("Written simplified financial report JSON → %s (total %d stocks)", FIN_FILE_NEW, len(report_bundle))

    # Overwrite prompts file
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump(new_prompts, f, ensure_ascii=False, indent=2)
    logging.info("Updated prompts file (inserted prediction summary) → %s", prompts_path)

    # Note old financial JSON status
    if os.path.isfile(FIN_FILE_OLD):
        logging.info("Note: Retained old financial JSON: %s", FIN_FILE_OLD)
    else:
        logging.info("Old financial JSON not detected (%s) — If needed for comparison, run old script to generate.", FIN_FILE_OLD)

    # Summary
    t_total = time.perf_counter() - t_start
    logging.info(
        "Processing complete: Total %d entries; Recognized codes %d; No codes %d; Fetch failures (no data) %d; Inserted real predictions %d; Total time %s.",
        total, cnt_ok, cnt_no_code, cnt_fail, cnt_real_pred, _fmt_dur(t_total)
    )


# ============================================================================
# CLI
# ============================================================================
def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Generate TuShare financial reports and insert prediction summaries into prompts JSON (insert predictions only if ann_date same month or interval coverage rule is met)."
    )
    p.add_argument(
        "--prompts",
        default="stage1_testdata.json",
        help="Input prompts JSON path (will overwrite in place and auto-backup).",
    )
    p.add_argument("--token", help="TuShare Token override env TUSHARE_TOKEN.")
    args = p.parse_args()

    if args.token:
        setattr(_get_pro, "_override_token", args.token)

    process_prompts_file(args.prompts)


if __name__ == "__main__":
    main()
