"""
TiresiasIQ Predictor — Adaptive Hybrid Model (v3)
------------------------------------------------
Features added / changed compared to v2.1:
- Per-user lightweight memory (counts, last_seen, context centroid) and global memory
- Online adaptation: update_with_log(...) updates priors, centroids, recency and running scaler without full retrain
- Short-term retrieval: nearest-neighbour scoring against recent logs (configurable window)
- Simple drift detector (Page-Hinkley style) per action to flag when retraining is recommended
- Persistent feature-space: `all_keywords` and `actions` are authoritative and kept fixed unless explicitly expanded
- Backwards-compatible API:
    - update_running_status(data) -> X, y  (for initial batch training)
    - predict(query, pred_time, user_id=None) -> PredictResult
    - update_with_log(log_row, user_id=None)  # online-adapt on new single row

Notes:
- This module uses sentence-transformers if available for better sentence embeddings; else falls back to spaCy.
- It avoids heavy dependencies for drift; uses a simple Page-Hinkley implementation.
- The code assumes logs are stored elsewhere (SQLite). update_with_log should be called after inserting a new log.

Author: Assistant
"""
from __future__ import annotations

import math
import time
import re
import os
import json
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    _SBERT = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    _SBERT = None

import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# load spaCy fallback
try:
    nlp = spacy.load('en_core_web_md')
except Exception:
    nlp = spacy.load('en_core_web_sm')

sentiment_analyzer = SentimentIntensityAnalyzer()

# ----------------------------- helpers & small classes -----------------------------
LIGHT_VERBS = set([
    "be","have","do","get","go","come","take","make","put","keep","let","seem",
    "want","need","try","feel","think","know","like","love","hate","say","tell","use",
    "ask","give","see","look","appear","consider","prefer","plan","hope","intend"
])


def _doc_dim() -> int:
    tok = nlp('probe')[0]
    return int(tok.vector.shape[0])


def normalize_action(text: str) -> str:
    if not text:
        return ''
    neg = text.lower().startswith('not_')
    base = text[4:] if neg else text
    doc = nlp(base)
    tok = next((t for t in doc if t.pos_ == 'VERB'), None)
    if tok is None:
        tok = next((t for t in doc if t.pos_ == 'NOUN'), None)
    if tok is None and len(doc) > 0:
        tok = doc[0]
    if tok is None:
        out = base.lower()
    else:
        particle = next((c.lower_ for c in tok.children if c.dep_ == 'prt'), None)
        lemma = tok.lemma_.lower()
        out = f"{lemma}_{particle}" if particle else lemma
    return f'not_{out}' if neg else out


def embed_text(text: str) -> np.ndarray:
    if not text:
        return np.zeros(_doc_dim(), dtype=float)
    if _SBERT is not None:
        try:
            v = _SBERT.encode([text], convert_to_numpy=True)[0]
            return v.astype(float)
        except Exception:
            pass
    d = nlp(text)
    if getattr(d, 'vector_norm', 0) and d.vector_norm > 0:
        return np.array(d.vector, dtype=float)
    tokens = [t.vector for t in d if t.has_vector]
    if tokens:
        return np.mean(tokens, axis=0)
    return np.zeros(_doc_dim(), dtype=float)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or not np.any(a) or not np.any(b):
        return 0.0
    return float(cosine_similarity([a], [b])[0][0])


# Simple Page-Hinkley drift detector
class PageHinkley:
    def __init__(self, delta=0.005, lambda_=50, alpha=0.99):
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.mean = 0.0
        self.mT = 0.0
        self.n = 0
        self.ph = 0.0

    def add(self, x: float) -> bool:
        # returns True if drift detected
        self.n += 1
        self.mean = self.alpha * self.mean + (1 - self.alpha) * x if self.n > 1 else x
        self.mT = min(self.mT, x - self.mean - self.delta) if self.n > 1 else x - self.mean - self.delta
        self.ph = max(self.ph, x - self.mean - self.mT - self.delta)
        if self.ph > self.lambda_:
            # reset after detection
            self.__init__(self.delta, self.lambda_, self.alpha)
            return True
        return False


# ----------------------------- result dataclass -----------------------------
@dataclass
class PredictResult:
    probability: float
    action: str
    extracted_action: str
    top_candidates: List[Tuple[str, float, float, float, float]] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


# ----------------------------- main Predictor -----------------------------
class Predictor:
    def __init__(self, model=None, all_keywords: Optional[List[str]] = None, actions: Optional[List[str]] = None,
                 pca_components: int = 16, w_model: float = 0.6, w_temporal: float = 0.25, w_semantic: float = 0.15,
                 recency_tau_days: float = 21.0, short_memory_size: int = 200,
                 dynamic_weights: bool = True, weight_lr: float = 0.05, weight_margin: float = 0.01, weight_min: float = 0.05,
                 decay_mode: str = 'exp', recency_alpha: float = 0.75,
                 enable_seasonality: bool = True, enable_holidays: bool = True, holidays: Optional[List[str]] = None,
                 enable_user_cycles: bool = True):
        # model, fixed feature-space
        self.model = model
        self.all_keywords = [k.strip().lower() for k in (all_keywords or [])]
        self.actions = [normalize_action(a) for a in (actions or [])]

        # blending weights
        s = max(1e-9, w_model + w_temporal + w_semantic)
        self.w_model = w_model / s; self.w_temporal = w_temporal / s; self.w_semantic = w_semantic / s

        # dynamic weight learning params
        self.dynamic_weights = bool(dynamic_weights)
        self.weight_lr = float(weight_lr)
        self.weight_margin = float(weight_margin)
        self.weight_min = float(weight_min)
        self._weight_history: deque = deque(maxlen=200)

        # pca for reducing embeddings when needed
        self.pca_components = max(1, int(pca_components))
        self.pca_: Optional[PCA] = None

        # priors and per-user memory
        self.priors_: Dict[str, Dict[str, np.ndarray]] = {}
        self.context_centroid_: Dict[str, np.ndarray] = {}
        self.last_seen_day_: Dict[str, float] = {}

        # per-user memory store: small in-memory structure
        self.user_memory: Dict[str, Dict[str, Any]] = {}

        # short-term memory (list of recent logs)
        self.short_memory: List[Dict[str, Any]] = []
        self.short_memory_size = int(short_memory_size)

        # online scaler (running mean & var)
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_var: Optional[np.ndarray] = None
        self.scaler_count: int = 0

        # drift detectors per action
        self.drift_detectors: Dict[str, PageHinkley] = {a: PageHinkley() for a in self.actions}
        self._drift_counts: Dict[str, Dict[str, Any]] = {}

        # recency & decay configuration
        self.recency_tau_days = float(max(1.0, recency_tau_days))
        self.decay_mode = str(decay_mode)
        self.recency_alpha = float(recency_alpha)

        # seasonality & holidays
        self.enable_seasonality = bool(enable_seasonality)
        self.enable_holidays = bool(enable_holidays)
        self.holidays = set()
        if holidays:
            for h in holidays:
                try:
                    # store only the date part (YYYY-MM-DD)
                    self.holidays.add(str(h)[:10])
                except Exception:
                    continue

        # user cycles
        self.enable_user_cycles = bool(enable_user_cycles)

        # data dir for drift persistence
        self._data_dir = os.path.join(os.path.dirname(__file__), 'data')
        try:
            os.makedirs(self._data_dir, exist_ok=True)
        except Exception:
            pass

        # cache for embeddings
        self._embed_cache: Dict[str, np.ndarray] = {}

    # --------------------------- feature builders ---------------------------
    def _extract_keywords(self, text: str) -> List[str]:
        d = nlp(text)
        keys = set()
        for ent in d.ents:
            keys.add(ent.text.lower())
        for tok in d:
            if tok.is_stop or tok.is_punct: continue
            if tok.pos_ in {"NOUN", "ADJ", "VERB"}:
                keys.add(tok.lemma_.lower())
        return sorted(keys)

    def _time_bits(self, iso: str) -> Dict[str, float]:
        from dateutil import parser
        try:
            dt = parser.isoparse(iso)
            hour = dt.hour
            dow = dt.weekday()
            is_weekend = 1 if dow >= 5 else 0
            month = dt.month
            hour_sin = math.sin(2*math.pi*hour/24)
            hour_cos = math.cos(2*math.pi*hour/24)
            dow_sin = math.sin(2*math.pi*dow/7)
            dow_cos = math.cos(2*math.pi*dow/7)
        except Exception:
            hour = 0; dow = 0; is_weekend = 0; month = 1; hour_sin = hour_cos = dow_sin = dow_cos = 0.0
        return {"hour": hour/23.0, "dow": dow/6.0, "is_weekend": float(is_weekend), "month": (month-1)/11.0,
                "hour_sin": hour_sin, "hour_cos": hour_cos, "dow_sin": dow_sin, "dow_cos": dow_cos,
                "raw_hour": hour, "raw_dow": dow}

    def _embed(self, text: str) -> np.ndarray:
        if text in self._embed_cache: return self._embed_cache[text]
        v = embed_text(text)
        self._embed_cache[text] = v
        return v

    def log_to_vector(self, keywords: str, polarity: float, subjectivity: float, action: str, user_time: str) -> np.ndarray:
        kws = {k.strip().lower() for k in keywords.split(',') if k.strip()}
        keyword_vec = [1.0 if k in kws else 0.0 for k in self.all_keywords]

        # action embedding reduced to pca_components (if pca_ present)
        a_vec = self._embed(action)
        if self.pca_ is not None:
            try:
                a_red = self.pca_.transform([a_vec])[0]
            except Exception:
                a_red = a_vec[:self.pca_components] if a_vec.shape[0] >= self.pca_components else np.pad(a_vec, (0, self.pca_components - a_vec.shape[0]))
        else:
            a_red = a_vec[:self.pca_components] if a_vec.shape[0] >= self.pca_components else np.pad(a_vec, (0, self.pca_components - a_vec.shape[0]))

        tb = self._time_bits(user_time)
        time_vec = [tb['hour'], tb['dow'], tb['is_weekend'], tb['month'], tb['hour_sin'], tb['hour_cos'], tb['dow_sin'], tb['dow_cos']]

        feat = np.array(keyword_vec + [polarity, subjectivity] + list(a_red) + time_vec, dtype=float)
        return feat

    # --------------------------- online scaler routines ---------------------------
    def _update_running_stats(self, x: np.ndarray):
        x = x.astype(float).reshape(-1)
        if self.scaler_mean is None:
            self.scaler_mean = x.copy()
            self.scaler_var = np.zeros_like(x)
            self.scaler_count = 1
            return
        self.scaler_count += 1
        delta = x - self.scaler_mean
        self.scaler_mean += delta / self.scaler_count
        delta2 = x - self.scaler_mean
        self.scaler_var += delta * delta2

    def _transform_with_running(self, x: np.ndarray) -> np.ndarray:
        if self.scaler_mean is None:
            return x
        std = np.sqrt((self.scaler_var / max(1, self.scaler_count - 1)) + 1e-9)
        return (x - self.scaler_mean) / std

    # --------------------------- priors & centroids updates ---------------------------
    def _ensure_action_space(self, action: str):
        # If a new action appears online, add it to action lists and init priors
        a = normalize_action(action)
        if a in self.actions: return a
        self.actions.append(a)
        # init priors with hierarchical memory structures
        self.priors_[a] = {
            "hour_hist": np.ones(24, dtype=float),
            "dow_hist": np.ones(7, dtype=float),
            "month_hist": np.ones(12, dtype=float),
            "week_hist": np.ones(53, dtype=float),  # ISO weeks 1..53
            "weekend_rate": 0.5,
            "holiday_rate": 0.5,
        }
        self.context_centroid_[a] = np.zeros(self.pca_components, dtype=float)
        self.last_seen_day_[a] = -1e12
        self.drift_detectors[a] = PageHinkley()
        self._drift_counts.setdefault(a, {"count": 0, "last_ts": 0})
        return a

    def _update_priors_online(self, action: str, user_time: str, context_vec: np.ndarray):
        a = self._ensure_action_space(action)
        from dateutil import parser
        try:
            dt = parser.isoparse(user_time)
            hour = dt.hour; dow = dt.weekday(); is_weekend = int(dow >= 5); day_float = dt.timestamp() / 86400.0
            month = dt.month  # 1..12
            iso_week = getattr(dt.isocalendar(), 'week', None)
            if iso_week is None:
                # Python <3.9 compat
                iso_week = dt.isocalendar()[1]
            week_idx = max(1, min(53, int(iso_week))) - 1
            is_holiday = 1 if (str(dt.date()) in self.holidays) else 0
        except Exception:
            hour = 0; dow = 0; is_weekend = 0; day_float = None; month = 1; week_idx = 0; is_holiday = 0
        pri = self.priors_.get(a)
        if pri is None:
            pri = {"hour_hist": np.ones(24, dtype=float), "dow_hist": np.ones(7, dtype=float),
                   "month_hist": np.ones(12, dtype=float), "week_hist": np.ones(53, dtype=float),
                   "weekend_rate": 0.5, "holiday_rate": 0.5}
            self.priors_[a] = pri
        # update histograms (simple additive smoothing)
        pri['hour_hist'][hour] += 1.0
        pri['dow_hist'][dow] += 1.0
        pri['month_hist'][max(0, min(11, month - 1))] += 1.0
        pri['week_hist'][week_idx] += 1.0
        # weekend & holiday moving averages
        prev_wr = pri.get('weekend_rate', 0.5)
        pri['weekend_rate'] = 0.95 * prev_wr + 0.05 * float(is_weekend)
        prev_hr = pri.get('holiday_rate', 0.5)
        pri['holiday_rate'] = 0.95 * prev_hr + 0.05 * float(is_holiday)
        # normalize histograms
        pri['hour_hist'] = pri['hour_hist'] / max(1.0, np.sum(pri['hour_hist']))
        pri['dow_hist'] = pri['dow_hist'] / max(1.0, np.sum(pri['dow_hist']))
        pri['month_hist'] = pri['month_hist'] / max(1.0, np.sum(pri['month_hist']))
        pri['week_hist'] = pri['week_hist'] / max(1.0, np.sum(pri['week_hist']))
        # update centroid via exponential moving average
        prev_cent = self.context_centroid_.get(a)
        if prev_cent is None or not np.any(prev_cent):
            self.context_centroid_[a] = context_vec.copy()
        else:
            self.context_centroid_[a] = 0.9 * prev_cent + 0.1 * context_vec
        # last seen day
        if day_float is not None:
            self.last_seen_day_[a] = max(self.last_seen_day_.get(a, -1e12), day_float)
        # drift detector update using 1 - temporal_prior as proxy error signal
        td = self._temporal_prior(a, user_time)
        err = 1.0 - td
        detector = self.drift_detectors.get(a)
        if detector and detector.add(err):
            # signal drift
            self._on_drift_detected(a)

    # --------------------------- update with a new log (online) ---------------------------
    def update_with_log(self, log_row: Tuple[str, str, float, float, str, str], user_id: Optional[str] = None):
        # log_row: (user_input, keywords, polarity, subjectivity, target_action, user_time)
        user_input, keywords, polarity, subjectivity, target_action, user_time = log_row
        # extract context vector
        ctx = self._embed(user_input)
        # ensure action space
        action = normalize_action((target_action or '').strip())
        action = self._ensure_action_space(action)
        # update priors and centroids
        self._update_priors_online(action, user_time, ctx[:self.pca_components] if ctx.shape[0] >= self.pca_components else np.pad(ctx, (0, self.pca_components - ctx.shape[0])))
        # update short memory
        entry = {'user_input': user_input, 'keywords': keywords, 'action': action, 'user_time': user_time, 'context': ctx}
        self.short_memory.insert(0, entry)
        if len(self.short_memory) > self.short_memory_size:
            self.short_memory.pop()
        # update user memory (cheap counters)
        if user_id is not None:
            um = self.user_memory.setdefault(user_id, {'counts': {}, 'last_seen': {}})
            um['counts'][action] = um['counts'].get(action, 0) + 1
            um['last_seen'][action] = time.time()
        # update running scaler on the feature vector for (action) used in NN
        feat = self.log_to_vector(keywords, polarity, subjectivity, action, user_time)
        try:
            self._update_running_stats(feat)
        except Exception:
            pass
        # dynamic weight update using the newly observed ground truth
        try:
            if self.dynamic_weights and len(self.actions) >= 2:
                self._update_weights_online(true_action=action, query_text=user_input, pred_time=user_time)
        except Exception:
            pass

    # --------------------------- temporal & semantic priors ---------------------------
    def _temporal_prior(self, action: str, pred_iso: str) -> float:
        pri = self.priors_.get(action)
        if pri is None:
            return 0.5
        tb = self._time_bits(pred_iso)
        hour = tb['raw_hour']; dow = tb['raw_dow']
        h_prob = float(pri['hour_hist'][hour]) if 0 <= hour < 24 else (1.0/24.0)
        d_prob = float(pri['dow_hist'][dow]) if 0 <= dow < 7 else (1.0/7.0)
        temporal_score = 0.6 * h_prob + 0.4 * d_prob
        # weekend adjustment
        if tb['is_weekend'] > 0.5:
            temporal_score *= (1.0 + 0.3 * (pri.get('weekend_rate', 0.5) - 0.5))
        # seasonality: month and ISO week
        if self.enable_seasonality:
            try:
                from dateutil import parser
                pred_dt = parser.isoparse(pred_iso)
                month = pred_dt.month
                iso_week = getattr(pred_dt.isocalendar(), 'week', None)
                if iso_week is None:
                    iso_week = pred_dt.isocalendar()[1]
                w_idx = max(1, min(53, int(iso_week))) - 1
                m_prob = float(pri.get('month_hist', np.ones(12)/12.0)[month-1])
                w_prob = float(pri.get('week_hist', np.ones(53)/53.0)[w_idx])
                # blend into temporal_score
                temporal_score = 0.5 * temporal_score + 0.3 * m_prob + 0.2 * w_prob
            except Exception:
                pass
        # holiday awareness
        if self.enable_holidays:
            try:
                from dateutil import parser
                pred_dt = parser.isoparse(pred_iso)
                if str(pred_dt.date()) in self.holidays:
                    temporal_score *= (1.0 + 0.3 * (pri.get('holiday_rate', 0.5) - 0.5))
            except Exception:
                pass
        # recency decay
        last_day = self.last_seen_day_.get(action)
        if last_day is not None:
            from dateutil import parser
            try:
                pred_dt = parser.isoparse(pred_iso)
                pred_day = pred_dt.timestamp() / 86400.0
                delta_days = max(0.0, pred_day - last_day)
                if self.decay_mode == 'power':
                    recency = 1.0 / pow(1.0 + delta_days, max(0.1, self.recency_alpha))
                else:
                    recency = math.exp(-delta_days / self.recency_tau_days)
                temporal_score = 0.7 * temporal_score + 0.3 * recency
            except Exception:
                pass
        return float(np.clip(temporal_score, 0.0, 1.0))

    def _semantic_prior(self, action: str, query_text: str) -> float:
        qv = self._embed(query_text)
        av = self.context_centroid_.get(action)
        if av is None or not np.any(av):
            av = self._embed(action)
        if not np.any(av) or not np.any(qv):
            return 0.5
        c = cosine(qv[:self.pca_components] if qv.shape[0] >= self.pca_components else qv, av[:self.pca_components] if av.shape[0] >= self.pca_components else av)
        return float((c + 1.0) / 2.0)

    # user-specific cycle modifier
    def _user_modifier(self, user_id: Optional[str], action: str, pred_iso: str) -> float:
        if not self.enable_user_cycles or not user_id:
            return 1.0
        um = self.user_memory.get(user_id)
        if not um:
            return 1.0
        cnt = float(um.get('counts', {}).get(action, 0))
        last_ts = um.get('last_seen', {}).get(action)
        rec = 1.0
        if last_ts:
            delta_days = max(0.0, (time.time() - last_ts) / 86400.0)
            if self.decay_mode == 'power':
                rec = 1.0 / pow(1.0 + delta_days, max(0.1, self.recency_alpha))
            else:
                rec = math.exp(-delta_days / self.recency_tau_days)
        # map counts to [0.95, 1.05]
        base = 1.0 + 0.05 * (1.0 - math.exp(-cnt / 10.0))
        return float(np.clip(base * (0.9 + 0.2 * rec), 0.9, 1.1))

    # --------------------------- prediction ---------------------------
    def predict(self, query: str, pred_time: str, user_id: Optional[str] = None) -> PredictResult:
        keywords, polarity, subjectivity, extracted_action, features = self.extract_features_from_text(query)
        # build candidate scores
        candidates = []
        for a in self.actions:
            x = self.log_to_vector(keywords, polarity, subjectivity, a, pred_time)
            # apply running scaler transform if available
            x_scaled = self._transform_with_running(x) if self.scaler_mean is not None else x
            p_model = float(self.model.predict(x_scaled.reshape(1, -1))[0][0]) if self.model is not None else 0.5
            t_base = self._temporal_prior(a, pred_time)
            u_mod = self._user_modifier(user_id, a, pred_time)
            t_prior = float(np.clip(t_base * u_mod, 0.0, 1.0))
            s_prior = self._semantic_prior(a, query)
            blended = self.w_model * p_model + self.w_temporal * t_prior + self.w_semantic * s_prior
            candidates.append((a, p_model, t_prior, s_prior, blended))
        # short-term retrieval booster: look into recent logs for matching action & time proximity
        # simple heuristic: if any recent short_memory entry has high semantic similarity with query and same action, boost that action
        # short-term retrieval booster: only if semantically similar AND time-close to pred_time
        qv = self._embed(query)
        tbq = self._time_bits(pred_time)  # uses user-provided pred_time
        for entry in self.short_memory[:50]:
            sim = cosine(qv, entry['context'])
            tbe = self._time_bits(entry['user_time'])  # uses user-provided log time
            same_dow = (tbq['raw_dow'] == tbe['raw_dow'])
            hour_diff = abs(tbq['raw_hour'] - tbe['raw_hour'])
            if sim > 0.75 and same_dow and hour_diff <= 2:
                # boost corresponding candidate blended score
                for i, (a, p, t, s, b) in enumerate(candidates):
                    if a == entry['action']:
                        candidates[i] = (a, p, t, s, min(1.0, b + 0.05 * sim))
        # Always evaluate the extracted action explicitly (even if it's not in action space yet)
        if extracted_action and extracted_action not in [a for a, *_ in candidates]:
            # ensure it has a candidate row too
            x_e_tmp = self.log_to_vector(keywords, polarity, subjectivity, extracted_action, pred_time)
            x_e_tmp_scaled = self._transform_with_running(x_e_tmp) if self.scaler_mean is not None else x_e_tmp
            p_e_tmp = float(self.model.predict(x_e_tmp_scaled.reshape(1, -1))[0][0]) if self.model is not None else 0.5
            t_e_base = self._temporal_prior(extracted_action, pred_time)
            u_mod_e = self._user_modifier(user_id, extracted_action, pred_time)
            t_e_tmp = float(np.clip(t_e_base * u_mod_e, 0.0, 1.0))
            s_e_tmp = self._semantic_prior(extracted_action, query)
            b_e_tmp = self.w_model * p_e_tmp + self.w_temporal * t_e_tmp + self.w_semantic * s_e_tmp
            candidates.append((extracted_action, p_e_tmp, t_e_tmp, s_e_tmp, b_e_tmp))
        # sort and pick best
        candidates.sort(key=lambda r: r[4], reverse=True)
        best = candidates[0]
        # Compute extracted-action scores explicitly for reporting
        extracted_scores = None
        if extracted_action:
            try:
                x_e = self.log_to_vector(keywords, polarity, subjectivity, extracted_action, pred_time)
                x_e_scaled = self._transform_with_running(x_e) if self.scaler_mean is not None else x_e
                p_e = float(self.model.predict(x_e_scaled.reshape(1, -1))[0][0]) if self.model is not None else 0.5
                t_e_base = self._temporal_prior(extracted_action, pred_time)
                u_mod_e = self._user_modifier(user_id, extracted_action, pred_time)
                t_e = float(np.clip(t_e_base * u_mod_e, 0.0, 1.0))
                s_e = self._semantic_prior(extracted_action, query)
                b_e = self.w_model * p_e + self.w_temporal * t_e + self.w_semantic * s_e
                extracted_scores = {"action": extracted_action, "model_score": p_e, "temporal_score": t_e, "semantic_score": s_e, "blended": b_e}
            except Exception:
                pass
        topk = candidates[:5]
        details = {
            "top_k": [(r[0], r[4]) for r in topk],
            "model_score": best[1],
            "temporal_score": best[2],
            "semantic_score": best[3],
            "blended": best[4],
            "top_action": best[0],
            "extracted_prediction": extracted_scores,
        }
        return PredictResult(probability=float(best[4]), action=best[0], extracted_action=extracted_action, top_candidates=topk, features=features, details=details)

    # --------------------------- dynamic weights ---------------------------
    def _update_weights_online(self, true_action: str, query_text: str, pred_time: str):
        """
        Passive-aggressive style online update on [w_model, w_temporal, w_semantic]
        to increase the margin between the true action and the top competing action.
        """
        # compile component vectors for each action
        comps = []  # (action, [p,t,s])
        for a in self.actions:
            # build features consistently with predict
            x = self.log_to_vector(','.join(self._extract_keywords(query_text)), 0.0, 0.0, a, pred_time)
            x_scaled = self._transform_with_running(x) if self.scaler_mean is not None else x
            p = float(self.model.predict(x_scaled.reshape(1, -1))[0][0]) if self.model is not None else 0.5
            t = self._temporal_prior(a, pred_time)
            s = self._semantic_prior(a, query_text)
            # user modifier is applied only at predict time; we keep base temporal here to avoid leakage
            comps.append((a, np.array([p, t, s], dtype=float)))
        # current weights vector
        w = np.array([self.w_model, self.w_temporal, self.w_semantic], dtype=float)
        # identify competitor
        true_vec = None
        best_other = None
        best_other_score = -1e9
        for a, v in comps:
            score = float(np.dot(w, v))
            if a == true_action:
                true_vec = v
            else:
                if score > best_other_score:
                    best_other_score = score
                    best_other = v
        if true_vec is None or best_other is None:
            return
        margin = float(np.dot(w, true_vec - best_other))
        if margin >= self.weight_margin:
            self._weight_history.append({"ok": True, "margin": margin})
            return
        # PA update: w <- w + eta*(true - other), then project to simplex
        eta = float(self.weight_lr)
        w_new = w + eta * (true_vec - best_other)
        # enforce non-negativity and minimum floor, then normalize to sum=1
        w_new = np.maximum(w_new, self.weight_min)
        w_new = w_new / max(1e-9, np.sum(w_new))
        self.w_model, self.w_temporal, self.w_semantic = [float(x) for x in w_new]
        self._weight_history.append({"ok": False, "margin": margin, "w": [self.w_model, self.w_temporal, self.w_semantic]})

    def _nudge_weights(self, toward: str = 'temporal', alpha: float = 0.02):
        w = np.array([self.w_model, self.w_temporal, self.w_semantic], dtype=float)
        if toward == 'temporal':
            w[1] += alpha
        elif toward == 'semantic':
            w[2] += alpha
        else:
            w[0] += alpha
        w = np.maximum(w, self.weight_min)
        w = w / max(1e-9, np.sum(w))
        self.w_model, self.w_temporal, self.w_semantic = [float(x) for x in w]

    def _on_drift_detected(self, action: str):
        print(f"[drift] detected for action {action}")
        info = self._drift_counts.setdefault(action, {"count": 0, "last_ts": 0})
        info["count"] += 1
        info["last_ts"] = time.time()
        # persist drift counts
        try:
            path = os.path.join(self._data_dir, 'drift_counts.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._drift_counts, f, indent=2)
        except Exception:
            pass
        # optionally nudge weights to rely a bit more on temporal/semantic when drift occurs
        if self.dynamic_weights:
            self._nudge_weights(toward='temporal', alpha=0.01)

    # --------------------------- helpers exposed ---------------------------
    def extract_features_from_text(self, text: str) -> Tuple[str, float, float, str, Dict[str, float]]:
        keywords = ', '.join(self._extract_keywords(text))
        vader = sentiment_analyzer.polarity_scores(text)
        polarity = float(vader.get('compound', 0.0))
        subjectivity = float(TextBlob(text).sentiment.subjectivity)
        action = self.extract_action(text)
        return keywords, polarity, subjectivity, action, {'polarity': polarity, 'subjectivity': subjectivity}

    def extract_action(self, text: str) -> str:
        m = re.search(r"(?:chance|probability|likelihood)\s+of\s+(.+?)(?:[\?\.!,$]|$)", text, flags=re.IGNORECASE)
        if m:
            phrase = m.group(1).strip()
            # retain common particles/objects like "meet with my sister", "study for the test"
            phrase = re.sub(r"^(me|my|us|our|you|your|the)\s+", "", phrase, flags=re.IGNORECASE)
            return normalize_action(phrase)
        d = nlp(text)
        root = next((t for t in d if t.dep_ == 'ROOT' and t.pos_ == 'VERB'), None)
        cand = root
        def heavy_child(tok):
            for ch in tok.children:
                if ch.pos_ == 'VERB' and ch.lemma_.lower() not in LIGHT_VERBS and ch.dep_ in {'xcomp','ccomp'}:
                    return ch
            return None
        if cand and cand.lemma_.lower() in LIGHT_VERBS:
            hc = heavy_child(cand)
            if hc:
                cand = hc
        if cand is None:
            cand = next((t for t in d if t.pos_ == 'VERB' and not t.is_stop), None)
        if cand is None:
            cand = next((t for t in d if t.pos_ == 'NOUN' and not t.is_stop), None)
        if cand is None:
            return ''
        negated = any(ch.dep_ == 'neg' for ch in cand.children)
        particle = next((ch.lower_ for ch in cand.children if ch.dep_ == 'prt'), None)
        action = f"{cand.lemma_.lower()}_{particle}" if particle else cand.lemma_.lower()
        if negated:
            action = f"not_{action}"
        return normalize_action(action)

    def summarize_action_context(self, action: str, max_lookback: int = 100) -> str:
        """
        Produce a short natural-language phrase for the given action based on recent logs.
        Heuristic: find the most recent entries with this action, extract salient noun chunks
        and keywords from the user_input, and attach common prepositional objects like
        "for <noun phrase>", "with <entity>", etc.
        """
        action_norm = normalize_action(action)
        # scan short memory
        texts = []
        for entry in self.short_memory[:max_lookback]:
            if normalize_action(entry.get('action','')) == action_norm:
                texts.append(entry.get('user_input',''))
                if len(texts) >= 5:
                    break
        if not texts:
            # fallback: just return the action verb as-is
            return action_norm.replace('_', ' ')
        # aggregate noun chunks and important tokens
        chunks = []
        for t in texts:
            try:
                d = nlp(t)
                for nc in d.noun_chunks:
                    chunks.append(nc.text.strip())
                # also include named entities
                for ent in d.ents:
                    chunks.append(ent.text.strip())
            except Exception:
                continue
        # pick the most frequent meaningful chunk
        if not chunks:
            return action_norm.replace('_', ' ')
        # score chunks by length and frequency
        from collections import Counter
        cnt = Counter([c.lower() for c in chunks if 2 <= len(c.split()) <= 5])
        if not cnt:
            cnt = Counter([c.lower() for c in chunks])
        phrase = next(iter(cnt.most_common(1)))[0]
        verb_phrase = action_norm.replace('_', ' ')
        # attach a preposition where it makes sense
        if any(w in phrase for w in ['test','exam','assignment','project','paper','final']):
            return f"{verb_phrase} for {phrase}"
        if any(w in phrase for w in ['sister','brother','mom','dad','friend','team','coworker']):
            return f"{verb_phrase} with {phrase}"
        if any(w in phrase for w in ['dinner','lunch','breakfast','movie','game']):
            return f"{verb_phrase} {phrase}"
        return f"{verb_phrase} {phrase}"

# End of predictor adaptive v3
