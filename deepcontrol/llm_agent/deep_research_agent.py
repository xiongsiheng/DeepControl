import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ----------------------------
# Action payload schemas
# ----------------------------

@dataclass
class ExpandInformationAction:
    doc_ids: List[str]
    max_chunks_per_doc: int

class DeepResearchAgent:
    """
    Protocol helper aligned to the prefix.

    Allowed model output per turn:
      Optional:
        <think>...</think>
      Then EXACTLY ONE action tag:
        - <search>...</search>
        - <expand>{"doc_ids":[...], "max_chunks_per_doc": ...}</expand>
        - <answer>...</answer>
    """

    # priority for detection (answer first)
    END_TAGS = [
        ("answer", "answer"),
        ("expand", "expand"),
        ("search", "search"),
    ]

    # -----------------------
    # Regex: THINK
    # -----------------------
    THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.S)

    # -----------------------
    # Regex: ACTION blocks (for locating full closed blocks)
    # - used in truncate_to_one_action() to find and re-compose output
    # -----------------------
    ACTION_BLOCK_PATTERNS = {
        "answer": re.compile(r"<answer>.*?</answer>", flags=re.S),
        # If you want to allow non-JSON inside expand tags, relax these patterns.
        # Keeping JSON-ish constraint reduces accidental matches.
        "expand": re.compile(
            r"<expand>\s*\{.*?\}\s*</expand>", flags=re.S
        ),
        "search": re.compile(r"<search>.*?</search>", flags=re.S),
    }

    # -----------------------
    # Regex: ACTION content extractors (fullmatch after normalization)
    # - used in detect_action() to extract inner content reliably
    # -----------------------
    ACTION_PATTERNS = {
        "answer": re.compile(r"^\s*<answer>(.*?)</answer>\s*$", flags=re.S),
        "expand": re.compile(
            r"^\s*<expand>(.*?)</expand>\s*$", flags=re.S
        ),
        "search": re.compile(r"^\s*<search>(.*?)</search>\s*$", flags=re.S),
    }

    # -----------------------
    # Truncation / normalization
    # -----------------------
    @staticmethod
    def truncate_to_one_action(text: str) -> str:
        """
        Normalize model output to:
          [optional think-block]
          exactly one CLOSED action tag-block
        Discard any other junk text before/between/after.

        Strategy:
          1) Find earliest CLOSED action block among all action types.
          2) Find the last think-block that ends BEFORE that action block starts.
          3) Return (think + newline + action) or just action.
        """
        if not text:
            return text

        s = text.strip()

        # 1) find earliest action block (by start index)
        best_action = None  # (start, end, action_str)
        for name, _ in DeepResearchAgent.END_TAGS[::-1]:
            # END_TAGS is "answer first" for detection; for locating earliest action,
            # we don't want priority; we want earliest start. We'll just scan all.
            pass

        # Scan all patterns to get earliest start
        for action_name, pat in DeepResearchAgent.ACTION_BLOCK_PATTERNS.items():
            m = pat.search(s)
            if not m:
                continue
            st, ed = m.start(), m.end()
            if best_action is None or st < best_action[0]:
                best_action = (st, ed, m.group(0).strip())

        if best_action is None:
            # no closed action tag found => return trimmed raw (driver will mark invalid)
            return s

        a_start, a_end, action_str = best_action

        # 2) find think-block: last one that ends before action starts
        think_str = ""
        last_end = -1
        for m in DeepResearchAgent.THINK_BLOCK_RE.finditer(s):
            if m.end() <= a_start and m.end() > last_end:
                think_str = m.group(0).strip()
                last_end = m.end()

        # 3) compose canonical output
        if think_str:
            return f"{think_str}\n{action_str}".strip()
        return action_str

    # -----------------------
    # Action detection
    # -----------------------
    @staticmethod
    def detect_action(text: str) -> Tuple[Optional[str], str]:
        """
        Returns (action_type, content) or (None, "").

        Expectation: caller already ran truncate_to_one_action(), but this function
        remains robust:
          - strip optional leading think-block (only if it is at the very beginning)
          - then require the remainder to FULLMATCH exactly one action tag
        """
        if not text:
            return None, ""

        s = text.strip()

        # Strip leading think block if present at start
        m_think = DeepResearchAgent.THINK_BLOCK_RE.match(s)
        if m_think:
            s = s[m_think.end():].strip()

        # Now s should be exactly one action tag
        for action_name, _ in DeepResearchAgent.END_TAGS:
            pat = DeepResearchAgent.ACTION_PATTERNS[action_name]
            m = pat.match(s)
            if m:
                return action_name, (m.group(1) or "").strip()

        return None, ""

    # -----------------------
    # JSON parsing helpers
    # -----------------------
    @staticmethod
    def parse_expand_json(
        raw: str,
        allowed_doc_ids: Optional[set] = None,
        default_max_chunks: int = 1,
        clamp_max_chunks: Tuple[int, int] = (1, 3),
    ) -> ExpandInformationAction:
        doc_ids: List[str] = []
        max_chunks = default_max_chunks

        # 1) strict JSON
        try:
            obj = json.loads(raw)
            doc_ids = obj.get("doc_ids", None)
            if doc_ids is None and "doc_id" in obj:
                # Compatibility: some models emit {"doc_id": "..."}.
                doc_ids = obj.get("doc_id")
            if doc_ids is None:
                doc_ids = []
            if "max_chunks_per_doc" in obj:
                max_chunks = int(obj["max_chunks_per_doc"])
        except Exception:
            # 2) relaxed fallback
            doc_ids = re.findall(r'"([^"]+)"', raw)
            if not doc_ids:
                doc_ids = re.findall(r"\b\d+\b", raw)

            mc = re.search(r"max_chunks_per_doc\s*[:=]\s*(\d+)", raw)
            if mc:
                max_chunks = int(mc.group(1))

        # --- normalize doc_ids ---
        if isinstance(doc_ids, (str, int)):
            doc_ids = [doc_ids]
        elif not isinstance(doc_ids, list):
            doc_ids = []

        doc_ids = [str(d).strip() for d in doc_ids if str(d).strip()]
        lo, hi = clamp_max_chunks
        max_chunks = max(lo, min(hi, int(max_chunks)))

        if allowed_doc_ids is not None:
            allowed_doc_ids = {str(d) for d in allowed_doc_ids}
            doc_ids = [d for d in doc_ids if d in allowed_doc_ids]

        return ExpandInformationAction(doc_ids=doc_ids, max_chunks_per_doc=max_chunks)
