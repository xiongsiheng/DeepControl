from typing import Any, Dict, List, Optional
import requests
import json

import threading
import time




def _print_failed_request(self, url: str, payload: dict, resp=None, exc=None):
    print("\n===== RETRIEVER REQUEST FAILED =====")
    print("URL:", url)
    print("Payload:", json.dumps(payload, ensure_ascii=False))
    if resp is not None:
        print("Status:", resp.status_code)
        print("Response headers:", dict(resp.headers))
        try:
            print("Response body:", resp.json())
        except Exception:
            print("Response body (raw):", resp.text)
    if exc is not None:
        print("Exception:", repr(exc))
    print("===================================\n")


class EntryRetrieverClient:
    """
    Adapter client aligned to the current retriever server.

    Server API:
      POST /retrieve
        {
          "queries": [str],
          "topk": int,
          "return_scores": bool
        }
        -> {
          "result": [
            [
              {
                "doc_id": str,     # ✅ corpus row idx
                "title": str,
                "snippet": str,
                "orig_id": str,    # optional
                "score": float
              }
            ]
          ]
        }

      POST /expand
        {
          "doc_ids": [str]      # corpus row idx
        }
        -> {
          "expanded": [
            {
              "doc_id": str,
              "contents": str
            }
          ]
        }
    """

    def __init__(self, base_url: str, topk: int = 8, timeout_s: int = 30, num_attempts: int = 1):
        self.base_url = base_url.rstrip("/")
        self.topk = topk
        self.timeout_s = timeout_s
        self.num_attempts = num_attempts

    def retrieve_entries(
        self,
        query: str,
        topk: Optional[int] = None,
        return_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        payload = {
            "queries": [query],
            "topk": int(topk or self.topk),
            "return_scores": bool(return_scores),
        }
        url = f"{self.base_url}/retrieve"

        for attempt in range(self.num_attempts):
          try:
            r = requests.post(f"{self.base_url}/retrieve", json=payload, timeout=self.timeout_s)
          
            r.raise_for_status()

            data = r.json()
            batch = data.get("result", [])
            if not batch:
                return []

            entries = []
            for e in batch[0]:
                item = {
                    "doc_id": e.get("doc_id"),
                    "title": e.get("title", ""),
                    "snippet": e.get("snippet", ""),
                    "full_text": e.get("full_text", "")
                }
                # optional debug field
                if "orig_id" in e:
                    item["orig_id"] = e.get("orig_id", "")

                if return_scores and "score" in e:
                    item["score"] = e["score"]
                entries.append(item)

            return entries
          
          except requests.HTTPError as e:
              self._print_failed_request(
                  url,
                  payload,
                  resp=e.response,
                  exc=e,
              )
              wait = 2 ** attempt
              time.sleep(wait)

          except Exception as e:
              self._print_failed_request(
                  url,
                  payload,
                  exc=e,
              )
              wait = 2 ** attempt
              time.sleep(wait)
        
        # after loop
        return []


    def expand(
        self,
        doc_ids: List[str],
        max_chunks_per_doc: int = 1,  # server does not support chunk, interface retained but ignored
    ) -> Dict[str, List[Dict[str, Any]]]:
        payload = {"doc_ids": doc_ids}
        url = f"{self.base_url}/expand"
        for attempt in range(self.num_attempts):
          try:
            r = requests.post(f"{self.base_url}/expand", json=payload, timeout=self.timeout_s)
                    
            r.raise_for_status()

            raw = r.json().get("expanded", [])
            expanded: Dict[str, List[Dict[str, Any]]] = {}
            for item in raw:
                doc_id = item.get("doc_id")
                contents = item.get("contents", "")
                expanded[doc_id] = [{"section": "full", "text": contents}]
            return expanded

          except requests.HTTPError as e:
              self._print_failed_request(
                  url,
                  payload,
                  resp=e.response,
                  exc=e,
              )
              wait = 2 ** attempt
              time.sleep(wait)

          except Exception as e:
              self._print_failed_request(
                  url,
                  payload,
                  exc=e,
              )
              wait = 2 ** attempt
              time.sleep(wait)

        # after loop
        return {}


    @staticmethod
    def format_search_results(entries: List[Dict[str, Any]], max_snippet_chars: int = 220) -> str:
        if len(entries) == 0:
            return 'None'
        lines = []
        for i, e in enumerate(entries, start=1):
            doc_id = e.get("doc_id", "")
            title = e.get("title", "")
            snippet = e.get("snippet", "") or ""

            if len(snippet) > max_snippet_chars:
                snippet = snippet[:max_snippet_chars] + "..."
            else:
                snippet = snippet + " ..."

            score = e.get("score", None)
            score_str = f" score={score:.4f}" if isinstance(score, (int, float)) else ""

            orig_id = e.get("orig_id", "")
            orig_str = f" orig_id={orig_id}" if orig_id else ""

            lines.append(
                f"Rank={i} doc_id={doc_id}{score_str}\n"
                f"Title: {title}\n"
                f"Snippet: {snippet}"
            )

        return "\n\n".join(lines).strip()

    @staticmethod
    def format_information(expanded: Dict[str, List[Dict[str, Any]]], max_chunk_chars: int = 1200) -> str:
        if len(expanded) == 0:
            return 'None'
        blocks = []
        for doc_id, chunks in expanded.items():
            blocks.append(f"[DOC {doc_id}]")
            for ch in chunks:
                text = (ch.get("text", "") or "").strip()
                blocks.append(text)
        return "\n".join(blocks).strip()