import os
import re
import time
import json
import uuid
import string
import argparse
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from openai import OpenAI  # pip install openai
from datasets import load_dataset

from deepcontrol.llm_agent.deep_research_agent import DeepResearchAgent
from deepcontrol.llm_agent.entry_retriever import EntryRetrieverClient


import multiprocessing as mp
from tqdm import tqdm


_RETRIEVER_SEM = None

def _init_worker(retriever_sem):
    global _RETRIEVER_SEM
    _RETRIEVER_SEM = retriever_sem



prefix = """Answer the given question.

You MUST follow the protocol below.

CONTROL
- A control message may appear anywhere in the conversation in the form:
  <control>...</control>
- You MUST follow the <control> message that appears in the context.


General rules
- Whenever you receive NEW information (from <search_results>, <information>), you MUST first reason inside <think>...</think>.
- You can call a search engine using: <search>query</search>.
  The environment will return snippets inside: <search_results>...</search_results>.
- If you want full text, you MUST decide inside <think>...</think>, then request expansion using: <expand>{"doc_ids": [id1, id2, ...]}</expand>
  The environment will return the expanded full text inside: <information>...</information>. You can expand multiple documents in one call by listing multiple doc_ids.
- If no further external knowledge is needed, output the final answer inside <answer>...</answer>.


Answer normalization rules (VERY IMPORTANT)
- The final answer MUST EXACTLY match the canonical short answer.
- Output the SHORTEST possible answer span.
- Do NOT add explanations, appositives, or parentheses.
- Do NOT add extra words, punctuation, or formatting.
- Use the most common name form that appears as a standalone answer.
- If multiple aliases exist, choose the most standard short form.
- Case-sensitive matching is required.

Examples:
Q: how many episodes are in series 7 game of thrones?
Correct: <answer>seven</answer>

Q: when does season 5 of bates motel come out?
Correct: <answer>February 20, 2017</answer>


Round definition
A round MUST be one of the following two sequences:

1) Answering round:
   <think>...</think>
   <search>...</search>
   <search_results>...</search_results>
   <think>...</think>
   <expand>...</expand>
   <information>...</information>
   <think>...</think>
   <answer>...</answer>

2) Continuing round:
   <think>...</think>
   <search>...</search>
   <search_results>...</search_results>
   <think>...</think>
   <expand>...</expand>
   <information>...</information>
   <think>...</think>

You may perform as many rounds as needed.

"""


def my_gpt_completion(
    model_name,
    messages,
    timeout=120,
    max_tokens=256,
    wait_time=0,
    temperature=0.7,
    api_key=None,
    source=None,
    base_url=None,
    verbose=False,
) -> str:
    if verbose:
        for m in messages:
            print(m["role"], ":", m["content"])
        print("-" * 20)


    if source == "local":
        if verbose:
            print("Using local model inference...")
            print("-" * 20)
        
        client = OpenAI(
            api_key="EMPTY",
            base_url=base_url or "http://127.0.0.1:8001/v1"
        )

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )        

        response = completion.choices[0].message.content

        time.sleep(wait_time)
        if verbose:
            print("response:\n", response)
            print("=" * 20)
        return response


    elif source == "deepseek":
        base_api_url = "https://api.deepseek.com"
        if api_key is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not set and api_key not provided.")

    elif source == "together":
        base_api_url = "https://api.together.xyz/v1/"
        if api_key is None:
            api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not set and api_key not provided.")

    elif source == "aliyun":
        base_api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if api_key is None:
            api_key = os.environ.get("ALIYUN_API_KEY")
        if not api_key:
            raise ValueError("ALIYUN_API_KEY not set and api_key not provided.")
        if messages[0]['role'] != 'system':
            messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}] + messages

    elif source == "openai":
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set and api_key not provided.")

    else:
        raise ValueError(f"Invalid source: {source}")

    client = OpenAI(api_key=api_key)
    if source in ("deepseek", "together", "aliyun"):
        client.base_url = base_api_url
    
    kwargs = {
        "model": model_name,
        "messages": messages,
        "timeout": timeout,
    }

    if not model_name.startswith("gpt-5"):
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens

    completion = client.chat.completions.create(**kwargs)

    response = completion.choices[0].message.content or ""

    time.sleep(wait_time)
    if verbose:
        print("response:\n", response)
        print("=" * 20)
    return response




# -----------------------------
# JSONL utilities
# -----------------------------
def write_jsonl_append(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def now_ts():
    """
    Returns:
      {
        "ts_epoch": float,        # seconds since epoch
        "ts_readable": str        # human-readable (English, ISO-like)
      }
    """
    now = datetime.now()
    return {
        "ts_epoch": now.timestamp(),
        "ts_readable": now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
        # keep 2 decimal places for seconds
    }





def load_done_qids(episode_jsonl: str) -> Tuple[set, int, int]:
    """
    Resume helper:
      - returns (done_qids, done_correct, done_incorrect)
    Policy (default):
      - if a line has "qid", we treat it as done (skip next run) to avoid duplicates.
    """
    done = set()
    ok = 0
    bad = 0
    if not episode_jsonl or (not os.path.exists(episode_jsonl)):
        return done, ok, bad

    with open(episode_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            qid = obj.get("qid")
            if qid is None:
                continue
            done.add(qid)
            if obj.get("final_correct") is True:
                ok += 1
            else:
                bad += 1
    return done, ok, bad


def load_qid_keywords(path: str) -> Dict[str, str]:
    """
    Load qid -> keyword mapping from jsonl.
    """
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            qid = obj.get("qid")
            kw = obj.get("keywords")
            paras = obj.get("paragraphs")
            if qid and kw:
                mapping[qid] = (kw, paras)
    return mapping


def find_doc_id_by_keyword(search_results: str, keyword: str) -> int:
    for m in re.finditer(
        r'Rank=\d+\s+doc_id=(\d+).*?\nTitle:\s*([^\n]+)',
        search_results,
        flags=re.DOTALL
    ):
        doc_id = int(m.group(1))
        title = m.group(2).strip()
        if title == keyword:
            return doc_id
    return -1


def load_verified_corpus(path: str) -> Dict[str, Dict[str, str]]:
    """
    Load verified corpus:
      qid -> {
        "paragraph": str,
        "simulated_cot": str
      }

    Only keep samples where correct == true and raw_output exists.
    """
    data = {}
    if not os.path.exists(path):
        return data

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("correct") is not True:
                continue

            qid = obj.get("qid")
            paragraph = obj.get("paragraph")
            raw_output = obj.get("raw_output")

            if not qid or not paragraph or not raw_output:
                continue

            data[qid] = {
                "paragraph": paragraph,
                "simulated_cot": raw_output,
            }

    return data



# -----------------------------
# QA correctness
# -----------------------------
def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def is_correct_nq(pred: str, gold_list: Any) -> bool:
    pred_n = normalize_answer(pred)
    if not pred_n:
        return False
    if gold_list is None:
        return False
    if isinstance(gold_list, (str, bytes)):
        gold_list = [gold_list]
    if not isinstance(gold_list, list):
        gold_list = list(gold_list)

    for g in gold_list:
        g_n = normalize_answer(g)
        if not g_n:
            continue
        if pred_n == g_n:
            return True
    return False


# -----------------------------
# Main driver
# -----------------------------
def run_tool_episode(
    retriever_url: str,
    question: str,
    golden_answers: Any,
    model: str,
    source: str,
    base_url: str,
    api_key: Optional[str] = None,
    question_index: Optional[int] = None,
    qid: Optional[str] = None,
    topk: int = 5,
    max_turns: int = 8,
    temperature: float = 0.6,
    max_tokens: int = 256,
    verbose: bool = False,
    episode_id: Optional[str] = None,
    retriever_timeout_s: int = 120,
) -> Dict[str, Any]:
    """
    Protocol-aligned driver (strict FSM).

    FSM states (per round):
      NEED_SEARCH
        - allow: search
      NEED_EXPAND
        - allow: expand
      NEED_ANSWER
        - allow: answer
    """
    episode_id = episode_id or str(uuid.uuid4())

    # episode-level trajectory
    traj_steps: List[Dict[str, Any]] = []

    # purely for logging / analysis, NOT exposed to agent
    retrieval_log = []   # List[Dict]

    agent = DeepResearchAgent()

    global _RETRIEVER_SEM
    retriever = EntryRetrieverClient(base_url=retriever_url, topk=topk, timeout_s=retriever_timeout_s)

    messages: List[Dict[str, str]] = [{"role": "user", "content": f"{prefix}\n\nQuestion: {question}\n"}]

    allowed_doc_ids: Optional[set] = None

    # round_id 1-based
    current_round_id = 1

    seen = {"search": 0, "expand": 0,  "answer": 0}

    NEED_SEARCH = "NEED_SEARCH"
    NEED_EXPAND = "NEED_EXPAND"
    NEED_ANSWER = "NEED_ANSWER"
    state = NEED_SEARCH

    def _invalid(reason: str):
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Invalid action: {reason}\n"
                    "Output exactly ONE of:\n"
                    "<search>...</search>\n"
                    "<expand>{\"doc_ids\":[...]}</expand>\n"
                    "<answer>...</answer>\n"
                ),
            }
        )

    final_answer = None
    final_correct = None

    for turn in range(max_turns):
        if turn == max_turns - 2:  # Force to end
            messages.append({
                "role": "user",
                "content": '<control>Stop searching.</control>'
            })

        raw = my_gpt_completion(
            model_name=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            source=source,
            base_url=base_url,
            verbose=verbose,
        )
        raw = agent.truncate_to_one_action(raw)
        
        act, content = agent.detect_action(raw)

        # record assistant output in conversation
        messages.append({"role": "assistant", "content": raw})

        # basic validity
        if act is None:
            _invalid("No valid action tag found.")
            continue
        if act not in seen:
            _invalid(f"Unknown action '{act}'.")
            continue

        # enforce FSM
        if state == NEED_SEARCH:
            if act == "expand":
                _invalid("You must <search> before <expand> in a round.")
                continue

        # passed checks => update trajectory state
        seen[act] += 1


        # ---- SEARCH ----
        if act == "search":
            if _RETRIEVER_SEM is not None:
                _RETRIEVER_SEM.acquire()
            try:
                entries = retriever.retrieve_entries(content, topk=topk, return_scores=True)
            finally:
                if _RETRIEVER_SEM is not None:
                    _RETRIEVER_SEM.release()

            # allowed_doc_ids = {e["doc_id"] for e in entries if e.get("doc_id") is not None}
            allowed_doc_ids = {str(e["doc_id"]) for e in entries if e.get("doc_id") is not None}

            sr = retriever.format_search_results(entries)
            obs_text = f"<search_results>\n{sr}\n</search_results>"
            messages.append({"role": "user", "content": obs_text})

            ts = now_ts()
            traj_steps.append(
                {
                    "episode_id": episode_id,
                    "turn": turn,
                    "round_id": current_round_id,
                    "state_before": NEED_SEARCH,
                    "action_type": "search",
                    "action": raw,
                    "query": content,
                    "observation": obs_text,
                    "topk": topk,
                    "ts_epoch": ts["ts_epoch"],
                    "ts_readable": ts["ts_readable"],
                }
            )

            for e in entries:
                retrieval_log.append({
                    "episode_id": episode_id,
                    "turn": turn,
                    "round_id": current_round_id,
                    "query": content,
                    "doc_id": str(e.get("doc_id")),
                    "title": e.get("title"),
                    "snippet": e.get("snippet"),
                    "score": e.get("score"),
                    "full_text": e.get("full_text", ""),
                    "expanded": False,
                })

            state = NEED_EXPAND
            continue

        # ---- EXPAND ----
        if act == "expand":
            if allowed_doc_ids is None:
                _invalid("No prior search_results in this round.")
                continue

            exp = agent.parse_expand_json(content, allowed_doc_ids=allowed_doc_ids)

            expanded = None
            if not exp.doc_ids:
                obs = "<information>\nNone\n</information>"
            else:
                if _RETRIEVER_SEM is not None:
                    _RETRIEVER_SEM.acquire()
                try:
                    expanded = retriever.expand(
                        exp.doc_ids,
                        max_chunks_per_doc=exp.max_chunks_per_doc
                    )
                finally:
                    if _RETRIEVER_SEM is not None:
                        _RETRIEVER_SEM.release()

                info = retriever.format_information(expanded)
                obs = f"<information>\n{info}\n</information>"

            messages.append({"role": "user", "content": obs})

            ts = now_ts()
            traj_steps.append(
                {
                    "episode_id": episode_id,
                    "turn": turn,
                    "round_id": current_round_id,
                    "state_before": NEED_EXPAND,
                    "action_type": "expand",
                    "action": raw,
                    "doc_ids": exp.doc_ids,
                    "max_chunks_per_doc": exp.max_chunks_per_doc,
                    "observation": obs,
                    "ts_epoch": ts["ts_epoch"],
                    "ts_readable": ts["ts_readable"],
                }
            )

            expanded_doc_ids = set(map(str, expanded.keys())) if expanded else set()

            for item in retrieval_log:
                if item["doc_id"] in expanded_doc_ids:
                    item["expanded"] = True

            state = NEED_ANSWER
            continue


        # ---- ANSWER ----
        if act == "answer":
            final_answer = (content or "").strip()
            final_correct = is_correct_nq(final_answer, golden_answers)

            ts = now_ts()
            traj_steps.append(
                {
                    "episode_id": episode_id,
                    "turn": turn,
                    "round_id": current_round_id,
                    "state_before": NEED_ANSWER,
                    "action_type": "answer",
                    "action": raw,
                    "final_answer": final_answer,
                    "final_correct": bool(final_correct),
                    "golden_answers": golden_answers,
                    "ts_epoch": ts["ts_epoch"],
                    "ts_readable": ts["ts_readable"],
                }
            )
            break

    # basic sanity
    ok = True
    if seen["expand"] > seen["search"]:
        ok = False

    return {
        "ok": ok,
        "seen": seen,
        "episode_id": episode_id,
        "final_answer": final_answer,
        "final_correct": final_correct,
        "traj_steps": traj_steps,
        "retrieval_log": retrieval_log,
        "messages": messages,
    }



def _worker_run_one(args):
    """
    Worker process: run one question and return one episode row.
    """
    (
        i,
        qid,
        question,
        golden_answers,
        retriever_url,
        model,
        source,
        base_url,
        api_key,
        topk,
        max_turns,
        temperature,
        max_tokens,
        verbose,
        retriever_timeout_s,
    ) = args

    try:
        report = run_tool_episode(
            retriever_url=retriever_url,
            question=question,
            golden_answers=golden_answers,
            model=model,
            source=source,
            base_url=base_url,
            api_key=api_key,
            topk=topk,
            max_turns=max_turns,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            question_index=i,
            qid=qid,
            episode_id=str(uuid.uuid4()),
            retriever_timeout_s=retriever_timeout_s,
        )
        err = None
    except Exception as e:
        report = {
            "ok": False,
            "seen": {},
            "episode_id": str(uuid.uuid4()),
            "final_answer": None,
            "final_correct": None,
            "traj_steps": [],
        }
        err = repr(e)


    ts = now_ts()
    episode_row = {
        "episode_id": report["episode_id"],
        "index": i,
        "qid": qid,
        "question": question,
        "golden_answers": golden_answers,
        "model": model,
        "source": source,
        "final_answer": report["final_answer"],
        "final_correct": bool(report["final_correct"]) if report["final_correct"] is not None else None,
        "seen": report["seen"],
        "ok": report["ok"],
        "traj_steps": report["traj_steps"],
        "retrieval_log": report.get("retrieval_log", []),
        "error": err,
        "ts_epoch": ts["ts_epoch"],
        "ts_readable": ts["ts_readable"],
    }

    return episode_row


def _writer_process(queue, out_jsonl: str):
    """
    Single writer process: consume episode rows from queue and append to JSONL files.
    """
    while True:
        item = queue.get()
        if item is None:
            break
        write_jsonl_append(out_jsonl, [item])



# -----------------------------
# Dataset loop
# -----------------------------
def run_over_dataset(
    dataset_repo_path: str,
    config_name: str,
    split: str,
    retriever_url: str,
    model: str,
    source: str,
    base_url: str,
    api_key: Optional[str],
    out_jsonl: str,
    max_samples: Optional[int] = None,
    topk: int = 10,
    max_turns: int = 20,
    temperature: float = 0.6,
    max_tokens: int = 256,
    verbose: bool = False,
    num_workers: int = 8,
    retriever_timeout_s: int = 120,
    retriever_max_concurrency: int = 16,
    resume: bool = False,
):
    ds = load_dataset(dataset_repo_path, config_name)[split]

    n = len(ds) if max_samples is None else min(len(ds), max_samples)
    print(f"[INFO] split={split}, total={len(ds)}, running={n}")

 
    done_qids = set()
    done_ok = 0
    done_fail = 0

    if os.path.exists(out_jsonl):
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue

                qid = row.get("qid")
                if qid is None:
                    continue

                if row.get("final_correct") is True:
                    done_ok += 1
                else:
                    done_fail += 1
                done_qids.add(qid)

    done_cnt = len(done_qids)

    # -------- prepare jobs --------
    jobs = []
    for i in range(n):
        ex = ds[i]
        qid = ex["id"]
        if resume and (qid in done_qids):
            continue
        q = (ex["question"] or "").strip()
        
        if q and q[-1] != "?":
            q += "?"
        
        gold = ex.get("golden_answers")
        if gold is None:
            gold = ex.get("answer")

        if verbose:
            print('[DEBUG]', i, qid, q, gold)

        jobs.append(
            (
                i,
                qid,
                q,
                gold,
                retriever_url,
                model,
                source,
                base_url,
                api_key,
                topk,
                max_turns,
                temperature,
                max_tokens,
                verbose,
                retriever_timeout_s,
            )
        )

    print(f"[INFO] Prepared {len(jobs)} jobs.")

    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=64)
    
    retriever_sem = ctx.Semaphore(retriever_max_concurrency)  # For example, 8~16

    writer = ctx.Process(
        target=_writer_process,
        args=(queue, out_jsonl),
    )
    writer.start()

    # retriever timeout counters (based on episode_row["error"] string)
    retr_timeout_cnt = 0
    retr_conn_timeout_cnt = 0
    retr_other_timeout_cnt = 0

    # start counts from already-done so postfix shows global stats
    ok_cnt = done_ok if resume else 0
    # fail_cnt = done_fail if resume else 0
    fail_cnt = 0

    start = time.time()

    # total must be full n, initial is already done
    with tqdm(total=n, initial=done_cnt, desc=f"{config_name} episodes", dynamic_ncols=True) as pbar:
        with ctx.Pool(processes=num_workers, initializer=_init_worker, initargs=(retriever_sem,)) as pool:
            for episode_row in pool.imap_unordered(_worker_run_one, jobs):
                queue.put(episode_row)

                # ---- timeout stats ----
                err = (episode_row.get("error") or "")
                if err:
                    # common retriever errors you saw:
                    # requests.exceptions.ReadTimeout / ConnectTimeout, urllib3 ReadTimeoutError, TimeoutError
                    if ("ReadTimeout" in err) or ("ReadTimeoutError" in err):
                        retr_timeout_cnt += 1
                    elif ("ConnectTimeout" in err) or ("ConnectTimeoutError" in err):
                        retr_conn_timeout_cnt += 1
                    elif ("TimeoutError" in err) or ("timed out" in err):
                        retr_other_timeout_cnt += 1

                # 1) Update count
                if episode_row.get("final_correct") is True:
                    ok_cnt += 1
                else:
                    fail_cnt += 1

                # 2) Update progress bar
                pbar.update(1)

                # 3) Update display (no need to update every time)
                if pbar.n % 100 == 0 or pbar.n == n:
                    elapsed = time.time() - start
                    pbar.set_postfix(
                        correct=ok_cnt,
                        incorrect=fail_cnt,
                        r_to=retr_timeout_cnt,
                        r_cto=retr_conn_timeout_cnt,
                        r_oto=retr_other_timeout_cnt,
                        # ep_per_min=f"{pbar.n / elapsed * 60:.1f}"
                    )

    queue.put(None)
    writer.join()






# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    # mode
    parser.add_argument("--mode", type=str, default="single", choices=["single", "dataset"])

    # model/retriever
    parser.add_argument("--retriever_url", type=str, default=os.environ.get("RETRIEVER_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--model", type=str)
    parser.add_argument("--source", type=str, choices=["openai", "together", "deepseek", "aliyun", "local"])
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)

    # episode params
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--verbose", action="store_true")

    # single question
    parser.add_argument("--question", type=str, default="Who won the first Nobel Prize in Physics?")
    parser.add_argument("--gold", type=str, nargs="*", default=["Wilhelm Conrad Röntgen", "Wilhelm Rontgen", "Röntgen"])

    # dataset params
    parser.add_argument("--dataset_repo_path", type=str, default="FlashRAG_datasets")
    parser.add_argument("--config_name", type=str, default="nq")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=None)

    # retriever
    parser.add_argument("--retriever_max_concurrency", type=int, default=16)
    parser.add_argument("--retriever_timeout_s", type=int, default=120)
    
    # running options
    parser.add_argument("--resume", action="store_true", help="resume from out_jsonl by skipping done qids")
    parser.add_argument("--num_workers", type=int, default=16)

    # outputs
    parser.add_argument("--out_jsonl", type=str, default="results/infer_fast.jsonl")

    args = parser.parse_args()


    if args.mode == "single":
        report = run_tool_episode(
            retriever_url=args.retriever_url,
            question=args.question,
            golden_answers=args.gold,
            model=args.model,
            source=args.source,
            base_url=args.base_url,
            api_key=args.api_key,
            topk=args.topk,
            max_turns=args.max_turns,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
            episode_id=str(uuid.uuid4()),
        )

        ts = now_ts()
        write_jsonl_append(
            args.out_jsonl,
            [
                {
                    "episode_id": report["episode_id"],
                    "split": "single",
                    "index": 0,
                    "question": args.question,
                    "golden_answers": args.gold,
                    "model": args.model,
                    "source": args.source,
                    "final_answer": report["final_answer"],
                    "final_correct": bool(report["final_correct"]) if report["final_correct"] is not None else None,
                    "seen": report["seen"],
                    "ok": report["ok"],
                    "traj_steps": report["traj_steps"],
                    "ts_epoch": ts["ts_epoch"],
                    "ts_readable": ts["ts_readable"],
                }
            ],
        )

        print(report["ok"], report["seen"], "episode_id=", report["episode_id"], "final_correct=", report["final_correct"])
        if not report["ok"]:
            for m in report["messages"]:
                print(m["role"], ":", m["content"])

    else:
        run_over_dataset(
            dataset_repo_path=args.dataset_repo_path,
            config_name=args.config_name,
            split=args.split,
            retriever_url=args.retriever_url,
            model=args.model,
            source=args.source,
            base_url=args.base_url,
            api_key=args.api_key,
            out_jsonl=args.out_jsonl,
            max_samples=args.max_samples,
            topk=args.topk,
            max_turns=args.max_turns,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
            num_workers=args.num_workers,
            retriever_timeout_s=args.retriever_timeout_s,
            retriever_max_concurrency=args.retriever_max_concurrency,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()