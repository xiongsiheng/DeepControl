import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import string
from typing import Any, Dict, List, Optional, Set, Tuple

from datasets import load_dataset
from openai import OpenAI
import requests
from tqdm import tqdm


PROMPT_TEMPLATE = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

INFORMATION_TEMPLATE = "<information>{search_results}</information>\n"

INVALID_ACTION_TEMPLATES = [
    "My previous action is invalid. If I want to search, I should put the query between <search> and </search>",
    "If I want to give the final answer, I should put the answer between <answer> and </answer>",
    "If I want to give final answer, I should put the final answer between <answer> and </answer>"
]


def normalize_question(question: str) -> str:
    question = (question or "").strip()
    if question and question[-1] != "?":
        question += "?"
    return question


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_correct(prediction: Optional[str], gold: Any) -> Optional[bool]:
    if gold is None:
        return None

    pred_norm = normalize_answer(prediction or "")
    if not pred_norm:
        return False

    if isinstance(gold, (str, bytes)):
        gold_list = [gold]
    elif isinstance(gold, list):
        gold_list = gold
    else:
        gold_list = list(gold)

    for item in gold_list:
        if pred_norm == normalize_answer(str(item)):
            return True
    return False


def extract_last_closed_tag(text: str, tag: str) -> Optional[str]:
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


def contains_invalid_action_template(text: str) -> bool:
    return any(s in text for s in INVALID_ACTION_TEMPLATES)


def get_query(text: str) -> Optional[str]:
    if contains_invalid_action_template(text):
        return None
    return extract_last_closed_tag(text, "search")


def get_answer(text: str) -> Optional[str]:
    if contains_invalid_action_template(text):
        return None
    return extract_last_closed_tag(text, "answer")


def build_client(source: str, api_key: Optional[str], base_url: Optional[str]) -> OpenAI:
    if source == "local":
        return OpenAI(api_key="EMPTY", base_url=base_url or "http://127.0.0.1:8001/v1")
    else:
        raise ValueError(f"Unsupported source: {source} for single turn prefill mode")


def merge_continuation(existing: str, new_text: str) -> str:
    if not existing:
        return new_text or ""
    if not new_text:
        return existing

    if new_text.startswith(existing):
        return new_text

    max_overlap = min(len(existing), len(new_text))
    for k in range(max_overlap, 0, -1):
        if existing[-k:] == new_text[:k]:
            return existing + new_text[k:]

    return existing + new_text


def truncate_after_last_answer(text: str) -> str:
    matches = list(re.finditer(r"</answer>", text, flags=re.DOTALL))
    if not matches:
        return text
    end = matches[-1].end()
    return text[:end]


def api_chat_completion(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    verbose: bool,
    continue_final_message: bool = False,
) -> str:
    if verbose:
        print("[Messages]")
        for i, msg in enumerate(messages):
            print(f"[{i}] role={msg['role']}")
            print(msg["content"])
            print("-" * 20)

    extra_body: Dict[str, Any] = {}
    if continue_final_message:
        extra_body["continue_final_message"] = True
        extra_body["add_generation_prompt"] = False

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=[
            "</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n",
            "</answer>", " </answer>", "</answer>\n", " </answer>\n", "</answer>\n\n", " </answer>\n\n",
        ],
        extra_body=extra_body if extra_body else None,
    )

    content = response.choices[0].message.content or ""

    if verbose:
        print("[Raw Response]")
        print(content)
        print("=" * 20)

    return content


def normalize_retriever_url(retriever_url: str) -> str:
    retriever_url = retriever_url.rstrip("/")
    if retriever_url.endswith("/retrieve"):
        return retriever_url
    return f"{retriever_url}/retrieve"


def retrieve_information(
    retriever_url: str,
    query: str,
    topk: int
) -> str:
    payload = {
        "queries": [query],
        "topk": topk,
        "return_scores": True,
    }
    response = requests.post(normalize_retriever_url(retriever_url), json=payload)
    response.raise_for_status()

    results = response.json().get("result", [])
    if not results:
        return "None"

    retrieval_result = results[0]
    if not retrieval_result:
        return "None"

    formatted = []
    for idx, doc_item in enumerate(retrieval_result, start=1):
        content = doc_item.get("document", {}).get("contents", "")
        if not content:
            continue
        lines = content.split("\n")
        title = lines[0] if lines else ""
        text = "\n".join(lines[1:]) if len(lines) > 1 else ""
        formatted.append(f"Doc {idx}(Title: {title}) {text}".rstrip())

    return "\n".join(formatted) if formatted else "None"


def run_single_question(
    client: OpenAI,
    model_name: str,
    retriever_url: str,
    question: str,
    topk: int,
    max_turns: int,
    max_tokens: int,
    temperature: float,
    verbose: bool,
    display: bool
) -> Dict[str, Any]:
    question = normalize_question(question)
    user_prompt = PROMPT_TEMPLATE.format(question=question)

    assistant_prefix = ""
    turns: List[Dict[str, str]] = []
    final_answer = None
    finished = False

    if display:
        print("\n################# [Start Reasoning + Searching] ##################\n")
        print(question)

    for _ in range(max_turns):
        if assistant_prefix:
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_prefix},
            ]
            raw_output = api_chat_completion(
                client=client,
                model_name=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=verbose,
                continue_final_message=True,
            )
        else:
            messages = [
                {"role": "user", "content": user_prompt},
            ]
            raw_output = api_chat_completion(
                client=client,
                model_name=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=verbose,
                continue_final_message=False,
            )

        prev_prefix = assistant_prefix
        assistant_prefix = merge_continuation(assistant_prefix, raw_output)

        output_text = assistant_prefix[len(prev_prefix):]

        # vLLM / API stop usually truncates the stop string, so we need to manually add the closing tag here.
        output_text = output_text.strip()
        if "<answer>" in output_text and not output_text.endswith("</answer>"):
            output_text += " </answer>"
            assistant_prefix += " </answer>"
        elif "<search>" in output_text and not output_text.endswith("</search>"):
            output_text += " </search>"
            assistant_prefix += " </search>"

        turns.append({"type": "assistant", "content": output_text})

        if display:
            print(output_text)

        # If the training template is matched, the retrieval is not performed, and the next round of generation continues.
        if contains_invalid_action_template(output_text):
            continue

        final_answer = get_answer(output_text)
        if final_answer is not None:
            finished = True
            break

        query = get_query(output_text)
        if not query:
            break

        search_results = retrieve_information(
            retriever_url=retriever_url,
            query=query,
            topk=topk
        )
        info_text = INFORMATION_TEMPLATE.format(search_results=search_results)

        assistant_prefix += info_text
        turns.append({"type": "information", "content": info_text})

        if display:
            print(info_text)

    merged_history = user_prompt + assistant_prefix

    return {
        "question": question,
        "final_answer": final_answer,
        "finished": finished,
        "turns": turns,
        "assistant_prefix": assistant_prefix,
        "merged_history": merged_history,
    }


def append_jsonl(path: str, rows: List[Dict[str, Any]]):
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_ids(path: str) -> Set[str]:
    done: Set[str] = set()
    if not path or not os.path.exists(path):
        return done

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            sample_id = row.get("id")
            if sample_id is not None:
                done.add(str(sample_id))
    return done


def run_dataset_one(
    sample_id: str,
    idx: int,
    question: str,
    gold: Any,
    source: str,
    api_key: Optional[str],
    base_url: Optional[str],
    model_name: str,
    retriever_url: str,
    topk: int,
    max_turns: int,
    max_tokens: int,
    temperature: float,
    verbose: bool
) -> Dict[str, Any]:
    client = build_client(source, api_key, base_url)
    report = run_single_question(
        client=client,
        model_name=model_name,
        retriever_url=retriever_url,
        question=question,
        topk=topk,
        max_turns=max_turns,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=verbose,
        display=False
    )
    return {
        "id": sample_id,
        "index": idx,
        "question": report["question"],
        "gold": gold,
        "final_answer": report["final_answer"],
        "correct": is_correct(report["final_answer"], gold),
        "finished": report["finished"],
        "turns": report["turns"],
        "assistant_prefix": report["assistant_prefix"],
        "merged_history": report["merged_history"],
    }


def parse_num_workers(value: str) -> int:
    num_workers = int(value)
    if num_workers <= 0:
        raise argparse.ArgumentTypeError("--num_workers must be a positive integer")
    return num_workers


def run_dataset(
    source: str,
    api_key: Optional[str],
    base_url: Optional[str],
    model_name: str,
    retriever_url: str,
    dataset_repo_path: str,
    config_name: str,
    split: str,
    topk: int,
    max_turns: int,
    max_tokens: int,
    temperature: float,
    max_samples: Optional[int],
    out_jsonl: str,
    verbose: bool,
    resume: bool,
    num_workers: int
):
    dataset = load_dataset(dataset_repo_path, config_name)[split]
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    done_ids = load_done_ids(out_jsonl) if resume else set()

    print(f"[INFO] split={split}, total={len(dataset)}, running={total}")

    jobs: List[Tuple[str, int, str, Any]] = []
    for idx in range(total):
        example = dataset[idx]
        sample_id = str(example.get("id", idx))
        if sample_id in done_ids:
            continue
        gold = example.get("golden_answers")
        if gold is None:
            gold = example.get("answer")
        jobs.append((sample_id, idx, example.get("question", ""), gold))

    if not jobs:
        print(f"[INFO] no pending samples, nothing to write to {out_jsonl}")
        return []

    rows: List[Dict[str, Any]] = []
    processed = 0

    if num_workers <= 1:
        client = build_client(source, api_key, base_url)
        for sample_id, idx, question, gold in tqdm(jobs, desc=f"{config_name}:{split}"):
            report = run_single_question(
                client=client,
                model_name=model_name,
                retriever_url=retriever_url,
                question=question,
                topk=topk,
                max_turns=max_turns,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=verbose,
                display=False
            )
            row = {
                "id": sample_id,
                "index": idx,
                "question": report["question"],
                "gold": gold,
                "final_answer": report["final_answer"],
                "correct": is_correct(report["final_answer"], gold),
                "finished": report["finished"],
                "turns": report["turns"],
                "assistant_prefix": report["assistant_prefix"],
                "merged_history": report["merged_history"],
            }
            rows.append(row)
            append_jsonl(out_jsonl, [row])
            processed += 1
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_job = {
                executor.submit(
                    run_dataset_one,
                    sample_id,
                    idx,
                    question,
                    gold,
                    source,
                    api_key,
                    base_url,
                    model_name,
                    retriever_url,
                    topk,
                    max_turns,
                    max_tokens,
                    temperature,
                    verbose
                ): (sample_id, idx)
                for sample_id, idx, question, gold in jobs
            }
            for future in tqdm(as_completed(future_to_job), total=len(future_to_job), desc=f"{config_name}:{split}"):
                sample_id, idx = future_to_job[future]
                try:
                    row = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed to process sample_id={sample_id}, index={idx}") from exc
                rows.append(row)
                append_jsonl(out_jsonl, [row])
                processed += 1

    rows.sort(key=lambda row: row["index"])

    print(f"[INFO] wrote {processed} rows to {out_jsonl}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="single", choices=["single", "dataset"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, default="local")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--retriever_url", type=str, default=os.environ.get("RETRIEVER_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--question", type=str, default="Who won the first Nobel Prize in Physics?")
    parser.add_argument("--dataset_repo_path", type=str, default="FlashRAG_datasets")
    parser.add_argument("--config_name", type=str, default="nq")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--out_jsonl", type=str, default="results/infer_vanilla_fast.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_workers", type=parse_num_workers, default=1)

    args = parser.parse_args()

    if args.mode == "single":
        client = build_client(args.source, args.api_key, args.base_url)
        report = run_single_question(
            client=client,
            model_name=args.model,
            retriever_url=args.retriever_url,
            question=args.question,
            topk=args.topk,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            verbose=args.verbose,
            display=True
        )
        print("\n################# [Finished] ##################\n")
        print(f"final_answer={report['final_answer']}")
        print(report["merged_history"])
    else:
        run_dataset(
            source=args.source,
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.model,
            retriever_url=args.retriever_url,
            dataset_repo_path=args.dataset_repo_path,
            config_name=args.config_name,
            split=args.split,
            topk=args.topk,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_samples=args.max_samples,
            out_jsonl=args.out_jsonl,
            verbose=args.verbose,
            resume=args.resume,
            num_workers=args.num_workers
        )


if __name__ == "__main__":
    main()