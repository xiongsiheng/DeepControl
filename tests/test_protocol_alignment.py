"""Training rollouts must show the policy exactly what inference shows it.

Checks, with the real chat template of a checkpoint (`TOKENIZER` env var):
  1. the data builder's first user turn equals the one infer_fast.py sends;
  2. the dataset's rendered prompt equals the chat-template rendering inference gets;
  3. a multi-turn rollout assembled the way generation.py assembles it (assistant output +
     <|im_end|> + environment turns) equals the chat-template rendering of the same
     conversation as an inference message list, control messages included.
Run: TOKENIZER=<path or hub id> python -m pytest -q tests/test_protocol_alignment.py
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deepcontrol import protocol as P  # noqa: E402

TOKENIZER = os.environ.get("TOKENIZER", "Qwen/Qwen2.5-3B-Instruct")
QUESTION = "who got the first nobel prize in physics"


def test_data_builder_prompt_matches_inference():
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "data_process", "nq_hotpotqa_search.py")
    spec = importlib.util.spec_from_file_location("nq_hotpotqa_search", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    make_prefix = mod.make_prefix
    src = open(os.path.join(os.path.dirname(__file__), "..", "infer_fast.py")).read()
    assert "prefix = PREFIX" in src
    assert make_prefix(QUESTION, "base") == f"{P.PREFIX}\n\nQuestion: {QUESTION}\n"


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(TOKENIZER)


def _dataset_prompt(tok, chat):
    # what verl/utils/dataset/rl_dataset.py renders for the first turn
    return tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)


def test_first_turn_rendering(tok):
    chat = [{"role": "user", "content": P.user_prompt(QUESTION)}]
    rl = _dataset_prompt(tok, chat)
    inf = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    assert rl == inf
    assert "You are Qwen, created by Alibaba Cloud." in rl  # default system message kept


def test_multi_turn_rollout_rendering(tok):
    from deepcontrol.llm_agent.generation import LLMGenerationManager as G

    a1 = "<think>I need the laureate.</think>\n<search>first nobel prize physics</search>"
    o1 = P.search_observation("Rank=1 doc_id=1 Title: Nobel Prize\nSnippet: ...")
    a2 = '<think>Expand it.</think>\n<expand>{"doc_ids": [1]}</expand>'
    o2 = P.information_observation("[doc_id=1] Wilhelm Röntgen ...")
    a3 = "<answer>Wilhelm Conrad Röntgen</answer>"
    msgs = [{"role": "user", "content": P.user_prompt(QUESTION)},
            {"role": "assistant", "content": a1}, {"role": "user", "content": o1},
            {"role": "user", "content": P.control_expand(["1"])},
            {"role": "assistant", "content": a2}, {"role": "user", "content": o2},
            {"role": "user", "content": P.CONTROL_STOP},
            {"role": "assistant", "content": a3}]
    inference = tok.apply_chat_template(msgs, tokenize=False)
    # generation.py: dataset prompt, then per turn model output + "<|im_end|>" and the
    # environment reply from _wrap_turns (which ends with the next assistant header).
    training = _dataset_prompt(tok, msgs[:1])
    training += a1 + "<|im_end|>" + G._wrap_turns(o1, [P.control_expand(["1"])])
    training += a2 + "<|im_end|>" + G._wrap_turns(o2, [P.CONTROL_STOP])
    training += a3 + "<|im_end|>\n"
    assert training == inference
    assert tok(training, add_special_tokens=False)["input_ids"] == tok(inference, add_special_tokens=False)["input_ids"]


def test_invalid_and_stop_texts_match_inference():
    src = open(os.path.join(os.path.dirname(__file__), "..", "infer_fast.py")).read()
    assert "invalid_action(reason)" in src and "CONTROL_STOP" in src
    assert P.CONTROL_STOP == "<control>Stop searching.</control>"
