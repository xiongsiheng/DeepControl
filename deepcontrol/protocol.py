"""Single source of the agent protocol shared by inference (infer_fast.py) and RL training.

Everything the model sees from the environment is defined here: the task prompt, the
control messages, the invalid-action feedback, and the per-episode action state machine.
Training rollouts and inference import the same strings and the same rules, so a policy is
trained on exactly the observations it receives at test time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

PREFIX = """Answer the given question.

You MUST follow the protocol below.

CONTROL
- A control message may appear anywhere in the conversation in the form:
  <control>...</control>
- You MUST follow the <control> message that appears in the context.


General rules
- This is only a wording rule and MUST NOT change whether you use search or expand. Never skip a search required by the routing rules. Do not use generic or reusable text inside <think>. Every think block must discuss question-specific entities, relations, or evidence. Never say that a question is straightforward, factual, or answerable directly from memory.
- If you choose a <search> or <expand> action, first output one complete <think>...</think> block. A direct <answer> action may either include or omit a think block.
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
A round MUST be one of the following sequences:

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

3) Direct-answer round:
   <answer>...</answer>

4) Reasoned direct-answer round:
   <think>...</think>
   <answer>...</answer>

You may perform as many rounds as needed.

"""


def user_prompt(question: str) -> str:
    """First user turn, as inference sends it."""
    return f"{PREFIX}\n\nQuestion: {question}\n"


# Control messages. STOP is the one inference injects before turn max_turns - 2; the
# training-time controller uses the same text. The other three are training-only.
CONTROL_STOP = "<control>Stop searching.</control>"
CONTROL_CONTINUE = "<control>Continue the search for one additional step</control>"
CONTROL_NO_EXPAND = "<control>Do not expand any retrieved documents. Continue the search.</control>"


def control_expand(doc_ids: List[str]) -> str:
    return f"<control>Expand the retrieved documents: [{', '.join(str(d) for d in doc_ids)}]</control>"


def invalid_action(reason: str) -> str:
    return (
        f"Invalid action: {reason}\n"
        "Output exactly ONE of:\n"
        "<search>...</search>\n"
        "<expand>{\"doc_ids\":[...]}</expand>\n"
        "<answer>...</answer>\n"
    )


EMPTY_INFORMATION = "<information>\nNone\n</information>"
NEED_SEARCH, NEED_EXPAND, NEED_ANSWER = "NEED_SEARCH", "NEED_EXPAND", "NEED_ANSWER"
ACTIONS = ("search", "expand", "answer")


@dataclass
class EpisodeState:
    """The inference state machine (infer_fast.run_tool_episode), one per episode.

    `expand` before any search is invalid; expansions may only name documents returned by
    the most recent search.
    """

    state: str = NEED_SEARCH
    allowed_doc_ids: Optional[Set[str]] = None
    seen: Dict[str, int] = field(default_factory=lambda: {a: 0 for a in ACTIONS})

    def check(self, act: Optional[str]) -> Optional[str]:
        """Return the invalid-action reason for `act`, or None when the action is valid."""
        if act is None:
            return "No valid action tag found."
        if act not in self.seen:
            return f"Unknown action '{act}'."
        if self.state == NEED_SEARCH and act == "expand":
            return "You must <search> before <expand> in a round."
        if act == "expand" and self.allowed_doc_ids is None:
            return "No prior search_results in this round."
        return None

    def on_search(self, entries: List[dict]) -> None:
        self.seen["search"] += 1
        self.allowed_doc_ids = {str(e["doc_id"]) for e in entries if e.get("doc_id") is not None}
        self.state = NEED_EXPAND

    def on_expand(self) -> None:
        self.seen["expand"] += 1
        self.state = NEED_ANSWER

    def on_answer(self) -> None:
        self.seen["answer"] += 1


def search_observation(formatted_results: str) -> str:
    return f"<search_results>\n{formatted_results}\n</search_results>"


def information_observation(formatted_information: str) -> str:
    return f"<information>\n{formatted_information}\n</information>"
