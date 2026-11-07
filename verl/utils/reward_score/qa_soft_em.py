# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) < 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()



def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    # if do_print:
    #     print(f"--------------------------------")
    #     print(f"Golden answers: {ground_truth['target']}")
    #     print(f"Extracted answer: {answer}")
    #     print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score

# -------------------------------------------------------------

def extract_last_assistant_message(text: str):
    parts = text.split("assistant")
    if len(parts) <= 1:
        return None
    return parts[-1].strip()


def extract_answer_from_last_assistant(text: str):
    last_msg = extract_last_assistant_message(text)
    if last_msg is None:
        return None

    matches = list(re.finditer(r'<answer>(.*?)</answer>', last_msg, re.DOTALL))
    if len(matches) == 0:
        return None

    return matches[-1].group(1).strip()


def soft_em_f1(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    pred_tokens = normalize_answer(prediction).split()
    if len(pred_tokens) == 0:
        return 0.0

    best_f1 = 0.0

    for gt in golden_answers:
        gt_tokens = normalize_answer(gt).split()
        if len(gt_tokens) == 0:
            continue

        common = {}
        for tok in pred_tokens:
            common[tok] = min(pred_tokens.count(tok), gt_tokens.count(tok))

        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        best_f1 = max(best_f1, f1)

    return best_f1


def compute_score_soft_f1_with_format_floor(
    solution_str,
    ground_truth,
    format_floor=0.1,
):
    """
    Reward logic:
    - No <answer> in last assistant -> 0
    - Has <answer>:
        reward = max(F1, format_floor)
    """

    answer = extract_answer_from_last_assistant(solution_str)

    # Case 1: no terminal answer
    if answer is None:
        return 0.0

    # Case 2: terminal answer exists
    f1 = soft_em_f1(answer, ground_truth["target"])

    return max(f1, format_floor)

# -------------------------------------------------------------

def parse_chatml_messages(text: str):
    """
    Parse ChatML-style conversation into a list of dicts:
    [{"role": "user"/"assistant"/"system", "content": "..."}]
    """
    pattern = r'(system|user|assistant)\n'
    splits = list(re.finditer(pattern, text))

    messages = []
    for i, m in enumerate(splits):
        role = m.group(1)
        start = m.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        content = text[start:end].strip()
        messages.append({"role": role, "content": content})

    return messages


def compute_base_qa_reward(solution_str, ground_truth, format_floor=0.1):
    answer = extract_answer_from_last_assistant(solution_str)
    if answer is None:
        return None  # Note: this returns None, not 0.

    f1 = soft_em_f1(answer, ground_truth["target"])
    return max(f1, format_floor)


def compute_tool_penalty(messages, penalty_per_hit=0.1):
    penalty = 0.0
    for msg in messages:
        if msg["role"] != "user":
            continue
        content = msg["content"]
        penalty += content.count("Invalid action") * penalty_per_hit
        penalty += content.count("Invalid input") * penalty_per_hit
        penalty += content.count("<information>\nNone\n</information>") * penalty_per_hit
    return penalty


def collect_user_documents(messages):
    """
    Collect all text from user messages as retrieved documents.
    """
    docs = []
    for msg in messages:
        if msg["role"] == "user":
            docs.append(msg["content"])
    return docs


def found_answer_in_documents(docs, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    norm_docs = [normalize_answer(doc) for doc in docs]

    for gt in golden_answers:
        norm_gt = normalize_answer(gt)
        if not norm_gt:
            continue
        for doc in norm_docs:
            if norm_gt in doc:
                return True
    return False


def extract_answer_from_last_message(last_msg):
    if last_msg.get("role") != "assistant":
        return None

    matches = list(re.finditer(
        r'<answer>(.*?)</answer>',
        last_msg.get("content", ""),
        re.DOTALL
    ))
    if not matches:
        return None

    answer = matches[-1].group(1).strip()

    # 🚫 If <answer> contains any tag, reject it outright.
    if re.search(r'<[^>]+>', answer):
        return None

    return answer




def has_protocol_violation(messages):
    """
    Hard check for protocol / grammar violations.
    Return True if violation detected.
    """

    if not messages:
        return True

    last = messages[-1]

    # 1. The last message must be assistant.
    if last.get("role") != "assistant":
        return True

    content = last.get("content", "").strip()

    # 2. Count action tags.
    action_tags = ["answer", "search", "expand"]
    present = {
        tag: bool(re.search(rf"<{tag}>", content))
        for tag in action_tags
    }

    num_actions = sum(present.values())

    # exactly ONE of
    if num_actions != 1:
        return True

    # 3. If <answer>, perform strict checks.
    if present["answer"]:
        # 3.1 There must be exactly one <answer>
        answers = list(re.finditer(
            r"<answer>(.*?)</answer>",
            content,
            re.DOTALL
        ))
        if len(answers) != 1:
            return True

        answer_text = answers[0].group(1).strip()

        # 3.2 <answer> must not contain any tags
        if re.search(r"<[^>]+>", answer_text):
            return True

        # 3.3 <answer> must be terminal (no trailing garbage)
        after  = content[answers[0].end():].strip()

        if after:
            return True

    return False



_CONTROL_RE = re.compile(r"<control>(.*?)</control>", re.DOTALL)
_ACTION_RE = re.compile(r"<(search|expand|answer)>(.*?)</\1>", re.DOTALL)


def _last_action(assistant_text):
    m = list(_ACTION_RE.finditer(assistant_text or ""))
    return (m[-1].group(1), m[-1].group(2)) if m else (None, "")


def count_control_noncompliance(messages):
    """Control messages the next assistant turn did not follow.

    Stop -> must answer; continue -> must search; expand [ids] -> must expand one of them;
    do-not-expand -> must not expand. A control with no following assistant turn is not
    counted.
    """
    from deepcontrol.protocol import CONTROL_CONTINUE, CONTROL_NO_EXPAND, CONTROL_STOP

    violations = 0
    for k, msg in enumerate(messages):
        if msg["role"] != "user":
            continue
        controls = ["<control>" + c + "</control>" for c in _CONTROL_RE.findall(msg["content"])]
        if not controls:
            continue
        nxt = next((m for m in messages[k + 1:] if m["role"] == "assistant"), None)
        if nxt is None:
            continue
        act, content = _last_action(nxt["content"])
        for c in controls:
            if c == CONTROL_STOP:
                violations += act != "answer"
            elif c == CONTROL_CONTINUE:
                violations += act != "search"
            elif c == CONTROL_NO_EXPAND:
                violations += act == "expand"
            elif c.startswith("<control>Expand the retrieved documents:"):
                wanted = set(re.findall(r"\d+", c))
                got = set(re.findall(r"\d+", content)) if act == "expand" else set()
                violations += not (wanted & got)
    return violations


_CHATML_SPECIAL_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>")


def count_format_violations(messages):
    """Assistant turns with text outside one optional <think> block and the first closed action
    (e.g. a copied <control> wrapper). Rollouts keep the sampled tokens up to the action
    (sampled_tokens), so such text is in the trajectory; inference drops it silently."""
    from deepcontrol.llm_agent.deep_research_agent import DeepResearchAgent

    def squash(x):
        return re.sub(r"\s+", " ", x).strip()

    n = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = _CHATML_SPECIAL_RE.sub("", msg.get("content", ""))
        n += squash(content) != squash(DeepResearchAgent.truncate_to_one_action(content))
    return n


def compute_final_reward(
    solution_str,
    ground_truth,
    format_floor=0.1,
    penalty_per_hit=0.2,
    penalty_cap=0.4,
    stray_text_penalty=0.0,
    **kwargs,
):
    """Composite reward.

    r = max(F1, format_floor) if the trajectory ends with an answer, else 0; minus
    min(penalty_per_hit * N, penalty_cap), where N counts tool-usage violations (invalid
    actions or inputs, empty expansions) and control non-compliance; stray_text_penalty per
    turn with text outside [think] + action joins it under the same cap. The penalty is
    applied after the floor, so a violating trajectory can score below the floor.
    """
    messages = parse_chatml_messages(solution_str)
    if not messages:
        return 0.0
    answer = extract_answer_from_last_message(messages[-1])
    reward = 0.0 if answer is None else max(soft_em_f1(answer, ground_truth["target"]), format_floor)
    n_tool = compute_tool_penalty(messages, penalty_per_hit=1.0)
    n_control = count_control_noncompliance(messages)
    n_stray = count_format_violations(messages) if stray_text_penalty else 0
    penalty = min(penalty_per_hit * (n_tool + n_control) + stray_text_penalty * n_stray, penalty_cap)
    return reward - penalty
