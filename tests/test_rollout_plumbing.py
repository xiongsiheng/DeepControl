"""_generate_with_gpu_padding keeps the caller's meta_info (greedy validation and the
128-token utility CoT) when it pads a batch to the GPU count, and turns off the per-call
log-prob recompute (the trainer recomputes once on the final trajectories).
Run: python -m pytest -q tests/test_rollout_plumbing.py
"""

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verl import DataProto  # noqa: E402
from deepcontrol.llm_agent.generation import LLMGenerationManager  # noqa: E402


class _FakeWorkerGroup:
    def __init__(self):
        self.calls = []

    def generate_sequences(self, batch):
        self.calls.append((batch.batch["input_ids"].shape[0], dict(batch.meta_info)))
        n = batch.batch["input_ids"].shape[0]
        return DataProto.from_dict({"responses": torch.arange(n).unsqueeze(1)})


def test_padding_keeps_meta_info():
    wg = _FakeWorkerGroup()
    fake = SimpleNamespace(config=SimpleNamespace(num_gpus=8), actor_rollout_wg=wg)
    ids = torch.ones(5, 4, dtype=torch.long)
    dp = DataProto.from_dict({"input_ids": ids, "attention_mask": ids, "position_ids": ids})
    dp.meta_info = {"do_sample": False, "response_length": 128}
    out = LLMGenerationManager._generate_with_gpu_padding(fake, dp)
    (n, meta), = wg.calls
    assert n == 8 and meta["do_sample"] is False and meta["response_length"] == 128
    assert meta["recompute_log_prob"] is False
    assert out.batch["responses"].shape[0] == 5



def test_controller_traces_follow_rows():
    import numpy as np
    traces = [[0.5], [], [0.7, 0.2], [0.1, 0.3]]  # ragged and equal-length rows mixed
    arr = np.empty(4, dtype=object)
    for i, t in enumerate(traces):
        arr[i] = t
    dp = DataProto.from_dict({"x": torch.arange(4).unsqueeze(1)}, non_tensors={"trace_utility": arr})
    dp.reorder(torch.tensor([2, 0, 3, 1]))
    assert [list(v) for v in dp.non_tensor_batch["trace_utility"]] == [traces[2], traces[0], traces[3], traces[1]]
    assert dp.batch["x"].squeeze(1).tolist() == [2, 0, 3, 1]
    parts = dp.chunk(2)
    back = DataProto.concat(parts)
    assert [list(v) for v in back.non_tensor_batch["trace_utility"]] == [traces[2], traces[0], traces[3], traces[1]]


def test_resume_replays_data_order():
    from torch.utils.data import DataLoader
    from verl.trainer.ppo.ray_trainer import epoch_batches

    dl = DataLoader(list(range(40)), batch_size=4, shuffle=True, drop_last=True)

    def run(epochs, start_epoch=0, saved=None, save_at=None):
        seen, ckpt = [], None
        for epoch in range(start_epoch, epochs):
            first = saved is not None and epoch == start_epoch
            epoch_rng, batches = epoch_batches(
                dl, epoch_rng=saved["epoch_rng"] if first else None,
                torch_rng=saved["torch_rng"] if first else None, skip=saved["step"] if first else 0)
            for idx, b in batches:
                torch.rand(3)  # other driver-side use of the torch RNG during a step
                seen.append((epoch, idx, b.tolist()))
                if (epoch, idx) == save_at:
                    ckpt = {"epoch_rng": epoch_rng, "torch_rng": torch.get_rng_state(), "step": idx + 1, "epoch": epoch}
        return seen, ckpt

    torch.manual_seed(0)
    full, ckpt = run(3, save_at=(1, 3))
    torch.manual_seed(12345)  # a fresh process starts from another state
    resumed, _ = run(3, start_epoch=ckpt["epoch"], saved=ckpt)
    done = [x for x in full if (x[0], x[1]) <= (1, 3)]
    assert resumed == full[len(done):]


class _CharTokenizer:
    """Tiny tokenizer: one id per character (pad 0), enough to check tensor layouts."""
    pad_token_id = 0

    def __call__(self, texts, add_special_tokens=False, **kw):
        return {"input_ids": [[1 + (ord(c) % 200) for c in t] for t in texts]}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["".join(chr(int(x) - 1) for x in row if int(x) != 0) for row in ids]


def _manager_with(wg):
    from deepcontrol.llm_agent.tensor_helper import TensorConfig, TensorHelper
    m = object.__new__(LLMGenerationManager)
    m.tokenizer = _CharTokenizer()
    m.actor_rollout_wg = wg
    m.tensor_fn = TensorHelper(TensorConfig(pad_token_id=0, max_prompt_length=4096, max_obs_length=512,
                                            max_start_length=1024))
    m.config = SimpleNamespace(num_gpus=1, utility_cot_tokens=128)
    return m


def test_gold_scoring_reads_the_target_positions():
    # The actor scores the last T positions of input_ids against `responses`
    # (dp_actor: logits[:, -T-1:-1]). This fake returns 0 where the token at that position is
    # the response token and -10 elsewhere, so a correct layout scores 0 for every row.
    class WG:
        world_size = 8

        def compute_log_prob(self, dp):
            ids, resp = dp.batch["input_ids"], dp.batch["responses"]
            T = resp.size(1)
            ok = (ids[:, -T:] == resp).float()
            return DataProto.from_dict({"old_log_probs": torch.where(ok > 0, 0.0, -10.0)})

    m = _manager_with(WG())
    qs = ["q", "a much longer question here"]
    ev = ["", "some evidence text that is long"]
    cots = ["short", ""]
    targets = [["Paris", "x"], ["a far longer target answer"]]
    scores = m._compute_target_mean_logp_from_question_evidence_cot(qs, ev, cots, targets)
    assert [round(s, 6) for s in scores] == [0.0, 0.0]


def test_cot_prompts_are_left_padded():
    class WG:
        def __init__(self):
            self.seen = None

        def generate_sequences(self, dp):
            self.seen = dp.batch["input_ids"].clone()
            return DataProto.from_dict({"responses": torch.ones(dp.batch["input_ids"].size(0), 3, dtype=torch.long)})

    wg = WG()
    m = _manager_with(wg)
    m._generate_cot_batch(["q", "a much longer question"], ["", "with evidence"], {})
    for row in wg.seen.tolist():
        first = next(i for i, x in enumerate(row) if x != 0)
        assert all(x != 0 for x in row[first:]), "padding after the prompt start"


def test_expand_confidence_gate():
    import numpy as np
    from deepcontrol.info_utility.novelty import NoveltyBank
    from deepcontrol.info_utility.utility import OnlineUtilityConfig, OnlineUtilityController, OnlineUtilityState
    from deepcontrol.protocol import control_expand

    def run(doc_scores):
        m = object.__new__(LLMGenerationManager)
        m.config = SimpleNamespace(
            utility_gamma=0.5, granularity_control=True, utility_eff_tau_low=0.3, utility_eff_tau_high=3.0,
            utility_novelty_max_length=512, utility_novelty_knn_k=5)
        m.novelty_encoder = SimpleNamespace(encode=lambda texts, max_length=512: np.eye(len(texts), 8, dtype=np.float32))
        m.utility_controller = OnlineUtilityController(OnlineUtilityConfig())
        m._generate_cot_batch = lambda qs, es, meta=None: ["" for _ in qs]
        m._compute_target_mean_logp_from_question_evidence_cot = lambda q, e, c, t, meta=None: list(doc_scores)
        entries = [{"doc_id": str(k), "snippet": f"s{k}", "full_text": f"text {k}"} for k in range(len(doc_scores))]
        recs = [{"act": "search", "valid": 1, "done": 0, "entries": entries, "expanded_info": ""}]
        controls = [[]]
        m._control_step(recs, torch.tensor([True]), [True], controls, [NoveltyBank(8)], [False], [0.0], [""], [""],
                        [-5.0], [""], ["q"], [["a"]], [OnlineUtilityState()], [[]], [[]], [[]], [None], [[]], [[]], {})
        return controls[0]

    assert run([-1.0, -2.5, -3.0]) == [control_expand(["0"])]   # margin 1.5: expand the best
    assert run([-3.0, -1.0, -2.5]) == [control_expand(["1"])]   # best is not rank 1
    assert run([-1.0, -1.5, -3.0]) == []                         # margin 0.5: no guidance


def test_answer_based_continue():
    import numpy as np
    from deepcontrol.info_utility.novelty import NoveltyBank
    from deepcontrol.info_utility.utility import OnlineUtilityConfig, OnlineUtilityController, OnlineUtilityState
    from deepcontrol.protocol import CONTROL_CONTINUE

    def answer_with(evidence, golds, answer="Lyon"):
        m = object.__new__(LLMGenerationManager)
        m.config = SimpleNamespace(
            utility_gamma=0.5, granularity_control=True, utility_eff_tau_low=0.3, utility_eff_tau_high=3.0,
            utility_novelty_max_length=512, utility_novelty_knn_k=5)
        m.novelty_encoder = SimpleNamespace(encode=lambda texts, max_length=512: np.eye(len(texts), 8, dtype=np.float32))
        m.utility_controller = OnlineUtilityController(OnlineUtilityConfig(continue_prob=1.0))
        m._generate_cot_batch = lambda qs, es, meta=None: ["" for _ in qs]
        m._compute_target_mean_logp_from_question_evidence_cot = lambda q, e, c, t, meta=None: [-1.0 for _ in q]
        recs = [{"act": "answer", "content": answer, "valid": 1, "done": 1, "entries": [], "expanded_info": ""}]
        controls = [[]]
        # an open search step whose evidence is `evidence`; the agent now answers
        m._control_step(recs, torch.tensor([True]), [True], controls, [NoveltyBank(8)], [True], [1.0], [evidence],
                        [""], [-5.0], [""], ["q"], [golds], [OnlineUtilityState()], [[]], [[]], [[]], [None],
                        [[]], [[]], {})
        return controls[0], recs[0]["done"]

    assert answer_with("Paris is the capital of France.", ["Paris"]) == ([], 1)          # answer in evidence
    assert answer_with("Lyon is a city in France.", ["Paris"]) == ([CONTROL_CONTINUE], 0)  # missing: continue
    assert answer_with("nothing relevant", ["yes"]) == ([], 1)                             # yes/no: skipped
    assert answer_with("nothing relevant", ["UK"]) == ([], 1)                              # too short: skipped
    assert answer_with("Lyon is a city in France.", ["Paris"], answer="paris") == ([], 1)  # right answer: no continue


def _gen_config(**kw):
    from deepcontrol.llm_agent.generation import GenerationConfig
    base = dict(max_turns=8, max_start_length=1024, max_prompt_length=4096, max_response_length=512,
                max_obs_length=1024, num_gpus=1, retriever_url="http://127.0.0.1:9", topk=5,
                enable_utility_controller=True)
    base.update(kw)
    return GenerationConfig(**base)


def test_control_modes():
    m = LLMGenerationManager(_CharTokenizer(), None, _gen_config(control_mode="fast"))
    assert m.fast_control and m.novelty_encoder is None and m.utility_controller is not None
    n = LLMGenerationManager(_CharTokenizer(), None, _gen_config(control_mode="normal"), is_validation=True)
    assert not n.fast_control
    o = LLMGenerationManager(_CharTokenizer(), None, _gen_config(control_mode="off"))
    assert o.utility_controller is None and not o.config.enable_utility_controller
    v = LLMGenerationManager(_CharTokenizer(), None, _gen_config(control_mode="fast"), is_validation=True)
    assert v.utility_controller is None   # validation runs without control
    import pytest
    with pytest.raises(ValueError):
        LLMGenerationManager(_CharTokenizer(), None, _gen_config(control_mode="utility"))


def test_fast_control_step():
    from deepcontrol.info_utility.utility import OnlineUtilityState
    from deepcontrol.protocol import CONTROL_CONTINUE, CONTROL_NO_EXPAND, CONTROL_STOP, control_expand

    m = LLMGenerationManager(_CharTokenizer(), None, _gen_config(control_mode="fast"))
    doc = lambda i, snip, full: {"doc_id": str(i), "snippet": snip, "full_text": full}

    def step(act, open_step, evidence, entries=(), golds=("Paris",), state=None, answer="Lyon"):
        recs = [{"act": act, "content": answer, "valid": 1, "done": int(act == "answer"), "entries": list(entries),
                 "expanded_info": ""}]
        controls = [[]]
        st = state or OnlineUtilityState()
        m._answer_control_step(recs, torch.tensor([True]), [True], controls, [open_step], [evidence], [""],
                               [list(golds)], [st], [None], [[]], [[]])
        return controls[0], recs[0]["done"], st

    assert step("answer", True, "Lyon only")[:2] == ([CONTROL_CONTINUE], 0)
    assert step("answer", True, "capital Paris")[:2] == ([], 1)
    assert step("answer", True, "Lyon only", answer="Paris")[:2] == ([], 1)   # right answer: no continue
    assert step("search", True, "capital Paris", [doc(1, "a", "b")])[0] == [CONTROL_STOP]
    assert step("search", False, "", [doc(1, "a", "b"), doc(2, "c", "is Paris")])[0] == [control_expand(["2"])]
    assert step("search", False, "", [doc(1, "a", "b")])[0] == [CONTROL_NO_EXPAND]
    assert step("search", False, "", [doc(1, "Paris", "Paris")])[0] == []
    used = OnlineUtilityState(intervention_sent=True)
    assert step("answer", True, "Lyon only", state=used)[:2] == ([], 1)    # one intervention per episode


def test_resume_keeps_configured_lr(tmp_path, monkeypatch):
    from verl.utils import train_state
    from verl.utils.torch_functional import get_constant_schedule_with_warmup
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda s: None)

    def build(lr):
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 2)
        opt = torch.optim.AdamW(m.parameters(), lr=lr)
        return m, opt, get_constant_schedule_with_warmup(opt, num_warmup_steps=2)

    m, opt, sch = build(5e-7)
    for _ in range(3):
        m(torch.ones(1, 4)).sum().backward()
        opt.step()
        sch.step()
    train_state.save_fsdp_train_state(str(tmp_path), 0, m, opt, sch)
    for lr in (5e-7, 3e-7):
        m2, opt2, sch2 = build(lr)
        train_state.load_fsdp_train_state(str(tmp_path), 0, m2, opt2, sch2)
        assert sch2.last_epoch == 3
        assert opt2.param_groups[0]["lr"] == lr
        torch.testing.assert_close(opt2.state_dict()["state"][0]["exp_avg"], opt.state_dict()["state"][0]["exp_avg"])
        opt2.step()
        sch2.step()
        assert opt2.param_groups[0]["lr"] == lr


class _SpecialCharTokenizer(_CharTokenizer):
    """_CharTokenizer plus <|im_end|> (id 250, also eos) for sampled-token checks."""
    eos_token_id = 250

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 250}[token]

    def decode(self, ids):
        return "".join("<|im_end|>" if int(x) == 250 else chr(int(x) - 1) for x in ids if int(x) != 0)


def test_sampled_tokens_cut_after_first_action():
    from verl.utils.reward_score.qa_soft_em import compute_final_reward, count_format_violations, parse_chatml_messages

    tok = _SpecialCharTokenizer()
    enc = lambda t: tok([t])["input_ids"][0]
    junk = "<control>\n<think>find it</think>\n<search>q</search>\n</control>"
    rows = [enc(junk) + [250, 0, 0], enc("no action at all") + [250]]
    width = max(len(r) for r in rows)
    responses = torch.tensor([r + [0] * (width - len(r)) for r in rows])

    m = LLMGenerationManager(tok, None, _gen_config(control_mode="fast"))
    ids, texts = m._cut_sampled_responses(responses)
    kept = "<control>\n<think>find it</think>\n<search>q</search>"
    assert texts == [kept + "<|im_end|>", "no action at all<|im_end|>"]
    # the sampled ids themselves, not a re-encoding of the cut text
    assert ids[0, :len(kept)].tolist() == rows[0][:len(kept)] and ids[0, len(kept)].item() == 250
    assert ids[0, len(kept) + 1:].eq(0).all()
    # the environment parses the action as inference does, so the stray prefix is no invalid action
    assert m._postprocess_action(texts[0]) == ("search", "q")

    def traj(*turns):  # decoded with special tokens, as the reward manager sees it
        out = "<|im_start|>user\nQuestion: q?<|im_end|>\n"
        for t in turns:
            out += f"<|im_start|>assistant\n{t}<|im_end|>\n<|im_start|>user\n<search_results>\nx\n</search_results><|im_end|>\n"
        return out.rsplit("<|im_start|>user", 1)[0]

    gt = {"target": ["Paris"]}
    answer = "<think>it is Paris</think>\n<answer>Paris</answer>"
    ok, bad = traj("<think>find it</think>\n<search>q</search>", answer), traj(kept, answer)
    assert count_format_violations(parse_chatml_messages(ok)) == 0
    assert count_format_violations(parse_chatml_messages(bad)) == 1
    assert compute_final_reward(ok, gt, stray_text_penalty=0.1) == 1.0
    assert abs(compute_final_reward(bad, gt, stray_text_penalty=0.1) - 0.9) < 1e-9
    assert compute_final_reward(bad, gt) == 1.0   # validation: no stray-text term


def test_offloaded_optimizer_state_is_loaded_only_for_the_step(monkeypatch):
    from verl.workers.actor import dp_actor

    calls = []
    monkeypatch.setattr(dp_actor, "load_fsdp_optimizer", lambda optimizer, device_id: calls.append("load"))
    monkeypatch.setattr(dp_actor, "offload_fsdp_optimizer", lambda optimizer: calls.append("offload"))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    actor = object.__new__(dp_actor.DataParallelPPOActor)
    actor.config = SimpleNamespace(grad_clip=1.0)
    actor.actor_module = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(actor.actor_module.parameters(), lr=0.1)
    real_step = opt.step
    opt.step = lambda: (calls.append("step"), real_step())
    actor.actor_optimizer = opt
    actor.actor_module(torch.ones(1, 2)).sum().backward()

    actor.optimizer_offload = True
    actor._optimizer_step()
    assert calls == ["load", "step", "offload"]
    calls.clear()
    actor.optimizer_offload = False
    actor._optimizer_step()
    assert calls == ["step"]
