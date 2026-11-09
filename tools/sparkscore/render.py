# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic prompt rendering: (state, questions) -> token ids.

The output is one shared prefix and one branch per question. A branch ends
exactly at the answer slot, and each option is one label token read there:

    prefix + branch_q            is what the model reads for question q
    option_ids_q[k]              is the token whose probability scores option k

The chat template is rendered once per tokenizer with placeholder strings and
cut at them, so serving a request is only tokenization. User content (the
state and the question blocks) is tokenized with special-token parsing off,
so text such as "<|im_end|>" inside a state stays plain text and cannot open
or close a turn. Labels are checked at load: each is exactly one token in its
real context, the ids are distinct, and the branch/label boundary tokenizes
the same jointly as piecewise. tests/test_render.py checks the rest against
the tokenizer's own apply_chat_template.
"""

import itertools
import string
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from tools.sparkscore import templates

DEFAULT_TOKENIZER = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
MAX_LABELS = 255


class Unsupported(ValueError):
    """A request the template cannot render for this engine (answered 422)."""


@dataclass(frozen=True)
class Branch:
    name: str  # the question's name in the request
    keys: list  # answer keys in request order (see templates.chat_v1.options)
    ids: list  # branch token ids, after the prefix, ending at the answer slot
    option_ids: list  # one label token id per shown option
    offset: int = 0  # options are shown as rotation(len(keys), offset); REVERSED: in reverse order


@dataclass(frozen=True)
class Rendered:
    prefix: list
    branches: list  # one or more per question, in request order
    # Templates that score option rotations in two passes set this: it takes
    # {name: {offset: probabilities in shown order}} from the first pass and
    # returns the branches of the second pass (often none).
    second_pass: object = field(default=None, compare=False, repr=False)
    # The template's own readout temperature for this request (d3-ultra's
    # depends on the number of questions); the server's --temperature
    # multiplies it.
    temperature: float = 1.0
    # A knowledge pack (templates.d3_ultra): tokens before the prefix that the
    # engine holds, so the prefix continues them. Empty: none.
    pack: list = field(default_factory=list)

    @property
    def input_tokens(self):
        return len(self.pack) + len(self.prefix) + sum(len(b.ids) for b in self.branches)


# Option rotations, as KnowLine's knowline_layout.py (PelaAI, Apache-2.0)
# defines them: a branch shows the options in a cyclic rotation, and the
# label distributions of a question's rotations are mapped back to request
# order and averaged with equal weight. Offset 0 is request order. d3-ultra's
# permutation average shows the options reversed instead (offset REVERSED).
REVERSED = "reversed"


def rotation(n, offset):
    """Original option indices in shown order."""
    if offset == REVERSED:
        return list(range(n - 1, -1, -1))
    offset %= n
    return list(range(offset, n)) + list(range(offset))


def to_original(shown, offset):
    out = [0.0] * len(shown)
    for j, i in enumerate(rotation(len(shown), offset)):
        out[i] = shown[j]
    return out


def combine(by_offset):
    """{offset: probabilities in shown order} -> their mean in request order.
    One offset 0 comes back unchanged."""
    rows = [to_original(p, o) for o, p in by_offset.items()]
    return [sum(r[i] for r in rows) / len(rows) for i in range(len(rows[0]))]


def load(template=templates.DEFAULT, tokenizer=None, revision=None, **options):
    """The renderer of a template. A module with its own Renderer class gets
    the options; the others use Renderer below. The template's own
    tokenizer, at its pinned revision, is the default."""
    module = templates.load(template)
    cls = getattr(module, "Renderer", Renderer)
    default = getattr(module, "TOKENIZER", DEFAULT_TOKENIZER), getattr(module, "REVISION", DEFAULT_REVISION)
    tokenizer = tokenizer or default[0]
    if revision is None and tokenizer == default[0]:
        revision = default[1]
    if cls is Renderer:
        if options:
            raise ValueError(f"template {template} takes no options ({', '.join(options)})")
        return Renderer(tokenizer, revision, template)
    return cls(tokenizer, revision, **options)


def _candidates():
    letters = string.ascii_uppercase
    return itertools.chain(letters, ("".join(p) for p in itertools.product(letters, repeat=2)))


class Renderer:
    def __init__(self, tokenizer=DEFAULT_TOKENIZER, revision=DEFAULT_REVISION, template=templates.DEFAULT):
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer

        hf = AutoTokenizer.from_pretrained(tokenizer, revision=revision)
        self.tokenizer = tokenizer
        self.revision = revision
        # Named without the local path, so it is safe to report.
        local = Path(tokenizer).is_dir()
        self.source = f"local:{Path(tokenizer).resolve().name}" if local else f"{tokenizer}@{(revision or 'main')[:12]}"
        self.template = templates.load(template)
        self.version = self.template.VERSION
        self.hf = hf
        # Markup (template text) parses special tokens; content does not.
        self._markup = Tokenizer.from_str(hf.backend_tokenizer.to_str())
        self._content = Tokenizer.from_str(hf.backend_tokenizer.to_str())
        self._content.encode_special_tokens = True

        state_slot, question_slot = (f"SPARKSCORE{uuid.uuid4().hex}" for _ in range(2))
        scaffold = self.chat(self.template.messages(state_slot, question_slot))
        if scaffold.count(state_slot) != 1 or scaffold.count(question_slot) != 1:
            raise ValueError("chat template dropped or repeated a content slot")
        head, rest = scaffold.split(state_slot)
        if question_slot not in rest:
            raise ValueError("chat template put the question before the state")
        mid, end = rest.split(question_slot)
        # Templates that trim message content (Qwen's do) are matched by
        # trimming the content here.
        padded = self.chat(self.template.messages(f"  {state_slot}  ", question_slot))
        self._trim = f"  {state_slot}" not in padded

        self.head = self._encode(self._markup, head)
        self.mid = self._encode(self._markup, mid)
        self.tail = self._encode(self._markup, end + self.template.CUE)
        self._end = end
        self.labels, self.option_ids = self._find_labels()
        self.option_text = {i: hf.decode([i]) for i in self.option_ids}

    def chat(self, messages):
        return self.hf.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    @staticmethod
    def _encode(tok, text):
        return tok.encode(text, add_special_tokens=False).ids

    def _find_labels(self):
        """MAX_LABELS labels whose token is one distinct id right after a real
        branch, A..Z first, then two-letter labels in order."""
        sample = {"type": "choice", "instructions": "q", "criteria": {"x": None, "y": None}}
        block = self.template.question_block(sample, ["A", "B"])
        branch_text = block + self._end + self.template.CUE
        branch = self._encode(self._content, block) + self.tail
        if self._encode(self._markup, branch_text) != branch:
            raise ValueError("branch text does not tokenize the same jointly and piecewise")
        labels, ids = [], []
        for label in _candidates():
            shown = self.template.LABEL_PREFIX + label
            full = self._encode(self._markup, branch_text + shown)
            if (
                len(full) == len(branch) + 1
                and full[:-1] == branch
                and full[-1] not in ids
                and self.hf.decode(full[-1:]) == shown
            ):
                labels.append(label)
                ids.append(full[-1])
            if len(labels) == MAX_LABELS:
                return labels, ids
        raise ValueError(f"tokenizer has only {len(labels)} single-token labels after {self.template.CUE!r}")

    def render(self, state, questions):
        """questions: {name: question} as validated by the server."""
        t = self.template
        state_text = t.text(state)
        blocks, keys = [], []
        for question in questions.values():
            k, _ = t.options(question)
            if len(k) > MAX_LABELS:
                raise ValueError(f"at most {MAX_LABELS} options per choice")
            blocks.append(t.question_block(question, self.labels[: len(k)]))
            keys.append(k)
        if self._trim:
            state_text = state_text.strip()
            blocks = [b.strip() for b in blocks]
        encoded = self._content.encode_batch([state_text, *blocks], add_special_tokens=False)
        prefix = self.head + encoded[0].ids + self.mid
        branches = [
            Branch(name, k, e.ids + self.tail, self.option_ids[: len(k)])
            for name, k, e in zip(questions, keys, encoded[1:])
        ]
        return Rendered(prefix, branches)
