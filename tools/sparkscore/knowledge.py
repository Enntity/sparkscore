# SPDX-License-Identifier: AGPL-3.0-only
"""Named knowledge: documents registered once and used by name.

A request can send `knowledge` (the text itself) or `knowledge_name` (a
document registered here). A registered document is warmed when it is added:
one small request on it makes Atlas encode its knowledge pack, so the first
real request does not pay for it.

Documents come from three places:
- `--knowledge NAME=PATH`, at startup;
- `--knowledge-dir DIR`: every `*.md` and `*.txt` file there, named by its
  file name without the extension, at startup;
- `PUT /v1/knowledge/NAME` with `{"text": "..."}`, at any time (kept in
  memory until the server stops or `DELETE /v1/knowledge/NAME`).
"""

import os
import re
import threading

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MAX_DOCUMENTS = 64
# The warm-up request: one yes/no question on an empty-ish state.
WARM_STATE = "(warm-up)"
WARM_QUESTIONS = {"warm": {"type": "noul", "instructions": "Is this a warm-up request?"}}


class KnowledgeError(ValueError):
    """A registration the library refuses: (HTTP status, detail)."""

    def __init__(self, status, detail):
        super().__init__(detail)
        self.status, self.detail = status, detail


class Library:
    def __init__(self, warm=None):
        """`warm(text)` -> (pack status or None, pack tokens): one request on the text (the Service's)."""
        self.docs = {}  # name -> {"text", "tokens", "pack"}
        self.lock = threading.Lock()
        self.warm = warm

    def text(self, name):
        with self.lock:
            doc = self.docs.get(name)
        if doc is None:
            raise KnowledgeError(422, f"no knowledge named {name!r}; GET /v1/knowledge lists them")
        return doc["text"]

    def put(self, name, text):
        if not isinstance(name, str) or not NAME.match(name):
            raise KnowledgeError(422, "a knowledge name is 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit")
        if not isinstance(text, str) or not text.strip():
            raise KnowledgeError(422, "knowledge text must be a nonempty string")
        with self.lock:
            if name not in self.docs and len(self.docs) >= MAX_DOCUMENTS:
                raise KnowledgeError(422, f"the library holds at most {MAX_DOCUMENTS} documents; DELETE one first")
        status, tokens = self.warm(text) if self.warm else (None, 0)
        entry = {"text": text, "tokens": tokens, "pack": status}
        with self.lock:
            self.docs[name] = entry
        return self.describe(name, entry)

    def delete(self, name):
        with self.lock:
            if self.docs.pop(name, None) is None:
                raise KnowledgeError(404, f"no knowledge named {name!r}")

    def list(self):
        with self.lock:
            return {"knowledge": [self.describe(n, d) for n, d in sorted(self.docs.items())]}

    @staticmethod
    def describe(name, doc):
        return {"name": name, "chars": len(doc["text"]), "pack_tokens": doc["tokens"], "pack": doc["pack"]}

    def load(self, specs=(), directory=None):
        """Startup documents: NAME=PATH specs and the *.md / *.txt files of a directory."""
        files = []
        for spec in specs:
            name, sep, path = spec.partition("=")
            if not sep:
                raise ValueError(f"--knowledge needs NAME=PATH, got {spec!r}")
            files.append((name, path))
        if directory:
            for entry in sorted(os.listdir(directory)):
                stem, ext = os.path.splitext(entry)
                if ext in (".md", ".txt") and os.path.isfile(os.path.join(directory, entry)):
                    files.append((stem, os.path.join(directory, entry)))
        out = []
        for name, path in files:
            with open(path, encoding="utf-8") as f:
                out.append(self.put(name, f.read()))
        return out
