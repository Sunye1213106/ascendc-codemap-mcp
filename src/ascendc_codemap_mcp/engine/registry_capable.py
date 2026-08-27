# -*- coding: utf-8 -*-
"""Registry competition + IsCapable predicates evaluated from source.

The predicate is parsed out of the C++ body and interpreted: nothing about a
particular operator or template class name is hard-coded. Root classification
comes from `source_resolver`, so a template whose IsCapable reads something the
accessor model does not know reports `Unknown` instead of silently passing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.anchors import arch_bucket, extract_registry
from ascendc_codemap_mcp.engine.cpp_expr import EvalUnknown, evaluate, parse_expr
from ascendc_codemap_mcp.engine.expr_ir import Call, Ref
from ascendc_codemap_mcp.engine.source_resolver import SourceResolver

# Legacy shorthand accepted by callers/tests, mapped onto canonical env keys.
LEGACY_ENV_KEYS = {
    "npu_arch": "PLATFORM_ARCH",
    "arch": "PLATFORM_ARCH",
    "actual_seq_qlen_present": "OPTIONAL_INPUT_PRESENCE[actual_seq_q_len]",
    "actual_seq_qlen_size": "INPUT_SHAPE[actual_seq_q_len].shape_size",
    "tnd_softmax_in": "ATTRIBUTE[tnd_softmax_in]",
}

COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
ENUM_RE = re.compile(r"enum\s+class\s+(\w+)\s*(?::\s*\w+\s*)?\{(.*?)\}", re.DOTALL)
# Declarator is the last identifier before '='; the type may carry *, & and <>.
DECLARATOR_RE = re.compile(r"(?P<name>\w+)\s*$")


@dataclass(frozen=True)
class EnumVal:
    """A scoped enum member: equal to its own name and ordered by its ordinal.

    Ascend C compares the same enum both ways — `npuArch == NpuArch::DAV_3510`
    treats it as an identity, while `GetAttrNum() > AttrIndex::TND_SOFTMAX_IN`
    treats it as an index — so the value has to satisfy both readings.
    """

    enum: str
    name: str
    ordinal: int | None = None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EnumVal):
            return (self.enum, self.name) == (other.enum, other.name)
        if isinstance(other, str):
            return other in (self.name, f"{self.enum}::{self.name}")
        if isinstance(other, int) and self.ordinal is not None:
            return self.ordinal == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.enum, self.name))

    def _ord(self) -> int:
        if self.ordinal is None:
            raise EvalUnknown(f"enum_ordinal_unknown:{self.enum}::{self.name}")
        return self.ordinal

    def __lt__(self, other: Any) -> bool:
        return self._ord() < int(other)

    def __le__(self, other: Any) -> bool:
        return self._ord() <= int(other)

    def __gt__(self, other: Any) -> bool:
        return self._ord() > int(other)

    def __ge__(self, other: Any) -> bool:
        return self._ord() >= int(other)

    def __int__(self) -> int:
        return self._ord()

    def __index__(self) -> int:
        return self._ord()

    def __str__(self) -> str:
        return f"{self.enum}::{self.name}"


# --- environment stubs ------------------------------------------------------
@dataclass
class TensorStub:
    name: str
    shape_size: int = 0

    def __bool__(self) -> bool:  # `tensor != nullptr` compares against None
        return True


@dataclass
class AttrsStub:
    count: int


def parse_enums(text: str) -> dict[str, dict[str, int]]:
    """`enum class AttrIndex { A = 0, B, C }` -> ordinals, for index comparisons."""
    out: dict[str, dict[str, int]] = {}
    for m in ENUM_RE.finditer(text):
        name = m.group(1)
        members: dict[str, int] = {}
        nxt = 0
        for part in m.group(2).split(","):
            part = COMMENT_RE.sub("", part).strip()
            if not part:
                continue
            if "=" in part:
                ident, val = part.split("=", 1)
                ident = ident.strip()
                try:
                    nxt = int(val.strip(), 0)
                except ValueError:
                    pass
            else:
                ident = part
            members[ident] = nxt
            nxt += 1
        out[name] = members
    return out


def normalize_env(env: dict[str, Any]) -> dict[str, Any]:
    out = dict(env)
    for legacy, canonical in LEGACY_ENV_KEYS.items():
        if legacy in env and canonical not in out:
            out[canonical] = env[legacy]
    return out


# --- statement model --------------------------------------------------------
@dataclass
class Stmt:
    kind: str  # decl | if | return | expr
    name: str = ""
    text: str = ""
    then: list["Stmt"] = field(default_factory=list)
    otherwise: list["Stmt"] = field(default_factory=list)


def _split_statements(body: str) -> list[str]:
    """Split a function body into top-level statements, brace/paren/string aware."""
    out: list[str] = []
    buf: list[str] = []
    depth_paren = depth_brace = 0
    in_str = in_chr = False
    i = 0
    while i < len(body):
        ch = body[i]
        if in_str:
            buf.append(ch)
            if ch == "\\":
                if i + 1 < len(body):
                    buf.append(body[i + 1])
                    i += 2
                    continue
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if in_chr:
            buf.append(ch)
            if ch == "'":
                in_chr = False
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch == "'":
            in_chr = True
        elif ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1
            if depth_brace == 0:
                buf.append(ch)
                out.append("".join(buf).strip())
                buf = []
                i += 1
                continue
        elif ch == ";" and depth_paren == 0 and depth_brace == 0:
            out.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


def _match_paren(src: str, start: int) -> int:
    depth = 0
    for j in range(start, len(src)):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    return -1


def parse_body(body: str) -> list[Stmt]:
    body = COMMENT_RE.sub("", body)
    stmts: list[Stmt] = []
    for raw in _split_statements(body):
        s = raw.strip()
        if not s:
            continue
        if s.startswith("if") and re.match(r"if\s*\(", s):
            open_i = s.index("(")
            close_i = _match_paren(s, open_i)
            cond = s[open_i + 1 : close_i]
            rest = s[close_i + 1 :].strip()
            then_src, else_src = _split_else(rest)
            stmts.append(
                Stmt(
                    kind="if",
                    text=cond,
                    then=parse_body(_unbrace(then_src)),
                    otherwise=parse_body(_unbrace(else_src)) if else_src else [],
                )
            )
            continue
        if s.startswith("return"):
            stmts.append(Stmt(kind="return", text=s[len("return") :].strip()))
            continue
        decl = _parse_decl(s)
        if decl is not None:
            stmts.append(decl)
            continue
        stmts.append(Stmt(kind="expr", text=s))
    return stmts


def _parse_decl(s: str) -> Stmt | None:
    """`const char *x = <init>` -> decl(x, init). Plain assignments are not decls."""
    eq = _top_level_assign(s)
    if eq < 0:
        return None
    lhs = s[:eq].strip()
    init = s[eq + 1 :].strip()
    if "(" in lhs or "[" in lhs or "." in lhs or "->" in lhs:
        return None
    lhs_clean = lhs.replace("*", " ").replace("&", " ")
    parts = lhs_clean.split()
    if len(parts) < 2:  # `x = y` is an assignment to an existing variable
        return None
    m = DECLARATOR_RE.search(lhs_clean)
    if not m:
        return None
    return Stmt(kind="decl", name=m.group("name"), text=init)


def _top_level_assign(s: str) -> int:
    """Index of a plain `=` outside parens/brackets/angle brackets/strings."""
    depth = angle = 0
    in_str = False
    i = 0
    while i < len(s):
        ch = s[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "<":
            angle += 1
        elif ch == ">":
            angle = max(0, angle - 1)
        elif ch == "=" and depth == 0 and angle == 0:
            prev = s[i - 1] if i else ""
            nxt = s[i + 1] if i + 1 < len(s) else ""
            if prev not in "=!<>+-*/%&|^" and nxt != "=":
                return i
        i += 1
    return -1


def _unbrace(src: str) -> str:
    src = src.strip()
    if src.startswith("{") and src.endswith("}"):
        return src[1:-1]
    return src


def _split_else(rest: str) -> tuple[str, str]:
    rest = rest.strip()
    if rest.startswith("{"):
        depth = 0
        for j, ch in enumerate(rest):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    then_src = rest[: j + 1]
                    after = rest[j + 1 :].strip()
                    if after.startswith("else"):
                        return then_src, after[len("else") :].strip()
                    return then_src, ""
        return rest, ""
    m = re.search(r"\belse\b", rest)
    if m:
        return rest[: m.start()], rest[m.end() :]
    return rest, ""


# --- predicate --------------------------------------------------------------
@dataclass
class CapablePred:
    class_name: str
    file: str
    line: int
    body: str
    roots: list[str] = field(default_factory=list)
    statements: list[Stmt] = field(default_factory=list)
    bindings: dict[str, str] = field(default_factory=dict)
    enums: dict[str, dict[str, int]] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)

    # -- evaluation ----------------------------------------------------------
    def _call_hook(self, call: Call, env: dict[str, Any]) -> Any:
        name = call.func[len("field:") :] if call.func.startswith("field:") else call.func
        args = call.args

        if name == "strcmp":
            a = self._eval(args[0], env) if len(args) > 0 else ""
            b = self._eval(args[1], env) if len(args) > 1 else ""
            return 0 if a == b else (1 if str(a) > str(b) else -1)
        if name.startswith("GetOptionalInput"):
            idx = self._index_arg(args)
            key = f"OPTIONAL_INPUT_PRESENCE[{idx}]"
            if key not in env:
                raise EvalUnknown(f"unbound:{key}")
            if not env[key]:
                return None
            size = env.get(f"INPUT_SHAPE[{idx}].shape_size", 0)
            return TensorStub(name=idx, shape_size=size)
        if name in ("GetShapeSize", "GetStorageShapeSize"):
            base = self._eval(args[0], env) if args else None
            if base is None:
                raise EvalUnknown("shape_size_of_null")
            return getattr(base, "shape_size", 0)
        if name == "GetAttrs":
            count = env.get("ATTRIBUTE[attr_num]")
            if count is None:
                count = len(self.enums.get("AttrIndex", {})) or 0
            return AttrsStub(count=count)
        if name == "GetAttrNum":
            base = self._eval(args[0], env) if args else None
            if isinstance(base, AttrsStub):
                return base.count
            return env.get("ATTRIBUTE[attr_num]", len(self.enums.get("AttrIndex", {})))
        if name.startswith("GetAttrPointer") or name.startswith("GetAttr"):
            idx = self._index_arg(args)
            key = f"ATTRIBUTE[{idx}]"
            if key in env:
                return env[key]
            raise EvalUnknown(f"unbound:{key}")
        if name.startswith("GetCurNpuArch") or name.startswith("GetSocVersion"):
            if "PLATFORM_ARCH" in env:
                return env["PLATFORM_ARCH"]
            raise EvalUnknown("unbound:PLATFORM_ARCH")
        raise EvalUnknown(f"call:{name}")

    def _index_arg(self, args) -> str:
        for a in args:
            if isinstance(a, Ref) and "::" in a.symbol:
                head, tail = a.symbol.split("::", 1)
                if head.endswith("Index"):
                    return tail.lower()
        return "?"

    def _resolve_ref(self, sym: str, env: dict[str, Any]) -> Any:
        if "::" in sym:
            head, tail = sym.split("::", 1)
            ordinals = self.enums.get(head, {})
            return EnumVal(enum=head, name=tail, ordinal=ordinals.get(tail))
        if sym in self.bindings:
            return self._eval(parse_expr(self.bindings[sym]), env)
        if sym in env:
            return env[sym]
        for pat, key in (
            (r"^npuArch$|^socVersion$", "PLATFORM_ARCH"),
            (r"^aivNum$|^coreNum$", "PLATFORM_CORE_COUNT"),
            (r"^ubSize$", "PLATFORM_MEMORY_SIZE"),
        ):
            if re.search(pat, sym) and key in env:
                return env[key]
        raise EvalUnknown(f"unbound:{sym}")

    def _eval(self, expr, env: dict[str, Any]) -> Any:
        return evaluate(
            expr,
            env,
            call_hook=self._call_hook,
            symbol_hook=self._resolve_ref,
        )

    def _exec(self, stmts: list[Stmt], env: dict[str, Any]) -> bool | None:
        for st in stmts:
            if st.kind == "decl":
                self.bindings[st.name] = st.text
                continue
            if st.kind == "return":
                return bool(self._eval(parse_expr(st.text), env))
            if st.kind == "if":
                cond = self._eval(parse_expr(st.text), env)
                branch = st.then if cond else st.otherwise
                got = self._exec(branch, env)
                if got is not None:
                    return got
                continue
        return None

    def evaluate_env(self, env: dict[str, Any]) -> bool | None:
        """True/False from the parsed body, or None when a symbol is unbound."""
        env = normalize_env(env)
        self.bindings = {}
        try:
            got = self._exec(self.statements, env)
        except EvalUnknown as exc:
            self.unresolved.append(str(exc))
            return None
        return bool(got) if got is not None else False

    # Backwards-compatible name used by existing callers/tests.
    def eval_arch35(self, env: dict[str, Any]) -> bool | None:
        return self.evaluate_env(env)


def _brace_body(src: str, open_idx: int) -> str:
    depth = 0
    for j in range(open_idx, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[open_idx + 1 : j]
    return src[open_idx + 1 :]


def _collect_roots(pred: CapablePred) -> tuple[list[str], list[str]]:
    """Root Sources for every guard and return value in the predicate body."""
    bindings = {s.name: s.text for s in _iter_stmts(pred.statements) if s.kind == "decl"}
    resolver = SourceResolver(bindings=bindings)
    roots: list[str] = []
    reasons: list[str] = []
    for st in _iter_stmts(pred.statements):
        if st.kind not in ("if", "return"):
            continue
        if st.kind == "return" and st.text.strip() in ("true", "false"):
            continue
        res = resolver.resolve(st.text)
        for r in res.roots:
            if r != "CONSTANT" and r not in roots:
                roots.append(r)
        for reason in res.reasons:
            if reason not in reasons:
                reasons.append(reason)
    return roots, reasons


def _iter_stmts(stmts: list[Stmt]):
    for st in stmts:
        yield st
        yield from _iter_stmts(st.then)
        yield from _iter_stmts(st.otherwise)


def extract_iscapable(path: str | Path, class_name: str | None = None) -> list[CapablePred]:
    src = Path(path).read_text(encoding="utf-8", errors="replace")
    enums = parse_enums(src)
    enums.update(_enums_from_siblings(Path(path)))
    out: list[CapablePred] = []
    seen: set[int] = set()

    for m in re.finditer(r"bool\s+(?:(\w+)::)?IsCapable\s*\(\s*\)\s*(?:const\s*)?(?:override\s*)?\{", src):
        if m.start() in seen:
            continue
        seen.add(m.start())
        cls = m.group(1)
        if not cls:
            before = src[: m.start()]
            cm = list(re.finditer(r"class\s+(\w+)", before))
            cls = cm[-1].group(1) if cm else "?"
        body = _brace_body(src, m.end() - 1).strip()
        pred = CapablePred(
            class_name=cls,
            file=str(path).replace("\\", "/"),
            line=src[: m.start()].count("\n") + 1,
            body=body,
            statements=parse_body(body),
            enums=enums,
        )
        pred.roots, pred.unresolved = _collect_roots(pred)
        out.append(pred)

    if class_name:
        out = [p for p in out if class_name in p.class_name]
    return out


def _enums_from_siblings(path: Path) -> dict[str, dict[str, int]]:
    """Index enums declared in the headers next to the tiling implementation."""
    out: dict[str, dict[str, int]] = {}
    for hdr in sorted(path.parent.glob("*.h")):
        try:
            out.update(parse_enums(hdr.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    return out


def classify_roots(body: str) -> list[str]:
    """Root Sources of a raw predicate body (kept for callers that only have text)."""
    pred = CapablePred(class_name="?", file="?", line=0, body=body, statements=parse_body(body))
    roots, _ = _collect_roots(pred)
    return roots


@dataclass
class Competition:
    arch: str
    ordered: list[dict]
    preds: dict[str, CapablePred] = field(default_factory=dict)

    def choose(self, env: dict[str, Any]) -> str | None:
        for r in self.ordered:
            pred = self.preds.get(r["class"])
            if pred is None:
                continue
            if pred.evaluate_env(env):
                return r["class"]
        return None

    def choose_with_reason(self, env: dict[str, Any]) -> dict[str, Any]:
        trace = []
        for r in self.ordered:
            pred = self.preds.get(r["class"])
            if pred is None:
                trace.append({"class": r["class"], "capable": None, "reason": "NO_PREDICATE"})
                continue
            verdict = pred.evaluate_env(env)
            trace.append(
                {
                    "class": r["class"],
                    "priority": r["priority"],
                    "capable": verdict,
                    "reason": None if verdict is not None else "UNKNOWN_SYMBOL",
                }
            )
            if verdict:
                return {"chosen": r["class"], "trace": trace, "lineage": self.selection_lineage(r["class"])}
        return {"chosen": None, "trace": trace, "lineage": None}

    def selection_lineage(self, chosen: str) -> str:
        """chosen(Tk) <=> forall i<k not capable(Ti) and capable(Tk)"""
        names = [r["class"] for r in self.ordered]
        if chosen not in names:
            raise ValueError(chosen)
        k = names.index(chosen)
        parts = [f"not capable({names[i]})" for i in range(k)]
        parts.append(f"capable({chosen})")
        return " and ".join(parts)


def build_arch35_competition(host_root: str | Path, *, op_name: str) -> Competition:
    return build_competition(host_root, arch="DAV_3510", op_name=op_name)


def build_competition(
    host_root: str | Path, arch: str = "DAV_3510", *, op_name: str
) -> Competition:
    regs = extract_registry(host_root, op_name)
    ordered = sorted(
        [r for r in regs if arch_bucket(r["arch_expr"]) == arch],
        key=lambda r: r["priority"],
    )
    preds: dict[str, CapablePred] = {}
    for r in ordered:
        found = extract_iscapable(r["file"], class_name=r["class"])
        if not found:
            found = [p for p in extract_iscapable(r["file"]) if r["class"] in p.class_name]
        if found:
            preds[r["class"]] = found[0]
    return Competition(arch=arch, ordered=ordered, preds=preds)
