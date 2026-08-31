# -*- coding: utf-8 -*-
"""Kernel Root Trace — source-rooted graph symmetric with Host UO.

Algorithm (no execution analysis):

  1. Collect source facts (Clang walks + lexical fallback)
  2. Build complete type / alias / member / call graph
  3. Seed AscendC / CANN terminal roots (+ known framework wrapper contracts)
  4. Single reverse fixed-point over WRAPS / ALIASES / CALLS
  5. Mark REACHED / UNRESOLVED / EXTERNAL with auditable gaps

Does **not** compute exec_rank, RAW/WAR/WAW, happens-before, pipeline,
buffer lifecycle, or engine scheduling. Flag APIs (Set/Wait, CrossCore*, IB*)
record identity-level pair appearance. TQue EnQue/DeQue stay outside that
check — CANN encapsulates their handshake.
"""

from __future__ import annotations

from ascendc_codemap_mcp.engine.paths import require_architecture
import os
import re
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.perf import TimeBudget, kernel_root_trace_budget_s
from ascendc_codemap_mcp.engine.ids import make_id, operation_site_id
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap, relation_id
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.identity import (
    bind_local_object,
    bind_or_create,
    find_declaration,
    is_forbidden_callable_name,
    trusted_scope_name,
)
from ascendc_codemap_mcp.engine.ir.evidence import TRUST_ADVISORY
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes import kernel_scan as kscan
from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text
from ascendc_codemap_mcp.engine.semantics import registry as semreg
from ascendc_codemap_mcp.engine.semantics.ascendc_storage import (
    ASCENDC_BUFFER_TYPES,
    ASCENDC_REGISTER_TYPES,
    REGISTER_MEMORY_SPACE,
    SHARE_BUFFER_CALLEES,
    STACK_BUFFER_CALLEES,
    TENSOR_METHOD_BRIDGES,
    TPIPE_METHOD_BRIDGES,
    TQUE_METHOD_BRIDGES,
    is_non_storage_type,
    is_storage_type_text,
    is_valid_storage_name,
    register_class_from_type,
    resolve_buffer_decl,
    storage_root_kind_from_space,
    tposition_from_type_text,
)
from ascendc_codemap_mcp.engine.semantics.ascendc_storage import (
    memory_space_from_type_text as _cann_memory_space,
)
from ascendc_codemap_mcp.engine.semantics.ascendc_vf import (
    AMBIGUOUS_VF_ROOTS,
    VF_ALIASES,
    architecture_has_vf,
    is_ambiguous_vf_name,
    is_cann_vf_api,
    is_vf_only_api,
    vf_root_spelling,
)
from ascendc_codemap_mcp.engine.pipe_lifetime import (
    continued_line_ranges,
    lifetime_edges,
    receiver_leaf,
    topo_pipe_ordinals,
)
from ascendc_codemap_mcp.engine.semantics.ascendc_util import is_cann_util_api
from ascendc_codemap_mcp.engine.semantics.ascendc_sync import (
    ASCENDC_SYNC_TYPES,
    FLAG_PAIR_MATE,
    SYNC_MECHANISM,
    SYNC_SPELLING_ALIASES,
    VF_GATED_SYNC,
    canonical_sync_name,
    flag_pair_key,
    is_flag_sync,
    is_sync_root,
    is_tpipe_callee,
    is_tque_callee,
    resolve_sync_site,
)

# ---------------------------------------------------------------------------
# Reason codes (auditable gaps)
# ---------------------------------------------------------------------------

REASON_NO_ASCENDC_ROOT = "NO_ASCENDC_ROOT_REACHED"
REASON_TYPE_UNRESOLVED = "TYPE_CANONICALIZATION_FAILED"
REASON_CALL_UNRESOLVED = "NO_ASCENDC_ROOT_REACHED"
REASON_EXTERNAL = "EXTERNAL_DECL_UNAVAILABLE"
REASON_UNPAIRED_FLAG_SYNC = "UNPAIRED_FLAG_SYNC"

#: This pass reads clang cursor rows, so its facts are authoritative. Nodes it
#: mints from a lexical fallback instead say so with their own label.
_PROV = "kernel_root_trace"

_ROOT_KIND_BY_CATEGORY: dict[str, str] = {
    "memory_transfer": "MEMORY_API",
    "memory_init": "MEMORY_API",
    "buffer_init": "MEMORY_API",
    "buffer_acquire": "MEMORY_API",
    "buffer_release": "MEMORY_API",
    "buffer_view": "MEMORY_API",
    "queue_enqueue": "MEMORY_API",
    "queue_dequeue": "MEMORY_API",
    "util": "UTIL_API",
    "sync_signal": "SYNC",
    "sync_wait": "SYNC",
    "sync_barrier": "SYNC",
    "reg_mask": "REGISTER",
    "reg_load": "REGISTER",
    "reg_store": "REGISTER",
    "reg_compute": "REGISTER",
    "vector": "COMPUTE_API",
    "vector_compute": "COMPUTE_API",
    "cube": "COMPUTE_API",
    "cube_compute": "COMPUTE_API",
    "cube_load": "COMPUTE_API",
    "cube_store": "COMPUTE_API",
    "memory_atomic": "MEMORY_API",
}

# AscendC / CANN terminal API spellings used as catalog roots (not project names).
_ASCENDC_API_ROOTS: frozenset[str] = frozenset(
    set(ASCENDC_BUFFER_TYPES)
    | set(ASCENDC_REGISTER_TYPES)
    | set(SYNC_MECHANISM)
    | {
        "DataCopy",
        "DataCopyPad",
        "InitBuffer",
        "AllocTensor",
        "FreeTensor",
        "EnQue",
        "DeQue",
        "Cast",
        "SetAtomicAdd",
        "SetAtomicNone",
        "SetAtomicType",
        "Mmad",
        "LoadData",
        "Fixpipe",
        "Matmul",
        # TPipe / tensor (kernel_tpipe.h, kernel_tensor.h, kernel_common.h)
        "TPipe",
        "GroupBarrier",
        "TQueSync",
        "GetTPipePtr",
        "GetBlockIdx",
        "GetSubBlockIdx",
        "SetGlobalBuffer",
        "GetPhyAddr",
        "InitOutput",
        "PopStackBuffer",
        "InitShareBufStart",
        "InitShareBufEnd",
        # Level-2 vector (arch22 and arch35). VF-only LoadAlign/CreateMask
        # come from is_cann_vf_api and are gated by architecture.
        "Interleave",
        "Duplicate",
        "ReduceSum",
        "Muls",
        "Adds",
        "Mul",
        "Add",
        "Sub",
        "Div",
        "Exp",
        "Abs",
        "Ceil",
        "Select",
        "sqrt",
        "Log",
    }
)

# Min/Max/Or/And/Xor also exist as project scalar/logic helpers. Prove only
# when the call looks like vector/Reg (3+ args or a typed tensor/register operand).
_VECTOR_AMBIGUOUS_ROOTS: frozenset[str] = frozenset(AMBIGUOUS_VF_ROOTS)

# Catalog spellings that are member contracts, never free-function roots.
_MEMBER_ONLY_ROOTS: frozenset[str] = frozenset({"Get"})

_CLASS_RE = re.compile(r"\b(?:class|struct)\s+(?P<name>[A-Za-z_]\w*)\b")
_USING_RE = re.compile(
    r"\busing\s+(?P<alias>[A-Za-z_]\w*)\s*=\s*(?P<target>[^;{]{1,400})\s*;"
)
_TYPEDEF_RE = re.compile(
    r"\btypedef\s+(?P<target>[\w:<>,\s*&]+?)\s+(?P<alias>[A-Za-z_]\w*)\s*;"
)
_MEMBER_RE = re.compile(
    r"(?P<type>(?:[\w:<>,\s*&]+?))\s+(?P<name>[A-Za-z_]\w*)\s*;"
)
_CONTINUATION_NAME_RE = re.compile(r"^\s*(?P<name>[A-Za-z_]\w*)\s*;\s*$")
_CXX_SKIP_BASE = frozenset(
    {
        "public",
        "private",
        "protected",
        "return",
        "if",
        "for",
        "while",
        "switch",
        "int",
        "float",
        "double",
        "bool",
        "char",
        "void",
        "auto",
        "size_t",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
        "half",
        "bfloat16_t",
        "constexpr",
        "consteval",
        "constinit",
    }
)


_VIEW_GETTERS = frozenset(
    {"Get", "GetTensor", "GetPre", "GetReused", "AllocTensor", "DeQue"}
)
_ASSIGN_LHS_RE = re.compile(r"\b([A-Za-z_]\w*)\s*=")
_MEMORY_TRANSFER_CALLEES = frozenset(
    {"DataCopy", "DataCopyPad", "Copy", "DataCopyScatter"}
)


def _budget_s() -> float:
    return kernel_root_trace_budget_s()


def _enabled() -> bool:
    raw = str(os.environ.get("UO_KERNEL_ROOT_TRACE") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _norm_file(path: str, root: str = "") -> str:
    return kscan.norm_file(path, root)


def _base_type_name(type_text: str) -> str:
    text = str(type_text or "").strip()
    text = re.sub(r"\b(?:const|volatile|static|mutable|typename|template)\b", " ", text)
    text = text.replace("&", " ").replace("*", " ")
    no_tpl = text.split("<", 1)[0].strip()
    token = no_tpl.split("::")[-1].strip()
    return token if token.isidentifier() else ""


def _persist_type_name(type_text: str) -> str:
    """Short type spelling for CodeMap attrs — never the class body."""
    return _base_type_name(type_text) or str(type_text or "").split("<", 1)[0].strip()[:120]


_SYNC_PRECEDES_CALLEES: frozenset[str] = frozenset(
    {
        "DataCopy",
        "DataCopyPad",
        "AllocTensor",
        "FreeTensor",
        "EnQue",
        "DeQue",
        "SetFlag",
        "WaitFlag",
        "CrossCoreSetFlag",
        "CrossCoreWaitFlag",
        "PipeBarrier",
        "DataSyncBarrier",
        "InitBuffer",
        "PopStackBuffer",
        "Cast",
    }
    | set(SYNC_MECHANISM)
    | set(STACK_BUFFER_CALLEES)
)


def _type_identity_key(type_text: str, *, usr: str = "", qualified: str = "") -> str:
    """Stable TYPE key: USR > qualified > templated spelling > short base name."""
    if usr:
        return f"usr:{usr}"
    if qualified:
        return f"q:{qualified}"
    text = str(type_text or "").strip()
    if not text:
        return ""
    # Keep template args so Buffer<TPosition::VECIN> ≠ bare / project Buffer.
    if "<" in text:
        compact = re.sub(r"\s+", "", text)
        return f"tpl:{compact}"
    return _base_type_name(text)


_TRACE_ARCHITECTURE = ""


def memory_space_from_type_text(type_text: str) -> str | None:
    """Physical tier for *type_text* on the chip this trace is running for.

    CANN ``GetPhyType`` reads C1 / C2 / CO2 / C2PIPE2GM out of an
    ``#if __NPU_ARCH__`` chain, so the tier is only decided once the
    architecture is bound. Same module-global binding as ``_vf_blocked``.
    """
    return _cann_memory_space(type_text, _TRACE_ARCHITECTURE)


def _vf_blocked(name: str) -> bool:
    """VF/Reg/SIMT roots are illegal on arch22."""
    if architecture_has_vf(_TRACE_ARCHITECTURE):
        return False
    short = canonical_sync_name(vf_root_spelling(name))
    if short in VF_GATED_SYNC or name in VF_GATED_SYNC:
        return True
    return is_vf_only_api(short)


def _is_ascendc_root_spelling(name: str) -> bool:
    """Catalog candidate check only — not REACHED proof."""
    if _vf_blocked(name):
        return False
    short = canonical_sync_name(vf_root_spelling(name))
    return (
        name in _ASCENDC_API_ROOTS
        or short in _ASCENDC_API_ROOTS
        or name in ASCENDC_BUFFER_TYPES
        or name in ASCENDC_REGISTER_TYPES
        or short in ASCENDC_SYNC_TYPES
        or is_cann_vf_api(name, architecture=_TRACE_ARCHITECTURE)
        or is_cann_util_api(name)
        or is_sync_root(name)
    )


_FRAMEWORK_DECL_MARKERS = (
    "/ascendc",
    "ascendc/",
    "/cann",
    "cann-",
    "tikcfw",
    "bisheng",
    "metadef",
    "kernel_operator",
    "basic_api",
    "impl/dav_",
    "include/aclnn",
)


def _is_framework_decl_file(path: str) -> bool:
    f = str(path or "").replace("\\", "/").lower()
    if not f:
        return False
    # Operator project sources are never AscendC catalog origins.
    if "/op_kernel/" in f or "/op_host/" in f:
        return False
    return any(m in f for m in _FRAMEWORK_DECL_MARKERS)


def _qualified_looks_ascendc(qualified: str) -> bool:
    q = str(qualified or "")
    if not q:
        return False
    return q.startswith("AscendC::") or "::AscendC::" in f"::{q}" or q.startswith("Cann::")


def _short_from_qualified(qualified: str, fallback: str = "") -> str:
    text = str(qualified or fallback or "").strip()
    if not text:
        return ""
    no_tpl = text.split("<", 1)[0].strip()
    short = no_tpl.split("::")[-1].strip()
    # Clang call-expr / ctor spellings: ``RegTensor()``, ``FixpipeConfig(CO2Layout, bool)``.
    if "(" in short:
        short = short.split("(", 1)[0].strip()
    return short


def _looks_like_framework_type_name(name: str) -> bool:
    """CANN catalog roots and param/config structs — never project class names."""
    if not name or not name[0].isupper():
        return False
    if not re.match(r"^[A-Za-z_]\w*$", name):
        return False
    if _is_ascendc_root_spelling(name):
        return True
    return name.endswith(("Params", "Config", "ExtParams", "PadParams")) or "Params" in name


_SPELLING_ALIASES: dict[str, str] = {"abs": "Abs", **VF_ALIASES, **SYNC_SPELLING_ALIASES}

_BUILTIN_SPELLINGS: frozenset[str] = frozenset(
    {
        "vector_bool",
        "vector_align",
        "__bs_f16",
        "memcpy",
        "conditional",
        "conditional_t",
    }
)


def _is_project_decl_file(path: str) -> bool:
    f = str(path or "").replace("\\", "/").lower()
    return "/op_kernel/" in f or "/op_host/" in f


def _is_compiler_builtin(
    name: str,
    *,
    usr: str = "",
    decl_file: str = "",
    qualified: str = "",
) -> bool:
    n = str(name or "")
    if n.startswith("__builtin_") or n.startswith("__bs_"):
        return True
    if n in _BUILTIN_SPELLINGS:
        return True
    q = str(qualified or "")
    if q.startswith("std::") or "::std::" in f"::{q}":
        return True
    f = str(decl_file or "").replace("\\", "/").lower()
    if "bisheng_prelude" in f:
        return True
    u = str(usr or "")
    if "c:@F@__builtin" in u or u.startswith("c:@S@vector_") or u.startswith("c:@S@__bs_"):
        return True
    return False


def _is_type_like_root(callee: str) -> bool:
    """RegTensor / LocalTensor / *Params are types, not execution operations."""
    short = str(callee or "").split("::")[-1]
    if short in ASCENDC_REGISTER_TYPES or short in ASCENDC_BUFFER_TYPES:
        return True
    if short in {"TPipe"} or short in ASCENDC_SYNC_TYPES:
        return True
    if short.endswith(("Params", "Config", "ExtParams", "PadParams")) or "Params" in short:
        return True
    return False


def _should_mint_operation(
    *,
    callee: str,
    is_root: bool,
    is_builtin: bool,
    is_project: bool,
) -> bool:
    """OPERATION nodes are source call sites of execution primitives, not every CallExpr."""
    if is_builtin:
        return False
    if _vf_blocked(callee):
        return False
    if _is_type_like_root(callee):
        return False
    if is_project and not is_root:
        return False
    # Or/Min/Max/And live in the VF catalog and as project scalars. Only mint
    # when this site was already proven as an AscendC root.
    if callee in _VECTOR_AMBIGUOUS_ROOTS or is_ambiguous_vf_name(callee):
        return bool(is_root)
    return bool(is_root or semreg.is_execution_primitive(callee))


def _receiver_looks_tensor(*texts: str) -> bool:
    blob = " ".join(str(t or "") for t in texts)
    return "LocalTensor" in blob or "GlobalTensor" in blob


def _usr_is_ascendc_tensor_method(usr: str) -> bool:
    u = str(usr or "")
    if "AscendC" not in u:
        return False
    return "LocalTensor" in u or "GlobalTensor" in u


def _prove_ascendc_api_root(
    *,
    callee: str,
    callee_qualified: str = "",
    callee_usr: str = "",
    callee_decl_file: str = "",
    receiver: str = "",
    receiver_type: str = "",
    receiver_canonical_type: str = "",
    has_identity: bool = False,
) -> tuple[bool, str]:
    """Return (proven, root_spelling). Spelling alone never proves member calls.

    Proof order:
      1. AscendC/CANN qualified name
      2. Framework declaration file (API catalog **or** param/config ctor)
      3. Free-function catalog match only when identity is unavailable (lexical)
         and there is no receiver (not a member call)
    """
    short = _short_from_qualified(callee_qualified, callee)
    short = _SPELLING_ALIASES.get(short, short)
    callee_canon = _SPELLING_ALIASES.get(callee, callee)
    in_catalog = bool(short and _is_ascendc_root_spelling(short)) or _is_ascendc_root_spelling(
        callee_canon
    )
    if in_catalog and not (short and _is_ascendc_root_spelling(short)):
        short = callee_canon
    if not short:
        short = callee_canon

    if _usr_is_ascendc_tensor_method(callee_usr) and callee in TENSOR_METHOD_BRIDGES:
        return True, callee
    if _receiver_looks_tensor(receiver_type, receiver_canonical_type) and callee in TENSOR_METHOD_BRIDGES:
        return True, callee
    # Get is a member contract. A CANN header hit is not proof of a free
    # AscendC::Get — Policy/Selector/TBuf all share the name.
    if short in _MEMBER_ONLY_ROOTS:
        if _qualified_looks_ascendc(callee_qualified) and not (
            receiver or receiver_type or receiver_canonical_type
        ):
            return True, short
        return False, ""

    # Framework-declared free/ctor symbols (DataCopyExtParams, FixpipeParams*, …)
    # need not sit in the compute-API catalog — the CANN header is the proof.
    if (
        _is_framework_decl_file(callee_decl_file)
        and not (receiver or receiver_type or receiver_canonical_type)
        and _looks_like_framework_type_name(short)
    ):
        if short in _VECTOR_AMBIGUOUS_ROOTS:
            # Min/Max still need call-shape evidence at the call site.
            pass
        else:
            return True, short

    if not in_catalog:
        if _qualified_looks_ascendc(callee_qualified):
            short = _SPELLING_ALIASES.get(
                _short_from_qualified(callee_qualified, callee),
                _short_from_qualified(callee_qualified, callee),
            )
            if short and _is_ascendc_root_spelling(short):
                return True, short
            # AscendC::DataCopyScatter and similar live in CANN headers even when
            # they are not in the small catalog.
            if _is_framework_decl_file(callee_decl_file) and not (
                receiver or receiver_type or receiver_canonical_type
            ):
                return True, short or callee
        if (
            _is_framework_decl_file(callee_decl_file)
            and "AscendC" in str(callee_usr or callee_qualified or "")
            and not (receiver or receiver_type or receiver_canonical_type)
        ):
            return True, short or callee
        return False, ""

    if _qualified_looks_ascendc(callee_qualified):
        return True, short
    if _is_framework_decl_file(callee_decl_file):
        return True, short
    # USR often encodes AscendC::Reg::RegTensor even when qualified is bare.
    if "AscendC" in str(callee_usr or "") and _is_ascendc_root_spelling(short):
        return True, short

    # Identity present but not AscendC/framework → project symbol, never root.
    if has_identity or callee_usr or callee_qualified or callee_decl_file:
        return False, ""

    # Lexical / unresolved free call: catalog match without receiver only.
    if receiver or receiver_type or receiver_canonical_type:
        return False, ""
    # Member-only contracts and Min/Max/Or never prove from the spelling alone.
    if short in _MEMBER_ONLY_ROOTS or short in _VECTOR_AMBIGUOUS_ROOTS or is_ambiguous_vf_name(short):
        return False, ""
    return True, short


_FLAG_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
#: A flag argument may be arithmetic on a base id, but not a call and not a
#: subscript: those name something this pass cannot follow to a single flag.
_FLAG_EXPR_OK = re.compile(r"^[\w\s+\-*/()]+$")


def _flag_identity(text: str) -> tuple[str, str]:
    """``(base flag, whole expression)`` for a sync flag argument.

    Cross-core flags are routinely written as ``id0_ + AIV0_AIV1_OFFSET``.
    Requiring a bare identifier dropped every such site, so a handshake with
    five sets and five waits was reported as three and four — a difference a
    reader is entitled to read as a bug in the operator. The offset is part of
    the identity, since two offsets off one base are two different flags.
    """
    expr = " ".join(str(text or "").split())
    if not expr or not _FLAG_EXPR_OK.match(expr):
        return "", ""
    names = _FLAG_IDENT_RE.findall(expr)
    if not names:
        return "", ""
    return names[0], expr


def _root_entity_id(spelling: str) -> str:
    return make_id("Root", "ascendc", spelling)


def _ensure_ascendc_root(codemap: CodeMap, spelling: str, *, root_kind: str) -> str:
    eid = _root_entity_id(spelling)
    codemap.upsert(
        EntityKind.TYPE,
        f"AscendC::{spelling}",
        eid=eid,
        attrs={
            "root_status": "REACHED",
            "root_kind": root_kind,
            "root": f"AscendC::{spelling}",
            "catalog": "ascendc",
            "spelling": spelling,
            # A catalog root is a node this pass synthesises to hang reached
            # types off, not something read out of the operator's source.
            "provenance": "catalog_root",
        },
        status="extracted",
        confidence=1.0,
    )
    return eid


def _category_root_kind(category: str, callee: str) -> str:
    canon = canonical_sync_name(callee)
    if canon in SYNC_MECHANISM or callee in SYNC_MECHANISM or category.startswith("sync_"):
        return "SYNC"
    if callee in ASCENDC_REGISTER_TYPES or category.startswith("reg_"):
        return "REGISTER"
    if category in _ROOT_KIND_BY_CATEGORY:
        return _ROOT_KIND_BY_CATEGORY[category]
    if callee in ASCENDC_BUFFER_TYPES:
        return "STORAGE"
    low = str(callee or "")
    if low.endswith(("Params", "Config", "ExtParams", "PadParams")) or "Params" in low:
        return "FRAMEWORK_TYPE"
    return "COMPUTE_API"


def _decl_fields(decl: Any) -> tuple[str, str, str, str, int]:
    if isinstance(decl, dict):
        return (
            str(decl.get("type_text") or decl.get("type") or ""),
            str(decl.get("name") or ""),
            str(decl.get("function") or decl.get("scope") or ""),
            str(decl.get("file") or ""),
            int(decl.get("line") or 0),
        )
    return (
        str(getattr(decl, "type_text", "") or getattr(decl, "type", "") or ""),
        str(getattr(decl, "name", "") or ""),
        str(getattr(decl, "function", "") or getattr(decl, "scope", "") or ""),
        str(getattr(decl, "file", "") or ""),
        int(getattr(decl, "line", 0) or 0),
    )


def _sync_object_kind(type_text: str) -> EntityKind | None:
    """Classify only explicit AscendC sync/storage type spellings."""
    text = re.sub(r"\s+", "", str(type_text or ""))
    if re.search(r"(?:^|::)(?:const|volatile)*TPipe(?:\*|\&|<|$)", text):
        return EntityKind.PIPE
    if re.search(r"(?:^|::)(?:const|volatile)*TQue(?:Bind)?(?:\*|\&|<|$)", text):
        return EntityKind.QUEUE
    if re.search(r"(?:^|::)(?:const|volatile)*HardEvent(?:Aic|Aiv)?(?:\*|\&|<|$)", text):
        return EntityKind.EVENT
    return None


def _type_is_pointer(type_text: str) -> bool:
    return "*" in re.sub(r"\s+", "", str(type_text or ""))


def _catalog_storage_root(type_text: str) -> str:
    """CANN storage/queue spelling in a type, or empty."""
    text = str(type_text or "")
    for spell in ("LocalTensor", "GlobalTensor", "TBufPool", "TBuf", "TQueBind", "TQue"):
        if spell in text:
            return spell
    return ""


def _is_catalog_storage_base(name: str) -> bool:
    """True when *name* itself is a catalog storage root, not a substring hit."""
    short = str(name or "").split("::")[-1]
    return short in ASCENDC_BUFFER_TYPES or short in {
        "LocalTensor",
        "GlobalTensor",
        "TBufPool",
        "TBuf",
        "TQueBind",
        "TQue",
    }


def _is_catalog_type_entity(ent: Any) -> bool:
    if ent is None:
        return False
    if ent.kind_name() != EntityKind.TYPE.value:
        return False
    if str(ent.attrs.get("catalog") or "") == "ascendc":
        return True
    return _is_catalog_storage_base(str(ent.name or ""))


def _is_storage_object(ent: Any) -> bool:
    """TBuf / TQue / InitBuffer-allocated BUFFER, QUEUE, or REGISTER — not a view TYPE."""
    if ent is None:
        return False
    kind = ent.kind_name()
    if kind == EntityKind.REGISTER.value:
        return True
    if kind == EntityKind.QUEUE.value:
        return True
    if kind != EntityKind.BUFFER.value:
        return False
    if str(ent.attrs.get("catalog") or "") == "ascendc":
        return False
    type_name = str(ent.attrs.get("type_name") or "")
    root = str(ent.attrs.get("root") or "").replace("AscendC::", "").split("::")[-1]
    if root in {"TBuf", "TQue", "TBufPool", "TQueBind"}:
        return True
    if "TBuf" in type_name or "TQue" in type_name:
        return True
    if "LocalTensor" in type_name or "GlobalTensor" in type_name:
        return False
    space = str(
        ent.attrs.get("physical_space") or ent.attrs.get("memory_space") or ""
    ).strip()
    if space and space != "UNKNOWN":
        return True
    return bool(ent.attrs.get("allocated"))


def _assign_lhs_on_line(text: str, line: int) -> str:
    if not text or int(line or 0) <= 0:
        return ""
    rows = str(text).splitlines()
    if line > len(rows):
        return ""
    hit = _ASSIGN_LHS_RE.search(rows[line - 1])
    name = str(hit.group(1) or "") if hit else ""
    return name if name.isidentifier() and not is_forbidden_callable_name(name) else ""


def _wraps_lock_type(type_text: str) -> bool:
    text = str(type_text or "")
    return "MutexID" in text or re.search(r"(?:^|::)Mutex(?:<|$)", text) is not None


def _propagate_wrap_flags(codemap: CodeMap) -> None:
    """Push wraps_storage / wraps_lock / wraps_flag up WRAPS (source composition)."""
    parents: dict[str, list[str]] = defaultdict(list)
    for rel in codemap.relations.values():
        if rel.kind_name() != RelationKind.WRAPS.value:
            continue
        parents[rel.dst].append(rel.src)

    def _seed_storage(ent: Any) -> bool:
        return bool(
            ent.attrs.get("wraps_storage")
            or _catalog_storage_root(str(ent.name or ""))
            or ent.kind_name() in {EntityKind.BUFFER.value, EntityKind.QUEUE.value}
        )

    def _push(start_id: str, flag: str) -> None:
        stack = list(parents.get(start_id, ()))
        seen: set[str] = set()
        while stack:
            src_id = stack.pop()
            if src_id in seen:
                continue
            seen.add(src_id)
            src = codemap.entities.get(src_id)
            if src is None:
                continue
            if not src.attrs.get(flag):
                src.attrs[flag] = True
            stack.extend(parents.get(src_id, ()))

    for ent in codemap.entities.values():
        if _seed_storage(ent):
            _push(ent.id, "wraps_storage")
        if ent.attrs.get("wraps_lock"):
            _push(ent.id, "wraps_lock")
        if ent.attrs.get("wraps_flag"):
            _push(ent.id, "wraps_flag")


def _receiver_is_tque(
    *,
    bid_recv: str,
    receiver_type: str,
    receiver_canonical: str,
    codemap: CodeMap,
) -> bool:
    if bid_recv and bid_recv in codemap.entities:
        ent = codemap.entities[bid_recv]
        if ent.kind_name() == EntityKind.QUEUE.value:
            return True
        blob = " ".join(
            str(ent.attrs.get(k) or "")
            for k in ("type_name", "root", "wrapper")
        )
        if "TQue" in blob:
            return True
    for text in (receiver_type, receiver_canonical):
        if _sync_object_kind(text) == EntityKind.QUEUE:
            return True
        if "TQue" in str(text or ""):
            return True
    return False


def _receiver_is_tpipe(
    *,
    bid_recv: str,
    receiver_type: str,
    receiver_canonical: str,
    codemap: CodeMap,
) -> bool:
    if bid_recv and bid_recv in codemap.entities:
        ent = codemap.entities[bid_recv]
        if ent.kind_name() == EntityKind.PIPE.value:
            return True
        blob = " ".join(
            str(ent.attrs.get(k) or "")
            for k in ("type_name", "root", "wrapper")
        )
        if "TPipe" in blob:
            return True
    for text in (receiver_type, receiver_canonical):
        if _sync_object_kind(text) == EntityKind.PIPE:
            return True
        if "TPipe" in str(text or ""):
            return True
    return False


def _looks_like_reg_or_vector_call(args: list[str] | None, targs: list[str] | None) -> bool:
    """True for CANN vector/Reg APIs, not 2-arg project scalar helpers."""
    args = [str(a) for a in (args or [])]
    targs = [str(a) for a in (targs or [])]
    blob = " ".join(targs + args)
    if any(
        tok in blob
        for tok in (
            "MaskReg",
            "RegTensor",
            "UnalignReg",
            "LocalTensor",
            "GlobalTensor",
            "preg_",
            "vreg_",
        )
    ):
        return True
    if re.search(r"\b(?:preg|vreg)\b", blob):
        return True
    return len(args) >= 3


_DTYPE_TARG_RE = re.compile(
    r"^(?:CALC_TYPE|half|float|double|bool|bfloat16_t|(?:u?int(?:8|16|32|64)_t)|T(?:\d+)?)$"
)


def _looks_like_typed_buffer_get(targs: list[str] | None) -> bool:
    """TBuf/TQue ``Get<DType>()`` — not a project Policy ``Get()``."""
    if not targs or len(targs) != 1:
        return False
    tok = str(targs[0] or "").split("::")[-1].strip()
    return bool(_DTYPE_TARG_RE.match(tok))


def _owner_from_receiver_type(type_text: str) -> str:
    """Class name for a member call, or empty when the type is a selector/conditional.

    ``std::conditional<…, PolicyA, PolicyB>`` names several Get methods; picking
    one Policy is a guess. cannbot locates via the buffer/receiver instead.
    """
    text = str(type_text or "")
    if not text:
        return ""
    compact = re.sub(r"\s+", "", text)
    if "std::conditional" in compact or "conditional_t" in compact:
        return ""
    if "Selector" in text:
        return ""
    base = _base_type_name(text)
    if base.lower() in {"conditional", "conditional_t", "type", "nullptr_t"}:
        return ""
    return base


_THIS_PREFIX_RE = re.compile(
    r"^(?:\(\s*\*\s*this\s*\)\s*(?:\.|->)|this\s*(?:\.|->))"
)


def _expr_storage_name(text: str) -> str:
    """Last identifier in a receiver/arg (`this->pipe` → `pipe`).

    Call expressions such as ``GetTPipePtr()`` are not names and return empty.
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    while True:
        nxt = _THIS_PREFIX_RE.sub("", raw, count=1).strip()
        if nxt == raw:
            break
        raw = nxt
    raw = raw.lstrip("&").strip().replace("->", ".")
    if "[" in raw:
        raw = raw.split("[", 1)[0]
    if "." in raw:
        raw = raw.rsplit(".", 1)[-1]
    raw = raw.strip()
    if not raw or "(" in raw or ")" in raw:
        return ""
    return raw if raw.isidentifier() else ""


def _identity_scopes(function: str, caller_qualified: str) -> list[str]:
    """Owner/method names that can key a class-member PIPE/BUFFER/QUEUE."""
    scopes: list[str] = []

    def add(value: str) -> None:
        text = str(value or "").strip()
        if not text:
            return
        text = text.split("<", 1)[0].strip()
        if text.endswith("()"):
            text = text[:-2].strip()
        if text and text not in scopes:
            scopes.append(text)

    add(function)
    q = str(caller_qualified or "").strip()
    add(q)
    for blob in (q, function):
        if "::" not in blob:
            continue
        head = blob.rsplit("::", 1)[0]
        add(head)
        add(head.split("::")[-1])
        add(blob.split("::")[-1])
    return scopes


def _short_type_name(type_text: str) -> str:
    return _base_type_name(type_text) or str(type_text or "").split("<")[0].split("::")[-1].strip()


def _backslash_logical_lines(text: str) -> list[tuple[int, str]]:
    """Join `\\` continuations so a #define body is one searchable line."""
    physical = str(text or "").splitlines()
    out: list[tuple[int, str]] = []
    i = 0
    while i < len(physical):
        start = i + 1
        chunk = physical[i]
        while chunk.rstrip().endswith("\\") and i + 1 < len(physical):
            chunk = chunk.rstrip()[:-1] + " " + physical[i + 1]
            i += 1
        out.append((start, chunk))
        i += 1
    return out


_TPIPE_DECL_RE = re.compile(
    r"(?:(?:AscendC|ascendc)::)?TPipe\s+(?:const\s+)?(?!\*)(?P<name>[A-Za-z_]\w*)"
)


def _tpipe_decl_lines(text: str) -> dict[str, int]:
    """Physical line of each non-pointer ``TPipe name`` decl, including ``#define`` bodies."""
    out: dict[str, int] = {}
    for idx, raw in enumerate(str(text or "").splitlines(), 1):
        line = raw.rstrip()
        if line.endswith("\\"):
            line = line[:-1]
        for match in _TPIPE_DECL_RE.finditer(line):
            name = match.group("name")
            if name and name not in out:
                out[name] = idx
    return out


def _split_top_level_args(text: str) -> list[str]:
    args: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            args.append(text[start:i].strip())
            start = i + 1
    args.append(text[start:].strip())
    return [a for a in args if a]


def _paren_arg_text(text: str, open_at: int) -> str:
    if open_at < 0 or open_at >= len(text) or text[open_at] != "(":
        return ""
    depth = 0
    for i in range(open_at, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_at + 1 : i]
    return ""


def _conditional_arg_parts(prefix: str) -> tuple[str, str, str] | None:
    """``(cond, then, else)`` of the first ``conditional`` template in *prefix*."""
    idx = str(prefix or "").lower().find("conditional")
    if idx < 0:
        return None
    lt = prefix.find("<", idx)
    if lt < 0:
        return None
    depth = 0
    for j in range(lt, len(prefix)):
        if prefix[j] == "<":
            depth += 1
        elif prefix[j] == ">":
            depth -= 1
            if depth == 0:
                parts = _split_top_level_args(prefix[lt + 1 : j])
                if len(parts) < 3:
                    return None
                return parts[0].strip(), parts[1].strip(), parts[2].strip()
    return None


def _conditional_branch_classes(prefix: str) -> list[str]:
    """True/false types of ``std::conditional<Cond, A, B>`` in a local decl prefix."""
    parts = _conditional_arg_parts(prefix)
    if not parts:
        return []
    names = [_short_type_name(p) for p in parts[1:3]]
    return [n for n in names if n]


_CONDITIONAL_SKIP_BASES = frozenset(
    {"nullptr_t", "nullptr", "conditional", "conditional_t", "type"}
)


def _conditional_storage_bases(type_text: str) -> list[str]:
    """Then/else class names of ``std::conditional<Cond, A, B>`` that can own storage."""
    out: list[str] = []

    def walk(text: str) -> None:
        parts = _conditional_arg_parts(text)
        if not parts:
            return
        for raw in parts[1:3]:
            raw = str(raw or "").strip()
            if "conditional" in raw.lower():
                walk(raw)
                continue
            short = _short_type_name(raw) or raw
            if not short or short in _CONDITIONAL_SKIP_BASES or short in _CXX_SKIP_BASE:
                continue
            if short not in out:
                out.append(short)

    walk(type_text)
    return out


def _member_type_weak(row: dict[str, Any]) -> bool:
    """Clang spelling that hides a ``std::conditional`` field — prefer lexical type_text."""
    text = str(row.get("type_text") or "")
    if "conditional" in text.lower():
        return False
    if not text.strip():
        return True
    base = (_base_type_name(text) or str(row.get("base_type") or "")).lower()
    return base in _CONDITIONAL_SKIP_BASES or base in {"void", ""}


def _conditional_storage_bases_any(*blobs: str) -> list[str]:
    for blob in blobs:
        hit = _conditional_storage_bases(blob)
        if hit:
            return hit
    return []


def _walk_conditional_polarity(text: str, acc: dict[str, tuple[str, str]]) -> None:
    parts = _conditional_arg_parts(text)
    if not parts:
        return
    cond, then, els = parts
    for raw, polarity in ((then, "then"), (els, "else")):
        if "conditional" in raw.lower():
            _walk_conditional_polarity(raw, acc)
            continue
        short = _short_type_name(raw) or raw.strip()
        if not short or short in _CONDITIONAL_SKIP_BASES or short in _CXX_SKIP_BASE:
            continue
        acc.setdefault(short, (cond, polarity))


def _conditional_polarity_map(*blobs: str) -> dict[str, tuple[str, str]]:
    acc: dict[str, tuple[str, str]] = {}
    for blob in blobs:
        if blob:
            _walk_conditional_polarity(str(blob), acc)
    return acc


def _branch_wrapper_names(codemap: Any, ent: Any) -> set[str]:
    names: set[str] = set()
    if ent is None:
        return names
    short = _short_type_name(ent.name)
    if (
        short
        and short not in _CONDITIONAL_SKIP_BASES
        and not _is_catalog_storage_base(short)
        and (
            ent.attrs.get("wraps_storage")
            or ent.attrs.get("wraps_lock")
            or ent.attrs.get("wraps_flag")
        )
    ):
        names.add(short)
    for _rel, other in codemap.neighbors(ent.id, kind=RelationKind.WRAPS, direction="out"):
        if other is None or other.kind_name() != EntityKind.TYPE.value:
            continue
        if not (
            other.attrs.get("wraps_storage")
            or other.attrs.get("wraps_lock")
            or other.attrs.get("wraps_flag")
        ):
            continue
        wrap = _short_type_name(other.name)
        if (
            wrap
            and wrap not in _CONDITIONAL_SKIP_BASES
            and not _is_catalog_storage_base(wrap)
        ):
            names.add(wrap)
    return names


def _branch_root_names(ent: Any) -> set[str]:
    roots: set[str] = set()
    if ent is None:
        return roots
    for root in ent.attrs.get("wrapped_roots") or []:
        short = str(root or "").replace("AscendC::", "").split("::")[-1]
        if short:
            roots.add(short)
    catalog = _catalog_storage_root(str(ent.name or ""))
    if catalog:
        roots.add(catalog)
    return roots


def _intersect_name_sets(sets: list[set[str]]) -> str:
    if not sets:
        return ""
    common = set(sets[0])
    for extra in sets[1:]:
        common &= extra
    if len(common) != 1:
        return ""
    return next(iter(common))


def _wrapper_from_branch_types(
    codemap: Any, ents: list[Any], *, expected: int = 0
) -> str:
    """Common storage wrapper only when every branch names the same wrapper."""
    live = [ent for ent in ents if ent is not None]
    if not live:
        return ""
    if expected and len(live) != expected:
        return ""
    sets = [_branch_wrapper_names(codemap, ent) for ent in live]
    if any(not names for names in sets):
        return ""
    return _intersect_name_sets(sets)


def _common_branch_root(ents: list[Any], *, expected: int = 0) -> str:
    """Common AscendC storage root only when every branch agrees."""
    live = [ent for ent in ents if ent is not None]
    if not live:
        return ""
    if expected and len(live) != expected:
        return ""
    sets = [_branch_root_names(ent) for ent in live]
    if any(not names for names in sets):
        return ""
    return _intersect_name_sets(sets)


def _member_type_sig(row: dict[str, Any]) -> str:
    return (
        _base_type_name(str(row.get("type_text") or ""))
        or str(row.get("base_type") or "")
        or str(row.get("type_text") or "")
    ).strip()


def _absorb_alt_member_type(existing: dict[str, Any], row: dict[str, Any]) -> None:
    """Keep a second strong type as a guarded alternative, do not drop it."""
    if _member_type_weak(row):
        return
    sig_r = _member_type_sig(row)
    if not sig_r or sig_r == _member_type_sig(existing):
        return
    text = str(row.get("type_text") or sig_r)
    current = str(existing.get("type_text") or "")
    alts = [str(x) for x in (existing.get("alt_type_texts") or []) if x]
    if text and text != current and text not in alts:
        alts.append(text)
    existing["alt_type_texts"] = alts


def _row_branch_bases(row: dict[str, Any], *blobs: str) -> list[str]:
    bases = _conditional_storage_bases_any(*blobs)
    alts = [str(x) for x in (row.get("alt_type_texts") or []) if x]
    if not alts:
        return bases
    if not bases:
        primary = _short_type_name(str(blobs[0] if blobs else "") or str(row.get("type_text") or ""))
        if (
            primary
            and primary not in _CONDITIONAL_SKIP_BASES
            and primary not in _CXX_SKIP_BASE
            and "conditional" not in str(row.get("type_text") or "").lower()
        ):
            bases.append(primary)
    for extra in alts:
        for name in _conditional_storage_bases(extra):
            if name not in bases:
                bases.append(name)
        if "conditional" in extra.lower():
            continue
        short = _short_type_name(extra) or _base_type_name(extra)
        if (
            short
            and short not in bases
            and short not in _CONDITIONAL_SKIP_BASES
            and short not in _CXX_SKIP_BASE
        ):
            bases.append(short)
    return bases


def _conditional_branch_attrs(
    name: str,
    branch: str,
    index: int,
    total: int,
    polarity: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    attrs: dict[str, Any] = {"via": "conditional_branch", "member": name}
    if branch in polarity:
        attrs["cond"] = polarity[branch][0]
        attrs["polarity"] = polarity[branch][1]
    elif total == 2:
        attrs["polarity"] = "then" if index == 0 else "else"
    return attrs


def _local_decl_owners(chunk: str, ident: str, before: int) -> list[str]:
    """Class names of the local ``Type ident;`` preceding a call in the same chunk."""
    if not ident or before <= 0:
        return []
    needle = re.compile(rf"\b{re.escape(ident)}\s*;")
    last = None
    for m in needle.finditer(chunk[:before]):
        last = m
    if last is None:
        return []
    prefix = chunk[: last.start()].rstrip()
    # A #define body is one logical line: ignore earlier decls / ``conditional`` aliases.
    cut = max(prefix.rfind(";"), prefix.rfind("{"), prefix.rfind("}"))
    if cut >= 0:
        prefix = prefix[cut + 1 :].rstrip()
    cond = _conditional_branch_classes(prefix)
    if cond:
        return cond
    if prefix.endswith(">"):
        depth = 0
        i = len(prefix) - 1
        while i >= 0:
            if prefix[i] == ">":
                depth += 1
            elif prefix[i] == "<":
                depth -= 1
                if depth == 0:
                    prefix = prefix[:i].rstrip()
                    break
            i -= 1
    m = re.search(r"([A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*)\s*$", prefix)
    if not m:
        return []
    name = _short_type_name(m.group(1))
    return [name] if name else []


_RECV_INIT_RE = re.compile(
    r"(?P<recv>[A-Za-z_]\w*)\s*(?:\.|->)\s*Init\s*\("
)
_ADDR_IDENT_RE = re.compile(r"&(?:\s*)(?P<name>[A-Za-z_]\w*)")
_CLASS_INHERIT_RE = re.compile(
    r"\b(?:class|struct)\s+(?P<name>[A-Za-z_]\w*)\b(?:\s*<[^;{]*>)?\s*:"
    r"(?P<bases>[^{;]+)\{",
    re.DOTALL,
)


def _inherit_pairs_from_text(text: str) -> list[tuple[str, str]]:
    """``class Kernel : public KernelBase<...>`` — independent of Clang BaseDecl."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for m in _CLASS_INHERIT_RE.finditer(str(text or "")):
        child = m.group("name")
        for part in _split_top_level_args(m.group("bases") or ""):
            cleaned = re.sub(r"\b(?:public|private|protected|virtual)\b", " ", part)
            parent = _short_type_name(cleaned)
            if not child or not parent or parent == child:
                continue
            key = (child, parent)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs


def _select_framework_bridge(
    *,
    callee: str,
    receiver: str,
    recv_is_wrapper: bool,
    recv_is_tque: bool,
    recv_is_tpipe: bool,
    receiver_type: str = "",
    receiver_canonical: str = "",
    callee_usr: str = "",
    targs: list[str] | None = None,
) -> tuple[str, str] | None:
    """Explicit CANN method contracts. Spelling alone never proves members."""
    del recv_is_wrapper
    # TBuf / TQueBind ``.template Get<T>()`` returns LocalTensor.
    if callee == "Get" and _looks_like_typed_buffer_get(targs):
        return ("LocalTensor", "STORAGE")
    # TPipe::InitBuffer / FetchEventID / AllocEventID. Receiver is the pipe; TQue is an argument.
    if callee in TPIPE_METHOD_BRIDGES and (recv_is_tpipe or (receiver and not recv_is_tque)):
        return TPIPE_METHOD_BRIDGES[callee]
    # EnQue/DeQue/Alloc/Free exist only on TQueBind in CANN.
    if callee in TQUE_METHOD_BRIDGES and (recv_is_tque or (receiver and not recv_is_tpipe)):
        return TQUE_METHOD_BRIDGES[callee]
    if callee in TENSOR_METHOD_BRIDGES and (
        receiver
        or _receiver_looks_tensor(receiver_type, receiver_canonical)
        or _usr_is_ascendc_tensor_method(callee_usr)
    ):
        return TENSOR_METHOD_BRIDGES[callee]
    return None


# ---------------------------------------------------------------------------
# Source scanners (complete graph — not storage-filtered)
# ---------------------------------------------------------------------------


def _scan_type_aliases(files: list[Path], *, root: str, deadline: float) -> list[dict[str, Any]]:
    from ascendc_codemap_mcp.engine.source_index import get_or_build

    if time.perf_counter() > deadline:
        return []
    return get_or_build(files, root=root, deadline=deadline).aliases_for(files)


def _scan_class_members(files: list[Path], *, root: str, deadline: float) -> list[dict[str, Any]]:
    """All class/struct field members in source scope (complete composition graph)."""
    from ascendc_codemap_mcp.engine.source_index import get_or_build

    if time.perf_counter() > deadline:
        return []
    return get_or_build(files, root=root, deadline=deadline).members_for(files)


def _existing_type_id(codemap: CodeMap, spell: str) -> str | None:
    name = str(spell or "").strip()
    if not name:
        return None
    preferred: str | None = None
    for hit in codemap.by_name(name, kind=EntityKind.TYPE):
        if str(hit.id).startswith("SRCTYPE::"):
            return hit.id
        if preferred is None:
            preferred = hit.id
    return preferred


def _collapse_duplicate_type_hashes(codemap: CodeMap) -> dict[str, str]:
    """Drop clang ``TYPE_<hash>`` nodes when a same-name ``SRCTYPE::`` exists.

    Rewrites incident edges onto the source type. Does not delete wrappers,
    buffers, or other live graph nodes. Returns old-id → canonical-id.
    """
    canonical: dict[str, str] = {}
    for ent in codemap.by_kind(EntityKind.TYPE):
        if str(ent.id).startswith("SRCTYPE::"):
            canonical.setdefault(str(ent.name), ent.id)
    if not canonical:
        return {}
    rewrite: dict[str, str] = {}
    for ent in list(codemap.by_kind(EntityKind.TYPE)):
        eid = str(ent.id)
        if not eid.startswith("TYPE_"):
            continue
        target = canonical.get(str(ent.name))
        if not target or target == eid:
            continue
        rewrite[eid] = target
    if not rewrite:
        return {}
    for eid in rewrite:
        codemap.entities.pop(eid, None)
    rebuilt: dict[str, Any] = {}
    for rel in list(codemap.relations.values()):
        src = rewrite.get(rel.src, rel.src)
        dst = rewrite.get(rel.dst, rel.dst)
        if src == dst or src not in codemap.entities or dst not in codemap.entities:
            continue
        rel.src = src
        rel.dst = dst
        rid = relation_id(rel.kind_name(), src, dst, attrs=rel.attrs)
        rel.id = rid
        rebuilt[rid] = rel
    codemap.relations.clear()
    codemap.relations.update(rebuilt)
    return rewrite


def _purge_root_trace_entities(codemap: CodeMap) -> None:
    """Drop previously minted root-trace nodes so finalize can rebuild them."""
    drop_kinds = {
        EntityKind.OPERATION.value,
        EntityKind.BUFFER.value,
        EntityKind.REGISTER.value,
        EntityKind.PIPE.value,
        EntityKind.EVENT.value,
        EntityKind.QUEUE.value,
    }
    drop_ids = {e.id for e in codemap.entities.values() if e.kind_name() in drop_kinds}
    for e in list(codemap.entities.values()):
        if e.kind_name() != EntityKind.TYPE.value:
            continue
        if e.attrs.get("catalog") == "ascendc":
            drop_ids.add(e.id)
        if e.attrs.get("role") in {
            "storage_wrapper_type",
            "project_wrapper_type",
            "type_alias",
            "source_type",
        }:
            drop_ids.add(e.id)
    for eid in drop_ids:
        codemap.entities.pop(eid, None)
    keep_rel = {
        RelationKind.WRAPS.value,
        RelationKind.ROOTED_AT.value,
        RelationKind.ALIASES.value,
        RelationKind.REFERENCES.value,
        RelationKind.CALLS.value,
        RelationKind.CONTAINS.value,
    }
    for rid, rel in list(codemap.relations.items()):
        if rel.src in drop_ids or rel.dst in drop_ids:
            if (
                rel.kind_name() == RelationKind.CALLS.value
                and rel.src not in drop_ids
                and rel.dst not in drop_ids
            ):
                continue
            if (
                rel.kind_name() == RelationKind.REFERENCES.value
                and rel.src not in drop_ids
                and rel.dst not in drop_ids
                and str(rel.attrs.get("provenance") or "") != "kernel_root_trace"
            ):
                continue
            if rel.kind_name() in keep_rel or rel.src in drop_ids or rel.dst in drop_ids:
                if str(rel.attrs.get("provenance") or "") == "kernel_root_trace" or (
                    rel.src in drop_ids or rel.dst in drop_ids
                ):
                    codemap.relations.pop(rid, None)


_LINK_SITE_SEEN: dict[str, set[tuple[str, int, int, str, str]]] = {}


def _reset_link_site_seen() -> None:
    _LINK_SITE_SEEN.clear()


def _link(
    codemap: CodeMap,
    kind: RelationKind,
    src: str,
    dst: str,
    *,
    attrs: dict[str, Any] | None = None,
    status: str = "confirmed",
    candidate: bool = False,
) -> None:
    """Topology-unique edge; accumulate call-site evidence under attrs['sites']."""
    payload = {**(attrs or {}), "provenance": "kernel_root_trace"}
    site = None
    if any(k in payload for k in ("file", "line", "column")):
        site = {
            "file": str(payload.get("file") or ""),
            "line": int(payload.get("line") or 0),
            "column": int(payload.get("column") or 0),
            "receiver": str(payload.get("receiver") or ""),
            "via": str(payload.get("via") or ""),
        }
    if candidate:
        rid = relation_id(
            kind.value if isinstance(kind, RelationKind) else str(kind),
            src,
            dst,
            attrs=payload,
        )
        existing = codemap.relations.get(rid)
        if existing is not None and str(existing.attrs.get("trust") or "") != TRUST_ADVISORY:
            rel = existing
        else:
            rel = codemap.mint_candidate_relation(
                kind,
                src,
                dst,
                provenance="lexical_source_calls",
                extra=payload,
                status=status,
            )
    else:
        rel = codemap.link(kind, src, dst, attrs=payload, status=status)
    if site is None:
        return
    key = (
        site["file"],
        site["line"],
        site["column"],
        site["receiver"],
        site["via"],
    )
    seen = _LINK_SITE_SEEN.get(rel.id)
    if seen is None:
        existing_sites = rel.attrs.get("sites")
        if not isinstance(existing_sites, list):
            existing_sites = []
            rel.attrs["sites"] = existing_sites
        seen = {
            (
                str(s.get("file") or ""),
                int(s.get("line") or 0),
                int(s.get("column") or 0),
                str(s.get("receiver") or ""),
                str(s.get("via") or ""),
            )
            for s in existing_sites
            if isinstance(s, dict)
        }
        _LINK_SITE_SEEN[rel.id] = seen
    if key in seen:
        return
    seen.add(key)
    sites = rel.attrs.get("sites")
    if not isinstance(sites, list):
        sites = []
        rel.attrs["sites"] = sites
    sites.append(site)
    # Keep first-seen file/line as the primary display site; do not overwrite.


def _link_backed_by(
    codemap: CodeMap,
    view_id: str,
    owner_id: str,
    *,
    space: str = "",
    via: str = "storage_owner",
    file: str = "",
    line: int = 0,
) -> None:
    """Tensor/View → storage object. Never a catalog LocalTensor/GlobalTensor TYPE."""
    if not view_id or not owner_id or view_id == owner_id:
        return
    if view_id not in codemap.entities or owner_id not in codemap.entities:
        return
    owner = codemap.entities.get(owner_id)
    if _is_catalog_type_entity(owner) or not _is_storage_object(owner):
        return
    payload: dict[str, Any] = {"via": via}
    space_s = str(space or "").strip()
    if space_s and space_s != "UNKNOWN":
        payload["physical_space"] = space_s
    if file:
        payload["file"] = file
        payload["line"] = int(line or 0)
    _link(codemap, RelationKind.BACKED_BY, view_id, owner_id, attrs=payload)


def _record_flag_pair_appearance(codemap: CodeMap, gaps: list[dict[str, Any]]) -> dict[str, int]:
    """Identity-level Set/Wait appearance. TQue ops never enter this check."""
    groups: dict[tuple[str, str, str], dict[str, list[str]]] = defaultdict(
        lambda: {"signals": [], "awaits": []}
    )
    event_ids: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for rel in codemap.relations.values():
        kn = rel.kind_name()
        if kn not in {RelationKind.SIGNALS.value, RelationKind.AWAITS.value}:
            continue
        op = codemap.entities.get(rel.src)
        event = codemap.entities.get(rel.dst)
        if op is None or event is None:
            continue
        callee = str(op.attrs.get("callee") or op.name or "")
        if is_tque_callee(callee) or not is_flag_sync(callee):
            continue
        identity = str(event.attrs.get("identity") or event.name or "")
        sync = {
            "mechanism": str(event.attrs.get("mechanism") or op.attrs.get("mechanism") or ""),
            "event": str(event.attrs.get("event_type") or ""),
        }
        key = flag_pair_key(identity, sync)
        if not key[1]:
            continue
        side = "signals" if kn == RelationKind.SIGNALS.value else "awaits"
        groups[key][side].append(op.id)
        event_ids[key].add(event.id)

    paired_keys = 0
    unpaired_keys = 0
    for key, sides in groups.items():
        paired = bool(sides["signals"] and sides["awaits"])
        if paired:
            paired_keys += 1
        else:
            unpaired_keys += 1
            present_id = (sides["signals"] or sides["awaits"])[0]
            present_op = codemap.entities.get(present_id)
            present = str((present_op.attrs.get("callee") if present_op else "") or "")
            present_canon = canonical_sync_name(present)
            gaps.append(
                {
                    "code": REASON_UNPAIRED_FLAG_SYNC,
                    "mechanism": key[0],
                    "identity": key[1],
                    "event": key[2],
                    "present": present,
                    "missing": FLAG_PAIR_MATE.get(present_canon, FLAG_PAIR_MATE.get(present, "")),
                    "entity_id": present_id,
                }
            )
        for oid in sides["signals"] + sides["awaits"]:
            ent = codemap.entities.get(oid)
            if ent is not None:
                ent.attrs["flag_paired"] = paired
        for eid in event_ids.get(key, ()):
            ev = codemap.entities.get(eid)
            if ev is not None:
                ev.attrs["paired"] = paired
                ev.attrs["signal_count"] = len(sides["signals"])
                ev.attrs["await_count"] = len(sides["awaits"])
    return {"flag_pairs": paired_keys, "unpaired_flag_sync": unpaired_keys}


def _propagate_reachability(codemap: CodeMap) -> None:
    """Single reverse fixed-point over WRAPS / ALIASES / CALLS from REACHED nodes."""
    reverse: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rel in codemap.relations.values():
        if str(rel.attrs.get("provenance") or "") != "kernel_root_trace":
            # Also follow CALLS from kernel call binder if present.
            if rel.kind_name() != RelationKind.CALLS.value:
                continue
        kn = rel.kind_name()
        if kn not in {
            RelationKind.WRAPS.value,
            RelationKind.ALIASES.value,
            RelationKind.CALLS.value,
            RelationKind.ROOTED_AT.value,
            RelationKind.BACKED_BY.value,
        }:
            continue
        # Lexical CALLS are candidates; the reverse climb only follows
        # resolved CallExpr / compiler-backed edges.
        if kn == RelationKind.CALLS.value and str(rel.attrs.get("trust") or "") == TRUST_ADVISORY:
            continue
        # Reverse: dst → src means "src reaches via dst"
        if kn == RelationKind.ROOTED_AT.value:
            continue
        reverse[rel.dst].append((rel.src, kn))

    queue: deque[str] = deque()
    seen: set[str] = set()
    for eid, e in codemap.entities.items():
        if e.attrs.get("root_status") == "REACHED":
            queue.append(eid)
            seen.add(eid)

    while queue:
        cur = queue.popleft()
        cur_e = codemap.entities.get(cur)
        if cur_e is None:
            continue
        cur_root = str(cur_e.attrs.get("root") or cur_e.name)
        cur_kind = str(cur_e.attrs.get("root_kind") or "")
        for parent, via in reverse.get(cur, []):
            pe = codemap.entities.get(parent)
            if pe is None:
                continue
            if str(pe.attrs.get("root_status") or "") == "GUARDED":
                continue
            if pe.attrs.get("root_status") == "REACHED" and pe.attrs.get("root"):
                # Already rooted; still allow ROOTED_AT edge refresh below.
                pass
            else:
                pe.attrs["root_status"] = "REACHED"
                pe.attrs["root"] = cur_root if cur_root.startswith("AscendC::") else (
                    cur_root if "::" in cur_root else f"AscendC::{cur_root.replace('AscendC::', '')}"
                )
                if not str(pe.attrs.get("root") or "").startswith("AscendC::") and cur_e.attrs.get("catalog") == "ascendc":
                    pe.attrs["root"] = cur_e.name
                pe.attrs["root_kind"] = cur_kind or pe.attrs.get("root_kind") or "STORAGE"
                trace = list(pe.attrs.get("trace") or [pe.name])
                if cur_e.name not in trace:
                    trace.append(cur_e.name)
                pe.attrs["trace"] = trace
                pe.status = "extracted"
                pe.confidence = max(float(pe.confidence or 0), 0.9)

            # Point ROOTED_AT at AscendC catalog when available.
            target = cur
            root_spell = str(pe.attrs.get("root") or "").replace("AscendC::", "")
            if root_spell and _is_ascendc_root_spelling(root_spell):
                target = _ensure_ascendc_root(
                    codemap, root_spell, root_kind=str(pe.attrs.get("root_kind") or "STORAGE")
                )
            elif cur_e.attrs.get("catalog") == "ascendc":
                target = cur
            _link(
                codemap,
                RelationKind.ROOTED_AT,
                parent,
                target,
                attrs={"via": f"{via}_closure"},
            )
            if parent not in seen:
                seen.add(parent)
                queue.append(parent)


def _link_owner_declares_methods(codemap: CodeMap, type_ents: dict[str, str]) -> None:
    """TYPE --DECLARES--> class METHOD so around/file:line can hop to Init/Lock."""
    for e in list(codemap.by_kind(EntityKind.METHOD)):
        owner = str(e.attrs.get("owner") or "")
        if not owner:
            q = str(e.attrs.get("qualified_name") or "")
            if "::" in q:
                owner = _base_type_name(q.rsplit("::", 1)[0]) or ""
        if not owner or owner not in type_ents:
            continue
        if e.name in _CXX_SKIP_BASE:
            continue
        _link(
            codemap,
            RelationKind.DECLARES,
            type_ents[owner],
            e.id,
            attrs={"via": "class_method"},
        )


def _normalize_settled_entities(codemap: CodeMap) -> None:
    """Four-state root_status: REACHED / PROJECT / BUILTIN / UNRESOLVED.

    Project types, compiler intrinsics and settled buffers are extracted — they
    are not locate-surface holes. Storage wrappers that never canonicalized
    stay UNRESOLVED.
    """
    settled = {"REACHED", "PROJECT", "BUILTIN"}
    for e in codemap.entities.values():
        rs = str(e.attrs.get("root_status") or "")
        kind = e.kind_name()
        name = e.name
        usr = str(e.attrs.get("usr") or e.attrs.get("callee_usr") or "")
        decl = str(e.attrs.get("callee_decl_file") or e.file or "")
        qualified = str(e.attrs.get("qualified_name") or e.attrs.get("callee_qualified") or "")
        if rs in settled:
            e.status = "extracted"
            continue
        if kind == EntityKind.TYPE.value:
            if e.attrs.get("catalog") == "ascendc":
                e.status = "extracted"
                continue
            if str(e.attrs.get("role") or "") == "storage_wrapper_type":
                continue
            if _is_compiler_builtin(name, usr=usr, decl_file=decl, qualified=qualified):
                e.attrs["root_status"] = "BUILTIN"
                e.attrs["root_kind"] = "STD"
            else:
                e.attrs["root_status"] = "PROJECT"
                e.attrs["root_kind"] = e.attrs.get("root_kind") or "PROJECT"
            e.status = "extracted"
            continue
        if kind == EntityKind.METHOD.value:
            if _is_compiler_builtin(name, usr=usr, decl_file=decl, qualified=qualified):
                e.attrs["root_status"] = "BUILTIN"
                e.attrs["root_kind"] = "BUILTIN"
            elif _usr_is_ascendc_tensor_method(usr):
                e.attrs["root_status"] = "REACHED"
                e.attrs["root"] = e.attrs.get("root") or "AscendC::LocalTensor"
                e.attrs["root_kind"] = e.attrs.get("root_kind") or "MEMORY_API"
            else:
                e.attrs["root_status"] = "PROJECT"
                e.attrs["root_kind"] = "PROJECT"
            e.status = "extracted"
            continue
        if kind == EntityKind.OPERATION.value:
            callee = str(e.attrs.get("callee") or name)
            if _is_compiler_builtin(
                callee,
                usr=usr,
                decl_file=decl,
                qualified=qualified,
            ):
                e.attrs["root_status"] = "BUILTIN"
                e.attrs["root_kind"] = "BUILTIN"
                e.attrs["root_proof"] = e.attrs.get("root_proof") or "compiler_builtin"
                e.status = "extracted"
            elif str(e.attrs.get("root_proof") or "").startswith("project_"):
                e.attrs["root_status"] = "PROJECT"
                e.attrs["root_kind"] = "PROJECT"
                e.status = "extracted"
            continue
        if kind == EntityKind.BUFFER.value and rs == "REACHED":
            e.status = "extracted"


def _add_clang_path(dst: set[str], raw: str) -> None:
    p = str(raw or "").replace("\\", "/")
    if not p:
        return
    dst.add(Path(p).name.lower())
    dst.add(p.lower())


def _cited_files_from_walk(wr: Any, dst: set[str]) -> None:
    """Union files named by a WalkResult so lexical skip covers included headers."""
    _add_clang_path(dst, str(getattr(wr, "path", "") or ""))
    for site in getattr(wr, "call_sites", None) or []:
        _add_clang_path(dst, getattr(site, "file", "") or "")
        _add_clang_path(dst, getattr(site, "callee_decl_file", "") or "")
    for decl in getattr(wr, "local_decls", None) or []:
        _add_clang_path(dst, getattr(decl, "file", "") or "")
    for decl in getattr(wr, "type_decls", None) or []:
        _add_clang_path(dst, getattr(decl, "file", "") or "")
    for decl in getattr(wr, "alias_decls", None) or []:
        _add_clang_path(dst, getattr(decl, "file", "") or "")
    fds = getattr(wr, "field_decls", None) or {}
    if isinstance(fds, dict):
        fds = fds.values()
    for fd in fds:
        _add_clang_path(dst, getattr(fd, "file", "") or "")
    for ctrl in getattr(wr, "controls", None) or []:
        _add_clang_path(dst, getattr(ctrl, "file", "") or "")
    fns = getattr(wr, "functions", None) or {}
    if isinstance(fns, dict):
        fns = fns.values()
    for rec in fns:
        _add_clang_path(dst, getattr(rec, "file", "") or "")


def finalize_kernel_root_trace(
    codemap: CodeMap,
    source_root: Path | str,
    *,
    architecture: str = "",
) -> CodeMap:
    if not _enabled():
        codemap.meta["kernel_root_trace"] = {"skipped": True, "reason": "UO_KERNEL_ROOT_TRACE=0"}
        return codemap

    t0 = time.perf_counter()
    budget = TimeBudget(_budget_s())
    deadline = budget.deadline
    root = str(Path(source_root).expanduser().resolve())
    arch = require_architecture(architecture or codemap.architecture)
    global _TRACE_ARCHITECTURE
    _TRACE_ARCHITECTURE = arch
    reachable, filter_strict = kscan.reachable_function_names(codemap)
    files = kscan.selected_kernel_files(codemap, Path(root))
    identity_filled = 0
    walk_confirm = 0

    _purge_root_trace_entities(codemap)
    _reset_link_site_seen()

    try:
        from ascendc_codemap_mcp.engine import tu_cache as _tu_cache

        _tu_cache.load_walk_bundle(Path(root), arch, path_substr="op_kernel")
    except Exception:  # noqa: BLE001
        pass
    if files:
        from ascendc_codemap_mcp.engine.source_index import get_or_build

        get_or_build(files, root=root, deadline=deadline, architecture=arch)

    # --- 1. Source facts -------------------------------------------------
    calls, decls, _controls, provenance = kscan.collect_call_sites_from_walks(
        Path(root),
        architecture=arch,
        reachable=reachable,
        filter_strict=filter_strict,
        deadline=deadline,
    )
    walk_stats = {}
    try:
        from ascendc_codemap_mcp.engine import tu_cache as _tu_cache

        walk_stats = dict(_tu_cache.stats() or {})
    except Exception:  # noqa: BLE001
        walk_stats = {}

    # Files already covered by a successful clang walk: skip full lexical merge
    # there (primitives-only lexical fill-in only). Uncovered files still get
    # the broader lexical fallback so we do not go silent.
    clang_files: set[str] = set()
    try:
        from ascendc_codemap_mcp.engine import tu_cache as _tu_cache

        for wr in _tu_cache.iter_cached_walks(
            Path(root), arch, path_substr="op_kernel", limit=96
        ):
            _cited_files_from_walk(wr, clang_files)
    except Exception:  # noqa: BLE001
        clang_files = set()

    uncovered = [
        f
        for f in files
        if f.name.lower() not in clang_files
        and str(f).replace("\\", "/").lower() not in clang_files
        and not any(str(f).replace("\\", "/").lower().endswith(cf) for cf in clang_files if "/" in cf)
    ]
    covered = [f for f in files if f not in uncovered]

    # Lexical fallback: uncovered files get full scan; covered files only
    # registry primitives (to catch APIs clang may have missed as dependent names).
    if files and time.perf_counter() < deadline:
        lexical: list = []
        if uncovered:
            lexical.extend(
                kscan.lexical_source_call_sites(
                    uncovered,
                    reachable=reachable,
                    filter_strict=filter_strict,
                    root=root,
                    deadline=deadline,
                    primitives_only=False,
                )
            )
        if covered and provenance.startswith("clang_walk"):
            lexical.extend(
                kscan.lexical_source_call_sites(
                    covered,
                    reachable=reachable,
                    filter_strict=filter_strict,
                    root=root,
                    deadline=deadline,
                    primitives_only=True,
                )
            )
        elif not provenance.startswith("clang_walk"):
            # No walk cache at all — full lexical across all selected files.
            lexical = kscan.lexical_source_call_sites(
                files,
                reachable=reachable,
                filter_strict=filter_strict,
                root=root,
                deadline=deadline,
                primitives_only=False,
            )
        walk_confirm = 0
        if lexical and provenance.startswith("clang_walk"):
            try:
                from ascendc_codemap_mcp.engine import tu_cache as _tu_cache

                walks = _tu_cache.iter_cached_walks(
                    Path(root), arch, path_substr="op_kernel", limit=96
                )
                walk_confirm = kscan.confirm_lexical_from_walks(lexical, walks)
            except Exception:  # noqa: BLE001
                walk_confirm = 0
        calls, added = kscan.merge_lexical_sites(calls, lexical, root=root)
        if added:
            provenance = f"{provenance}+lexical_source_calls"
        identity_filled = 0
        if files and time.perf_counter() < deadline:
            from ascendc_codemap_mcp.engine.passes.kernel_call_identity import (
                build_source_symbol_index,
                enrich_call_sites,
            )

            calls = [kscan.site_as_dict(s) for s in (calls or [])]
            sym_index = build_source_symbol_index(files, root=root, deadline=deadline)
            identity_filled = enrich_call_sites(calls, sym_index, root=root)
        lex_decls = kscan.lexical_buffer_decls(
            files,
            reachable=reachable,
            filter_strict=False,
            deadline=deadline,
        )
        decls = list(decls or []) + list(lex_decls or [])

    # Gated source n is this_op + sibling_op. Fill those primitives on the
    # same TimeBudget; family_common cube templates stay clang-only.
    extra_arch = kscan.kernel_corpus(
        Path(root),
        arch,
        deadline=deadline,
    )
    extra_added = 0
    gated_fill_complete = True
    priority: list[Path] = []
    for path in extra_arch:
        owner = kscan.kernel_file_owner(path, Path(root))
        if owner in {"this_op", "sibling_op"}:
            priority.append(path)
    if priority:
        extra_sites = kscan.lexical_source_call_sites(
            priority,
            reachable=reachable,
            filter_strict=False,
            root=root,
            deadline=deadline,
            primitives_only=True,
        )
        calls, extra_added = kscan.merge_lexical_sites(calls, extra_sites, root=root)
        if extra_added:
            provenance = f"{provenance}+arch_kernel_primitives"
        if budget.expired():
            gated_fill_complete = False
    try:
        from ascendc_codemap_mcp.engine.diagnostics.source_api import count_source_kernel_apis

        source_api_gated = count_source_kernel_apis(
            Path(root), arch, files=priority or extra_arch
        )
    except Exception:  # noqa: BLE001
        source_api_gated = {}

    kernel_backend = (
        "clang"
        if provenance.startswith("clang_walk")
        else ("lexical" if "lexical" in provenance else "none")
    )

    aliases_lex = _scan_type_aliases(files, root=root, deadline=deadline) if files else []
    members_lex = _scan_class_members(files, root=root, deadline=deadline) if files else []
    clang_graph = kscan.collect_type_graph_from_walks(
        Path(root), architecture=arch, deadline=deadline
    )
    # Clang-first; lexical regex only fills parse gaps.
    aliases = list(clang_graph.get("aliases") or [])
    members = list(clang_graph.get("members") or [])
    clang_types = list(clang_graph.get("types") or [])
    clang_bases = list(clang_graph.get("bases") or [])
    seen_alias = {
        (str(r.get("alias") or ""), str(r.get("file") or ""), int(r.get("line") or 0))
        for r in aliases
    }
    for row in aliases_lex:
        key = (str(row.get("alias") or ""), str(row.get("file") or ""), int(row.get("line") or 0))
        if key in seen_alias:
            continue
        row = dict(row)
        row.setdefault("provenance", "lexical_regex")
        aliases.append(row)
        seen_alias.add(key)
    seen_member: dict[tuple[str, str, str], dict[str, Any]] = {}
    members_merged: list[dict[str, Any]] = []
    for r in members:
        key = (
            str(r.get("owner") or ""),
            str(r.get("member") or ""),
            _norm_file(str(r.get("file") or ""), root),
        )
        if key in seen_member:
            _absorb_alt_member_type(seen_member[key], r)
            continue
        seen_member[key] = r
        members_merged.append(r)
    for row in members_lex:
        key = (
            str(row.get("owner") or ""),
            str(row.get("member") or ""),
            _norm_file(str(row.get("file") or ""), root),
        )
        existing = seen_member.get(key)
        if existing is not None:
            if _member_type_weak(existing) and not _member_type_weak(row):
                existing["type_text"] = row.get("type_text") or existing.get("type_text")
                existing["base_type"] = row.get("base_type") or existing.get("base_type")
            else:
                _absorb_alt_member_type(existing, row)
            continue
        row = dict(row)
        row.setdefault("provenance", "lexical_regex")
        members_merged.append(row)
        seen_member[key] = row
    members = members_merged

    alias_to_target: dict[str, str] = {
        str(row["alias"]): str(row["target"]) for row in aliases if row.get("alias")
    }

    def _resolve_alias_chain(type_text: str) -> str:
        base = _base_type_name(type_text)
        seen: set[str] = set()
        while base and base in alias_to_target and base not in seen:
            seen.add(base)
            type_text = alias_to_target[base]
            base = _base_type_name(type_text)
        return type_text

    type_ents: dict[str, str] = {}

    # --- 2. Complete type / alias / member graph -------------------------
    for row in aliases:
        alias = str(row["alias"])
        target = str(row["target"])
        tid = make_id("Type", "alias", alias, row["file"], int(row["line"]))
        resolved = _resolve_alias_chain(target)
        root_spell = _base_type_name(resolved)
        reached = _is_ascendc_root_spelling(root_spell)
        ent = codemap.upsert(
            EntityKind.TYPE,
            alias,
            eid=tid,
            attrs={
                "role": "type_alias",
                "alias_of": target,
                "resolved_type": resolved,
                "root_status": "REACHED" if reached else "UNRESOLVED",
                "root_kind": (
                    "REGISTER"
                    if reached and root_spell in ASCENDC_REGISTER_TYPES
                    else ("STORAGE" if reached and root_spell in ASCENDC_BUFFER_TYPES else (
                        "SYNC" if reached and root_spell in SYNC_MECHANISM else (
                            "COMPUTE_API" if reached else ""
                        )
                    ))
                ),
                "root": f"AscendC::{root_spell}" if reached else "",
                "trace": [alias, _base_type_name(target)] + ([root_spell] if reached else []),
                "provenance": _PROV,
            },
            file=str(row["file"]),
            line=int(row["line"]),
            status="extracted" if reached else "partial",
            confidence=1.0 if reached else 0.5,
        )
        type_ents[alias] = ent.id
        # Always ALIASES to target type node (complete graph).
        tbase = _base_type_name(target)
        if tbase and tbase not in type_ents:
            if _is_ascendc_root_spelling(tbase):
                type_ents[tbase] = _ensure_ascendc_root(
                    codemap,
                    tbase,
                    root_kind="STORAGE" if tbase in ASCENDC_BUFFER_TYPES else (
                        "REGISTER" if tbase in ASCENDC_REGISTER_TYPES else "COMPUTE_API"
                    ),
                )
            else:
                existing = _existing_type_id(codemap, tbase)
                if existing:
                    type_ents[tbase] = existing
                    continue
                mid = make_id("Type", "alias_target", tbase, row["file"], int(row["line"]))
                ment = codemap.upsert(
                    EntityKind.TYPE,
                    tbase,
                    eid=mid,
                    attrs={
                        "role": "source_type",
                        "root_status": "UNRESOLVED",
                        "provenance": _PROV,
                        # This node stands where the alias names the type, not
                        # where the type is declared. Without the mark, the
                        # `using T = std::conditional_t<…, A, B>` line counted
                        # as a second definition site of A, and a card said
                        # "2 definition sites (all listed)" about a class
                        # declared exactly once.
                        "reference_only": True,
                    },
                    file=str(row["file"]),
                    line=int(row["line"]),
                    status="partial",
                    confidence=0.5,
                )
                type_ents[tbase] = ment.id
        if tbase and tbase in type_ents:
            _link(codemap, RelationKind.ALIASES, ent.id, type_ents[tbase], attrs={"via": "using"})
        if reached:
            rid = _ensure_ascendc_root(
                codemap,
                root_spell,
                root_kind=str(ent.attrs.get("root_kind") or "STORAGE"),
            )
            _link(codemap, RelationKind.ROOTED_AT, ent.id, rid)

    # Every class that appears as an owner or member type gets a TYPE node.
    for row in members:
        owner = str(row["owner"])
        if owner not in type_ents:
            existing = _existing_type_id(codemap, owner)
            if existing:
                type_ents[owner] = existing
                continue
            oid = make_id("Type", "class", owner, row["file"], int(row["line"]))
            ent = codemap.upsert(
                EntityKind.TYPE,
                owner,
                eid=oid,
                attrs={
                    "role": "source_type",
                    "root_status": "UNRESOLVED",
                    "root_kind": "",
                    "root": "",
                    "trace": [owner],
                    "provenance": _PROV,
                },
                file=str(row["file"]),
                line=int(row["line"]),
                status="partial",
                confidence=0.5,
            )
            type_ents[owner] = ent.id

    # Seed clang TypeDecl nodes (USR / qualified identity).
    for row in clang_types:
        name = str(row.get("name") or "")
        if not name:
            continue
        ikey = _type_identity_key(
            name,
            usr=str(row.get("usr") or ""),
            qualified=str(row.get("qualified_name") or ""),
        ) or name
        if ikey in type_ents:
            continue
        existing = _existing_type_id(codemap, name)
        if existing:
            type_ents[ikey] = existing
            type_ents.setdefault(name, existing)
            continue
        display = str(row.get("qualified_name") or name)
        mid = make_id(
            "Type",
            "clang",
            str(row.get("usr") or display),
            row.get("file") or "",
            int(row.get("line") or 0),
        )
        ment = codemap.upsert(
            EntityKind.TYPE,
            name,
            eid=mid,
            attrs={
                "role": "source_type",
                "root_status": "UNRESOLVED",
                "qualified_name": str(row.get("qualified_name") or ""),
                "usr": str(row.get("usr") or ""),
                "trace": [name],
                "provenance": _PROV,
            },
            file=str(row.get("file") or ""),
            line=int(row.get("line") or 0),
            status="partial",
            confidence=0.6,
        )
        type_ents[ikey] = ment.id
        if name and name not in type_ents:
            type_ents[name] = ment.id

    wraps_edges: list[tuple[str, str, dict[str, Any]]] = []
    for row in members:
        owner = str(row["owner"])
        if owner not in type_ents:
            continue
        type_text = str(row["type_text"])
        resolved = _resolve_alias_chain(type_text)
        canon = str(row.get("canonical_type") or "")
        resolved_base = _base_type_name(resolved) or str(row.get("base_type") or "")
        branch_bases = _row_branch_bases(row, type_text, resolved, canon)
        if branch_bases:
            targets = [(base, base, base) for base in branch_bases]
        else:
            if not resolved_base or resolved_base in _CXX_SKIP_BASE:
                continue
            member_ikey = _type_identity_key(
                resolved,
                usr=str(row.get("referenced_type_usr") or ""),
                qualified="",
            ) or resolved_base
            display = resolved.strip() if "<" in resolved else resolved_base
            targets = [(resolved_base, member_ikey, display)]
        for resolved_base, member_ikey, display in targets:
            # Reuse an already-materialised owner/class node for the same short name.
            if member_ikey not in type_ents and resolved_base in type_ents and "<" not in display:
                type_ents[member_ikey] = type_ents[resolved_base]
            if member_ikey not in type_ents:
                existing = _existing_type_id(codemap, resolved_base) or _existing_type_id(
                    codemap, display
                )
                if existing:
                    type_ents[member_ikey] = existing
                    type_ents.setdefault(resolved_base, existing)
                elif _is_ascendc_root_spelling(resolved_base):
                    type_ents[member_ikey] = _ensure_ascendc_root(
                        codemap,
                        resolved_base,
                        root_kind=(
                            "STORAGE"
                            if resolved_base in ASCENDC_BUFFER_TYPES
                            else (
                                "REGISTER"
                                if resolved_base in ASCENDC_REGISTER_TYPES
                                else "COMPUTE_API"
                            )
                        ),
                    )
                    type_ents.setdefault(resolved_base, type_ents[member_ikey])
                else:
                    mid = make_id("Type", "member_type", member_ikey, row["file"], int(row["line"]))
                    ment = codemap.upsert(
                        EntityKind.TYPE,
                        display,
                        eid=mid,
                        attrs={
                            "role": "source_type",
                            "root_status": "UNRESOLVED",
                            "type_name": _persist_type_name(resolved),
                            "spelling_base": resolved_base,
                            "provenance": _PROV,
                        },
                        file=str(row["file"]),
                        line=int(row["line"]),
                        status="partial",
                        confidence=0.5,
                    )
                    type_ents[member_ikey] = ment.id
                    type_ents.setdefault(resolved_base, ment.id)
            wraps_edges.append(
                (
                    type_ents[owner],
                    type_ents[member_ikey],
                    {
                        "member": row["member"],
                        "type_name": resolved_base
                        if branch_bases
                        else _persist_type_name(type_text),
                        "file": row["file"],
                        "line": row["line"],
                    },
                )
            )

    for src, dst, attrs in wraps_edges:
        _link(codemap, RelationKind.WRAPS, src, dst, attrs=attrs)
        src_e = codemap.entities.get(src)
        dst_e = codemap.entities.get(dst)
        if src_e is None or dst_e is None:
            continue
        type_name = str(attrs.get("type_name") or "")
        catalog = _catalog_storage_root(type_name) or _catalog_storage_root(dst_e.name)
        if catalog:
            src_e.attrs["wraps_storage"] = True
            src_e.attrs.setdefault("wrapped_roots", [])
            roots = src_e.attrs.get("wrapped_roots")
            if isinstance(roots, list) and catalog not in roots:
                roots.append(catalog)
        if _wraps_lock_type(type_name) or _wraps_lock_type(dst_e.name):
            src_e.attrs["wraps_lock"] = True
        # CONTAINS/DECLARES belong on the member instance (FIELD/BUFFER),
        # not on the member's TYPE. TYPE→TYPE WRAPS stays as composition.

    # Inheritance edges from Clang BaseDecl.
    for row in clang_bases:
        derived = str(row.get("derived") or "")
        base = str(row.get("base") or "")
        if not derived or not base:
            continue
        dkey = _type_identity_key(
            derived, usr=str(row.get("derived_usr") or "")
        ) or derived
        bkey = _type_identity_key(base, usr=str(row.get("base_usr") or "")) or base
        if dkey not in type_ents and derived in type_ents:
            type_ents[dkey] = type_ents[derived]
        if bkey not in type_ents:
            if _is_ascendc_root_spelling(base):
                type_ents[bkey] = _ensure_ascendc_root(
                    codemap,
                    base,
                    root_kind="STORAGE" if base in ASCENDC_BUFFER_TYPES else "COMPUTE_API",
                )
            else:
                mid = make_id("Type", "base", bkey, row.get("file") or "", int(row.get("line") or 0))
                ment = codemap.upsert(
                    EntityKind.TYPE,
                    base,
                    eid=mid,
                    attrs={
                        "role": "source_type",
                        "root_status": "UNRESOLVED",
                        "usr": str(row.get("base_usr") or ""),
                        "provenance": _PROV,
                    },
                    file=str(row.get("file") or ""),
                    line=int(row.get("line") or 0),
                    status="partial",
                    confidence=0.5,
                )
                type_ents[bkey] = ment.id
                type_ents.setdefault(base, ment.id)
        if type_ents.get(dkey) and type_ents.get(bkey):
            _link(
                codemap,
                RelationKind.WRAPS,
                type_ents[dkey],
                type_ents[bkey],
                attrs={
                    "via": "inherits",
                    "file": row.get("file") or "",
                    "line": int(row.get("line") or 0),
                },
            )

    # --- 3. Seed AscendC / CANN roots ------------------------------------
    for spell in sorted(ASCENDC_BUFFER_TYPES):
        type_ents.setdefault(spell, _ensure_ascendc_root(codemap, spell, root_kind="STORAGE"))
    for spell in sorted(ASCENDC_REGISTER_TYPES):
        type_ents.setdefault(spell, _ensure_ascendc_root(codemap, spell, root_kind="REGISTER"))
    for spell in sorted(SYNC_MECHANISM):
        if _vf_blocked(spell):
            continue
        type_ents.setdefault(spell, _ensure_ascendc_root(codemap, spell, root_kind="SYNC"))
    for spell in sorted(_ASCENDC_API_ROOTS - ASCENDC_BUFFER_TYPES - set(ASCENDC_REGISTER_TYPES) - set(SYNC_MECHANISM)):
        if _vf_blocked(spell):
            continue
        type_ents.setdefault(
            spell,
            _ensure_ascendc_root(codemap, spell, root_kind=_category_root_kind("", spell)),
        )

    rewrite = _collapse_duplicate_type_hashes(codemap)
    if rewrite:
        for key, eid in list(type_ents.items()):
            type_ents[key] = rewrite.get(eid, eid)
    _propagate_wrap_flags(codemap)

    # --- 4. BUFFER / REGISTER decl sites ---------------------------------
    buffer_by_key: dict[tuple[str, str], str] = {}
    buffer_by_name: dict[str, str] = {}
    buffer_by_file: dict[tuple[str, str], str] = {}
    gaps: list[dict[str, Any]] = []

    def _index_storage(eid: str, *, scope: str, name: str, nfile: str) -> None:
        if not eid or not name:
            return
        if scope:
            buffer_by_key[(scope, name)] = eid
        buffer_by_name[name] = eid
        nf = str(nfile or "").replace("\\", "/")
        if nf:
            buffer_by_file[(nf, name)] = eid

    def _lookup_storage(
        name: str,
        scopes: list[str],
        nfile: str = "",
        kinds: set[str] | None = None,
    ) -> str:
        if not name:
            return ""

        def _ok(eid: str) -> bool:
            if not eid:
                return False
            if not kinds:
                return True
            ent = codemap.entities.get(eid)
            return ent is not None and ent.kind_name() in kinds

        nf = str(nfile or "").replace("\\", "/")

        def _in_file(eid: str) -> bool:
            ent = codemap.entities.get(eid) if eid else None
            if ent is None:
                return False
            return str(getattr(ent, "file", "") or "").replace("\\", "/") == nf

        # A scope is an unqualified function name, so two classes that both
        # spell a method ProcessSink share one key and the index keeps whichever
        # was written last. Asking for the file first keeps a local of one class
        # from resolving to the same-named local of the other.
        scope_hits = [
            eid for eid in (buffer_by_key.get((s, name)) for s in scopes) if _ok(eid or "")
        ]
        if nf:
            for hit in scope_hits:
                if _in_file(hit or ""):
                    return hit or ""
            hit = buffer_by_file.get((nf, name))
            if _ok(hit or ""):
                return hit or ""
        if scope_hits:
            return scope_hits[0] or ""
        hit = buffer_by_name.get(name) or ""
        return hit if _ok(hit) else ""

    def _unique_pipe_in_scopes(scopes: list[str]) -> str:
        found: list[str] = []
        seen: set[str] = set()
        scope_set = {s for s in scopes if s}
        if not scope_set:
            return ""
        for (scope, _name), eid in buffer_by_key.items():
            if scope not in scope_set or eid in seen:
                continue
            ent = codemap.entities.get(eid)
            if ent is None or ent.kind_name() != EntityKind.PIPE.value:
                continue
            seen.add(eid)
            found.append(eid)
        return found[0] if len(found) == 1 else ""

    def _lookup_instance_pipe(name: str, nfile: str) -> str:
        """Prefer a non-pointer TPipe; skip ``TPipe *pipeIn`` parameters."""
        if not name:
            return ""
        hit = _lookup_storage(name, [], nfile, kinds={EntityKind.PIPE.value})
        ent = codemap.entities.get(hit) if hit else None
        if (
            ent is not None
            and not ent.attrs.get("pointer")
            and ent.attrs.get("catalog") != "ascendc"
        ):
            return hit
        nf = str(nfile or "").replace("\\", "/")
        file_hits: list[str] = []
        other_hits: list[str] = []
        for e in codemap.by_kind(EntityKind.PIPE):
            if e.name != name:
                continue
            if e.attrs.get("catalog") == "ascendc" or e.attrs.get("pointer"):
                continue
            ef = str(e.file or "").replace("\\", "/")
            if nf and ef == nf:
                file_hits.append(e.id)
            else:
                other_hits.append(e.id)
        if len(file_hits) == 1:
            return file_hits[0]
        if file_hits:
            return file_hits[0]
        return other_hits[0] if len(other_hits) == 1 else ""

    tpipe_decl_by_file: dict[str, dict[str, int]] = {}
    kernel_file_text: dict[str, str] = {}
    for path in files or []:
        pth = Path(path)
        if not pth.is_file():
            continue
        try:
            text = read_text(pth)
        except OSError:
            continue
        nfile = _norm_file(str(pth), root)
        kernel_file_text[nfile] = text
        tpipe_decl_by_file[nfile] = _tpipe_decl_lines(text)

    def _collect_lexical_init_pipe_args() -> None:
        """``opPost.Init(..., &pipePost)`` in #define bodies clang never turns into Init calls."""
        for path in files or []:
            pth = Path(path)
            if not pth.is_file():
                continue
            try:
                text = read_text(pth)
            except OSError:
                continue
            nfile = _norm_file(str(pth), root)
            phys = text.splitlines()
            decl_lines = tpipe_decl_by_file.get(nfile) or _tpipe_decl_lines(text)
            tpipe_decl_by_file[nfile] = decl_lines
            lexical_inherit.extend(_inherit_pairs_from_text(text))
            for start_line, logical in _backslash_logical_lines(text):
                for m in _RECV_INIT_RE.finditer(logical):
                    recv = m.group("recv")
                    args = _paren_arg_text(logical, m.end() - 1)
                    if not args:
                        continue
                    inst_ids: list[str] = []
                    first_name = ""
                    for am in _ADDR_IDENT_RE.finditer(args):
                        name = am.group("name")
                        aid = _lookup_instance_pipe(name, nfile)
                        if not aid:
                            continue
                        inst_ids.append(aid)
                        if not first_name:
                            first_name = name
                    if not inst_ids:
                        continue
                    owners = _local_decl_owners(logical, recv, m.start())
                    if not owners:
                        continue
                    aline = start_line
                    if first_name and first_name in decl_lines:
                        aline = decl_lines[first_name]
                    elif first_name:
                        token = f"TPipe {first_name}"
                        for pi, raw in enumerate(phys):
                            if token in raw or re.search(
                                rf"\bTPipe\b[^;\n]*\b{re.escape(first_name)}\b", raw
                            ):
                                aline = pi + 1
                                break
                    for owner in owners:
                        for aid in inst_ids:
                            pipe_arg_sites.append((owner, aid, nfile, aline))

    pipe_arg_sites: list[tuple[str, str, str, int]] = []
    lexical_inherit: list[tuple[str, str]] = []
    initbuffer_links: list[tuple[str, str, str, int, str]] = []
    buf_count = 0
    reg_count = 0
    pipe_count = 0
    event_count = 0
    queue_count = 0

    # Member fields as BUFFER anchors when typed as storage/wrapper/alias-to-them.
    for row in members:
        type_text = str(row.get("type_text") or "")
        expanded = _resolve_alias_chain(type_text)
        name = str(row["member"])
        if not is_valid_storage_name(name):
            continue
        base = _base_type_name(expanded) or str(row.get("base_type") or "")
        sync_kind = _sync_object_kind(expanded) or _sync_object_kind(type_text)
        if sync_kind is not None:
            nfile = str(row["file"])
            line = int(row["line"])
            owner = str(row["owner"])
            if (owner, name) in buffer_by_key:
                continue
            root_spell = (
                "TPipe"
                if sync_kind == EntityKind.PIPE
                else ("TQue" if sync_kind == EntityKind.QUEUE else "HardEvent")
            )
            root_id = _ensure_ascendc_root(codemap, root_spell, root_kind="SYNC")
            pipe_attrs = {
                    "scope": trusted_scope_name(owner),
                    "type_name": _persist_type_name(type_text),
                    "tposition": tposition_from_type_text(expanded)
                    or tposition_from_type_text(type_text)
                    or "",
                    "memory_space": memory_space_from_type_text(expanded)
                    or memory_space_from_type_text(type_text)
                    or "",
                    "root_status": "REACHED",
                    "root_kind": "SYNC",
                    "root": f"AscendC::{root_spell}",
                    "provenance": str(row.get("provenance") or "kernel_root_trace"),
                    "allocated": False,
                    "pointer": _type_is_pointer(type_text) or _type_is_pointer(expanded),
            }
            if sync_kind == EntityKind.PIPE and not pipe_attrs["pointer"]:
                pipe_attrs["role"] = "launch_instance"
                pipe_attrs["kernel_file"] = nfile
            ent = bind_local_object(
                codemap,
                sync_kind,
                name,
                file=nfile,
                line=line,
                scope=owner,
                root=root,
                attrs=pipe_attrs,
                status="extracted",
                confidence=1.0,
            )
            if ent is None:
                continue
            _link(codemap, RelationKind.ROOTED_AT, ent.id, root_id)
            if owner in type_ents:
                _link(
                    codemap,
                    RelationKind.CONTAINS,
                    type_ents[owner],
                    ent.id,
                    attrs={"member": name, "via": "class_member"},
                )
                _link(
                    codemap,
                    RelationKind.DECLARES,
                    type_ents[owner],
                    ent.id,
                    attrs={"member": name, "via": "class_member"},
                )
            _index_storage(ent.id, scope=owner, name=name, nfile=nfile)
            pipe_count += int(sync_kind == EntityKind.PIPE)
            event_count += int(sync_kind == EntityKind.EVENT)
            queue_count += int(sync_kind == EntityKind.QUEUE)
            continue
        branch_bases = _row_branch_bases(
            row, type_text, expanded, str(row.get("canonical_type") or "")
        )
        branch_ents = [
            codemap.entities[type_ents[b]]
            for b in branch_bases
            if b in type_ents and type_ents[b] in codemap.entities
        ]
        owner_ent = codemap.entities.get(type_ents[base]) if base in type_ents else None
        wraps_storage = bool(owner_ent and owner_ent.attrs.get("wraps_storage"))
        wraps_storage = wraps_storage or any(
            e.attrs.get("wraps_storage") for e in branch_ents
        )
        wraps_lock = bool(owner_ent and owner_ent.attrs.get("wraps_lock"))
        wraps_lock = wraps_lock or any(e.attrs.get("wraps_lock") for e in branch_ents)
        catalog_root = (
            _catalog_storage_root(expanded)
            or _catalog_storage_root(type_text)
            or _catalog_storage_root(str(row.get("canonical_type") or ""))
        )
        space = (
            memory_space_from_type_text(expanded)
            or memory_space_from_type_text(type_text)
            or memory_space_from_type_text(str(row.get("canonical_type") or ""))
            or ""
        )
        known = (
            is_storage_type_text(expanded)
            or is_storage_type_text(type_text)
            or is_storage_type_text(str(row.get("canonical_type") or ""))
            or catalog_root
            or wraps_storage
            or base in alias_to_target
            or (
                owner_ent is not None
                and str(owner_ent.attrs.get("root_status") or "") == "REACHED"
                and "LocalTensor" in str(owner_ent.attrs.get("root") or "")
            )
            or any(
                str(e.attrs.get("root_status") or "") == "REACHED"
                and "LocalTensor" in str(e.attrs.get("root") or "")
                for e in branch_ents
            )
            or (bool(branch_bases) and bool(space))
        )
        if not known:
            continue
        nfile = str(row["file"])
        line = int(row["line"])
        owner = str(row["owner"])
        resolved = resolve_buffer_decl(expanded, _TRACE_ARCHITECTURE) or resolve_buffer_decl(
            type_text, _TRACE_ARCHITECTURE
        )
        if not space:
            space = "UNKNOWN"
        root_spell = catalog_root
        is_wrapper = bool((wraps_storage or branch_bases) and not catalog_root)
        wrapper_spell = ""
        if branch_bases:
            wrapper_spell = _wrapper_from_branch_types(
                codemap, branch_ents, expected=len(branch_bases)
            )
            root_spell = _common_branch_root(branch_ents, expected=len(branch_bases))
        elif is_wrapper:
            wrapper_spell = base if wraps_storage else ""
            if not root_spell and owner_ent is not None:
                roots = list(owner_ent.attrs.get("wrapped_roots") or [])
                root_spell = str(roots[0] if roots else "")
        elif not root_spell and base in ASCENDC_BUFFER_TYPES:
            root_spell = base
        elif not root_spell and owner_ent is not None:
            root_spell = str(owner_ent.attrs.get("root") or "").replace("AscendC::", "")
        type_name = (
            "|".join(branch_bases) if branch_bases else _persist_type_name(type_text)
        )
        if branch_bases and not root_spell:
            root_status = "GUARDED"
        elif root_spell:
            root_status = "REACHED"
        else:
            root_status = "UNRESOLVED"
        attrs = {
            "memory_space": space,
            "tposition": tposition_from_type_text(expanded)
            or tposition_from_type_text(type_text)
            or "",
            "scope": trusted_scope_name(owner),
            "type_name": type_name,
            "role": "storage_wrapper" if is_wrapper else "storage",
            "wrapper": wrapper_spell,
            "root_status": root_status,
            "root_kind": "STORAGE" if root_spell else "",
            "root": f"AscendC::{root_spell}" if root_spell else "",
            "trace": [name]
            + (branch_bases or ([base] if base else []))
            + ([root_spell] if root_spell else []),
            "allocated": bool(catalog_root in {"TBuf", "TBufPool"}),
            "wraps_lock": wraps_lock,
            "provenance": _PROV,
        }
        if branch_bases:
            attrs["conditional_flag"] = True
        ent = bind_local_object(
            codemap,
            EntityKind.BUFFER,
            name,
            file=nfile,
            line=line,
            scope=owner,
            root=root,
            attrs=attrs,
            status="extracted" if root_spell else "partial",
            confidence=0.9 if root_spell else 0.4,
        )
        if ent is None:
            continue
        bid = ent.id
        _index_storage(ent.id, scope=trusted_scope_name(owner), name=name, nfile=nfile)
        buf_count += 1
        if owner in type_ents:
            _link(
                codemap,
                RelationKind.CONTAINS,
                type_ents[owner],
                ent.id,
                attrs={"member": name, "via": "class_member"},
            )
            _link(
                codemap,
                RelationKind.DECLARES,
                type_ents[owner],
                ent.id,
                attrs={"member": name, "via": "class_member"},
            )
        if base in type_ents and _is_catalog_storage_base(base):
            _link(codemap, RelationKind.WRAPS, ent.id, type_ents[base], attrs={"via": "member_type"})
        polarity = _conditional_polarity_map(
            type_text, expanded, str(row.get("canonical_type") or ""), *list(row.get("alt_type_texts") or [])
        )
        for idx, branch in enumerate(branch_bases):
            if branch in type_ents:
                _link(
                    codemap,
                    RelationKind.WRAPS,
                    ent.id,
                    type_ents[branch],
                    attrs=_conditional_branch_attrs(
                        name, branch, idx, len(branch_bases), polarity
                    ),
                )
        if root_spell:
            rid = _ensure_ascendc_root(codemap, root_spell, root_kind="STORAGE")
            _link(codemap, RelationKind.WRAPS, ent.id, rid, attrs={"via": "storage_root"})
            _link(codemap, RelationKind.ROOTED_AT, ent.id, rid)
            _link(
                codemap,
                RelationKind.INSTANCE_OF,
                ent.id,
                rid,
                attrs={"via": "view_type", "file": nfile, "line": line},
            )

    for decl in decls or []:
        type_text, name, function, file, line = _decl_fields(decl)
        if not name or not is_valid_storage_name(name):
            continue
        expanded = _resolve_alias_chain(type_text)
        base = _base_type_name(expanded)
        sync_kind = _sync_object_kind(expanded) or _sync_object_kind(type_text)
        if sync_kind is not None:
            nfile = _norm_file(file, root)
            line = int(line)
            this_ptr = _type_is_pointer(type_text) or _type_is_pointer(expanded)
            if sync_kind == EntityKind.PIPE and not this_ptr:
                mapped = (tpipe_decl_by_file.get(nfile) or {}).get(name)
                if mapped:
                    line = mapped
            existing_key = buffer_by_key.get((function, name))
            if existing_key:
                continue
            existing_name = buffer_by_name.get(name)
            if existing_name:
                exist_ent = codemap.entities.get(existing_name)
                # A TPipe* member / other-file namesake must not suppress a local instance.
                skip_namesake = True
                if (
                    sync_kind == EntityKind.PIPE
                    and not this_ptr
                    and exist_ent is not None
                    and (
                        exist_ent.attrs.get("pointer")
                        or str(exist_ent.file or "").replace("\\", "/") != nfile
                    )
                ):
                    skip_namesake = False
                if skip_namesake:
                    continue
            root_spell = (
                "TPipe"
                if sync_kind == EntityKind.PIPE
                else ("TQue" if sync_kind == EntityKind.QUEUE else "HardEvent")
            )
            root_id = _ensure_ascendc_root(codemap, root_spell, root_kind="SYNC")
            pipe_attrs = {
                    "scope": trusted_scope_name(function),
                    "type_name": _persist_type_name(type_text),
                    "tposition": tposition_from_type_text(expanded)
                    or tposition_from_type_text(type_text)
                    or "",
                    "memory_space": memory_space_from_type_text(expanded)
                    or memory_space_from_type_text(type_text)
                    or "",
                    "root_status": "REACHED",
                    "root_kind": "SYNC",
                    "root": f"AscendC::{root_spell}",
                    "provenance": "kernel_root_trace",
                    "allocated": False,
                    "pointer": _type_is_pointer(type_text) or _type_is_pointer(expanded),
            }
            if sync_kind == EntityKind.PIPE and not pipe_attrs["pointer"]:
                pipe_attrs["role"] = "launch_instance"
                pipe_attrs["kernel_file"] = nfile
            ent = bind_local_object(
                codemap,
                sync_kind,
                name,
                file=nfile,
                line=line,
                scope=function,
                root=root,
                attrs=pipe_attrs,
                status="extracted",
                confidence=1.0,
            )
            if ent is None:
                continue
            _link(codemap, RelationKind.ROOTED_AT, ent.id, root_id)
            _index_storage(ent.id, scope=trusted_scope_name(function), name=name, nfile=nfile)
            pipe_count += int(sync_kind == EntityKind.PIPE)
            event_count += int(sync_kind == EntityKind.EVENT)
            queue_count += int(sync_kind == EntityKind.QUEUE)
            continue
        branch_bases = _conditional_storage_bases_any(type_text, expanded)
        branch_ents = [
            codemap.entities[type_ents[b]]
            for b in branch_bases
            if b in type_ents and type_ents[b] in codemap.entities
        ]
        owner_ent = codemap.entities.get(type_ents[base]) if base in type_ents else None
        wraps_storage = bool(owner_ent and owner_ent.attrs.get("wraps_storage"))
        wraps_storage = wraps_storage or any(
            e.attrs.get("wraps_storage") for e in branch_ents
        )
        catalog_root = _catalog_storage_root(expanded) or _catalog_storage_root(type_text)
        space_hint = memory_space_from_type_text(expanded) or memory_space_from_type_text(
            type_text
        )
        known = (
            is_storage_type_text(expanded)
            or is_storage_type_text(type_text)
            or catalog_root
            or wraps_storage
            or base in alias_to_target
            or (
                owner_ent is not None
                and str(owner_ent.attrs.get("root_status") or "") == "REACHED"
            )
            or (bool(branch_bases) and bool(space_hint))
        )
        if not known:
            continue
        if is_non_storage_type(expanded):
            continue
        nfile = _norm_file(file, root)
        reg_class = register_class_from_type(expanded) or register_class_from_type(type_text)
        if reg_class:
            if not nfile or int(line or 0) <= 0:
                continue
            root_id = _ensure_ascendc_root(
                codemap, _base_type_name(expanded) or "RegTensor", root_kind="REGISTER"
            )
            ent = bind_local_object(
                codemap,
                EntityKind.REGISTER,
                name,
                file=nfile,
                line=line,
                scope=function,
                root=root,
                attrs={
                    "register_class": reg_class,
                    "memory_space": REGISTER_MEMORY_SPACE,
                    "type_name": _persist_type_name(type_text),
                    "scope": trusted_scope_name(function),
                    "root_status": "REACHED",
                    "root_kind": "REGISTER",
                    "root": codemap.entities[root_id].name,
                    "trace": [name, _base_type_name(expanded) or type_text],
                    "provenance": "kernel_root_trace",
                },
                status="extracted",
                confidence=1.0,
            )
            if ent is None:
                continue
            _link(codemap, RelationKind.ROOTED_AT, ent.id, root_id)
            _index_storage(ent.id, scope=function, name=name, nfile=nfile)
            reg_count += 1
            continue

        resolved = resolve_buffer_decl(expanded, _TRACE_ARCHITECTURE) or resolve_buffer_decl(
            type_text, _TRACE_ARCHITECTURE
        )
        space = memory_space_from_type_text(expanded) or memory_space_from_type_text(type_text) or "UNKNOWN"
        is_wrapper = bool((wraps_storage or branch_bases) and not catalog_root)
        wrapper_spell = base if (is_wrapper and not branch_bases) else ""
        project_reached = bool(
            owner_ent is not None
            and str(owner_ent.attrs.get("root_status") or "") == "REACHED"
        ) or any(
            str(e.attrs.get("root_status") or "") == "REACHED" for e in branch_ents
        )
        root_status = "REACHED"
        root_spell = catalog_root
        if branch_bases:
            wrapper_spell = _wrapper_from_branch_types(
                codemap, branch_ents, expected=len(branch_bases)
            )
            root_spell = _common_branch_root(branch_ents, expected=len(branch_bases))
            if not root_spell:
                root_status = "GUARDED"
        elif is_wrapper:
            roots = list(owner_ent.attrs.get("wrapped_roots") or []) if owner_ent else []
            root_spell = str(roots[0] if roots else "LocalTensor")
        elif base in ASCENDC_BUFFER_TYPES:
            root_spell = base
        elif project_reached:
            root_spell = str(owner_ent.attrs.get("root") or "").replace(
                "AscendC::", ""
            ) or "LocalTensor"
        elif space != "UNKNOWN" and (resolved or is_storage_type_text(expanded)):
            root_spell = storage_root_kind_from_space(space)
        else:
            root_status = "UNRESOLVED"
        if branch_bases and not root_spell:
            root_status = "GUARDED"
        elif root_spell:
            root_status = "REACHED"

        attrs = {
            "memory_space": space,
            "tposition": tposition_from_type_text(expanded)
            or tposition_from_type_text(type_text)
            or "",
            "scope": trusted_scope_name(function),
            "type_name": (
                "|".join(branch_bases) if branch_bases else _persist_type_name(type_text)
            ),
            "role": (
                "storage_wrapper"
                if is_wrapper
                else ("project_wrapper" if project_reached else "cann_storage")
            ),
            "wrapper": wrapper_spell,
            "root_status": root_status,
            "root_kind": "STORAGE" if root_spell else "",
            "root": f"AscendC::{root_spell}" if root_spell else "",
            "trace": [name]
            + ([wrapper_spell] if wrapper_spell else [])
            + ([root_spell] if root_spell else []),
            "allocated": bool(catalog_root in {"TBuf", "TBufPool"}),
            "wraps_lock": bool(owner_ent and owner_ent.attrs.get("wraps_lock")),
            "provenance": "kernel_root_trace",
        }
        if branch_bases:
            attrs["conditional_flag"] = True
        if root_status == "UNRESOLVED":
            attrs["gap_code"] = REASON_NO_ASCENDC_ROOT
        ent = bind_local_object(
            codemap,
            EntityKind.BUFFER,
            name,
            file=nfile,
            line=line,
            scope=function,
            root=root,
            attrs=attrs,
            status="extracted" if root_status == "REACHED" else "partial",
            confidence=1.0 if root_status == "REACHED" else 0.4,
        )
        if ent is None:
            continue
        if root_status == "UNRESOLVED":
            gaps.append(
                {
                    "code": REASON_NO_ASCENDC_ROOT,
                    "entity_id": ent.id,
                    "name": name,
                    "file": nfile,
                    "line": line,
                }
            )
        _index_storage(ent.id, scope=trusted_scope_name(function), name=name, nfile=nfile)
        buf_count += 1
        polarity = _conditional_polarity_map(type_text, expanded)
        for idx, branch in enumerate(branch_bases):
            if branch in type_ents:
                _link(
                    codemap,
                    RelationKind.WRAPS,
                    ent.id,
                    type_ents[branch],
                    attrs=_conditional_branch_attrs(
                        name, branch, idx, len(branch_bases), polarity
                    ),
                )
        if root_spell:
            rid = _ensure_ascendc_root(codemap, root_spell, root_kind="STORAGE")
            if is_wrapper and wrapper_spell in type_ents and _is_catalog_storage_base(
                wrapper_spell
            ):
                _link(codemap, RelationKind.WRAPS, ent.id, type_ents[wrapper_spell])
            if base in type_ents and _is_catalog_storage_base(base):
                _link(codemap, RelationKind.WRAPS, ent.id, type_ents[base], attrs={"via": "decl_type"})
            _link(codemap, RelationKind.WRAPS, ent.id, rid, attrs={"via": "storage_root"})
            _link(codemap, RelationKind.ROOTED_AT, ent.id, rid)
            _link(
                codemap,
                RelationKind.INSTANCE_OF,
                ent.id,
                rid,
                attrs={"via": "view_type", "file": nfile, "line": line},
            )

    # --- 5. METHOD + OPERATION call sites (all source calls) -------------
    method_ents: dict[str, str] = {}  # identity key → METHOD entity id

    def _ensure_method(
        name: str,
        *,
        file: str = "",
        line: int = 0,
        usr: str = "",
        qualified: str = "",
        owner: str = "",
    ) -> str:
        short = str(name or "").split("::")[-1]
        if is_forbidden_callable_name(short):
            return ""
        q = str(qualified or "").strip()
        if not q and owner:
            q = f"{owner}::{short}"
        # Bare short "qualified" from lexical enclosing-func is not a real
        # qualifier — collapse it so caller/callee METHOD nodes unify.
        if q and "::" not in q and q.split("::")[-1] == short:
            q = ""
        if usr:
            key = f"usr:{usr}"
        elif q:
            key = f"q:{q}"
        elif owner:
            key = f"q:{owner}::{short}"
        else:
            key = f"name:{short}"
        if key in method_ents:
            return method_ents[key]
        existing = find_declaration(
            codemap, EntityKind.METHOD, symbol=short, owner=owner, file=file
        )
        if existing is None and owner:
            existing = find_declaration(
                codemap, EntityKind.METHOD, symbol=short, owner=owner
            )
        if existing is not None:
            method_ents[key] = existing.id
            return existing.id
        if not usr:
            hits = [
                e
                for e in codemap.by_name(short, kind=EntityKind.METHOD)
                if str(e.name or "").split("::")[-1] == short
            ]
            if owner:
                owned = [
                    e
                    for e in hits
                    if str(e.attrs.get("owner") or e.attrs.get("lexical_owner") or "")
                    == owner
                ]
                if len(owned) == 1:
                    method_ents[key] = owned[0].id
                    return owned[0].id
            if len(hits) == 1:
                method_ents[key] = hits[0].id
                return hits[0].id
            return ""
        ent = bind_or_create(
            codemap,
            EntityKind.METHOD,
            short,
            file=file,
            line=line,
            owner=owner,
            architecture=str(getattr(codemap, "architecture", "") or ""),
            attrs={
                "role": "source_method",
                "root_status": "UNRESOLVED",
                "root_kind": "",
                "root": "",
                "trace": [q or short],
                "usr": usr,
                "qualified_name": q,
                "spelling": short,
                "provenance": "kernel_root_trace",
            },
            status="partial",
        )
        if ent is None:
            return ""
        method_ents[key] = ent.id
        return ent.id

    op_count = 0
    seen_op_ids: set[str] = set()
    for i, site in enumerate(calls or []):
        if i % 200 == 0 and time.perf_counter() > deadline:
            gated_fill_complete = False
            break
        d = site if isinstance(site, dict) else kscan.site_as_dict(site)
        callee = str(d.get("callee") or "").split("::")[-1]
        if not callee or is_forbidden_callable_name(callee):
            continue
        file = str(d.get("file") or "")
        line = int(d.get("line") or 0)
        column = int(d.get("column") or 0)
        category, _engine, conf = semreg.classify(callee)
        receiver = str(d.get("receiver") or "")
        function = str(d.get("caller") or "")
        if function and is_forbidden_callable_name(function.split("::")[-1]):
            function = ""
        nfile = _norm_file(file, root)
        caller_qualified = str(d.get("caller_qualified") or "")
        caller_usr = str(d.get("caller_usr") or "")
        callee_qualified = str(d.get("callee_qualified") or "")
        callee_usr = str(d.get("callee_usr") or "")
        callee_decl_file = str(d.get("callee_decl_file") or "")
        callee_decl_line = int(d.get("callee_decl_line") or 0)
        identity_kind = str(d.get("identity_kind") or "")
        receiver_type = str(d.get("receiver_type") or "")
        receiver_canonical = str(d.get("receiver_canonical_type") or "")
        has_identity = bool(callee_usr or callee_qualified or callee_decl_file)

        # Receiver for TPipe/TQue CANN method contracts.
        recv_name = _expr_storage_name(receiver)
        identity_scopes = _identity_scopes(function, caller_qualified)
        bid_recv = ""
        if recv_name:
            bid_recv = _lookup_storage(recv_name, identity_scopes, nfile)
        elif receiver:
            bid_recv = (
                buffer_by_key.get((function, receiver)) or buffer_by_name.get(receiver) or ""
            )
        recv_is_wrapper = False
        if bid_recv and bid_recv in codemap.entities:
            be = codemap.entities[bid_recv]
            recv_is_wrapper = be.attrs.get("role") in {
                "storage_wrapper",
                "project_wrapper",
            } or bool(be.attrs.get("wraps_lock") or be.attrs.get("wraps_storage"))
        recv_is_tque = _receiver_is_tque(
            bid_recv=bid_recv,
            receiver_type=receiver_type,
            receiver_canonical=receiver_canonical,
            codemap=codemap,
        )
        recv_is_tpipe = _receiver_is_tpipe(
            bid_recv=bid_recv,
            receiver_type=receiver_type,
            receiver_canonical=receiver_canonical,
            codemap=codemap,
        )

        args = [str(a) for a in (d.get("args") or [])]
        targs = [str(a) for a in (d.get("template_args") or [])]
        callee_owner = ""
        if callee_qualified and "::" in callee_qualified:
            callee_owner = (
                callee_qualified.rsplit("::", 1)[0].split("::")[-1].split("<")[0]
            )
        elif receiver_canonical or receiver_type:
            callee_owner = _owner_from_receiver_type(
                receiver_canonical or receiver_type
            )
        for arg in args:
            raw = str(arg).strip()
            if not raw.startswith("&"):
                continue
            aname = _expr_storage_name(raw)
            if not aname:
                continue
            aid = _lookup_storage(
                aname, identity_scopes, nfile, kinds={EntityKind.PIPE.value}
            )
            ae = codemap.entities.get(aid) if aid else None
            if ae is None or ae.attrs.get("pointer"):
                continue
            pipe_arg_sites.append((callee_owner, aid, nfile, line))
        if callee == "InitBuffer":
            obj_name = _expr_storage_name(args[0] if args else "")
            if not obj_name and args:
                obj_name = str(args[0]).lstrip("&").replace("->", ".").split(".")[-1]
            obj_id = (
                _lookup_storage(
                    obj_name,
                    identity_scopes,
                    nfile,
                    kinds={EntityKind.BUFFER.value, EntityKind.QUEUE.value},
                )
                if obj_name
                else ""
            )
            pipe_id = bid_recv
            if pipe_id and pipe_id in codemap.entities:
                if codemap.entities[pipe_id].kind_name() != EntityKind.PIPE.value:
                    pipe_id = ""
            if not pipe_id and recv_name:
                pipe_id = _lookup_storage(
                    recv_name, identity_scopes, nfile, kinds={EntityKind.PIPE.value}
                )
            if not pipe_id and recv_is_tpipe and not recv_name:
                pipe_id = _unique_pipe_in_scopes(identity_scopes)
            obj_ok = False
            if obj_id and obj_id in codemap.entities:
                obj_kind = codemap.entities[obj_id].kind_name()
                obj_ok = obj_kind in {
                    EntityKind.BUFFER.value,
                    EntityKind.QUEUE.value,
                }
            if obj_ok:
                obj = codemap.entities[obj_id]
                obj.attrs["allocated"] = True
                obj.attrs["root_status"] = "REACHED"
                obj.status = "extracted"
            if pipe_id and obj_ok:
                _link(
                    codemap,
                    RelationKind.BINDS,
                    pipe_id,
                    obj_id,
                    attrs={
                        "via": "InitBuffer",
                        "file": nfile,
                        "line": line,
                        "receiver": receiver,
                    },
                )
                initbuffer_links.append((pipe_id, obj_id, nfile, line, receiver))
                qent = codemap.entities.get(pipe_id)
                if qent is not None and qent.kind_name() == EntityKind.QUEUE.value:
                    _link_backed_by(
                        codemap,
                        pipe_id,
                        obj_id,
                        space=str(obj.attrs.get("memory_space") or ""),
                        via="InitBuffer",
                        file=nfile,
                        line=line,
                    )
        if callee in _VIEW_GETTERS and bid_recv:
            recv_ent = codemap.entities.get(bid_recv)
            if recv_ent is not None and _is_storage_object(recv_ent):
                lhs = _assign_lhs_on_line(kernel_file_text.get(nfile, ""), line)
                view_id = (
                    _lookup_storage(lhs, identity_scopes, nfile) if lhs else ""
                )
                if view_id and view_id != bid_recv:
                    _link_backed_by(
                        codemap,
                        view_id,
                        bid_recv,
                        space=str(recv_ent.attrs.get("memory_space") or ""),
                        via=callee,
                        file=nfile,
                        line=line,
                    )
        if callee in STACK_BUFFER_CALLEES:
            obj_name = _expr_storage_name(args[0] if args else "")
            if not obj_name and args:
                obj_name = str(args[0]).lstrip("&").replace("->", ".").split(".")[-1]
            obj_id = _lookup_storage(obj_name, identity_scopes, nfile) if obj_name else ""
            if obj_id and obj_id in codemap.entities:
                obj = codemap.entities[obj_id]
                obj.attrs["allocated"] = True
                obj.attrs["root_status"] = "REACHED"
                obj.attrs["stack_pop"] = True
                obj.status = "extracted"
                stack_pipe = bid_recv
                if stack_pipe and stack_pipe in codemap.entities:
                    if (
                        codemap.entities[stack_pipe].kind_name()
                        != EntityKind.PIPE.value
                    ):
                        stack_pipe = ""
                if stack_pipe:
                    _link(
                        codemap,
                        RelationKind.BINDS,
                        stack_pipe,
                        obj_id,
                        attrs={
                            "via": "PopStackBuffer",
                            "file": nfile,
                            "line": line,
                            "receiver": receiver,
                        },
                    )
        # Incomplete member-Get / empty Or: spelling alone is a guess.
        if callee == "Get" and not receiver and not targs and not has_identity:
            continue
        if callee in _VECTOR_AMBIGUOUS_ROOTS and not args and not targs and not has_identity:
            continue
        bridge = _select_framework_bridge(
            callee=callee,
            receiver=receiver,
            recv_is_wrapper=recv_is_wrapper,
            recv_is_tque=recv_is_tque,
            recv_is_tpipe=recv_is_tpipe,
            receiver_type=receiver_type,
            receiver_canonical=receiver_canonical,
            callee_usr=callee_usr,
            targs=targs,
        )
        proven, proven_spell = _prove_ascendc_api_root(
            callee=callee,
            callee_qualified=callee_qualified,
            callee_usr=callee_usr,
            callee_decl_file=callee_decl_file,
            receiver=receiver,
            receiver_type=receiver_type,
            receiver_canonical_type=receiver_canonical,
            has_identity=has_identity,
        )
        if (
            not proven
            and not bridge
            and not receiver
            and _looks_like_reg_or_vector_call(args, targs)
            and (
                callee in _VECTOR_AMBIGUOUS_ROOTS
                or is_cann_vf_api(callee, architecture=_TRACE_ARCHITECTURE)
                or is_ambiguous_vf_name(callee)
            )
        ):
            proven, proven_spell = True, vf_root_spelling(callee)
        is_root = bool(proven or bridge)
        root_kind = ""
        root_spell = ""
        if bridge:
            root_spell, root_kind = bridge
        elif proven:
            root_spell = proven_spell
            root_kind = _category_root_kind(category, proven_spell)
        is_builtin = _is_compiler_builtin(
            callee,
            usr=callee_usr,
            decl_file=callee_decl_file,
            qualified=callee_qualified,
        )
        is_scalar_minmax = bool(
            callee in {"Min", "Max"}
            and not receiver
            and not _looks_like_reg_or_vector_call(args, targs)
        )
        is_project = bool(
            not is_root
            and not is_builtin
            and (
                (
                    callee_qualified
                    and callee_decl_file
                    and not _qualified_looks_ascendc(callee_qualified)
                    and not _is_framework_decl_file(callee_decl_file)
                )
                or (
                    _is_project_decl_file(callee_decl_file)
                    and not _qualified_looks_ascendc(callee_qualified)
                    and not _is_framework_decl_file(callee_decl_file)
                )
                or is_scalar_minmax
                or bool(receiver)
            )
        )
        # The declaration line itself is not a call site.
        if (
            not is_root
            and not receiver
            and callee_decl_line == line
            and _norm_file(callee_decl_file, root) == nfile
        ):
            continue

        mint_op = _should_mint_operation(
            callee=callee,
            is_root=is_root,
            is_builtin=is_builtin,
            is_project=is_project,
        )
        if not mint_op and (is_builtin or _is_type_like_root(callee)):
            continue
        oid = (
            operation_site_id(
                file=file, line=line, column=column, callee=callee, root=root
            )
            if mint_op
            else ""
        )
        trace = [callee_qualified or callee]
        if bridge or proven:
            trace.append(f"AscendC::{root_spell}")
        elif is_project:
            trace.append(callee_qualified)
        project_proof = (
            "project_method"
            if identity_kind == "method" or (is_project and receiver)
            else ("project_free" if is_project else "")
        )
        if is_root:
            op_root_status, op_root_kind = "REACHED", (root_kind or "")
            op_root = f"AscendC::{root_spell}" if root_spell else ""
            op_proof = (
                "framework_bridge"
                if bridge
                else ("qualified_or_decl" if proven and has_identity else "lexical_free_catalog")
            )
        elif is_builtin:
            op_root_status, op_root_kind, op_root = "BUILTIN", "BUILTIN", ""
            op_proof = "compiler_builtin"
        elif is_project:
            op_root_status, op_root_kind, op_root = "PROJECT", "PROJECT", ""
            op_proof = project_proof or "project_free"
        elif category == "UNKNOWN" and nfile and line > 0:
            op_root_status, op_root_kind, op_root = "PROJECT", "PROJECT", ""
            op_proof = "project_unclassified"
        else:
            op_root_status, op_root_kind, op_root = "UNRESOLVED", "", ""
            op_proof = project_proof
        attrs = {
            "callee": callee,
            "callee_qualified": callee_qualified,
            "callee_usr": callee_usr,
            "callee_decl_file": callee_decl_file,
            "callee_decl_line": callee_decl_line,
            "identity_kind": identity_kind,
            "category": (
                category
                if category != "UNKNOWN"
                else (
                    "framework_bridge"
                    if bridge
                    else (
                        "compiler_builtin"
                        if is_builtin
                        else ("project_symbol" if (is_project or op_root_status == "PROJECT") else "UNKNOWN")
                    )
                )
            ),
            "function": function,
            "args": args,
            "template_args": targs,
            "receiver": receiver,
            "receiver_type": receiver_type,
            "receiver_canonical_type": receiver_canonical,
            "root_status": op_root_status,
            "root_kind": op_root_kind,
            "root": op_root,
            "wrapper": callee if bridge else "",
            "trace": trace,
            "provenance": str(d.get("provenance") or provenance),
            "column": column,
            "root_proof": op_proof,
            "owner": kscan.kernel_file_owner(nfile or file, Path(root)),
            "instantiation_n": int(d.get("instantiation_n") or 1),
        }
        if targs:
            attrs["template_arg_sets"] = [list(targs)]
        if callee in STACK_BUFFER_CALLEES:
            attrs["mechanism"] = "stack"
        elif callee in SHARE_BUFFER_CALLEES:
            attrs["mechanism"] = "share"
        elif is_tque_callee(callee):
            attrs["mechanism"] = "tque"
        elif is_tpipe_callee(callee):
            attrs["mechanism"] = "tpipe"
        elif is_flag_sync(callee) or is_sync_root(callee):
            attrs["mechanism"] = str(resolve_sync_site(callee, args, targs).get("mechanism") or "")
        if not is_root and not is_builtin:
            if not nfile or line <= 0 or not callee.isidentifier():
                continue
        if (
            mint_op
            and not is_root
            and not is_project
            and not is_builtin
            and (category != "UNKNOWN" or conf not in {"", "unresolved"})
        ):
            attrs["gap_code"] = REASON_CALL_UNRESOLVED
            gaps.append(
                {
                    "code": REASON_CALL_UNRESOLVED,
                    "entity_id": oid,
                    "callee": callee,
                    "callee_qualified": callee_qualified,
                    "file": nfile,
                    "line": line,
                }
            )
        ent = None
        if mint_op and oid:
            existing = codemap.entities.get(oid)
            if existing is not None:
                existing.attrs["instantiation_n"] = int(
                    existing.attrs.get("instantiation_n") or 1
                ) + int(attrs.get("instantiation_n") or 1)
                if targs:
                    sets = existing.attrs.setdefault("template_arg_sets", [])
                    if isinstance(sets, list) and list(targs) not in sets:
                        sets.append(list(targs))
                ent = existing
            else:
                ent = codemap.upsert(
                    EntityKind.OPERATION,
                    callee,
                    eid=oid,
                    attrs=attrs,
                    file=nfile,
                    line=line,
                    status="extracted" if is_root else "partial",
                    confidence=(
                        1.0
                        if is_root and conf == "confirmed"
                        else (0.85 if bridge else 0.5)
                    ),
                )
                seen_op_ids.add(oid)
                op_count += 1

        if ent is not None:
            reads, writes = semreg.arg_effects(callee, args, receiver=receiver)
            for names, kind in (
                (reads, RelationKind.READS),
                (writes, RelationKind.WRITES),
            ):
                for raw in names:
                    aname = _expr_storage_name(raw) or str(raw or "").strip()
                    if not aname or not aname.isidentifier():
                        continue
                    aid = _lookup_storage(aname, identity_scopes, nfile)
                    if not aid:
                        continue
                    _link(
                        codemap,
                        kind,
                        ent.id,
                        aid,
                        attrs={
                            "via": "arg_effect",
                            "file": nfile,
                            "line": line,
                            "arg": aname,
                        },
                    )
            if callee in _MEMORY_TRANSFER_CALLEES or category == "memory_transfer":
                src_ids = [
                    _lookup_storage(
                        _expr_storage_name(raw) or str(raw or "").strip(),
                        identity_scopes,
                        nfile,
                    )
                    for raw in reads
                ]
                dst_ids = [
                    _lookup_storage(
                        _expr_storage_name(raw) or str(raw or "").strip(),
                        identity_scopes,
                        nfile,
                    )
                    for raw in writes
                ]
                engine = str(_engine or "")
                for sid in src_ids:
                    for did in dst_ids:
                        if not sid or not did or sid == did:
                            continue
                        _link(
                            codemap,
                            RelationKind.FLOWS_TO,
                            sid,
                            did,
                            attrs={
                                "via": "MemoryTransfer",
                                "callee": callee,
                                "file": nfile,
                                "line": line,
                                "engine": engine,
                            },
                        )

        # Flag identity only. TQue EnQue/DeQue have no user event; CANN owns that
        # handshake, so they never get SIGNALS/AWAITS.
        if ent is not None and is_flag_sync(callee) and not is_tque_callee(callee):
            sync = resolve_sync_site(callee, args, targs)
            raw_flag = str(sync.get("flag") or (args[0] if args else "")).strip()
            base, identity = _flag_identity(raw_flag)
            if base:
                event_id = make_id("Event", "sync", function, identity)
                event = codemap.upsert(
                    EntityKind.EVENT,
                    identity,
                    eid=event_id,
                    attrs={
                        "scope": function,
                        "identity": identity,
                        "identity_base": base,
                        "event_type": str(sync.get("event") or ""),
                        "mechanism": str(sync.get("mechanism") or ""),
                        "cross_core": bool(sync.get("cross_core")),
                        "provenance": str(d.get("provenance") or provenance),
                    },
                    file=nfile,
                    line=line,
                    status="extracted",
                    confidence=1.0,
                )
                relation_kind = (
                    RelationKind.SIGNALS
                    if "Wait" in FLAG_PAIR_MATE.get(canonical_sync_name(callee), "")
                    else RelationKind.AWAITS
                )
                _link(
                    codemap,
                    relation_kind,
                    ent.id,
                    event.id,
                    attrs={
                        "identity_arg": identity,
                        "file": nfile,
                        "line": line,
                        "column": column,
                    },
                )
                event_count += 1
                for role in ("src_pipe", "dst_pipe"):
                    pipe_name = str(sync.get(role) or "")
                    if not pipe_name:
                        continue
                    if not pipe_name.startswith("PIPE_"):
                        pipe_name = f"PIPE_{pipe_name}"
                    pipe_id = make_id("Pipe", "hard_event", pipe_name)
                    pipe = codemap.upsert(
                        EntityKind.PIPE,
                        pipe_name,
                        eid=pipe_id,
                        attrs={
                            "role": role,
                            "catalog": "ascendc",
                            "root_status": "REACHED",
                            "root_kind": "SYNC",
                            "root": f"AscendC::{pipe_name}",
                            "provenance": str(d.get("provenance") or provenance),
                        },
                        file=nfile,
                        line=line,
                        status="extracted",
                        confidence=1.0,
                    )
                    _link(
                        codemap,
                        RelationKind.REFERENCES,
                        event.id,
                        pipe.id,
                        attrs={"role": role},
                    )
                    pipe_count += 1

        # Source METHOD CALLS graph (caller → callee method or this op).
        caller_mid = (
            _ensure_method(
                function.split("::")[-1] if function else "",
                file=nfile,
                line=line,
                usr=caller_usr,
                qualified=caller_qualified or function,
            )
            if function
            else ""
        )
        callee_mid = ""
        if not proven and not bridge:
            owner_guess = ""
            if callee_qualified and "::" in callee_qualified:
                owner_guess = callee_qualified.rsplit("::", 1)[0].split("::")[-1]
                if owner_guess.lower() in {"conditional", "conditional_t"}:
                    owner_guess = ""
            elif receiver_type or receiver_canonical:
                owner_guess = _owner_from_receiver_type(receiver_canonical or receiver_type)
            callee_mid = _ensure_method(
                callee,
                file=_norm_file(callee_decl_file, root) or nfile,
                line=callee_decl_line or line,
                usr=callee_usr,
                qualified=callee_qualified,
                owner=owner_guess,
            )
            if is_project and callee_mid:
                me = codemap.entities.get(callee_mid)
                if me is not None:
                    me.status = "extracted"
                    me.confidence = max(float(me.confidence or 0), 0.9)
                    me.attrs["decl_file"] = _norm_file(callee_decl_file, root) or nfile
                    me.attrs["decl_line"] = callee_decl_line or line
                    if receiver:
                        me.attrs.setdefault("receivers", [])
                        recs = me.attrs.get("receivers")
                        if isinstance(recs, list) and receiver not in recs:
                            recs.append(receiver)
                bind_via = (
                    "project_decl"
                    if callee_decl_file
                    else ("method_receiver" if receiver else "project_free")
                )
                if ent is not None:
                    _link(
                        codemap,
                        RelationKind.BINDS,
                        ent.id,
                        callee_mid,
                        attrs={
                            "via": bind_via,
                            "file": nfile,
                            "line": line,
                            "column": column,
                            "decl_file": _norm_file(callee_decl_file, root),
                            "decl_line": callee_decl_line,
                            "receiver": receiver,
                        },
                    )
        if caller_mid and callee_mid and caller_mid != callee_mid:
            _link(
                codemap,
                RelationKind.CALLS,
                caller_mid,
                callee_mid,
                attrs={
                    "via": "source_call",
                    "file": nfile,
                    "line": line,
                    "column": column,
                    "receiver": receiver,
                },
                candidate=not has_identity,
            )
        if caller_mid and ent is not None:
            _link(
                codemap,
                RelationKind.CALLS,
                caller_mid,
                ent.id,
                attrs={
                    "via": "call_site",
                    "file": nfile,
                    "line": line,
                    "column": column,
                    "receiver": receiver,
                },
                candidate=not has_identity,
            )
        if ent is not None and is_root and root_spell:
            rid = _ensure_ascendc_root(codemap, root_spell, root_kind=root_kind or "COMPUTE_API")
            _link(
                codemap,
                RelationKind.ROOTED_AT,
                ent.id,
                rid,
                attrs={"via": "framework_method_bridge" if bridge else "ascendc_catalog"},
            )
            if caller_mid:
                # Direct edge so fixed-point can climb methods that call rooted ops.
                _link(
                    codemap,
                    RelationKind.CALLS,
                    caller_mid,
                    rid,
                    attrs={
                        "via": "rooted_call",
                        "file": nfile,
                        "line": line,
                        "column": column,
                    },
                    status="partial",
                )
        if ent is not None:
            # Operands are storage too, not just receivers. An AscendC vector
            # instruction is a free function -- `Add(vregDst, vregSrc0, vregSrc1)`
            # -- so the tensors and registers it touches arrive as arguments,
            # and a receiver-only edge can never see them. That is why every
            # REGISTER in the product carried one ROOTED_AT to its type and
            # nothing else: extracted, named, and unreachable by any walk from
            # the kernel.
            #
            # The binding is the compiler's, not a guess: `args` is the argument
            # list clang resolved at this call site, and `_lookup_storage` walks
            # the enclosing scope chain the way C++ name lookup does. Restricted
            # to arguments that *are* an identifier, because a bare name in a
            # scope denotes one declaration, while `buf.Get()[i]` would need the
            # expression's type to say what it denotes -- and a wrong operand
            # edge is worse than a missing one.
            for arg in args:
                symbol = str(arg).strip()
                if not symbol.isidentifier():
                    continue
                sid = _lookup_storage(symbol, identity_scopes, nfile)
                if not sid or sid == ent.id or sid == bid_recv:
                    continue
                _link(
                    codemap,
                    RelationKind.REFERENCES,
                    ent.id,
                    sid,
                    attrs={"symbol": symbol, "via": "operand"},
                )
        if ent is not None and bid_recv:
            _link(
                codemap,
                RelationKind.REFERENCES,
                ent.id,
                bid_recv,
                attrs={"symbol": receiver},
            )
            if not is_project:
                _link(
                    codemap,
                    RelationKind.CALLS,
                    bid_recv,
                    ent.id,
                    attrs={
                        "via": "method_receiver",
                        "file": nfile,
                        "line": line,
                        "column": column,
                    },
                    status="partial",
                )

    _collect_lexical_init_pipe_args()
    for ent in list(codemap.by_kind(EntityKind.PIPE)):
        if ent.attrs.get("pointer"):
            continue
        name = str(ent.name or "")
        ef = str(ent.file or "").replace("\\", "/")
        leaf = ef.rsplit("/", 1)[-1]
        for nfile, decls in tpipe_decl_by_file.items():
            if name not in decls:
                continue
            nf = str(nfile).replace("\\", "/")
            if ef != nf and leaf != nf.rsplit("/", 1)[-1]:
                continue
            line = int(decls[name])
            if line > 0 and int(ent.line_start or 0) != line:
                ent.line_start = line
            break
    inherit_from: dict[str, set[str]] = defaultdict(set)
    for rel in list(codemap.relations.values()):
        if rel.kind_name() != RelationKind.WRAPS.value:
            continue
        if str(rel.attrs.get("via") or "") != "inherits":
            continue
        src_e = codemap.entities.get(rel.src)
        dst_e = codemap.entities.get(rel.dst)
        if src_e is None or dst_e is None:
            continue
        child = _short_type_name(src_e.name)
        parent = _short_type_name(dst_e.name)
        if child and parent:
            inherit_from[child].add(parent)
    for child, parent in lexical_inherit:
        inherit_from[child].add(parent)

    def _owner_ancestors(name: str) -> set[str]:
        start = _short_type_name(name)
        found = {start} if start else set()
        stack = [start] if start else []
        while stack:
            cur = stack.pop()
            for base in inherit_from.get(cur, ()):
                if base and base not in found:
                    found.add(base)
                    stack.append(base)
        return found

    def _owners_match(ptr_scope: str, site_owner: str) -> bool:
        ps = _short_type_name(ptr_scope)
        so = _short_type_name(site_owner)
        if not ps or not so:
            return False
        if ps == so:
            return True
        return ps in _owner_ancestors(so)

    for pipe_id, obj_id, nfile, line, receiver in initbuffer_links:
        ptr = codemap.entities.get(pipe_id)
        if ptr is None or not ptr.attrs.get("pointer"):
            continue
        owner = str(ptr.attrs.get("scope") or "")
        if not owner:
            continue
        seen_inst: set[str] = set()
        for callee_owner, instance_id, afile, aline in pipe_arg_sites:
            if (
                not _owners_match(owner, callee_owner)
                or instance_id == pipe_id
                or instance_id in seen_inst
            ):
                continue
            inst = codemap.entities.get(instance_id)
            if inst is None or inst.attrs.get("pointer"):
                continue
            seen_inst.add(instance_id)
            _link(
                codemap,
                RelationKind.ALIASES,
                pipe_id,
                instance_id,
                attrs={"via": "pipe_ptr", "file": afile, "line": aline},
            )
            _link(
                codemap,
                RelationKind.BINDS,
                instance_id,
                obj_id,
                attrs={
                    "via": "InitBuffer",
                    "file": nfile,
                    "line": line,
                    "receiver": receiver,
                },
            )

    pair_stats = _record_flag_pair_appearance(codemap, gaps)

    by_scope: dict[str, list[Any]] = defaultdict(list)
    by_file_pipes: dict[str, list[Any]] = defaultdict(list)
    for e in codemap.by_kind(EntityKind.PIPE):
        if e.attrs.get("catalog") == "ascendc":
            continue
        if e.attrs.get("pointer"):
            continue
        by_scope[str(e.attrs.get("scope") or "")].append(e)
        by_file_pipes[_norm_file(str(e.file or ""), root)].append(e)
    destroy_by_file: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for op in codemap.by_kind(EntityKind.OPERATION):
        callee = str(op.attrs.get("callee") or op.name or "").split("::")[-1]
        if callee != "Destroy":
            continue
        recv = receiver_leaf(str(op.attrs.get("receiver") or ""))
        if not recv:
            continue
        destroy_by_file[_norm_file(str(op.file or ""), root)].append(
            (int(op.line_start or 0), recv)
        )
    for nfile, pipes in by_file_pipes.items():
        text = kernel_file_text.get(nfile) or ""
        if not text:
            leaf = str(nfile or "").replace("\\", "/").rsplit("/", 1)[-1]
            for key, body in kernel_file_text.items():
                if str(key).replace("\\", "/").rsplit("/", 1)[-1] == leaf:
                    text = body
                    break
        ranges = continued_line_ranges(text) if text else []
        constructs = [(e.name, int(e.line_start or 0)) for e in pipes]
        file_destroys = destroy_by_file.get(nfile) or []
        if not file_destroys:
            leaf = str(nfile or "").replace("\\", "/").rsplit("/", 1)[-1]
            for key, rows in destroy_by_file.items():
                if str(key).replace("\\", "/").rsplit("/", 1)[-1] == leaf:
                    file_destroys = rows
                    break
        edges = lifetime_edges(constructs, file_destroys, ranges) if ranges else []
        if not edges:
            continue
        line_of = {e.name: int(e.line_start or 0) for e in pipes}
        ordinals = topo_pipe_ordinals([e.name for e in pipes], edges, line_of=line_of)
        by_name = {e.name: e for e in pipes}
        for name, ordinal in ordinals.items():
            pipe = by_name.get(name)
            if pipe is None:
                continue
            pipe.attrs["pipe_ordinal"] = ordinal
        for src_name, dst_name in edges:
            src = by_name.get(src_name)
            dst = by_name.get(dst_name)
            if src is None or dst is None:
                continue
            _link(
                codemap,
                RelationKind.PRECEDES,
                src.id,
                dst.id,
                attrs={"via": "pipe_destroy"},
            )
    for _scope, pipes in by_scope.items():
        pipes.sort(key=lambda item: (int(item.line_start or 0), item.name))
        for idx, pipe in enumerate(pipes, start=1):
            if pipe.attrs.get("pipe_ordinal"):
                continue
            pipe.attrs["pipe_ordinal"] = idx

    lock_roots = {"Lock", "Unlock", "AllocMutexID", "ReleaseMutexID"}
    flag_roots = set(FLAG_PAIR_MATE)
    marked_lock: set[str] = set()
    marked_flag: set[str] = set()
    call_rels = [
        rel
        for rel in list(codemap.relations.values())
        if rel.kind_name() == RelationKind.CALLS.value
    ]
    for rel in call_rels:
        dst = codemap.entities.get(rel.dst)
        src = codemap.entities.get(rel.src)
        if dst is None or src is None:
            continue
        leaf = str(dst.attrs.get("callee") or dst.name or "").split("::")[-1]
        root_leaf = str(dst.attrs.get("root") or "").split("::")[-1]
        is_lock = leaf in lock_roots or root_leaf in lock_roots
        is_flag = leaf in flag_roots or root_leaf in flag_roots
        if not (is_lock or is_flag):
            continue
        qn = str(src.attrs.get("qualified_name") or "")
        owner = qn.rsplit("::", 1)[0].split("::")[-1] if "::" in qn else ""
        if not owner or owner not in type_ents:
            continue
        te = codemap.entities.get(type_ents[owner])
        if te is None:
            continue
        if is_lock and te.id not in marked_lock:
            marked_lock.add(te.id)
            te.attrs["wraps_lock"] = True
            lock_id = _ensure_ascendc_root(codemap, "Lock", root_kind="SYNC")
            _link(codemap, RelationKind.WRAPS, te.id, lock_id, attrs={"via": "calls_cann_lock"})
        if is_flag and te.id not in marked_flag:
            marked_flag.add(te.id)
            te.attrs["wraps_flag"] = True
            flag_id = _ensure_ascendc_root(
                codemap, leaf if leaf in flag_roots else root_leaf, root_kind="SYNC"
            )
            _link(codemap, RelationKind.WRAPS, te.id, flag_id, attrs={"via": "calls_cann_flag"})
    _propagate_wrap_flags(codemap)

    # Bounded lexical statement order.  PRECEDES is adjacency in one source
    # function/file, not flag pairing and not a happens-before relation.
    def _sync_op(item: Any) -> bool:
        name = str(item.attrs.get("callee") or item.name or "")
        return name in _SYNC_PRECEDES_CALLEES

    ordered_ops: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for operation in codemap.by_kind(EntityKind.OPERATION):
        if not _sync_op(operation):
            continue
        ordered_ops[
            (str(operation.attrs.get("function") or ""), str(operation.file or ""))
        ].append(operation)
    precedes_count = 0
    for (_function, _file), operations in sorted(ordered_ops.items()):
        operations.sort(
            key=lambda item: (
                int(item.line_start or 0),
                int(item.attrs.get("column") or 0),
                item.id,
            )
        )
        for previous, current in zip(operations, operations[1:]):
            if precedes_count >= 500:
                break
            _link(
                codemap,
                RelationKind.PRECEDES,
                previous.id,
                current.id,
                attrs={"via": "same_function_source_order"},
            )
            precedes_count += 1
        if precedes_count >= 500:
            break

    # --- 6. Single fixed-point -------------------------------------------
    _propagate_reachability(codemap)

    # Project declaration identity is not an AscendC root. Undo any
    # CALLS/WRAPS climb that would stamp catalog REACHED on those sites.
    for e in codemap.by_kind(EntityKind.OPERATION):
        proof = str(e.attrs.get("root_proof") or "")
        if proof == "compiler_builtin":
            e.attrs["root_status"] = "BUILTIN"
            e.attrs["root_kind"] = "BUILTIN"
            e.attrs["root"] = ""
            e.status = "extracted"
            e.confidence = max(float(e.confidence or 0), 0.9)
            continue
        if not proof.startswith("project_"):
            continue
        e.attrs["root_status"] = "PROJECT"
        e.attrs["root"] = ""
        e.attrs["root_kind"] = "PROJECT"
        e.status = "extracted"
        e.confidence = max(float(e.confidence or 0), 0.9)

    # Propagate REACHED onto METHOD entities that CALL a REACHED node.
    # (Fixed-point already walks CALLS; refresh METHOD attrs from ROOTED_AT.)
    for e in codemap.by_kind(EntityKind.METHOD):
        if e.attrs.get("root_status") == "REACHED":
            continue
        for rel, other in codemap.neighbors(e.id, kind=RelationKind.CALLS, direction="out"):
            if other.attrs.get("root_status") == "REACHED" or other.attrs.get("catalog") == "ascendc":
                e.attrs["root_status"] = "REACHED"
                e.attrs["root"] = other.attrs.get("root") or other.name
                e.attrs["root_kind"] = other.attrs.get("root_kind") or e.attrs.get("root_kind") or ""
                trace = list(e.attrs.get("trace") or [e.name])
                if other.name not in trace:
                    trace.append(other.name)
                e.attrs["trace"] = trace
                e.status = "extracted"
                break

    _link_owner_declares_methods(codemap, type_ents)
    _normalize_settled_entities(codemap)

    # --- 7. Gaps for still-unresolved source types that participate in WRAPS
    unresolved_types = 0
    for e in codemap.entities.values():
        if e.kind_name() != EntityKind.TYPE.value:
            continue
        if e.attrs.get("catalog") == "ascendc":
            continue
        if e.attrs.get("root_status") != "UNRESOLVED":
            continue
        # Only gap types that appear in a WRAPS edge (participated in composition).
        participates = bool(
            codemap.neighbors(e.id, kind=RelationKind.WRAPS, direction="both")
        )
        if not participates:
            continue
        unresolved_types += 1
        gaps.append(
            {
                "code": REASON_NO_ASCENDC_ROOT,
                "entity_id": e.id,
                "name": e.name,
                "file": e.file,
                "line": e.line_start,
            }
        )

    elapsed = time.perf_counter() - t0
    gap_counts = Counter(str(g.get("code") or "") for g in gaps)
    reached_bufs = 0
    reached_regs = 0
    reached_ops = 0
    tque_ops = 0
    tpipe_ops = 0
    project_resolved_ops = 0
    for e in codemap.by_kind(EntityKind.BUFFER):
        if e.attrs.get("root_status") == "REACHED":
            reached_bufs += 1
    for e in codemap.by_kind(EntityKind.REGISTER):
        if e.attrs.get("root_status") == "REACHED":
            reached_regs += 1
    for e in codemap.by_kind(EntityKind.OPERATION):
        if e.attrs.get("root_status") == "REACHED":
            reached_ops += 1
        callee = str(e.attrs.get("callee") or e.name or "")
        if is_tque_callee(callee):
            tque_ops += 1
        if is_tpipe_callee(callee):
            tpipe_ops += 1
        if str(e.attrs.get("root_proof") or "").startswith("project_"):
            project_resolved_ops += 1
    signals = awaits = wraps = rooted_at = alias_rels = binds = 0
    for r in codemap.relations.values():
        kind = r.kind_name()
        if kind == RelationKind.SIGNALS.value:
            signals += 1
        elif kind == RelationKind.AWAITS.value:
            awaits += 1
        elif kind == RelationKind.WRAPS.value:
            wraps += 1
        elif kind == RelationKind.ROOTED_AT.value:
            if str(r.attrs.get("provenance") or "") == "kernel_root_trace":
                rooted_at += 1
        elif kind == RelationKind.ALIASES.value:
            alias_rels += 1
        elif kind == RelationKind.BINDS.value:
            binds += 1
    quality = {
        "operations": op_count,
        "buffers": buf_count,
        "registers": reg_count,
        "pipes": len(codemap.by_kind(EntityKind.PIPE)),
        "events": len(codemap.by_kind(EntityKind.EVENT)),
        "queues": len(codemap.by_kind(EntityKind.QUEUE)),
        "precedes": precedes_count,
        "signals": signals,
        "awaits": awaits,
        "tque_ops": tque_ops,
        "tpipe_ops": tpipe_ops,
        "flag_pairs": int(pair_stats.get("flag_pairs") or 0),
        "unpaired_flag_sync": int(pair_stats.get("unpaired_flag_sync") or 0),
        "reached_operations": reached_ops,
        "reached_buffers": reached_bufs,
        "reached_registers": reached_regs,
        "wraps": wraps,
        "rooted_at": rooted_at,
        "aliases": alias_rels,
        "project_resolved_ops": project_resolved_ops,
        "identity_filled": identity_filled,
        "binds": binds,
    }
    meta = {
        "architecture": arch,
        "elapsed_s": round(elapsed, 3),
        "budget_s": _budget_s(),
        "provenance": provenance,
        "kernel_backend": kernel_backend,
        "walk_cache_stats": walk_stats,
        "clang_covered_files": len(covered),
        "lexical_uncovered_files": len(uncovered),
        "walk_cache_confirms": walk_confirm,
        "selected_files": len(files),
        "class_members": len(members),
        "type_aliases": len(aliases),
        "clang_type_decls": len(clang_types),
        "clang_base_decls": len(clang_bases),
        "operations": op_count,
        "buffers": buf_count,
        "registers": reg_count,
        "pipes": len(codemap.by_kind(EntityKind.PIPE)),
        "events": len(codemap.by_kind(EntityKind.EVENT)),
        "queues": len(codemap.by_kind(EntityKind.QUEUE)),
        "precedes": precedes_count,
        "reached_operations": reached_ops,
        "reached_buffers": reached_bufs,
        "reached_registers": reached_regs,
        "unresolved_types": unresolved_types,
        "gap_count": len(gaps),
        "gap_counts": dict(gap_counts),
        "gaps": gaps[:200],
        "identity_filled": identity_filled,
        "corpus_n": len(extra_arch),
        "gated_corpus_n": len(priority),
        "arch_kernel_primitives_added": extra_added,
        "gated_fill_complete": gated_fill_complete,
        "budget_expired": budget.expired(),
        "source_api_gated": source_api_gated,
        "quality": quality,
    }
    try:
        from ascendc_codemap_mcp.engine.perf import record_pass

        record_pass(
            "kernel_root_trace",
            source_files=len(files),
            gated_fill_complete=gated_fill_complete,
            budget_expired=budget.expired(),
            cache_walk_loads=int((walk_stats or {}).get("pickle_load") or 0),
        )
    except Exception:  # noqa: BLE001
        pass
    codemap.meta["kernel_root_trace"] = meta
    codemap.meta["kernel_backend"] = kernel_backend
    # Thin compat for older query helpers (not an execution model).
    codemap.meta["kernel_execution"] = {
        "operations": op_count,
        "buffers": buf_count,
        "registers": reg_count,
        "elapsed_s": meta["elapsed_s"],
        "root_trace": True,
        "kernel_backend": kernel_backend,
    }
    _TRACE_ARCHITECTURE = ""
    return codemap
