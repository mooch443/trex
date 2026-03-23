#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMMONS_ROOT = REPO_ROOT / "Application" / "src" / "commons" / "common"
COMMONS_SOURCES = COMMONS_ROOT / "CommonsSources.cmake"
INVENTORY_PATH = REPO_ROOT / "Application" / "Tests" / "commons_module_inventory.json"
EXPORTS_DIR = COMMONS_ROOT / "modules" / "generated"
SMOKE_DIR = REPO_ROOT / "Application" / "Tests" / "generated"

GROUP_TO_MODULE = {
    "COMMONS_MISC_HEADERS": "commons.misc",
    "COMMONS_FILE_HEADERS": "commons.file",
    "COMMONS_PROCESSING_HEADERS": "commons.processing",
    "COMMONS_VIDEO_HEADERS": "commons.video",
    "COMMONS_GUI_HEADERS": "commons.gui",
    "COMMONS_GUI_DYN_HEADERS": "commons.gui.dyn",
    "COMMONS_HTTP_HEADERS": "commons.http",
}

HEADER_TO_SMOKE_FILE = {
    "commons.types": "commons_module_smoke_types.inc",
    "commons.misc": "commons_module_smoke_misc.inc",
    "commons.file": "commons_module_smoke_file.inc",
    "commons.processing": "commons_module_smoke_processing.inc",
    "commons.video": "commons_module_smoke_video.inc",
    "commons.gui": "commons_module_smoke_gui.inc",
    "commons.gui.dyn": "commons_module_smoke_gui_dyn.inc",
    "commons.http": "commons_module_smoke_http.inc",
}

MODULE_TO_EXPORT_FILE = {
    "commons.types": "commons.exports.types.inc",
    "commons.misc": "commons.exports.misc.inc",
    "commons.file": "commons.exports.file.inc",
    "commons.processing": "commons.exports.processing.inc",
    "commons.video": "commons.exports.video.inc",
    "commons.gui": "commons.exports.gui.inc",
    "commons.gui.dyn": "commons.exports.gui_dyn.inc",
    "commons.http": "commons.exports.http.inc",
}

MODULE_TO_INCLUDE_FILE = {
    "commons": "commons.includes.all.inc",
    "commons.types": "commons.includes.types.inc",
    "commons.misc": "commons.includes.misc.inc",
    "commons.file": "commons.includes.file.inc",
    "commons.processing": "commons.includes.processing.inc",
    "commons.video": "commons.includes.video.inc",
    "commons.gui": "commons.includes.gui.inc",
    "commons.gui.dyn": "commons.includes.gui_dyn.inc",
    "commons.http": "commons.includes.http.inc",
}

MODULE_HEADER_EXCLUSIONS = {
    "misc/ThreadedAnalysis_impl.h": "implementation_header_included_by_public_header",
    "misc/RBSettings.h": "tracker_dependent_public_header",
    "modules/commons.exports.hpp": "aggregator_header",
}

MANUAL_MACROS = [
    {
        "header": "misc/EnumClass.h",
        "name": "ENUM_CLASS",
        "kind": "macro",
        "module": "commons.misc",
        "smoke": "macro_section",
    },
    {
        "header": "misc/EnumClass.h",
        "name": "ENUM_CLASS_HAS_DOCS",
        "kind": "macro",
        "module": "commons.misc",
        "smoke": "macro_section",
    },
    {
        "header": "misc/EnumClass.h",
        "name": "ENUM_CLASS_DOCS",
        "kind": "macro",
        "module": "commons.misc",
        "smoke": "macro_section",
    },
    {
        "header": "misc/create_struct.h",
        "name": "CREATE_STRUCT",
        "kind": "macro",
        "module": "commons.misc",
        "smoke": "macro_section",
    },
    {
        "header": "gui/ControlsAttributes.h",
        "name": "ATTRIBUTE_ALIAS",
        "kind": "macro",
        "module": "commons.gui",
        "smoke": "macro_section",
    },
    {
        "header": "gui/ControlsAttributes.h",
        "name": "NUMBER_ALIAS",
        "kind": "macro",
        "module": "commons.gui",
        "smoke": "macro_section",
    },
]

MACRO_EXCLUSIONS = {
    ("misc/GlobalSettings.h", "SETTING"): "named_api_replacement",
    ("misc/GlobalSettings.h", "BOOL_SETTING"): "named_api_replacement",
    ("misc/GlobalSettings.h", "OPTIONAL_SETTING"): "named_api_replacement",
    ("misc/GlobalSettings.h", "READ_SETTING"): "named_api_replacement",
    ("misc/GlobalSettings.h", "READ_SETTING_WITH_DEFAULT"): "named_api_replacement",
}

DECLARATION_KEYWORDS = {
    "alignas",
    "constexpr",
    "consteval",
    "constinit",
    "const",
    "volatile",
    "inline",
    "extern",
    "static",
    "friend",
    "virtual",
    "explicit",
    "mutable",
    "register",
    "thread_local",
    "typename",
    "signed",
    "unsigned",
    "short",
    "long",
    "auto",
    "decltype",
    "noexcept",
    "requires",
    "operator",
}

STATEMENT_SKIP_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "catch",
    "static_assert",
}

NO_SMOKE_KINDS = {"concept"}
TYPE_LIKE_KINDS = {"class", "struct", "class_template"}
ALLOWED_NAMESPACE_ROOTS = {"cmn", "pv", "EnumMeta"}
IGNORED_SYMBOL_NAMES = {
    "ALIAS_NAME",
    "ALIAS_NAMETag",
    "ATTRIBUTE_ALIAS",
    "CHANGE_SETTER",
    "_CHANGE_SETTER",
    "ENUM_CLASS",
    "NAME",
    "NUMBER_ALIAS",
    "T",
    "class_name",
    "defined",
    "DEBUG_CV",
    "false",
    "toStr",
    "to_html",
    "true",
}

MANUAL_SYMBOL_EXCLUSIONS = {
    ("file/DataFormat.h", "cmn::mappedRegion"): "platform_build_gated",
    ("file/DataFormat.h", "cmn::mappedRegionS"): "platform_build_gated",
    ("misc/types.h", "cmn::CV_MAX_THICKNESS"): "internal_linkage_constant",
    ("misc/CircularGraph.h", "cmn::periodic::make_clockwise"): "member_function_parser_artifact",
    ("misc/SpriteMap.h", "cmn::Map"): "wrong_namespace_parser_artifact",
    ("misc/SpriteProperty.h", "cmn::Property"): "deduction_guide_parser_artifact",
    ("misc/metastring.h", "cmn::utils::Rt"): "template_parameter_parser_artifact",
    ("gui/DrawSFBase.h", "cmn::gui::Path"): "sfml_build_gated",
    ("gui/DrawSFBase.h", "cmn::gui::SpriteWithTexture"): "sfml_build_gated",
    ("gui/DrawSFBase.h", "cmn::gui::SpriteWithRenderTexture"): "sfml_build_gated",
    ("gui/DrawSFBase.h", "cmn::gui::SFBase"): "sfml_build_gated",
    ("gui/HttpClient.h", "cmn::gui::HttpClient"): "http_build_gated",
    ("gui/LabeledField.h", "cmn::gui::has_set"): "wrong_namespace_parser_artifact",
    ("gui/types/Entangled.h", "cmn::gui::detail::set"): "requires_expression_parser_artifact",
    ("gui/types/Drawable.h", "cmn::gui::Handler"): "internal_forward_decl_only",
    ("gui/types/ListItemTypes.h", "cmn::gui::TREX_TYPE_EXPORT"): "macro_parser_artifact",
    ("misc/colors.h", "cmn::gui::cmn::cmap::Viridis"): "wrong_namespace_parser_artifact",
    ("misc/colors.h", "cmn::gui::cmn::cmap::BlackToPink"): "wrong_namespace_parser_artifact",
    ("misc/colors.h", "cmn::gui::cmn::cmap::BlueToRed"): "wrong_namespace_parser_artifact",
    ("misc/colors.h", "cmn::gui::cmn::cmap::PinkFoam"): "wrong_namespace_parser_artifact",
    ("misc/colors.h", "cmn::gui::cmn::cmap::BlueToYellow"): "wrong_namespace_parser_artifact",
    ("misc/colors.h", "cmn::gui::cmn::cmap::BlackToWhite"): "wrong_namespace_parser_artifact",
    ("misc/colors.h", "cmn::gui::cmn::cmap::BlackToCyan"): "wrong_namespace_parser_artifact",
    ("misc/colors.h", "cmn::gui::cmn::cmap::BlueToCyan"): "wrong_namespace_parser_artifact",
    ("misc/colors.h", "cmn::gui::cmn::cmap::BlackToGreen"): "wrong_namespace_parser_artifact",
    ("misc/colors.h", "cmn::gui::cmn::cmap::ColorMap"): "wrong_namespace_parser_artifact",
}


@dataclass(frozen=True)
class Symbol:
    header: str
    module: str
    namespace: str
    name: str
    kind: str
    smoke: str
    reason: str | None = None

    @property
    def qualified(self) -> str:
        if self.namespace:
            return f"{self.namespace}::{self.name}"
        return self.name


MANUAL_SYMBOLS = {
    "misc/SpriteMap.h": [
        Symbol("misc/SpriteMap.h", "commons.misc", "cmn::sprite", "Map", "class", "using"),
    ],
    "misc/SpriteProperty.h": [
        Symbol("misc/SpriteProperty.h", "commons.misc", "cmn::sprite", "Property", "class", "using"),
    ],
    "gui/LabeledField.h": [
        Symbol("gui/LabeledField.h", "commons.gui", "cmn::gui::dyn", "has_set", "class_template", "using"),
    ],
    "misc/colors.h": [
        Symbol("misc/colors.h", "commons.misc", "cmn::cmap", "Viridis", "class", "using"),
        Symbol("misc/colors.h", "commons.misc", "cmn::cmap", "BlackToPink", "class", "using"),
        Symbol("misc/colors.h", "commons.misc", "cmn::cmap", "BlueToRed", "class", "using"),
        Symbol("misc/colors.h", "commons.misc", "cmn::cmap", "PinkFoam", "class", "using"),
        Symbol("misc/colors.h", "commons.misc", "cmn::cmap", "BlueToYellow", "class", "using"),
        Symbol("misc/colors.h", "commons.misc", "cmn::cmap", "BlackToWhite", "class", "using"),
        Symbol("misc/colors.h", "commons.misc", "cmn::cmap", "BlackToCyan", "class", "using"),
        Symbol("misc/colors.h", "commons.misc", "cmn::cmap", "BlueToCyan", "class", "using"),
        Symbol("misc/colors.h", "commons.misc", "cmn::cmap", "BlackToGreen", "class", "using"),
        Symbol("misc/colors.h", "commons.misc", "cmn::cmap", "ColorMap", "class", "using"),
    ],
}


def parse_set_blocks(text: str) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    current_name: str | None = None
    current_values: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if current_name is None:
            match = re.match(r"set\((\w+)\s*$", line)
            if match:
                current_name = match.group(1)
                current_values = []
            continue
        if line == ")":
            blocks[current_name] = current_values
            current_name = None
            current_values = []
            continue
        current_values.append(line)
    return blocks


def module_for_header(header: str, group_membership: dict[str, str]) -> str:
    return group_membership.get(header, "commons.types")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_comments_and_literals(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            out.append(" ")
            out.append(" ")
            i += 2
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if ch == "/" and nxt == "*":
            out.append(" ")
            out.append(" ")
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            if i + 1 < n:
                out.append(" ")
                out.append(" ")
                i += 2
            continue
        if ch in {'"', "'"}:
            quote = ch
            out.append(" ")
            i += 1
            while i < n:
                cur = text[i]
                out.append("\n" if cur == "\n" else " ")
                if cur == "\\" and i + 1 < n:
                    out.append("\n" if text[i + 1] == "\n" else " ")
                    i += 2
                    continue
                if cur == quote:
                    i += 1
                    break
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def strip_preprocessor_blocks(text: str) -> str:
    lines: list[str] = []
    in_directive = False
    for raw_line in text.splitlines():
        stripped = raw_line.lstrip()
        if in_directive:
            lines.append("")
            in_directive = raw_line.rstrip().endswith("\\")
            continue
        if stripped.startswith("#"):
            lines.append("")
            in_directive = raw_line.rstrip().endswith("\\")
            continue
        lines.append(raw_line)
    return "\n".join(lines)


TOKEN_RE = re.compile(r"::|==|!=|<=|>=|<<=|>>=|<<|>>|->\*|->|&&|\|\||[A-Za-z_]\w*|[{}()\[\];,<>*=:&~+\-/%]|\.{3}")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def join_operator(tokens: list[str]) -> str:
    if not tokens:
        return "operator"
    return "operator" + "".join(tokens)


def unique_symbols(symbols: list[Symbol]) -> list[Symbol]:
    seen: set[tuple[str, str, str, str]] = set()
    result: list[Symbol] = []
    for symbol in symbols:
        key = (symbol.qualified, symbol.kind, symbol.header, symbol.module)
        if key in seen:
            continue
        seen.add(key)
        result.append(symbol)
    return result


def is_allowed_namespace(namespace: str) -> bool:
    if not namespace:
        return False
    root = namespace.split("::", 1)[0]
    return root in ALLOWED_NAMESPACE_ROOTS


def filter_symbols(symbols: list[Symbol]) -> list[Symbol]:
    filtered = []
    for symbol in symbols:
        if symbol.name in IGNORED_SYMBOL_NAMES:
            continue
        if symbol.name.startswith("operator_") or symbol.name.endswith("operator"):
            continue
        if not is_allowed_namespace(symbol.namespace):
            continue
        filtered.append(symbol)
    return unique_symbols(filtered)


def type_like_names_for_module(module_name: str, inventory: dict) -> set[str]:
    names: set[str] = set()
    for header in inventory["headers"]:
        if header["module"] != module_name:
            continue
        for symbol in header["symbols"]:
            if symbol["kind"] in TYPE_LIKE_KINDS:
                names.add(symbol["qualified"])
    return names


def sanitize_for_identifier(text: str) -> str:
    return re.sub(r"\W+", "_", text).strip("_")


def parse_macro_invocations(header: str, module: str, text: str) -> tuple[list[Symbol], list[dict[str, str]]]:
    symbols: list[Symbol] = []
    exclusions: list[dict[str, str]] = []
    scope_stack: list[tuple[list[str], int]] = []
    brace_depth = 0
    lines = text.splitlines()
    in_macro_definition = False
    for line_no, raw_line in enumerate(lines, start=1):
        line = re.sub(r"//.*", "", raw_line)
        stripped = raw_line.lstrip()
        if in_macro_definition:
            brace_depth += line.count("{")
            brace_depth -= line.count("}")
            in_macro_definition = raw_line.rstrip().endswith("\\")
            continue
        if stripped.startswith("#define"):
            in_macro_definition = raw_line.rstrip().endswith("\\")
            continue
        for match in re.finditer(r"\bnamespace\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*\{", line):
            namespace_name = match.group(1).split("::")
            scope_stack.append((namespace_name, brace_depth + 1))
        for match in re.finditer(r"\b(?:class|struct)\s+([A-Za-z_]\w*)\b(?:\s+final\b)?(?:\s*:\s*[^{]+)?\s*\{", line):
            scope_stack.append(([match.group(1)], brace_depth + 1))
        current_namespace = "::".join(segment for names, _depth in scope_stack for segment in names)

        for pattern, kind in (
            (r"\bCREATE_STRUCT\s*\(\s*([A-Za-z_]\w*)", "class"),
            (r"\bATTRIBUTE_ALIAS\s*\(\s*([A-Za-z_]\w*)", "alias"),
            (r"\bNUMBER_ALIAS\s*\(\s*([A-Za-z_]\w*)", "alias"),
        ):
            for match in re.finditer(pattern, line):
                name = match.group(1)
                symbols.append(Symbol(header, module, current_namespace, name, kind, "using"))
                if "ALIAS" in pattern:
                    symbols.append(Symbol(header, module, current_namespace, f"{name}Tag", "struct", "using"))

        for match in re.finditer(r"\bENUM_CLASS\s*\(\s*([A-Za-z_]\w*)", line):
            name = match.group(1)
            enum_namespace = f"{current_namespace}::{name}" if current_namespace else name
            symbols.append(Symbol(header, module, enum_namespace, "Class", "enum_class", "using"))
            exclusions.append(
                {
                    "header": header,
                    "name": f"{enum_namespace}::helpers",
                    "kind": "group",
                    "reason": "internal_linkage_enum_helpers",
                    "line": str(line_no),
                }
            )

        for match in re.finditer(r"#define\s+([A-Za-z_]\w*)", raw_line):
            macro_name = match.group(1)
            reason = MACRO_EXCLUSIONS.get((header, macro_name))
            if reason:
                exclusions.append(
                    {
                        "header": header,
                        "name": macro_name,
                        "kind": "macro",
                        "reason": reason,
                        "line": str(line_no),
                    }
                )

        brace_depth += line.count("{")
        brace_depth -= line.count("}")
        while scope_stack and brace_depth < scope_stack[-1][1]:
            scope_stack.pop()
    return filter_symbols(symbols), exclusions


def parse_regular_symbols(header: str, module: str, text: str) -> tuple[list[Symbol], list[dict[str, str]]]:
    tokens = tokenize(strip_comments_and_literals(strip_preprocessor_blocks(text)))
    i = 0
    namespace_stack: list[str] = []
    scope_stack: list[tuple[str, int, int]] = []
    brace_depth = 0
    pending_template = False
    symbols: list[Symbol] = []
    exclusions: list[dict[str, str]] = []

    def namespace_path() -> str:
        return "::".join(namespace_stack)

    def at_namespace_scope() -> bool:
        return all(kind == "namespace" for kind, _depth, _segments in scope_stack)

    def push_scope(kind: str, namespace_segments: int = 0) -> None:
        nonlocal brace_depth
        brace_depth += 1
        scope_stack.append((kind, brace_depth, namespace_segments))

    def pop_scopes() -> None:
        nonlocal brace_depth
        while scope_stack and brace_depth < scope_stack[-1][1]:
            kind, _depth, namespace_segments = scope_stack.pop()
            if kind == "namespace" and namespace_segments:
                del namespace_stack[-namespace_segments:]

    def scan_to_statement_end(start: int) -> tuple[list[str], int, str]:
        stmt: list[str] = []
        paren_depth = 0
        angle_depth = 0
        idx = start
        while idx < len(tokens):
            tok = tokens[idx]
            if tok == "(":
                paren_depth += 1
            elif tok == ")":
                paren_depth = max(paren_depth - 1, 0)
            elif tok == "<":
                angle_depth += 1
            elif tok == ">":
                angle_depth = max(angle_depth - 1, 0)
            if tok in {";", "{"} and paren_depth == 0 and angle_depth == 0:
                return stmt, idx, tok
            stmt.append(tok)
            idx += 1
        return stmt, idx, ""

    def extract_function_name(stmt: list[str]) -> tuple[str | None, int | None]:
        if not stmt:
            return None, None
        if stmt[0] in STATEMENT_SKIP_KEYWORDS:
            return None, None
        try:
            open_paren = stmt.index("(")
        except ValueError:
            return None, None
        prefix = stmt[:open_paren]
        if not prefix:
            return None, None
        if "operator" in prefix:
            op_index = prefix.index("operator")
            return join_operator(prefix[op_index + 1 :]), op_index
        for tok in reversed(prefix):
            if re.fullmatch(r"[A-Za-z_]\w*", tok) and tok not in DECLARATION_KEYWORDS:
                return tok, prefix.index(tok)
        return None, None

    def extract_variable_name(stmt: list[str]) -> str | None:
        if not stmt or stmt[0] in STATEMENT_SKIP_KEYWORDS:
            return None
        for idx in range(len(stmt) - 1, -1, -1):
            tok = stmt[idx]
            if re.fullmatch(r"[A-Za-z_]\w*", tok) and tok not in DECLARATION_KEYWORDS:
                prev = stmt[idx - 1] if idx > 0 else ""
                if prev != "::":
                    return tok
        return None

    while i < len(tokens):
        tok = tokens[i]
        if tok == "{":
            push_scope("block")
            i += 1
            continue
        if tok == "}":
            brace_depth = max(brace_depth - 1, 0)
            pop_scopes()
            i += 1
            continue
        if tok == "template" and at_namespace_scope():
            depth = 0
            while i < len(tokens):
                if tokens[i] == "<":
                    depth += 1
                elif tokens[i] == ">":
                    depth -= 1
                    if depth <= 0:
                        i += 1
                        break
                i += 1
            pending_template = True
            continue
        if tok == "namespace":
            j = i + 1
            parts: list[str] = []
            namespace_alias = False
            while j < len(tokens):
                nxt = tokens[j]
                if nxt == "{":
                    break
                if nxt == "=":
                    namespace_alias = True
                if re.fullmatch(r"[A-Za-z_]\w*", nxt):
                    parts.append(nxt)
                j += 1
            if not namespace_alias and j < len(tokens) and tokens[j] == "{":
                joined = "::".join(parts)
                if joined:
                    namespace_parts = joined.split("::")
                    namespace_stack.extend(namespace_parts)
                    push_scope("namespace", len(namespace_parts))
                    i = j + 1
                    continue
            i = j + 1
            continue
        if tok in {"class", "struct"} and at_namespace_scope():
            name = tokens[i + 1] if i + 1 < len(tokens) and re.fullmatch(r"[A-Za-z_]\w*", tokens[i + 1]) else None
            stmt, end_idx, terminator = scan_to_statement_end(i + 2)
            if name:
                symbols.append(
                    Symbol(
                        header=header,
                        module=module,
                        namespace=namespace_path(),
                        name=name,
                        kind="class_template" if pending_template else tok,
                        smoke="using",
                    )
                )
            pending_template = False
            if terminator == "{":
                push_scope("class")
            i = end_idx + 1
            continue
        if tok == "enum" and at_namespace_scope():
            j = i + 1
            if j < len(tokens) and tokens[j] in {"class", "struct"}:
                j += 1
            name = tokens[j] if j < len(tokens) and re.fullmatch(r"[A-Za-z_]\w*", tokens[j]) else None
            stmt, end_idx, terminator = scan_to_statement_end(j + 1)
            if name:
                symbols.append(
                    Symbol(
                        header=header,
                        module=module,
                        namespace=namespace_path(),
                        name=name,
                        kind="enum_class" if "class" in tokens[i : j + 1] else "enum",
                        smoke="using",
                    )
                )
            if terminator == "{":
                push_scope("enum")
            pending_template = False
            i = end_idx + 1
            continue
        if tok == "using" and at_namespace_scope():
            if i + 1 < len(tokens) and tokens[i + 1] == "namespace":
                stmt, end_idx, _terminator = scan_to_statement_end(i + 2)
                i = end_idx + 1
                continue
            if i + 1 < len(tokens) and re.fullmatch(r"[A-Za-z_]\w*", tokens[i + 1]):
                alias_name = tokens[i + 1]
                stmt, end_idx, _terminator = scan_to_statement_end(i + 2)
                if "=" in stmt:
                    symbols.append(
                        Symbol(
                            header=header,
                            module=module,
                            namespace=namespace_path(),
                            name=alias_name,
                            kind="alias_template" if pending_template else "alias",
                            smoke="using",
                        )
                    )
                    pending_template = False
                    i = end_idx + 1
                    continue
            stmt, end_idx, _terminator = scan_to_statement_end(i + 1)
            pending_template = False
            i = end_idx + 1
            continue
        if tok == "typedef" and at_namespace_scope():
            stmt, end_idx, _terminator = scan_to_statement_end(i + 1)
            alias_name = extract_variable_name(stmt)
            if alias_name:
                symbols.append(Symbol(header, module, namespace_path(), alias_name, "typedef", "using"))
            pending_template = False
            i = end_idx + 1
            continue
        if tok == "concept" and at_namespace_scope():
            if i + 1 < len(tokens) and re.fullmatch(r"[A-Za-z_]\w*", tokens[i + 1]):
                symbols.append(Symbol(header, module, namespace_path(), tokens[i + 1], "concept", "excluded", "requires_witness_type"))
            stmt, end_idx, _terminator = scan_to_statement_end(i + 2)
            pending_template = False
            i = end_idx + 1
            continue
        if at_namespace_scope():
            stmt, end_idx, terminator = scan_to_statement_end(i)
            if end_idx == i:
                i += 1
                continue
            fn_name, fn_name_index = extract_function_name(stmt)
            if fn_name:
                if fn_name_index is not None and fn_name_index > 0 and stmt[fn_name_index - 1] == "::":
                    pending_template = False
                    if terminator == "{":
                        push_scope("function")
                    i = end_idx + 1
                    continue
                symbols.append(
                    Symbol(
                        header,
                        module,
                        namespace_path(),
                        fn_name,
                        "function_template" if pending_template else "function",
                        "using",
                    )
                )
                pending_template = False
                if terminator == "{":
                    push_scope("function")
                i = end_idx + 1
                continue
            if terminator == "{":
                pending_template = False
                depth = 1
                i = end_idx + 1
                while i < len(tokens) and depth > 0:
                    if tokens[i] == "{":
                        depth += 1
                    elif tokens[i] == "}":
                        depth -= 1
                    i += 1
                if i < len(tokens) and tokens[i] == ";":
                    i += 1
                continue
            variable_name = extract_variable_name(stmt)
            if (
                variable_name
                and "::" not in stmt
                and any(tok in stmt for tok in {"=", "constinit", "constexpr", "extern", "inline"})
            ):
                symbols.append(Symbol(header, module, namespace_path(), variable_name, "variable", "using"))
                pending_template = False
                i = end_idx + 1
                continue
            pending_template = False
            i = end_idx + 1
            continue
        i += 1
    return filter_symbols(symbols), exclusions


def build_inventory() -> dict:
    blocks = parse_set_blocks(COMMONS_SOURCES.read_text(encoding="utf-8"))
    public_headers = [header for header in blocks["COMMONS_PUBLIC_HEADERS"]]
    group_membership: dict[str, str] = {}
    for group_name, module_name in GROUP_TO_MODULE.items():
        for header in blocks.get(group_name, []):
            group_membership[header] = module_name

    inventory_headers = []
    all_symbols: list[Symbol] = []
    all_exclusions: list[dict[str, str]] = []

    for header in public_headers:
        module_name = module_for_header(header, group_membership)
        header_path = COMMONS_ROOT / header
        text = header_path.read_text(encoding="utf-8")
        header_exclusion_reason = MODULE_HEADER_EXCLUSIONS.get(header)
        if header_exclusion_reason:
            inventory_headers.append(
                {
                    "path": header,
                    "module": module_name,
                    "sha256": sha256_text(text),
                    "symbols": [],
                    "excluded": [
                        {
                            "header": header,
                            "name": header,
                            "kind": "header",
                            "reason": header_exclusion_reason,
                        }
                    ],
                }
            )
            all_exclusions.append(
                {
                    "header": header,
                    "name": header,
                    "kind": "header",
                    "reason": header_exclusion_reason,
                }
            )
            continue
        macro_symbols, macro_exclusions = parse_macro_invocations(header, module_name, text)
        regular_symbols, regular_exclusions = parse_regular_symbols(header, module_name, text)

        header_symbols = unique_symbols(macro_symbols + regular_symbols)
        exclusions = macro_exclusions + regular_exclusions

        if header == "modules/commons.exports.hpp":
            exclusions.append(
                {
                    "header": header,
                    "name": "modules/commons.exports.hpp",
                    "kind": "header",
                    "reason": "aggregator_header",
                }
            )

        exclusions.extend(
            {
                "header": hdr,
                "name": qualified,
                "kind": "symbol",
                "reason": reason,
            }
            for (hdr, qualified), reason in MANUAL_SYMBOL_EXCLUSIONS.items()
            if hdr == header
        )
        header_symbols = [
            symbol
            for symbol in header_symbols
            if MANUAL_SYMBOL_EXCLUSIONS.get((header, symbol.qualified)) is None
        ]
        header_symbols = unique_symbols(header_symbols + MANUAL_SYMBOLS.get(header, []))

        inventory_headers.append(
            {
                "path": header,
                "module": module_name,
                "sha256": sha256_text(text),
                "symbols": [
                    {
                        "name": symbol.name,
                        "namespace": symbol.namespace,
                        "qualified": symbol.qualified,
                        "kind": symbol.kind,
                        "smoke": symbol.smoke,
                        **({"reason": symbol.reason} if symbol.reason else {}),
                    }
                    for symbol in header_symbols
                ],
                "excluded": exclusions,
            }
        )
        all_symbols.extend(header_symbols)
        all_exclusions.extend(exclusions)

    inventory = {
        "version": 1,
        "source": "COMMONS_PUBLIC_HEADERS",
        "headers": inventory_headers,
        "macros": MANUAL_MACROS,
        "summary": {
            "headers": len(public_headers),
            "symbols": len(unique_symbols(all_symbols)),
            "excluded": len(all_exclusions),
        },
    }
    return inventory


def generate_smoke_file(module_name: str, inventory: dict) -> str:
    type_like_names = type_like_names_for_module(module_name, inventory)
    lines = [
        "// Generated by Application/Tests/generate_commons_module_surface.py.",
        "// Do not edit by hand.",
        "",
    ]
    if module_name == "commons.http":
        lines.append("#if WITH_MHD")
        lines.append("")
    namespace_name = module_name.replace(".", "_")
    for header in inventory["headers"]:
        if header["module"] != module_name:
            continue
        symbols_by_namespace: dict[str, list[dict]] = defaultdict(list)
        for symbol in header["symbols"]:
            if symbol["smoke"] != "using":
                continue
            symbols_by_namespace[symbol["namespace"]].append(symbol)
        if not symbols_by_namespace:
            continue
        lines.append(f"// {header['path']}")
        for namespace in sorted(symbols_by_namespace):
            if not namespace:
                continue
            if namespace in type_like_names:
                lines.append(f"namespace commons_module_smoke::{namespace_name} {{")
                seen_names: set[str] = set()
                for symbol in sorted(symbols_by_namespace[namespace], key=lambda item: item["name"]):
                    if symbol["name"] in seen_names:
                        continue
                    seen_names.add(symbol["name"])
                    alias_name = sanitize_for_identifier(f"{namespace}_{symbol['name']}")
                    lines.append(f"using {alias_name} = ::{namespace}::{symbol['name']};")
                lines.append("}")
                lines.append("")
                continue
            lines.append(f"namespace commons_module_smoke::{namespace_name}::{namespace} {{")
            seen_names: set[str] = set()
            for symbol in sorted(symbols_by_namespace[namespace], key=lambda item: item["name"]):
                if symbol["name"] in seen_names:
                    continue
                seen_names.add(symbol["name"])
                lines.append(f"using ::{namespace}::{symbol['name']};")
            lines.append("}")
            lines.append("")
    if module_name == "commons.http":
        lines.append("#endif")
    return "\n".join(lines).rstrip() + "\n"


def generate_exports(module_name: str, inventory: dict) -> str:
    type_like_names = type_like_names_for_module(module_name, inventory)
    symbols_by_namespace: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for header in inventory["headers"]:
        if header["module"] != module_name:
            continue
        for symbol in header["symbols"]:
            if symbol["smoke"] == "excluded":
                continue
            if symbol["kind"] == "macro":
                continue
            if symbol["namespace"] in type_like_names:
                continue
            namespace = symbol["namespace"]
            symbols_by_namespace[namespace].append((symbol["name"], header["path"]))

    lines = [
        "// Generated by Application/Tests/generate_commons_module_surface.py.",
        "// Do not edit by hand.",
        "",
    ]
    for namespace in sorted(symbols_by_namespace):
        if not namespace:
            continue
        lines.append(f"export namespace {namespace} {{")
        current_header = None
        seen_names: set[str] = set()
        for name, header in sorted(set(symbols_by_namespace[namespace]), key=lambda item: (item[1], item[0])):
            if name in seen_names:
                continue
            seen_names.add(name)
            if header != current_header:
                lines.append(f"// {header}")
                current_header = header
            lines.append(f"using ::{namespace}::{name};")
        lines.append("}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def generate_include_file(module_name: str, inventory: dict) -> str:
    lines = [
        "// Generated by Application/Tests/generate_commons_module_surface.py.",
        "// Do not edit by hand.",
        "",
    ]
    for header in inventory["headers"]:
        if module_name != "commons" and header["module"] != module_name:
            continue
        if any(excluded["kind"] == "header" and excluded["reason"] == MODULE_HEADER_EXCLUSIONS.get(header["path"]) for excluded in header["excluded"]):
            continue
        if header["path"] == "http/httpd.h":
            lines.append("#if WITH_MHD")
            lines.append(f"#include <{header['path']}>")
            lines.append("#endif")
        else:
            lines.append(f"#include <{header['path']}>")
    return "\n".join(lines).rstrip() + "\n"


def write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def generate_all() -> dict[str, str]:
    inventory = build_inventory()
    outputs: dict[str, str] = {}
    outputs[str(INVENTORY_PATH)] = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    for module_name, filename in MODULE_TO_EXPORT_FILE.items():
        outputs[str(EXPORTS_DIR / filename)] = generate_exports(module_name, inventory)
    for module_name, filename in MODULE_TO_INCLUDE_FILE.items():
        outputs[str(EXPORTS_DIR / filename)] = generate_include_file(module_name, inventory)
    for module_name, filename in HEADER_TO_SMOKE_FILE.items():
        outputs[str(SMOKE_DIR / filename)] = generate_smoke_file(module_name, inventory)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    outputs = generate_all()

    if args.check:
        mismatches = []
        for path_str, content in outputs.items():
            path = Path(path_str)
            if not path.exists():
                mismatches.append(f"missing: {path}")
                continue
            if path.read_text(encoding="utf-8") != content:
                mismatches.append(f"out-of-date: {path}")
        if mismatches:
            for mismatch in mismatches:
                print(mismatch, file=sys.stderr)
            print("Run Application/Tests/generate_commons_module_surface.py to refresh the checked-in surface files.", file=sys.stderr)
            return 1
        return 0

    for path_str, content in outputs.items():
        write_if_changed(Path(path_str), content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
