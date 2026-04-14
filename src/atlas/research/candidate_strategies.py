from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas.common.models import StrategyContext, StrategyDecision
from atlas.strategies.base import BaseStrategy

ALLOWED_IMPORTS = {
    "__future__",
    "math",
    "statistics",
    "typing",
    "atlas.common.models",
    "atlas.strategies.base",
}

FORBIDDEN_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}

SAFE_TOP_LEVEL_NODES = (
    ast.Assign,
    ast.AnnAssign,
    ast.ClassDef,
    ast.Expr,
    ast.Import,
    ast.ImportFrom,
)


@dataclass(slots=True)
class CandidateValidationResult:
    status: str
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    class_name: str | None = None
    module_path: str | None = None

    @property
    def valid(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "issues": self.issues,
            "warnings": self.warnings,
            "class_name": self.class_name,
            "module_path": self.module_path,
        }


def validate_candidate_module(
    *,
    candidate_name: str,
    module_path: Path,
    validation_context: StrategyContext,
    strategy_config: Any,
) -> tuple[CandidateValidationResult, type[BaseStrategy] | None]:
    issues: list[str] = []
    source_code = module_path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(source_code, filename=str(module_path))
    except SyntaxError as exc:
        return (
            CandidateValidationResult(
                status="failed",
                issues=[f"syntax error at line {exc.lineno}: {exc.msg}"],
                module_path=str(module_path),
            ),
            None,
        )

    issues.extend(_validate_ast(tree))
    if issues:
        return (
            CandidateValidationResult(
                status="failed",
                issues=issues,
                module_path=str(module_path),
            ),
            None,
        )

    try:
        strategy_cls = load_strategy_class_from_path(module_path, candidate_name)
    except Exception as exc:  # pragma: no cover - defensive
        return (
            CandidateValidationResult(
                status="failed",
                issues=[f"module import failed: {exc}"],
                module_path=str(module_path),
            ),
            None,
        )

    try:
        strategy = strategy_cls(strategy_config)
        decision = strategy.generate(validation_context)
    except Exception as exc:
        return (
            CandidateValidationResult(
                status="failed",
                issues=[f"strategy smoke test failed: {exc}"],
                class_name=strategy_cls.__name__,
                module_path=str(module_path),
            ),
            None,
        )

    if not isinstance(decision, StrategyDecision):
        return (
            CandidateValidationResult(
                status="failed",
                issues=["strategy smoke test did not return StrategyDecision"],
                class_name=strategy_cls.__name__,
                module_path=str(module_path),
            ),
            None,
        )

    return (
        CandidateValidationResult(
            status="passed",
            class_name=strategy_cls.__name__,
            module_path=str(module_path),
        ),
        strategy_cls,
    )


def load_strategy_class_from_path(module_path: Path, candidate_name: str) -> type[BaseStrategy]:
    module_name = _build_module_name(candidate_name, module_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to create import spec for {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    strategy_classes = [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, BaseStrategy) and cls is not BaseStrategy and cls.__module__ == module_name
    ]
    if len(strategy_classes) != 1:
        raise RuntimeError("Candidate module must define exactly one BaseStrategy subclass.")
    return strategy_classes[0]


def _build_module_name(candidate_name: str, module_path: Path) -> str:
    digest = hashlib.sha1(str(module_path).encode("utf-8")).hexdigest()[:10]
    return f"atlas_generated_{candidate_name}_{digest}"


def _validate_ast(tree: ast.AST) -> list[str]:
    issues: list[str] = []
    body = getattr(tree, "body", [])
    for node in body:
        if not isinstance(node, SAFE_TOP_LEVEL_NODES):
            issues.append(f"top-level statement {type(node).__name__} is not allowed")
        elif isinstance(node, ast.Expr) and not isinstance(getattr(node, "value", None), ast.Constant):
            issues.append("top-level expressions are not allowed")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    issues.append(f"import '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name not in ALLOWED_IMPORTS:
                issues.append(f"import from '{module_name}' is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            issues.append(f"call to '{node.func.id}' is not allowed")
    return issues
