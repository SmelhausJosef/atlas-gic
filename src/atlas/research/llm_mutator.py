from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Any

from atlas.common.config import AppConfig


@dataclass(slots=True)
class MutationCandidate:
    candidate_name: str
    source: str
    notes: str
    parent_candidate_name: str | None = None
    strategy_name: str | None = None
    class_name: str | None = None
    strategy_code: str | None = None
    mutation_spec: dict[str, Any] = field(default_factory=dict)
    family_name: str | None = None
    iteration_index: int = 0

    @property
    def is_generated(self) -> bool:
        return self.strategy_code is not None


class LLMMutator:
    def __init__(self, config: AppConfig):
        self.config = config

    def generate(
        self,
        parent_candidate_name: str,
        parent_strategy_source: str,
        baseline_summary: dict[str, Any],
        *,
        remaining_budget: int | None = None,
    ) -> list[MutationCandidate]:
        return self.propose_strategy_families(
            parent_candidate_name,
            parent_strategy_source,
            baseline_summary,
            remaining_budget=remaining_budget,
        )

    def propose_strategy_families(
        self,
        parent_candidate_name: str,
        parent_strategy_source: str,
        baseline_summary: dict[str, Any],
        *,
        remaining_budget: int | None = None,
    ) -> list[MutationCandidate]:
        if not self.config.research.use_llm:
            return []
        candidate_limit = self.config.research.llm_candidate_count
        if remaining_budget is not None:
            candidate_limit = max(0, min(candidate_limit, remaining_budget))
        if candidate_limit == 0:
            return []
        prompt = self._build_idea_prompt(
            parent_candidate_name=parent_candidate_name,
            strategy_source=parent_strategy_source,
            baseline_summary=baseline_summary,
            candidate_limit=candidate_limit,
        )
        payload = self._invoke_codex(prompt)
        return self._parse_candidates(
            payload,
            default_parent_candidate_name=parent_candidate_name,
            default_family_name=None,
            default_iteration_index=0,
            max_candidates=candidate_limit,
        )

    def tune_strategy_family(
        self,
        *,
        family_name: str,
        iteration_index: int,
        parent_candidate_name: str,
        parent_strategy_source: str,
        baseline_summary: dict[str, Any],
        family_history: list[dict[str, Any]],
    ) -> MutationCandidate | None:
        if not self.config.research.use_llm:
            return None
        prompt = self._build_tuning_prompt(
            family_name=family_name,
            iteration_index=iteration_index,
            parent_candidate_name=parent_candidate_name,
            parent_strategy_source=parent_strategy_source,
            baseline_summary=baseline_summary,
            family_history=family_history,
        )
        payload = self._invoke_codex(prompt)
        candidates = self._parse_candidates(
            payload,
            default_parent_candidate_name=parent_candidate_name,
            default_family_name=family_name,
            default_iteration_index=iteration_index,
            max_candidates=1,
        )
        return candidates[0] if candidates else None

    def _invoke_codex(self, prompt: str) -> dict[str, Any]:
        if self.config.research.provider not in {"codex_sdk", "codex_cli", "codex"}:
            raise RuntimeError(f"Unsupported LLM provider: {self.config.research.provider}")
        codex_bin = os.environ.get("ATLAS_CODEX_BIN", "").strip() or "codex"
        resolved_bin = shutil.which(codex_bin)
        if resolved_bin is None:
            raise RuntimeError(
                f"Codex CLI executable '{codex_bin}' was not found on PATH. Install Codex CLI or set ATLAS_CODEX_BIN."
            )

        with tempfile.TemporaryDirectory(prefix="atlas-codex-home-") as codex_home:
            codex_home_path = Path(codex_home)
            self._prepare_isolated_codex_home(codex_home_path)
            with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".txt", delete=False) as handle:
                output_path = Path(handle.name)
            try:
                command = [
                    resolved_bin,
                    "-a",
                    "never",
                    "-s",
                    "danger-full-access",
                    "exec",
                    "--skip-git-repo-check",
                    "--color",
                    "never",
                    "--output-last-message",
                    str(output_path),
                    "-m",
                    self.config.research.model,
                    prompt,
                ]
                env = os.environ.copy()
                env["CODEX_HOME"] = str(codex_home_path)
                result = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    cwd=Path.cwd(),
                    timeout=600,
                    check=False,
                    env=env,
                )
                response_text = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else ""
                if result.returncode != 0:
                    error_text = (result.stderr or result.stdout or response_text).strip()
                    raise RuntimeError(f"Codex CLI execution failed: {error_text or f'exit code {result.returncode}'}")
                if not response_text:
                    response_text = (result.stdout or "").strip()
                if not response_text:
                    raise RuntimeError("Codex CLI returned no final message.")
                return self._extract_json_payload(response_text)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Codex CLI execution timed out after 600 seconds.") from exc
            finally:
                output_path.unlink(missing_ok=True)

    def _prepare_isolated_codex_home(self, codex_home_path: Path) -> None:
        codex_home_path.mkdir(parents=True, exist_ok=True)
        auth_path = Path.home() / ".codex" / "auth.json"
        if not auth_path.exists():
            raise RuntimeError("Codex auth.json was not found in ~/.codex. Run `codex login` first.")
        shutil.copy2(auth_path, codex_home_path / "auth.json")
        (codex_home_path / "config.toml").write_text(
            '\n'.join(
                [
                    f'model = "{self.config.research.model}"',
                    'approval_policy = "never"',
                    'sandbox_mode = "danger-full-access"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def _build_idea_prompt(
        self,
        *,
        parent_candidate_name: str,
        strategy_source: str,
        baseline_summary: dict[str, Any],
        candidate_limit: int,
    ) -> str:
        current_strategy_config = self.config.strategy.model_dump()
        return f"""
You are generating new strategy ideas for a deterministic BTC perpetual 5-second backtest research system.

Rules:
- Return JSON only.
- Propose complete strategy modules, not config tweaks.
- Each candidate is a new strategy family idea.
- The module must define exactly one class that subclasses BaseStrategy and implements generate(context) -> StrategyDecision.
- Keep all logic self-contained inside the candidate module.
- Allowed imports only: from __future__ import annotations, import math, import statistics, from atlas.common.models import StrategyContext, StrategyDecision, from atlas.strategies.base import BaseStrategy.
- Never import os, sys, pathlib, subprocess, socket, requests, httpx, asyncio, or anything else.
- Never call open, eval, exec, compile, input, __import__, or breakpoint.
- Do not access files, network, subprocesses, environment variables, or mutate any external state.
- Use the existing StrategyConfig fields when practical; if you need constants, hardcode them inside the class.
- Propose at most {candidate_limit} family ideas.
- Parent candidate for this round is "{parent_candidate_name}".
- Include a compact mutation_spec with objective, hypothesis, and risk_notes.

Current strategy config:
{json.dumps(current_strategy_config, indent=2)}

Parent strategy source:
```python
{self._truncate_text(strategy_source, max_chars=7000)}
```

Baseline evaluation summary:
{json.dumps(baseline_summary, indent=2)}

Return JSON in this exact shape:
{{
  "candidates": [
    {{
      "candidate_name": "short_slug",
      "family_name": "short_slug",
      "notes": "why this new strategy idea may improve the score",
      "parent_candidate_name": "{parent_candidate_name}",
      "class_name": "GeneratedStrategy",
      "strategy_code": "from __future__ import annotations\\n\\nimport math\\n\\nfrom atlas.common.models import StrategyContext, StrategyDecision\\nfrom atlas.strategies.base import BaseStrategy\\n\\n\\nclass GeneratedStrategy(BaseStrategy):\\n    def generate(self, context: StrategyContext) -> StrategyDecision:\\n        return StrategyDecision(target_position=0.0, confidence=0.0, reason='flat', tags=['flat'])\\n",
      "mutation_spec": {{
        "objective": "discover a new short-horizon edge",
        "hypothesis": "this logic targets a different structure than the parent baseline",
        "risk_notes": "hardcoded constants may overfit or collapse trade count"
      }}
    }}
  ]
}}
""".strip()

    def _build_tuning_prompt(
        self,
        *,
        family_name: str,
        iteration_index: int,
        parent_candidate_name: str,
        parent_strategy_source: str,
        baseline_summary: dict[str, Any],
        family_history: list[dict[str, Any]],
    ) -> str:
        current_strategy_config = self.config.strategy.model_dump()
        return f"""
You are refining an existing strategy family for a deterministic BTC perpetual 5-second backtest research system.

Rules:
- Return JSON only.
- Produce exactly one tuned strategy module for the same family.
- This is iteration {iteration_index} for family "{family_name}".
- The module must define exactly one class that subclasses BaseStrategy and implements generate(context) -> StrategyDecision.
- Keep all logic self-contained inside the candidate module.
- Allowed imports only: from __future__ import annotations, import math, import statistics, from atlas.common.models import StrategyContext, StrategyDecision, from atlas.strategies.base import BaseStrategy.
- Never import os, sys, pathlib, subprocess, socket, requests, httpx, asyncio, or anything else.
- Never call open, eval, exec, compile, input, __import__, or breakpoint.
- Do not access files, network, subprocesses, environment variables, or mutate any external state.
- Improve the family by reducing failure modes found in history while preserving the idea.
- Include a compact mutation_spec with objective, hypothesis, and risk_notes.

Current strategy config:
{json.dumps(current_strategy_config, indent=2)}

Parent candidate name:
{parent_candidate_name}

Parent strategy source:
```python
{self._truncate_text(parent_strategy_source, max_chars=7000)}
```

Baseline evaluation summary:
{json.dumps(baseline_summary, indent=2)}

Family iteration history:
{json.dumps(family_history[-4:], indent=2)}

Return JSON in this exact shape:
{{
  "candidates": [
    {{
      "candidate_name": "{family_name}_iter_{iteration_index}",
      "family_name": "{family_name}",
      "notes": "why this refinement should improve the family",
      "parent_candidate_name": "{parent_candidate_name}",
      "class_name": "GeneratedStrategy",
      "strategy_code": "from __future__ import annotations\\n\\nimport math\\n\\nfrom atlas.common.models import StrategyContext, StrategyDecision\\nfrom atlas.strategies.base import BaseStrategy\\n\\n\\nclass GeneratedStrategy(BaseStrategy):\\n    def generate(self, context: StrategyContext) -> StrategyDecision:\\n        return StrategyDecision(target_position=0.0, confidence=0.0, reason='flat', tags=['flat'])\\n",
      "mutation_spec": {{
        "objective": "improve this family after the last backtest",
        "hypothesis": "this refinement should reduce the main failure mode from family history",
        "risk_notes": "refinement may overfit to recent history"
      }}
    }}
  ]
}}
""".strip()

    def _extract_json_payload(self, response_text: str) -> dict[str, Any]:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError("Codex SDK response did not contain a JSON object.")
        return json.loads(response_text[start : end + 1])

    def _parse_candidates(
        self,
        payload: dict[str, Any],
        default_parent_candidate_name: str,
        *,
        default_family_name: str | None,
        default_iteration_index: int,
        max_candidates: int | None = None,
    ) -> list[MutationCandidate]:
        raw_candidates = payload.get("candidates", [])
        candidates: list[MutationCandidate] = []
        limit = max_candidates or self.config.research.llm_candidate_count
        for index, raw in enumerate(raw_candidates[:limit], start=1):
            family_name = self._sanitize_candidate_name(str(raw.get("family_name") or default_family_name or raw.get("candidate_name") or f"family_{index}"))
            candidate_name = self._sanitize_candidate_name(
                str(raw.get("candidate_name") or f"{family_name}_iter_{default_iteration_index}")
            )
            notes = str(raw.get("notes") or "generated by Codex SDK")
            raw_parent = raw.get("parent_candidate_name")
            class_name = str(raw.get("class_name") or "GeneratedStrategy")
            strategy_code = self._sanitize_strategy_code(raw.get("strategy_code"))
            if not strategy_code:
                continue
            candidates.append(
                MutationCandidate(
                    candidate_name=candidate_name,
                    source="codex_sdk",
                    notes=notes,
                    parent_candidate_name=str(raw_parent) if raw_parent else default_parent_candidate_name,
                    strategy_name=candidate_name,
                    class_name=class_name,
                    strategy_code=strategy_code,
                    mutation_spec=self._sanitize_mutation_spec(raw.get("mutation_spec")),
                    family_name=family_name,
                    iteration_index=default_iteration_index,
                )
            )
        return candidates

    def _sanitize_mutation_spec(self, raw_spec: Any) -> dict[str, str]:
        if not isinstance(raw_spec, dict):
            return {}
        cleaned: dict[str, str] = {}
        for key in ("objective", "hypothesis", "risk_notes"):
            value = raw_spec.get(key)
            if isinstance(value, str) and value.strip():
                cleaned[key] = value.strip()
        return cleaned

    def _sanitize_strategy_code(self, raw_code: Any) -> str | None:
        if not isinstance(raw_code, str):
            return None
        stripped = raw_code.strip()
        if not stripped:
            return None
        return stripped + "\n"

    def _sanitize_candidate_name(self, raw_name: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_name.strip()).strip("_").lower()
        return cleaned or "codex_generated_candidate"

    def _truncate_text(self, value: str, *, max_chars: int) -> str:
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 64] + "\n# ... truncated for prompt budget ...\n"
