"""Agent (the execution subject).

In MSEF, the agent is the execution subject for tasks within a stage: the execution engine provides stage partitioning and dependency orchestration, while the concrete tasks inside each stage are completed autonomously by the agent. At every stage the agent builds a prompt (fusing task-reduction results, compressed context, and mounted memory), calls the model backend, parses structured output, and lets quality gates and human approval constrain it at stage boundaries.

This module contains no business decisions (gates/approval/rollback are handled by the engine); it only handles "generation" and "parsing".
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from msef.llm.base import LLMBackend, parse_json_block
from msef.types import Stage, StageOutput, Task

logger = logging.getLogger(__name__)

# Per-stage prompt templates
SYSTEM_PROMPT = (
    "You are the software-engineering agent in the MSEF multi-stage execution framework. "
    "Complete the requested stage and return a JSON object with the required fields. "
    "All numeric quality fields are floats from 0 to 1, where 1 is best. Do not add prose."
)

STAGE_INSTRUCTION: Dict[str, str] = {
    Stage.PLAN.value: (
        'Planning stage. Return JSON: {"plan_elements": ["scope", "dependencies", '
        '"sequence", "risks", "rollback"], "files": ["affected files"], '
        '"risks": ["identified risks"], "operations": [{"name": "operation", '
        '"risk_level": "high|extreme|medium|low|minimal", "impact_scope": 1, '
        '"reversibility": 0.0}], "plan_completeness": 0.0, '
        '"impact_coverage": 0.0, "risk_sufficiency": 0.0}'
    ),
    Stage.CODE.value: (
        'Coding stage. Return JSON: {"files": ["changed files"], '
        '"syntax_ok": true, "compile_ok": true, "spec_ok": 0.0, '
        '"changes": "change summary", "operations": []}'
    ),
    Stage.TEST.value: (
        'Testing stage. Return JSON: {"unit_pass": 0.0, "integration_pass": 0.0, '
        '"new_defects": 0.0}'
    ),
    Stage.REVIEW.value: (
        'Review stage. Return JSON: {"spec_ok": 0.0, "risk_items": '
        '[{"level": "high|medium|low|extreme"}], "alerts": 0, "complexity": 0.0, '
        '"dependency_scope": 0.0, "history_defect_density": 0.0}'
    ),
    Stage.DELIVER.value: (
        'Delivery stage. Return JSON: {"summary": "change summary", '
        '"changes": [], "risks": [], "tests": [], "rollback": "rollback plan", '
        '"report_complete": true, "risks_flagged": true}'
    ),
}


def _merge_output(output: StageOutput, payload: Dict[str, Any]) -> StageOutput:
    """Merge JSON fields returned by the LLM into the stage output (tolerates missing fields)."""
    content = dict(output.content)
    content.update(payload)
    files = payload.get("files") or output.files
    return StageOutput(
        stage=output.stage,
        content=content,
        files=list(dict.fromkeys(files)) if isinstance(files, list) else output.files,
        meta=output.meta,
        model=output.model,
        extra=output.extra,
    )


class Agent:
    """Stage execution agent.

    Args:
        backend: LLM backend (real model or simulated).
        memory_mount_func: Optional memory mounting function callable(query) -> (mounted_text, entries), used to inject memory retrieval results into the prompt (context mounting).
    """

    def __init__(self, backend: LLMBackend, memory_mount_func=None) -> None:
        self.backend = backend
        self.memory_mount_func = memory_mount_func
        self.token_usage = 0

    # ------------------------------------------------------------------
    def execute(self, stage: Stage, task: Task, subtasks: List = None,
                compressed_context: str = "", mounted_memory: str = "",
                prior_outputs: Optional[Dict[Stage, StageOutput]] = None,
                model: Optional[str] = None) -> StageOutput:
        """Execute one stage and return structured output."""
        instruction = STAGE_INSTRUCTION[stage.value]
        prompt = self._build_prompt(stage, task, subtasks, compressed_context,
                                    mounted_memory, prior_outputs)
        raw = self.backend.complete(prompt, system=SYSTEM_PROMPT)
        self.token_usage += self.backend.count_tokens(prompt) + self.backend.count_tokens(raw)
        payload = parse_json_block(raw)
        output = StageOutput(
            stage=stage,
            content={},
            files=[],
            meta={
                "task_id": task.id,
                "stage": stage.value,
                "instruction": instruction,
            },
            model=model or getattr(self.backend, "name", "unknown"),
        )
        return _merge_output(output, payload)

    # ------------------------------------------------------------------
    def _build_prompt(self, stage: Stage, task: Task, subtasks: List,
                      compressed_context: str, mounted_memory: str,
                      prior_outputs: Optional[Dict[Stage, StageOutput]]) -> str:
        parts = [
            f"Task ID: {task.id}",
            f"Task title: {task.title}",
            f"Task description: {task.description}",
            f"Task type: {task.task_type.value}",
            f"Acceptance criteria: {task.acceptance_criteria}",
        ]
        if subtasks:
            parts.append(
                "Task decomposition (subtasks):\n" + "\n".join(
                    f"- [{s.files}] {s.description}" for s in subtasks
                )
            )
        if compressed_context:
            parts.append(f"Compressed task context:\n{compressed_context[:4000]}")
        if mounted_memory:
            parts.append(f"Mounted memory (project/team/personal):\n{mounted_memory[:3000]}")
        if prior_outputs:
            prev = prior_outputs.get(stage)
            if prev is not None:
                parts.append(f"Previous stage output:\n{json.dumps(prev.content, ensure_ascii=False)[:2000]}")
        parts.append(STAGE_INSTRUCTION[stage.value])
        return "\n\n".join(parts)
