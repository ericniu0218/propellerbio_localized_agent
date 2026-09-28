"""Document drafting templates: SOP, study report, deviation report, and
regulatory section templates. The Specialist fills these in; the Director
reviews high-stakes ones (regulatory/clinical) before they're finalized.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel

from app.tools.base import Tool, ToolResult

_TEMPLATES: dict[str, str] = {
    "sop": (
        "# Standard Operating Procedure: {title}\n\n"
        "## 1. Purpose\n{purpose}\n\n"
        "## 2. Scope\n{scope}\n\n"
        "## 3. Responsibilities\n{responsibilities}\n\n"
        "## 4. Procedure\n{procedure}\n\n"
        "## 5. References\n{references}\n"
    ),
    "study_report": (
        "# Study Report: {title}\n\n"
        "## 1. Objective\n{objective}\n\n"
        "## 2. Methods\n{methods}\n\n"
        "## 3. Results\n{results}\n\n"
        "## 4. Discussion\n{discussion}\n\n"
        "## 5. Conclusion\n{conclusion}\n\n"
        "## 6. References\n{references}\n"
    ),
    "deviation_report": (
        "# Deviation Report: {title}\n\n"
        "## 1. Deviation Description\n{description}\n\n"
        "## 2. Root Cause\n{root_cause}\n\n"
        "## 3. Impact Assessment\n{impact}\n\n"
        "## 4. Corrective/Preventive Action\n{capa}\n\n"
        "## 5. Approval\n{approval}\n"
    ),
    "regulatory_section": (
        "# Regulatory Submission Section: {title}\n\n"
        "## 1. Summary\n{summary}\n\n"
        "## 2. Supporting Data\n{supporting_data}\n\n"
        "## 3. Risk Assessment\n{risk_assessment}\n\n"
        "## 4. References\n{references}\n"
    ),
}


class DocTemplateArgs(BaseModel):
    template: Literal["sop", "study_report", "deviation_report", "regulatory_section"]
    fields: dict[str, str]


class DocTemplateTool(Tool):
    name: ClassVar[str] = "doc_template"
    description: ClassVar[str] = (
        "Render an SOP, study report, deviation report, or regulatory "
        "section template with the given fields. Missing fields are left "
        "as an explicit placeholder rather than silently dropped."
    )
    args_model: ClassVar[type[BaseModel]] = DocTemplateArgs

    async def run(self, args: DocTemplateArgs) -> ToolResult:
        template = _TEMPLATES[args.template]
        safe_fields = {
            key: args.fields.get(key, f"[[MISSING: {key}]]")
            for key in _field_names(template)
        }
        rendered = template.format(**safe_fields)
        missing = [k for k, v in safe_fields.items() if v.startswith("[[MISSING")]
        summary = (
            f"Rendered {args.template} template."
            + (f" Missing fields: {', '.join(missing)}." if missing else "")
        )
        return ToolResult(summary=summary, content=rendered, citations=[])


def _field_names(template: str) -> list[str]:
    import string

    return [name for _, name, _, _ in string.Formatter().parse(template) if name]
