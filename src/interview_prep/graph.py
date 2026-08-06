"""Compile the extraction and validation slice as a LangGraph workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    extract_requirements,
    mark_extraction_ready,
    report_validation_errors,
    route_after_validation,
    validate_requirements,
)
from .schemas import WorkflowInput, WorkflowState


def build_graph():
    """Build a fixed workflow with one deterministic conditional edge."""

    builder = StateGraph(WorkflowState, input_schema=WorkflowInput)
    builder.add_node("extract_requirements", extract_requirements)
    builder.add_node("validate_requirements", validate_requirements)
    builder.add_node("mark_extraction_ready", mark_extraction_ready)
    builder.add_node("report_validation_errors", report_validation_errors)

    builder.add_edge(START, "extract_requirements")
    builder.add_edge("extract_requirements", "validate_requirements")
    builder.add_conditional_edges(
        "validate_requirements",
        route_after_validation,
        {
            "ready": "mark_extraction_ready",
            "invalid": "report_validation_errors",
        },
    )
    builder.add_edge("mark_extraction_ready", END)
    builder.add_edge("report_validation_errors", END)
    return builder.compile()


graph = build_graph()
