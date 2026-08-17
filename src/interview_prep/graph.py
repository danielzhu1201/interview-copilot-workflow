"""Compile the complete fixed Interview Prep Workflow V1 graph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    assemble_package,
    assess_gaps,
    build_strategy,
    extract_candidate_evidence,
    extract_requirements,
    generate_questions,
    match_evidence,
    parse_interview_round,
    report_errors,
    route_after_validation,
    validate_inputs,
    validate_package,
)
from .schemas import WorkflowInput, WorkflowState


def build_graph():
    """Build the fixed graph with one code-owned output branch."""

    builder = StateGraph(WorkflowState, input_schema=WorkflowInput)
    builder.add_node("validate_inputs", validate_inputs)
    builder.add_node("parse_interview_round", parse_interview_round)
    builder.add_node("extract_candidate_evidence", extract_candidate_evidence)
    builder.add_node("extract_requirements", extract_requirements)
    builder.add_node("match_evidence", match_evidence)
    builder.add_node("assess_gaps", assess_gaps)
    builder.add_node("build_strategy", build_strategy)
    builder.add_node("generate_questions", generate_questions)
    builder.add_node("validate_package", validate_package)
    builder.add_node("assemble_package", assemble_package)
    builder.add_node("report_errors", report_errors)

    builder.add_edge(START, "validate_inputs")
    builder.add_edge("validate_inputs", "parse_interview_round")
    builder.add_edge("parse_interview_round", "extract_candidate_evidence")
    builder.add_edge("extract_candidate_evidence", "extract_requirements")
    builder.add_edge("extract_requirements", "match_evidence")
    builder.add_edge("match_evidence", "assess_gaps")
    builder.add_edge("assess_gaps", "build_strategy")
    builder.add_edge("build_strategy", "generate_questions")
    builder.add_edge("generate_questions", "validate_package")
    builder.add_conditional_edges(
        "validate_package",
        route_after_validation,
        {
            "valid": "assemble_package",
            "invalid": "report_errors",
        },
    )
    builder.add_edge("assemble_package", END)
    builder.add_edge("report_errors", END)
    return builder.compile()


graph = build_graph()
