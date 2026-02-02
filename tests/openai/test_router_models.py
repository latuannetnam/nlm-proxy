from nlm_proxy.openai.router import NotebookRole, NotebookSelection, MultiRoutingDecision, RequestType
from enum import Enum

def test_notebook_role_enum():
    assert NotebookRole.PRIMARY.value == "primary"
    assert NotebookRole.SECONDARY.value == "secondary"

def test_multi_routing_decision_properties():
    primary = NotebookSelection(notebook_id="1", role=NotebookRole.PRIMARY, reasoning="main")
    secondary1 = NotebookSelection(notebook_id="2", role=NotebookRole.SECONDARY, reasoning="supp1")
    secondary2 = NotebookSelection(notebook_id="3", role=NotebookRole.SECONDARY, reasoning="supp2")

    decision = MultiRoutingDecision(
        request_type=RequestType.NOTEBOOKLM,
        notebooks=[secondary1, primary, secondary2],
        reasoning="complex query"
    )

    assert decision.primary_notebook == primary
    assert len(decision.secondary_notebooks) == 2
    assert secondary1 in decision.secondary_notebooks
    assert secondary2 in decision.secondary_notebooks

def test_multi_routing_no_primary():
    decision = MultiRoutingDecision(
        request_type=RequestType.LLM_TASK,
        notebooks=[],
        reasoning="general task"
    )
    assert decision.primary_notebook is None
    assert decision.secondary_notebooks == []
