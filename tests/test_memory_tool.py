from agents.memory_tool import recall_facts, remember_fact


def test_recall_facts_global_empty(patched_db):
    assert recall_facts.invoke({"scope": "global"}) == "No remembered facts for this scope."


def test_remember_and_recall_global(patched_db):
    remember_fact.invoke({"fact": "Always CC accounts@ on invoice emails.", "scope": "global"})
    result = recall_facts.invoke({"scope": "global"})
    assert "Always CC accounts@ on invoice emails." in result


def test_remember_fact_requires_entity_name_for_scoped_fact(patched_db):
    result = remember_fact.invoke({"fact": "Late payer", "scope": "customer"})
    assert "entity_name is required" in result


def test_remember_fact_rejects_unknown_entity(patched_db):
    result = remember_fact.invoke(
        {"fact": "Late payer", "scope": "customer", "entity_name": "Nonexistent Co"}
    )
    assert "No customer named 'Nonexistent Co'" in result


def test_remember_and_recall_scoped_to_entity(patched_db):
    remember_fact.invoke(
        {"fact": "Frequent late payer.", "scope": "customer", "entity_name": "Blue Fern Cafe"}
    )
    blue_fern = recall_facts.invoke({"scope": "customer", "entity_name": "Blue Fern Cafe"})
    acme = recall_facts.invoke({"scope": "customer", "entity_name": "Acme Roasters"})
    assert "Frequent late payer." in blue_fern
    assert acme == "No remembered facts for this scope."


def test_recall_facts_requires_entity_name_for_scoped_lookup(patched_db):
    result = recall_facts.invoke({"scope": "supplier"})
    assert "entity_name is required" in result
