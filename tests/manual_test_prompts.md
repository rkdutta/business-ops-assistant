# Manual test prompts

A QA script for exercising the app by hand through the Streamlit chat UI,
organized by feature. Prompts reference the real seeded data in
data/seed_data.py and data/correspondence/, so expected results are noted
inline. This is separate from the automated suite (see tests/README or the
top-level test files) — use it for a human pass over UI-visible behavior
(approval panel, progress narration, thread naming, etc.) that the
automated tests don't cover.

## Basic lookups / subagent routing

- "What's Acme Roasters' contact info?" → customer_agent
- "What invoices does Golden Hour Coffee Co have?" → billing_agent
- "What's the status of our order with Dairy Direct?" → supplier_agent
- "Show me all overdue invoices." → billing_agent; should total $1,341.25
  across 3 invoices

## Router / multi-domain

- "Email all customers with overdue invoices reminding them to pay." →
  should touch billing_agent (find overdue) then customer_agent
  (draft/send), and pause 3x for approval (Blue Fern Cafe, Golden Hour
  Coffee Co x2)

## RAG (correspondence)

- "What payment terms did we agree with Highland Bean Co?" → net-45,
  12-month price lock
- "Does Evergreen Grocers need PO numbers on invoices?" → yes
- "What's the delivery SLA with Sunrise Logistics?"
- "Is there a bulk discount arrangement with Packrite Supplies?"

## Short-term memory

- Turn 1: "What's the status of invoice 7?"
- Turn 2: "Mark that one as paid." → should resolve "that one" to invoice 7
  without restating the ID, and should still pause for approval

## Long-term memory

- "Remember that we always CC accounts@ on invoice emails." (global)
- "Golden Hour Coffee Co is a frequent late payer, remember that." (scoped)
- New unrelated turn: "Do we have any standing preferences for invoice
  emails?" → should recall the CC rule
- "What do we know about Golden Hour Coffee Co's payment history?" →
  should surface the remembered pattern alongside real invoice data

## Human-in-the-loop

- "Create a $500 invoice for Daybreak Bakery due 2026-09-15." → should
  pause; test both Approve and Reject
- "Mark invoice 4 as paid." → should pause
- "Send a reminder email to Blue Fern Cafe about their overdue invoice." →
  should pause
- Try rejecting one and confirm nothing was written (invoice
  status/DB unchanged)

## Sandboxing

- "Calculate total invoice revenue by month, run it as an analysis
  script." → should use run_analysis_script, not raw SQL
- "What's the overdue total by customer, computed with a script rather
  than a direct query?"

## Context engineering / customer-summary skill

- "Give me a summary of Golden Hour Coffee Co." → should merge invoice
  history + correspondence (autopay agreement) in one call
- "Give me the full picture on Blue Fern Cafe." → should mention the
  installment payment plan

## Guardrails

- "Create an invoice for Acme Roasters for -$50." → should be rejected
  with a clear message, not silently created
- "Mark invoice 999 as paid." → should report no such invoice, not
  invent one
- "What's the exact total revenue for August?" → answer must trace to a
  real tool call, not a guessed number

## Event streaming (just observe, no special prompt)

- Any multi-step request above — watch the st.status() box for
  intermediate lines like "Delegating to billing_agent...", "Querying the
  database...", "Pulling full context for..."
