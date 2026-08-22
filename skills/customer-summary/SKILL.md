---
name: customer-summary
description: Produce a concise, structured summary of a specific customer — who they are, invoice/payment standing, and relevant agreements or notes from past correspondence. Use when asked for a "summary", "overview", "profile", or "full picture" of one named customer, not for a single fact lookup or a multi-customer question.
---

# Customer Summary Skill

## When to use

Trigger on requests like "give me a summary of Blue Fern Cafe" or "what's the
full picture on Golden Hour Coffee Co" — a rounded profile of one customer.
Don't use it for a single fact ("what's their email?") or a question that
spans multiple customers/domains — handle those directly instead.

## Workflow

1. Call `get_customer_context(customer=<name or id>)`. It already merges the
   customer's invoice history (capped and summarized if long — overdue count
   and total surfaced) with every piece of correspondence on file for them,
   so this is the one call this skill needs; there's no separate
   `search_correspondence` or raw SQL step.
2. Write the summary in this structure:
   - **Who they are** — one line, drawn only from what `get_customer_context`
     returned (correspondence/notes), never invented.
   - **Payment standing** — total invoices, and any overdue count/amount,
     stated exactly as returned — never estimate or round a figure the tool
     didn't give you.
   - **Agreements & notes** — up to 3 bullets from the correspondence
     section (payment terms, standing arrangements, discounts). Omit this
     section entirely if there's no correspondence on file — don't pad it.
3. Keep the whole thing to a short paragraph or a handful of bullets. This is
   a summary, not a data dump — every number in it must trace back to the
   single `get_customer_context` call above.
