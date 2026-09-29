# UX and Plain-Language Rules

Status: **draft, awaiting approval before implementation** — companion to
`docs/server/WORKFLOW.md`. Written for whoever builds the frontend; the backend
enforces the parts of this that are contracts (event fields, error shape),
and everything visual is their call, not dictated here.

Audience reminder: expert scientists/engineers, not AI users. **If a
feature needs a tutorial, it's designed wrong.** Every rule below exists to
make that true.

## 1. No jargon, by default

The default view never shows: model names, "tokens," "context,"
"embeddings," "temperature," "prompt," or agent internals of any kind.
These only appear if the user turns on **Show technical details** in
settings. The backend always sends both the plain and technical fields on
every event — the frontend decides which to render based on that one
setting; the backend never has to be asked twice for the same data in two
shapes.

## 2. Plain-language agent panel

The right-side panel shows a friendly name and a human status per agent,
computed backend-side (in `app/copy/`) from the agent's role **and its
current action**, not just its role — the same agent shows a different
label depending on what it's doing right now:

| Role | Action | Friendly label + status |
|---|---|---|
| Director | planning | "Planner: breaking your request into steps" |
| Director | reviewing | "Planner: double-checking the results" |
| Specialist | searching docs/literature | "Researcher: reading N documents" |
| Specialist | extracting tables/data | "Assistant: pulling tables out of your PDF" |
| Specialist | drafting | "Assistant: writing a draft" |
| Runner | classifying/formatting | "Assistant: getting things organized" |
| any | blocked on human approval | **"Waiting for you"** + a clear button |

This table is the seed of `docs/COPY_GUIDE.md` (written in build-order
step 6) and lives in code next to it so the two never drift apart.

## 3. Trust signals

- **"Checked" badge**: shown when the Director reviewed and approved the
  answer (maps to the existing `review.approved` event / a completed
  task's review outcome).
- **"Needs a human check" badge**: shown on high-stakes content, or any
  answer that hasn't gone through Director review yet (e.g. still mid-task,
  or explicitly flagged). Never both badges at once; "needs a human check"
  always wins if both could apply.
- Badges are a color + one short label, never a paragraph explaining the
  review pipeline.

## 4. Errors

Every user-facing error is one sentence describing what happened, in plain
words, plus what to do next. Never a stack trace, an exception class name,
or an HTTP status code as the visible text. Examples:

- Bad: `RuntimeError: BudgetExceededError: specialist exceeded budget`
- Good: "This task used more thinking than expected and stopped early.
  Try breaking it into smaller questions, or ask again — it usually works
  the second time."
- Bad: `403 Forbidden`
- Good: "You don't have access to that yet. Ask the owner to share it with
  you or your team."

The backend enforces the shape (a plain `message` + optional `next_step`
string on every error response) via one exception handler; individual
endpoints don't hand-write error text.

## 5. The standard Brief

Every condensed answer — a chat response, a summary card, a compare
result — renders in this order, always:

```
Bottom line
  1–2 sentences. The answer, not a preamble.

Key findings
  • short bullet
  • short bullet

Key numbers
  | metric | value | source |

Open questions / risks
  • anything uncertain, conflicting, or worth a second look

Sources
  [1] doc name, section — click to open the exact passage, highlighted
```

The length slider (**One line · Short · One page · Full detail**) changes
how much of this is shown, not what's computed:

- **One line** — Bottom line only.
- **Short** — Bottom line + top 3 key findings.
- **One page** — everything, bullets/table trimmed to fit roughly a page.
- **Full detail** — everything, untrimmed, full source excerpts.

Moving the slider is instant (a render, not a re-ask) — see
`docs/server/WORKFLOW.md` §5.

## 6. Starter buttons and empty states

An empty chat shows 3–4 buttons based on the user's department (admin-set,
per `docs/server/WORKFLOW.md` §1): defaults are "Summarize files," "Compare
results," "Draft a report," "Catch me up," and a department can swap these
for its own starter recipes. Nobody sees a blank box with no next step.

## 7. Long tasks, undo, and control

- Starting a task never blocks the chat — the user can keep typing or
  navigate away and gets a notification when it's done.
- Any running task can be **stopped**, the request **edited**, and
  **retried** in one click each — no need to start a new chat from
  scratch to correct a typo in a long-running request.
- Every finished answer has four buttons, always in the same place: **Copy,
  Download** (Word/PDF/Excel, whichever fits the content), **Share, Save
  as Recipe**.

## 8. Accessibility and environment

Keyboard-navigable throughout (tab order follows visual order; every
button reachable and actionable without a mouse), readable contrast
(WCAG AA minimum), and no dependency on a GPU, extension, or non-standard
browser — it must work on a standard company laptop with whatever browser
IT ships.

## 9. Usability checklist (design every screen against this)

A first-time user, given no instructions, must be able to do all of the
following in under 5 minutes:

1. **Log in** — click "Sign in," land in My Space. No setup screen first.
2. **Upload files and get a summary** — drag a file onto the chat, see a
   summary card with a bottom line and citations, no dialog explaining
   what's happening.
3. **Ask a question with sources** — type a plain-English question, get a
   brief-format answer, click a citation, see the source passage.
4. **Share a result with their team** — one Share button, pick a Team
   Space, done — no separate "sharing settings" page to find first.
5. **Run a starter recipe** — click one of the empty-state buttons, fill
   in the one or two fields it asks for, get a result.

If any of these needs a tooltip, a help article, or asking a coworker, the
screen it happened on gets redesigned before ship — this checklist is the
acceptance test for the whole UX layer, not just an aspiration.
