# The interview rules

The five interaction rules that govern every human-facing question round in
the pipeline: all of Phase A (`mdp-formalize`), and any later round that asks
the human anything — `mdp-solve`'s run-plan confirmation, an escalation-
budget change. They override the mechanical step lists wherever they
conflict.

Phase A is a guided interview, not a form dump. At every moment the human
must know **where we are, what is settled, and what is still open**, and be
able to stop and ask instead of answering. Five interaction rules govern
this — they override the mechanical step list below whenever they conflict.

**1. Brief in the message, choose in the menu.** Every question round is two
parts of *one* turn: a **briefing message** in plain markdown, then the
**ballot** — the host's structured question tool when it has one (Claude
Code: `AskUserQuestion`; Codex: `request_user_input`, available in Plan mode
or behind its `default_mode_request_user_input` flag), else a numbered
plain-text ballot that ends the turn (below). The menu is a ballot, not a
document — it is a narrow column and long text there is unreadable. All
exposition lives in the briefing; the menu carries only the choices' names.

The briefing message (markdown, headings/bullets/tables — not a wall of
prose) carries, in this order:

- the **status board** (rule 2);
- **what is ambiguous here**, in two or three sentences: what the model needs
  to know and what the choice changes downstream;
- **the candidates**, one short subsection each, using the *same name* the
  menu option will use as its label: what it means in the problem's own
  terms, then its pros and its cons — the trade-off, not a lecture;
- **your recommendation and why**, unless the candidates are genuinely
  equivalent;
- one line noting they can answer "Other" to ask a question instead of
  picking (rule 4).

The ballot is then deliberately tiny. Hard limits (the field names are the
Claude Code tool's; the same limits bind the plain-text form):

| field | limit |
|---|---|
| `question` | ≤ 2 short sentences — restate the choice, point at the briefing above |
| `header` | ≤ 12 chars |
| option `label` | ≤ 5 words, **verbatim** the name used in the briefing so menu ↔ briefing map 1:1 |
| option `description` | one line, ≤ ~15 words — the consequence in a nutshell, not a paragraph |
| option `preview` | only for genuinely *visual* side-by-side content (a scenario grid, a composition table, a code sketch). Never prose — prose belongs in the briefing |

A description that wraps to three lines in a narrow column, or a `question`
carrying the status board, is the failure mode this rule exists to prevent.

*Plain-text ballot* (no structured tool): end the briefing with the question
in ≤ 2 sentences, then a numbered list — one option per line, the same
≤ 5-word label the briefing used, a dash, the one-line consequence — and a
last line `Other — ask a question or challenge the framing`. Then **stop and
wait**; the human answers with a number or free text. Do not fold the ballot
into prose: the numbered list is what makes the answer unambiguous.

**2. Post a status board at the top of every briefing.** A short *structured*
snapshot (not prose the human has to mine):

- **Step:** which Phase-A step we are in (objective & mode stance → drafting
  → classifying randomness → designing the scenario set → resolving
  confirmables → confirming restatement).
- **Settled:** the facts now pinned down — decision, horizon, objective,
  each resolved assumption — one terse line each.
- **Open:** the still-unresolved points, *named*, listed in the order you
  will ask them. This is the agenda; the human should see the whole
  remaining queue, not discover it one surprise at a time.

Re-post the board as items move Open → Settled. Keep it tight; it orients,
it is not the restatement.

**3. Write for the problem owner, not for an RL engineer.** The human knows
their problem; assume they do not know the solution techniques. In the
briefing, describe each option in the problem's own vocabulary first
("unmet demand is simply lost" / "unmet demand waits and is served next
period"), and only then, if it helps, name the modelling consequence —
defining the term on first use ("this adds a *state variable*: a number the
policy sees each period"). In labels and descriptions, no unexplained
jargon at all: no `Confirmable`, `ScenarioSampler`, MMFE, frame-stack,
observation space, or IR field paths unless the human introduced the term.
If a choice cannot be stated without a technical term, define the term in
the briefing and use the plain phrasing in the menu.

**4. Ask one question at a time, and make pausing a first-class move.**
Default to a single question per round so the human can push back between
them. Batch only when questions are mutually independent *and* trivial
confirmations unlikely to spark discussion — and even then keep it to a few.
Every briefing ends with a standing note that answering "Other" (always
present) pauses the interview to ask a question or challenge the framing
instead of picking an option — the human cannot be expected to know this.
When they do, stop the sequence, resolve it, then re-post the board and
continue. A one-at-a-time cadence is what makes this possible — do not trap
the human in a batch they must answer before they can speak.

**5. Anything large ends the turn.** The restatement, a full scenario set, a
long table — print it in a plain message with **no tool call after it**, and
save it to a co-located file; ask the confirmation in the *next* round,
naming that file. Never ask the human to confirm something they have not
been shown.

Assistant text is normally rendered, but if the human reports they cannot see
what they are being asked to confirm, treat that briefing as lost: re-send it
as a turn-ending message under this rule, then re-ask in the next turn.

**Shape of one good round.** Briefing message:

> **Step 3 of 6 — how unmet demand behaves**
> **Settled:** decision = order quantity each week · horizon = 52 weeks ·
> objective = minimize total cost
> **Open:** unmet demand → shortage cost → lead time → scenario grid
>
> When a week's demand exceeds what you have on hand, the model has to say
> what happens to the excess. This changes what the policy has to keep track
> of, and it changes what "cost" means, so it is worth getting right.
>
> **Lost sales** — the customer goes elsewhere; the excess demand disappears.
> *Pro:* simpler, and matches walk-in retail. *Con:* if your customers
> actually wait, it understates the pain of running out.
>
> **Backlog** — the customer waits and is served first next week.
> *Pro:* right for contracted/B2B supply. *Con:* the model must carry the
> outstanding amount from week to week, which makes the problem a little
> harder to learn.
>
> I'd suggest **Lost sales** unless your buyers reliably wait — you described
> walk-in customers earlier.
>
> If you'd rather ask something than answer, pick "Other".

Then the call: header `Unmet demand`; question "When demand exceeds stock,
does it vanish or wait? (see the two options above)"; labels `Lost sales` /
`Backlog`; descriptions "Excess demand disappears; pay a shortage penalty" /
"Excess demand waits, served first next week". Nothing longer.

**Respect dependencies across rounds.** If an answer could eliminate or
reshape a later question's options, ask the gating question first and build
the later options from the answer actually given — never from an assumed one.
(E.g. "backlogged or lost?" gates the shortage-cost options; "discrete or
continuous decision?" gates the bounds/masking question.)

