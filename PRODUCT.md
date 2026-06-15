# Product

## Register

product

## Users

ML infrastructure engineers, MLOps/SRE operators, and researchers evaluating or running the
`wmcp-jepa-serve` backend. They arrive with a task in flow: send a WMCP call (`encode`, `rollout`,
`score`, `plan`) against the LeWM Push-T checkpoint, watch live Prometheus metrics (request rate,
p95 latency, compute time, queue wait, validation errors), drive synthetic traffic, and confirm
the service is healthy and serving the real backend (`backend`/`revision`, not `mock`). The
canonical surface is the **WMCP LeWM Operations Console** at `:8088`. Context is a second monitor or
a split screen next to a terminal and Grafana; the user is verifying, debugging, or demoing, not
browsing.

## Product Purpose

Give a single browser surface that exercises the WMCP serving API end to end and shows its
operational shape in real time. Success is an operator trusting the readout at a glance: is the
backend real, is it ready, what's the live latency/error profile, did my request return the expected
tensor shapes. It complements Grafana/Prometheus/Tempo (linked, not replaced) by being the
hands-on request workbench — the place you *act* on the service, not just observe it.

## Brand Personality

**Instrument panel** — calm, precise, trustworthy. The console should read like scientific/lab
instrumentation, not a marketing dashboard: exact numbers, quiet confidence, no flash. Voice is
terse and technical (operator-to-operator), favoring real metric names and exact units over friendly
paraphrase. Three words: **precise, grounded, legible.** The earthy editorial palette already in the
code (warm paper + olive/moss/clay/teal/amber) is the deliberate carrier of this — warmth and
trust without resorting to dark-mode-cool or gradient-hype. Preserve that identity.

## Anti-references

- **Generic dark-mode SaaS dashboard** — the default neon-on-charcoal observability look (stock
  Grafana skin, glowing line charts on black). The warm light palette is a deliberate escape from
  this; don't regress toward it.
- **Gradient / glassy "AI startup" aesthetic** — purple gradients, glassmorphism, decorative blur,
  the hero-metric template (giant number + tiny label + gradient accent as a stat-card cliché).
- **Crowded enterprise admin (Bootstrap-era)** — undifferentiated gray tables, default form
  controls, no hierarchy; the unloved internal-tool look.
- **Toy / playful** — oversized rounded corners, bright primary toy colors, cartoon affordances;
  undermines the "trustworthy infrastructure" read. (Cards top out at ~12–16px radius here.)

## Design Principles

1. **The readout is the product.** Every pixel earns its place by helping an operator judge service
   health or a request's correctness faster. Decoration that doesn't convey state is noise.
2. **Exact over approximate.** Show real metric names, real units, real tensor shapes. Don't
   paraphrase `wmcp_request_latency_seconds` p95 into something "friendlier" — operators trust the
   precise word.
3. **Identity is the warmth.** The escape from dark-mode-default *is* the brand. Carry calm and
   trust through the committed earthy palette, full borders, and defined shadows — never through
   gradients, glass, or hype.
4. **Earned familiarity.** Standard tool affordances (selects, number inputs, buttons, disclosure
   rows) behave exactly as expected. The console disappears into the task; novelty is reserved for
   the data, not the chrome.
5. **Legible under load.** Dense live data must stay readable while it updates — stable layout, no
   reflow jitter, contrast that holds when numbers churn.

## Accessibility & Inclusion

Target **WCAG 2.1 AA**. Body text ≥4.5:1 against its surface; large/display text ≥3:1; visible
`:focus-visible` rings on every interactive control (already present — keep them); full keyboard
operability for the request workbench and traffic controls; honor `prefers-reduced-motion` for any
sparkline/transition motion (crossfade or instant fallback). Status and chart colors should not rely
on hue alone where feasible — pair with a label, shape, or text state (e.g. `ok`/`error` words
alongside the olive/red).
