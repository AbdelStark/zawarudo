---
name: WMCP LeWM Operations Console
description: Field-instrument telemetry console for an action-conditioned world-model serving backend.
colors:
  instrument-teal: "#176f72"
  deep-olive: "#4e6630"
  moss: "#7b9652"
  clay: "#b45b42"
  amber: "#d79a27"
  signal-red: "#b3384b"
  graphite-olive: "#22231e"
  sage-paper: "#edf0ea"
  lab-bone: "#fcfbf4"
  input-bone: "#fefff9"
  slate-sage: "#687068"
  hairline-sage: "#c8d0c7"
  quiet-sage: "#e5e8dd"
typography:
  display:
    fontFamily: "Avenir Next, IBM Plex Sans, sans-serif"
    fontSize: "4rem"
    fontWeight: 600
    lineHeight: 0.95
    letterSpacing: "0"
  headline:
    fontFamily: "Avenir Next, IBM Plex Sans, sans-serif"
    fontSize: "1.45rem"
    fontWeight: 600
    lineHeight: 1.05
    letterSpacing: "0"
  metric:
    fontFamily: "IBM Plex Sans, Aptos, Segoe UI, sans-serif"
    fontSize: "2.35rem"
    fontWeight: 700
    lineHeight: 1.05
    letterSpacing: "0"
  body:
    fontFamily: "IBM Plex Sans, Aptos, Segoe UI, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "0"
  label:
    fontFamily: "IBM Plex Sans, Aptos, Segoe UI, sans-serif"
    fontSize: "0.78rem"
    fontWeight: 800
    lineHeight: 1.2
    letterSpacing: "0"
  mono:
    fontFamily: "SFMono-Regular, Cascadia Code, Consolas, monospace"
    fontSize: "0.86rem"
    fontWeight: 400
    lineHeight: 1.45
    letterSpacing: "0"
rounded:
  sm: "6px"
  md: "8px"
  pill: "999px"
spacing:
  xs: "8px"
  sm: "10px"
  md: "16px"
  lg: "24px"
components:
  button-default:
    backgroundColor: "{colors.lab-bone}"
    textColor: "{colors.graphite-olive}"
    rounded: "{rounded.sm}"
    padding: "8px 13px"
    height: "38px"
  button-primary:
    backgroundColor: "{colors.graphite-olive}"
    textColor: "{colors.sage-paper}"
    rounded: "{rounded.sm}"
    padding: "8px 13px"
    height: "38px"
  input-field:
    backgroundColor: "{colors.input-bone}"
    textColor: "{colors.graphite-olive}"
    rounded: "{rounded.sm}"
    padding: "7px 9px"
    height: "42px"
  badge:
    backgroundColor: "#e3eeee"
    textColor: "{colors.instrument-teal}"
    rounded: "{rounded.pill}"
    padding: "6px 10px"
  panel:
    backgroundColor: "{colors.lab-bone}"
    textColor: "{colors.graphite-olive}"
    rounded: "{rounded.md}"
    padding: "24px"
  metric-tile:
    backgroundColor: "{colors.graphite-olive}"
    textColor: "{colors.sage-paper}"
    padding: "18px"
---

# Design System: WMCP LeWM Operations Console

## 1. Overview

**Creative North Star: "The Field Instrument"**

This is calibrated equipment, not a marketing dashboard. The console reads like a warm-enamel field
instrument: a steady housing in muted sage and bone, hairline gridlines, exact dials, and one teal
needle that moves when the system moves. Every surface exists to help an operator judge service
health or a request's correctness at a glance — is the backend real, is it ready, what is the live
latency and error profile, did the call return the expected tensor shapes. Decoration that does not
convey state is noise and is removed.

Its defining move is the refusal of the genre default. Observability tooling reaches for neon-on-
charcoal by reflex; this console answers in a warm, light, earth-toned register instead — sage paper,
lab bone, deep olive, terracotta clay, instrument teal. That warmth *is* the brand: trust carried by
material and precision, never by gradients, glass, or hype. The voice is terse and technical,
operator-to-operator. It shows the real metric name and the real unit; it does not paraphrase
`wmcp_request_latency_seconds` into something friendlier.

What it explicitly rejects: the generic dark-mode SaaS dashboard (stock Grafana skin, glowing lines
on black); the gradient/glassy "AI startup" aesthetic (purple gradients, glassmorphism, the
hero-metric stat-card cliché); the crowded Bootstrap-era enterprise admin (undifferentiated gray
tables, no hierarchy); and anything toy (oversized rounded corners, bright primary toy colors).

**Key Characteristics:**
- Warm, light, earth-toned — a deliberate escape from dark-mode observability.
- Hairline-ruled surfaces; depth from borders and tonal steps, not heavy shadow.
- One teal accent reserved for live signal, selection, and focus.
- Heavy weight confined to small uppercase micro-labels; body and data stay calm.
- Exact metric names and units everywhere; the data is the only ornament.

## 2. Colors

A warm earth palette: muted sage and bone neutrals carrying a small committee of saturated
instrument colors, each tied to a specific operational meaning.

### Primary
- **Instrument Teal** (`#176f72`): The single live-signal accent. Sparkline strokes, the selected
  badge, focus rings, and the "this is the moving part" cue. Reserved for signal, selection, and
  focus — never decoration.

### Secondary
- **Deep Olive** (`#4e6630`): Affirmative state and micro-labels — the eyebrow kicker and the `ok`
  response status. The calm, settled green.
- **Moss** (`#7b9652`): The lighter growth green. Sparkline area fill (as a transparency) and tonal
  tints mixed into dark metric tiles.

### Tertiary
- **Clay** (`#b45b42`): Warm terracotta. A rotating tint on dark metric tiles and a warm structural
  accent; never an alarm color.
- **Amber** (`#d79a27`): The one action highlight — the Refresh control. Draws the eye to the
  manual-poll affordance without shouting.

### Neutral
- **Graphite Olive** (`#22231e`): Primary ink for text, and the fill for dark surfaces (primary
  buttons, metric tiles). A warm near-black, never pure `#000`.
- **Sage Paper** (`#edf0ea`): The body background. A warm, desaturated sage-gray — the workbench
  surface the instruments rest on.
- **Lab Bone** (`#fcfbf4`): Panel and surface fill. A warm off-white that lifts panels a half-step
  off the sage paper.
- **Input Bone** (`#fefff9`): The brightest field — input, textarea, and chart canvas backgrounds,
  signalling "you write here / data renders here."
- **Slate Sage** (`#687068`): Muted text — secondary labels and table headers. Verified at 4.80:1 on
  the status surface and 4.93:1 on Lab Bone (clears the 4.5:1 floor); keep it on bone/paper tints, not
  on darker fills.
- **Hairline Sage** (`#c8d0c7`): The borders-and-dividers line. The ruling that defines every edge.
- **Quiet Sage** (`#e5e8dd`): The quietest tint, for recessed zones.

### Semantic
- **Signal Red** (`#b3384b`): Error state only — `error` response status, error-rate emphasis. Earns
  its saturation by being rare.

### Named Rules
**The One Needle Rule.** Instrument Teal is the only live-signal color. It marks what is moving or
selected — sparkline, current selection, focus. If teal is on more than a small fraction of a resting
screen, it has stopped meaning "signal" and become decoration. Pull it back.

**The Warm-Light Rule.** The background is warm and light, always. Reverting to charcoal/black to
"look like observability tooling" is forbidden — that regression erases the brand.

## 3. Typography

**Display Font:** Avenir Next (with IBM Plex Sans, sans-serif fallback)
**Body Font:** IBM Plex Sans (with Aptos, Segoe UI fallback)
**Label/Mono Font:** SFMono-Regular (with Cascadia Code, Consolas fallback)

**Character:** A humanist geometric display (Avenir Next) over a technical humanist sans (IBM Plex
Sans), with a true monospace for payloads and data — a contrast pairing, not two near-identical
sans. Plex carries the instrument's precision; Avenir gives the headings their composed, editorial
calm. The mono is where the operator reads truth: JSON, tensor shapes, raw response bodies.

### Hierarchy
- **Display** (Avenir Next, 600, `4rem`, line-height 0.95): The masthead H1 only ("LeWM serving
  telemetry"). Steps down structurally at breakpoints (`3.1rem` ≤1040px, `2.35rem` ≤720px) — fixed
  rem per breakpoint, not fluid clamp. Letter-spacing held at `0`; never tracked tighter.
- **Headline** (Avenir Next, 600, `1.45rem`, line-height 1.05): Panel titles ("Send WMCP calls",
  "Live service shape").
- **Metric** (IBM Plex Sans, 700, `2.35rem`): The big live numbers in the metrics grid. The
  data-display register — large, exact, set on the dark tiles.
- **Body** (IBM Plex Sans, 400, `1rem`, line-height 1.5): Default running text and response prose.
  Cap prose at 65–75ch; data rows and tables may run denser.
- **Label** (IBM Plex Sans, 800, `0.78rem`, uppercase): Micro-labels — eyebrow kicker, control
  labels, chart titles, table headers. The heavy weight is a deliberate, *confined* accent here.
- **Mono** (SFMono-Regular, 400, `0.86rem`, line-height 1.45): Request/response payloads, the JSON
  editor, log bodies. Where exactness is read.

### Named Rules
**The Confined-Weight Rule.** Weight 800 lives only on small uppercase micro-labels, where it reads
as machined precision. Headings stay at 600, body at 400, data at 700. Do not let 800/900 weights
spread into headings, prose, or large numbers — broadcast bolding is the opposite of refined.

**The Untracked-Display Rule.** Display letter-spacing stays at `0` and never goes below `-0.04em`.
This is a measured, calm headline, not a cramped fashion grotesque.

## 4. Elevation

Depth is **border-defined**: every surface is drawn by the 1px Hairline Sage ruling and separated
from neighbors by single-pixel gaps that read as fine seams. Surfaces are flat at rest. Shadow is
reserved for state — the response to hover, focus, or active lift — not painted under everything as
ambient decoration.

> **Direction note.** The current `styles.css` applies one large ambient shadow
> (`0 18px 45px rgb(35 45 38 / 12%)`) to panels, the status band, and the metrics grid *at rest*,
> paired with the 1px border. That always-on border-plus-wide-shadow pairing is the pattern to dial
> back: keep the hairline border as the resting edge and move the soft lift onto interaction.

### Shadow Vocabulary
- **Ambient lift** (`box-shadow: 0 18px 45px rgb(35 45 38 / 12%)`): A soft, dark-green-tinted float.
  Use it as a *state* response (panel on hover/focus, a raised control), not as the resting style of
  every surface.

### Named Rules
**The Flat-At-Rest Rule.** Surfaces are defined by the hairline border and tonal steps
(`sage-paper` → `lab-bone` → `input-bone` lighten as you move toward where the operator acts).
Shadows appear as feedback to state, then settle back. A screen at rest should read as ruled paper,
not floating cards.

## 5. Components

Components feel **refined and restrained**: crisp but quiet, precise without heaviness. Hairline
borders, modest 6px corners, one calm hover, no flourish.

### Buttons
- **Shape:** Lightly squared corners (6px radius); minimum height 38px. Full pill (999px) is for
  badges only, never buttons.
- **Default:** Lab Bone fill (`#fcfbf4`) with a 1px Graphite-Olive border and Graphite-Olive label.
- **Primary:** Inverted — Graphite-Olive fill (`#22231e`) with Sage-Paper label. One primary action
  per cluster ("Send").
- **Hover:** A single calm `translateY(-1px)` lift over 140ms ease. No shadow bloom, no color flip.
- **Focus:** 3px Instrument-Teal-tinted outline, 2px offset (shared focus treatment, below).
- **Disabled:** opacity 0.48, no lift, `not-allowed` cursor.

### Badges / Chips
- **Style:** Full pill (999px), a pale teal-tint background with Instrument-Teal text and a 1px
  hairline border. Used for inline status tags ("uri pixels", "waiting").
- **State:** Read-only status markers, not interactive filters.

### Cards / Panels
- **Corner Style:** 8px radius.
- **Background:** Lab Bone (`#fcfbf4`).
- **Border:** 1px Hairline Sage — the defining edge (see Elevation).
- **Shadow Strategy:** Flat at rest; ambient lift reserved for state (see Elevation).
- **Internal Padding:** `clamp(16px, 2vw, 24px)`.
- **Note:** Never nest a card inside a card. The status band and metrics grid are single ruled
  surfaces subdivided by 1px seams, not stacks of cards.

### Inputs / Fields
- **Style:** Input Bone fill (`#fefff9`), 1px Hairline Sage border, 6px radius, 42px control height.
  The textarea is the mono JSON editor (≥410px tall, resizable vertically).
- **Focus:** Shared ring — 3px outline in teal mixed 45% with white, 2px offset.
- **Error / Disabled:** Error states surface in Signal Red on the response side; pair color with the
  literal `error` word, never hue alone.

### Navigation
- **Style:** The masthead service links (Grafana, Prometheus) render as Default buttons — bone fill,
  hairline border, heavy label. Same 1px hover lift. On mobile they reflow to a two-column grid.

### Signature: Metrics Grid
- A seamless run of dark Graphite-Olive tiles separated by 1px gaps, each carrying a 0.78rem
  uppercase Moss-tinted label and a 2.35rem Metric number. Every fourth tile rotates a faint
  teal/olive/clay tint mixed into the ink so the row reads as one instrument cluster, not eight
  cards. This is the console's hero element — keep it dense, exact, and unboxed.

### Signature: Sparklines
- SVG line charts on an Input-Bone canvas with a faint graph-paper grid, a 3.4px Instrument-Teal
  stroke, and a Moss area fill at ~20% opacity. The teal needle is the live signal; the grid is the
  instrument face.

## 6. Do's and Don'ts

### Do:
- **Do** keep the background warm and light (Sage Paper `#edf0ea`); carry trust through material and
  precision, not effects.
- **Do** reserve Instrument Teal (`#176f72`) for live signal, selection, and focus — the One Needle.
- **Do** define surfaces with the 1px Hairline Sage (`#c8d0c7`) border and tonal steps; keep them
  flat at rest and let shadow respond to state.
- **Do** confine weight 800 to small uppercase micro-labels; headings at 600, body at 400.
- **Do** show real metric names and units (`wmcp_request_latency_seconds`, p95, `/s`) and exact
  tensor shapes. The data is the ornament.
- **Do** pair status color with a word or shape (`ok` / `error`) so meaning never rests on hue alone
  (WCAG 2.1 AA; colorblind-safe).
- **Do** keep `:focus-visible` rings on every control and honor `prefers-reduced-motion` — crossfade
  or cut instead of the hover lift.

### Don't:
- **Don't** revert to a **generic dark-mode SaaS dashboard** — neon-on-charcoal, stock Grafana skin,
  glowing lines on black. The warm light palette is the brand.
- **Don't** introduce the **gradient / glassy "AI startup" aesthetic** — purple gradients,
  glassmorphism, decorative blur, or the hero-metric stat-card cliché (giant number + tiny label +
  gradient accent).
- **Don't** ship the **crowded Bootstrap-era enterprise admin** look — undifferentiated gray tables,
  default form controls, no hierarchy.
- **Don't** go **toy** — no oversized rounded corners (cards stay 8px, never 24px+), no bright
  primary toy colors, no cartoon affordances.
- **Don't** pair a 1px border with a wide ambient shadow on resting surfaces (the ghost-card tell);
  pick the border at rest, the shadow on state.
- **Don't** let teal spread across a resting screen, and don't track the display heading below
  `-0.04em`.
- **Don't** nest cards, and don't paraphrase metric names into "friendlier" copy.
