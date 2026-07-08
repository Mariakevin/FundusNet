---
name: FundusNet
description: AI-powered retinal disease screening for clinical technicians
colors:
  primary: "#C05621"
  primary-light: "#DD6B20"
  base: "#1A1A2E"
  base-light: "#2D2D44"
  success: "#2D6A4F"
  danger: "#9B2C2C"
  info: "#2B6CB0"
  accent-gold: "#B7791F"
  bg: "#F8F6F3"
  bg-alt: "#F0EDE8"
  surface: "#FFFFFF"
  surface-hover: "#FDFCFB"
  ink: "#1A1A2E"
  ink-secondary: "#4A5568"
  ink-muted: "#A0AEC0"
  border: "#E2DDD5"
  border-light: "#EDE9E3"
typography:
  display:
    fontFamily: "DM Serif Display, Georgia, serif"
    fontSize: "clamp(1.875rem, 4vw, 2.25rem)"
    fontWeight: 400
    lineHeight: 1.2
  body:
    fontFamily: "DM Sans, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "DM Sans, system-ui, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: "0.06em"
rounded:
  sm: "6px"
  md: "10px"
  lg: "14px"
  xl: "20px"
  full: "9999px"
spacing:
  xs: "0.25rem"
  sm: "0.5rem"
  md: "0.75rem"
  lg: "1rem"
  xl: "1.25rem"
  "2xl": "1.5rem"
  "3xl": "2rem"
  "4xl": "2.5rem"
  "5xl": "3rem"
  "6xl": "4rem"
  "8xl": "5rem"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "#ffffff"
    rounded: "{rounded.sm}"
    padding: "0.75rem 1.25rem"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink-secondary}"
    rounded: "{rounded.sm}"
    padding: "0.75rem 1.25rem"
  card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.lg}"
    padding: "1.5rem"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "0.75rem 1rem"
  badge-success:
    backgroundColor: "rgba(45, 106, 79, 0.08)"
    textColor: "{colors.success}"
    rounded: "{rounded.full}"
    padding: "0.25rem 0.75rem"
  badge-warning:
    backgroundColor: "rgba(245, 158, 11, 0.15)"
    textColor: "#D97706"
    rounded: "{rounded.full}"
    padding: "0.25rem 0.75rem"
---

# Design System: FundusNet

## 1. Overview

**Creative North Star: "The Precision Instrument"**

FundusNet's design language is clinical precision made visible. Every element communicates competence: dense information architecture, sharp typographic hierarchy, and a warm-but-serious palette that feels like a trusted medical device rather than a consumer app. The system rejects the sterility of hospital-blue interfaces and the casualness of mobile health apps in favor of something rarer — a tool that feels technically excellent and humanly warm at the same time.

The amber primary accent (#C05621) carries the warmth of fundus photography itself — the orange-red glow of a retinal scan. Paired with deep indigo (#1A1A2E) for authority and cream (#F8F6F3) for approachability, the palette bridges clinical trust and technical sophistication. DM Serif Display headings lend editorial weight; DM Sans body text keeps information dense but readable.

**Key Characteristics:**
- Information-dense but never cluttered — hierarchy through typography and spacing, not hiding
- Warm amber accents that evoke the fundus imagery the tool analyzes
- Clinical gravitas without coldness — serious but never intimidating
- Ensemble transparency — model status and confidence are always visible
- Refusal as a feature — uncertainty is communicated clearly, not buried

## 2. Colors

The palette is warm-forward with clinical precision: amber carries action, indigo carries authority, cream carries approachability.

### Primary
- **Fundus Amber** (#C05621): The signature accent. Used on primary buttons, active nav states, focus rings, confidence values, and upload zone highlights. Evokes the retinal imagery the system analyzes.
- **Fundus Amber Light** (#DD6B20): Hover state for primary elements. Slightly warmer and lighter.

### Neutral
- **Deep Indigo** (#1A1A2E): Text, brand icon background, headings. The authority anchor — dark enough for clinical gravitas, warm enough to avoid sterile-black.
- **Indigo Light** (#2D2D44): Secondary surfaces, avatar backgrounds. A softer echo of the base.
- **Warm Paper** (#F8F6F3): Page background. A deliberately warm off-white that avoids both the sterile-white hospital default and the overly-saturated cream AI default.
- **Linen** (#F0EDE8): Alt background, table headers, stat card hover states. One step warmer than the page bg.
- **Pure Surface** (#FFFFFF): Cards, modals, upload zones. Clean separation from the warm bg.
- **Ink Secondary** (#4A5568): Body text, descriptions, secondary labels. Warm gray with enough contrast for readability.
- **Ink Muted** (#A0AEC0): Placeholder text, timestamps, inactive labels. Use sparingly — never for body text.
- **Border** (#E2DDD5): Card borders, table dividers, input strokes. Warm-tinted, never cold gray.
- **Border Light** (#EDE9E3): Subtle dividers within cards and tables.

### Semantic
- **Forest Green** (#2D6A4F): Healthy status, success badges, positive confidence. Evokes "all clear" without the generic-green cliché.
- **Signal Red** (#9B2C2C): Abnormal findings, danger states, error alerts. Clinical urgency without alarmism.
- **Clinical Blue** (#2B6CB0): Informational alerts, info badges, secondary actions. Restrained — used on <5% of surfaces.
- **Diagnostic Gold** (#B7791F): Warning states, moderate confidence, needs-review badges. The middle ground between healthy and abnormal.

### Named Rules
**The Amber Accent Rule.** Fundus Amber (#C05621) appears on ≤15% of any given screen. Its rarity is the point — it draws the eye to exactly one action or data point per view. When everything is amber, nothing is.

**The Warmth Constraint.** All neutrals carry a slight warm tint (hue 40-100 in OKLCH). Cold grays (#888, #999, #ccc) are prohibited. If a neutral feels too warm, adjust lightness — not hue.

## 3. Typography

**Display Font:** DM Serif Display (with Georgia, serif fallback)
**Body Font:** DM Sans (with system-ui, sans-serif fallback)

**Character:** The serif/sans pairing creates clinical authority without stuffiness. DM Serif Display's low-contrast strokes feel editorial and precise — like a medical journal heading. DM Sans is warm geometric with open counters — technical enough for data density, human enough for long reading sessions.

### Hierarchy
- **Display** (400 weight, clamp(1.875rem, 4vw, 2.25rem), 1.2 line-height): Page titles, hero headings. Used once per page maximum.
- **Title** (400 weight, 1.5rem, 1.2): Section headings within cards. Serif for visual hierarchy separation from body.
- **Subtitle** (400 weight, 1.125rem, 1.6): Page descriptions, secondary headings. Body font, slightly larger.
- **Body** (400 weight, 1rem, 1.6): All readable content. Max line length: 65-75ch. Color: ink-secondary (#4A5568).
- **Label** (600 weight, 0.75rem, 0.06em spacing, uppercase): Table headers, stat labels, eyebrow text. Used for structural hierarchy, never for body copy.

### Named Rules
**The Serif Reserve Rule.** DM Serif Display is reserved for headings and display text only. Never use it for body copy, labels, or interactive elements. The serif's authority comes from its scarcity.

**The Muted Text Floor.** Muted text (#A0AEC0) is for timestamps and inactive states only. Any text the user must read — body copy, descriptions, instructions — uses ink-secondary (#4A5568) or darker. If contrast is even close, bump toward ink — light gray "for elegance" is the most common readability failure.

## 4. Elevation

The system uses a subtle, ambient shadow vocabulary rather than dramatic lifts. Shadows are structural — they separate surfaces (cards from bg, modals from page) rather than creating depth theater. The warm-tinted shadow color (rgba with indigo base) keeps shadows from feeling cold or harsh.

### Shadow Vocabulary
- **Subtle** (`0 1px 3px rgba(26,26,46,0.04), 0 1px 2px rgba(26,26,46,0.03)`): Resting cards, table rows. Barely perceptible — creates separation without weight.
- **Standard** (`0 4px 12px rgba(26,26,46,0.06), 0 2px 4px rgba(26,26,46,0.03)`): Hovered cards, dropdown menus. Visible lift on interaction.
- **Medium** (`0 10px 24px rgba(26,26,46,0.07), 0 4px 8px rgba(26,26,46,0.03)`): Modals, processing cards, elevated panels. Clear surface separation.
- **Large** (`0 24px 48px rgba(26,26,46,0.09)`): Auth cards, primary dialogs. Maximum structural lift.

### Named Rules
**The Flat-By-Default Rule.** Surfaces are flat at rest with subtle shadows. Shadows intensify only on state change (hover, focus, modal). If a shadow is visible on a resting card, it's too dark.

## 5. Components

### Buttons
- **Shape:** Gently curved edges (6px radius). Not pill-round, not sharp-square.
- **Primary:** Fundus Amber background, white text, 12px 20px padding. Used for the single primary action per view (New Analysis, Start Analysis, Get Started).
- **Secondary/Default:** White surface background, ink-secondary text, 1px border. Used for secondary actions (View All, Monitor, Cancel).
- **Danger:** Signal Red background, white text. Used for destructive actions (Delete, Sign Out icon).
- **Ghost/Icon:** Transparent background, 34px square, border on hover. Used in navbar (sign out) and table actions.
- **Hover:** Background shifts one step warmer (amber → amber-light, white → bg-alt). Border gains amber tint.
- **Focus:** 3px amber outline with 2px offset. Always visible, never hidden.

### Cards
- **Corner Style:** Gently curved (14px radius).
- **Background:** Pure white (#FFFFFF) on warm paper bg (#F8F6F3).
- **Shadow Strategy:** Subtle at rest, standard on hover. Cards lift slightly on interaction.
- **Border:** 1px solid warm border (#E2DDD5). Separates from bg without heaviness.
- **Internal Padding:** 1.5rem (24px) standard, 2rem (32px) for auth/processing cards.

### Inputs / Fields
- **Style:** 1.5px stroke border, white background, 6px radius. Warm border color (#E2DDD5) at rest.
- **Focus:** Border shifts to Fundus Amber with 3px amber glow ring. Clear, unmissable.
- **Error:** Border shifts to Signal Red. Error text below in same red.
- **Placeholder:** Muted text (#A0AEC0). Never body text color — always clearly distinguishable from real content.

### Navigation
- **Style:** Sticky top bar, white surface, subtle bottom border. Brand icon in indigo square (38px, 10px radius).
- **Typography:** DM Sans 500 weight, 0.875rem. Active state gains amber background tint + amber text + amber border.
- **Hover:** Background shifts to amber-bg (8% opacity amber). Subtle, not loud.
- **User Area:** Pill-shaped container with avatar initial, username, and sign-out icon.
- **Mobile:** Wraps naturally. Links stack below brand. No hamburger — all links visible.

### Badges / Chips
- **Success:** Forest green text on forest green 8% background, 1px forest border. Used for Healthy status, loaded models.
- **Warning:** Amber text on amber 15% background, 1px amber border. Used for Cataract/Glaucoma, uncertainty states.
- **Danger:** Red text on red 8% background. Used for abnormal findings, offline models.
- **Shape:** Full pill (9999px radius). Always inline-flex with icon + text.

### Tables
- **Header:** Uppercase labels, 0.06em tracking, muted text, warm bg background. Structural hierarchy, not data.
- **Rows:** Hover state shifts to warm bg. Bottom border on each row (border-light).
- **Image Thumbnails:** 42px square, 6px radius, object-fit cover. Consistent across all table rows.

### Upload Zone
- **Style:** Dashed border (2px), 20px radius. Generous padding (5rem). Centered content.
- **Hover:** Border shifts to amber, background gains amber tint (8% opacity). Clear affordance.
- **Icon:** 68px amber-bg circle with 32px amber icon. Large enough to be unmistakable.

### Processing / Loading
- **Style:** Centered card, generous padding (3rem). Amber-bg circle with spinner.
- **Spinner:** 22px, 2.5px border, amber top-color. Subtle, not flashy.
- **Status Text:** Small, secondary color. Updates dynamically during processing.

## 6. Do's and Don'ts

### Do:
- **Do** use Fundus Amber (#C05621) as the single accent color — it appears on ≤15% of any screen.
- **Do** use DM Serif Display for headings only — never body copy, labels, or buttons.
- **Do** use uppercase tracked labels (0.06em spacing) for structural hierarchy — table headers, stat labels, section eyebrows.
- **Do** show model status, confidence percentages, and uncertainty estimates prominently — ensemble transparency is the core value.
- **Do** use warm-tinted shadows (indigo-based rgba) — never cold gray shadows.
- **Do** communicate refusal clearly — "The model is uncertain" with amber warning styling, not buried in fine print.
- **Do** use the fundus-glow radial gradient as a signature element on hero sections — it evokes the retinal imagery.
- **Do** maintain 4.5:1 contrast on all body text — bump toward ink if close.

### Don't:
- **Don't** use SaaS-dashboard aesthetics — no generic metric-card templates, no gradient accent bars, no hero-metric big-number-small-label layouts.
- **Don't** use cold-clinical medical UI — no sterile hospital-blue palettes, no harsh white backgrounds, no generic "healthcare" visual language.
- **Don't** use consumer-app casualness — no playful illustrations, no rounded-everything mobile-app feel, no whimsical empty states.
- **Don't** use gradient text (background-clip: text) — decorative, never meaningful. Use solid colors.
- **Don't** use glassmorphism — blurs and glass cards are decorative, not functional.
- **Don't** use identical card grids — same-sized cards with icon + heading + text repeated endlessly.
- **Don't** use border-left or border-right greater than 1px as a colored accent stripe — rewrite with background tints or nothing.
- **Don't** use cold grays (#888, #999, #ccc) for any neutral — all neutrals carry warm tint.
- **Don't** use DM Serif Display below 1.5rem — it loses legibility at small sizes.
- **Don't** animate layout properties unless truly needed — motion is for feedback, not decoration.
