# Product

## Register

product

## Users

Medical technicians performing retinal screening in clinical settings. They upload fundus images and receive AI-assisted classifications (Healthy, Cataract, Glaucoma, Retina Disease) with confidence scores, Grad-CAM explainability, and uncertainty estimates. The primary workflow is: upload image → review result → decide if escalation to ophthalmologist is needed. Speed and accuracy are critical — technicians screen multiple patients per session and need quick, trustworthy AI triage to prioritize manual review.

## Product Purpose

FundusNet provides automated diabetic retinopathy screening through deep learning ensemble inference. It exists to accelerate clinical decision-making by giving technicians a fast, reliable second opinion on fundus images. Success means: fewer missed diagnoses, faster throughput, and confident escalation decisions. The system deliberately refuses classification when uncertain rather than guessing — clinical safety over completeness.

## Brand Personality

Technical excellence. Sophisticated, modern, AI-forward — showcases the engineering behind the diagnosis. The interface should feel like a precision instrument: dense with information but never cluttered, fast but never rushed, technical but never opaque. Voice: confident, precise, clinical authority without coldness. Three words: precise, sophisticated, trustworthy.

## Anti-references

- **No SaaS-dashboard aesthetic**: No generic metric-card templates, no gradient accent bars, no hero-metric layouts with big-number-small-label. This is a medical tool, not a B2B product page.
- **No cold-clinical medical UI**: No sterile hospital-blue palettes, no harsh white backgrounds, no generic "healthcare" visual language. The warmth of the current amber/cream system is intentional.
- **No consumer-app casualness**: No playful illustrations, no rounded-everything mobile-app feel, no whimsical empty states. This needs clinical gravitas — technicians trust tools that feel serious.
- **No AI slop tells**: No gradient text, no glassmorphism, no identical icon-heading-text card grids, no numbered section markers on every section.

## Design Principles

1. **Information density with clarity**: Every pixel carries meaning. Show model status, confidence levels, and uncertainty — but never overwhelm. Use hierarchy, not hiding.
2. **Refusal over false confidence**: The system says "I don't know" when uncertain. The interface must make this feel like a feature, not a failure — clinical safety is the product.
3. **Ensemble transparency**: Show which models agree, which disagree, and why. The multi-model architecture isn't hidden — it's the core value proposition.
4. **Speed is trust**: Fast inference means fast results. The UI must never feel like it's slowing down the clinical workflow. Loading states exist but don't linger.
5. **Clinical precision in every detail**: From confidence percentages to preprocessing visualizations, the interface communicates precision. Numbers matter, labels matter, accuracy matters.

## Accessibility & Inclusion

- WCAG AA compliance (4.5:1 body text, 3:1 large text)
- Full keyboard navigation with visible focus states
- Screen reader support with ARIA labels on all interactive elements
- Reduced motion support via `prefers-reduced-motion`
- Skip-to-content link for keyboard users
- Semantic HTML structure with proper heading hierarchy
- Color is never the sole indicator of status — always paired with text or icons
