---
name: RankAI Structural System
colors:
  surface: '#f7faf6'
  surface-dim: '#d7dbd7'
  surface-bright: '#f7faf6'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f1f4f1'
  surface-container: '#ebefeb'
  surface-container-high: '#e5e9e5'
  surface-container-highest: '#e0e3e0'
  on-surface: '#181d1a'
  on-surface-variant: '#3f4944'
  inverse-surface: '#2d312f'
  inverse-on-surface: '#eef2ee'
  outline: '#6f7a74'
  outline-variant: '#bec9c3'
  surface-tint: '#086b53'
  primary: '#005440'
  on-primary: '#ffffff'
  primary-container: '#0f6e56'
  on-primary-container: '#9aedcf'
  inverse-primary: '#84d6b9'
  secondary: '#595e6f'
  on-secondary: '#ffffff'
  secondary-container: '#dbdff4'
  on-secondary-container: '#5d6274'
  tertiary: '#78352b'
  on-tertiary: '#ffffff'
  tertiary-container: '#954c41'
  on-tertiary-container: '#ffd3cc'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#a0f3d4'
  primary-fixed-dim: '#84d6b9'
  on-primary-fixed: '#002117'
  on-primary-fixed-variant: '#00513e'
  secondary-fixed: '#dee2f7'
  secondary-fixed-dim: '#c1c6da'
  on-secondary-fixed: '#161b2a'
  on-secondary-fixed-variant: '#414657'
  tertiary-fixed: '#ffdad4'
  tertiary-fixed-dim: '#ffb4a8'
  on-tertiary-fixed: '#3b0804'
  on-tertiary-fixed-variant: '#743329'
  background: '#f7faf6'
  on-background: '#181d1a'
  surface-variant: '#e0e3e0'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  sidebar_width: 220px
  gutter: 1.5rem
  margin_mobile: 1rem
  margin_desktop: 2rem
  stack_sm: 0.5rem
  stack_md: 1rem
  stack_lg: 1.5rem
---

## Brand & Style
The design system focuses on high-utility, professional efficiency for HR and talent acquisition. It delivers a "Modern-Analytical" aesthetic—combining the precision of enterprise software with the approachability of a contemporary SaaS tool. 

The visual narrative is driven by **Minimalism** and **Flat Design**. To ensure the AI-generated data feels trustworthy rather than opaque, the UI prioritizes extreme legibility, generous whitespace, and a strict absence of decorative elements like gradients or heavy shadows. The interface uses high-contrast structural blocks to separate navigation, data visualization, and deep-dive candidate profiles.

## Colors
The palette is rooted in a warm, sophisticated off-white background to reduce eye strain during long sessions. 

- **Primary (#0F6E56):** A deep teal used for primary actions, success states, and key data highlights. It suggests growth and stability.
- **Secondary/Surface (#1A1F2E):** A dark navy reserved for high-level navigation (Sidebar) and primary headings to provide strong visual anchoring.
- **System States:** Amber is utilized for manual review triggers, while Red is strictly reserved for bias flags and critical system errors.
- **Neutrals:** Use `#E8E6E1` for borders and `#F1EFED` for alternating table row tints to maintain the warm tone of the background.

## Typography
This design system utilizes **Inter** exclusively to leverage its systematic, neutral, and highly legible qualities. 

Headings are set in the dark navy secondary color with tighter letter spacing for a compact, professional look. Body text should maintain a 1.5x line-height ratio to ensure maximum readability in data-heavy tables. Labels and metadata use a slightly heavier weight (600) at smaller sizes to remain prominent within complex dashboard layouts.

## Layout & Spacing
The layout follows a **Fixed-Fluid** hybrid model. A fixed left sidebar at 220px provides persistent navigation, while the main content area utilizes a fluid grid that expands to fill the viewport.

- **Grid:** Use a 12-column layout for the main dashboard content.
- **Breakpoints:** Mobile (<768px) collapses the sidebar into a drawer; Tablet (768px-1280px) reduces horizontal margins; Desktop (>1280px) utilizes full spacing.
- **Rhythm:** An 8px base unit governs all padding and margins. Vertical stacking of cards and sections should follow the `stack_lg` (24px) spacing to maintain the "generous whitespace" brand requirement.

## Elevation & Depth
In alignment with the **Flat Design** style, this design system avoids shadows entirely. Depth is communicated through **Tonal Layers** and **Low-contrast Outlines**:

- **Level 0 (Base):** Background color `#F8F7F4`.
- **Level 1 (Cards/Containers):** Pure white `#FFFFFF` with a 1px solid border in `#E8E6E1`.
- **Interactive States:** Hover states on table rows use a subtle background tint of `#F1EFED` and a 4px primary teal left-accent border to indicate selection/focus.
- **Navigation:** The sidebar uses the dark navy `#1A1F2E` to create a stark, "recessed" vertical plane.

## Shapes
The shape language is **Soft (0.25rem)**. This provides a subtle modern touch without feeling overly "bubbly" or consumer-oriented. 

- **Small elements:** Checkboxes, tags, and buttons use 4px (`0.25rem`) corner radii.
- **Large elements:** Metric cards and data tables use 8px (`0.5rem`) corner radii for the outer container.
- **Status Pills:** Verdict badges and chips use a "Pill" style (full rounding) to differentiate them from functional buttons.

## Components
- **Buttons:** 
  - **Primary:** Solid teal (`#0F6E56`) with white text. No shadow, flat color.
  - **Secondary/Ghost:** Transparent background with teal border and teal text.
- **Data Tables:**
  - Headers are uppercase `label-md` in navy. 
  - Alternate row colors (`#F1EFED`) starting from the second row.
  - 4px Teal accent bar on the left edge of a row during hover.
- **Metric Cards:**
  - White background, 1px `#E8E6E1` border.
  - Large `display-lg` numbers for key stats.
- **Badges (Verdicts):** 
  - **High Match:** Deep teal background, white text.
  - **Potential:** Light teal background, dark teal text.
  - **Review:** Amber background, dark navy text.
- **Input Fields:** 
  - White fill, 1px `#E8E6E1` border. On focus, the border thickens to 2px in teal.
- **Sidebar Items:** 
  - Dark navy background. Active items feature a subtle teal tint and a vertical bar on the left.