# Dual Theme (Minimal / Synthwave) Design Spec
**Date:** 2026-03-11
**Status:** Approved

## Overview

Add a second "clean white AI startup" theme to Hand Magic and a toggle button in the header that switches between it and the existing retro synthwave theme using a scanline wipe animation.

---

## 1. Themes

### Synth (existing, default)
No changes. All current CSS is the synth theme.

### Minimal
- Background: `#ffffff`
- Text: `#0a0a0a`
- Subtext: `#6b7280`
- Borders: `#e5e5e5`
- Card background: `#fafafa`
- Accent: `#0a0a0a` (black buttons, black borders)
- Font: Inter (Google Fonts), fallback system-ui
- No glows, no box-shadows, no CRT overlay, no perspective grid
- Buttons: clean black border, hover = black fill + white text
- Inputs: white bg, gray border, gray focus ring

---

## 2. Theme Switching

### Attribute
`data-theme` on `<html>`: `"synth"` (default) or `"minimal"`.

All minimal overrides live in `[data-theme="minimal"] { ... }` blocks in `main.css`.

### Persistence
`localStorage.getItem("hm-theme")` — read on page load, applied before first paint to avoid flash.

---

## 3. Toggle Button

Location: top-right of the header, inline with the credit text.

Labels:
- When in synth mode: `[ MINIMAL ]`
- When in minimal mode: `[ ✦ SYNTH ]`

Styling: matches the active theme (black border in minimal, cyan border in synth).

---

## 4. Scanline Wipe Transition

A `<div id="theme-wipe">` is always in the DOM, fixed-position, full-screen, pointer-events none, hidden by default.

On toggle click:
1. `theme-wipe` becomes visible — fills with repeating horizontal lines (CSS `repeating-linear-gradient`)
2. CSS animation slides it **down** from `translateY(-100%)` to `translateY(0)` over 300ms
3. At 300ms midpoint: swap `data-theme` attribute
4. Continue animation sliding `theme-wipe` from `translateY(0)` to `translateY(100%)` over 300ms
5. At 600ms: hide `theme-wipe`, animation done

Implemented entirely in CSS + a small JS function. No libraries.

---

## 5. Files Changed

| File | Change |
|---|---|
| `app/static/css/main.css` | Add Inter font import, `[data-theme="minimal"]` overrides, `#theme-wipe` styles + keyframe |
| `app/templates/index.html` | Add `data-theme` to `<html>`, add toggle button, add `#theme-wipe` div, add theme init script |
| `app/static/js/studio.js` | No changes needed |
| `app/static/js/generate.js` | No changes needed |

No backend changes required.
