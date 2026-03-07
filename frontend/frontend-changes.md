# Frontend Changes - Dark/Light Theme Toggle Feature

## Summary
Added a dark/light theme toggle feature that allows users to switch between themes with a smooth transition animation. The theme preference is persisted in localStorage.

## Changes Made

### 1. index.html
- Added theme toggle button with sun and moon SVG icons in the top-right corner
- Updated CSS cache-busting version from `?v=12` to `?v=13`
- Updated JS cache-busting version from `?v=11` to `?v=12`

**New elements added:**
- Theme toggle button (`#themeToggle`) with accessibility attributes
- Sun icon SVG (visible in light mode)
- Moon icon SVG (visible in dark mode)

### 2. style.css
- **Added CSS variables for light theme** under `[data-theme="light"]` selector
- **Added smooth transition animations** for all theme-affected elements
- **Added theme toggle button styles** with hover and focus states
- **Updated existing elements** to use CSS variables instead of hardcoded colors

**Key CSS additions:**

#### New CSS Variables (Light Theme)
```css
[data-theme="light"] {
    --background: #f8fafc;
    --surface: #ffffff;
    --surface-hover: #f1f5f9;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border-color: #e2e8f0;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --code-bg: rgba(0, 0, 0, 0.05);
    --error-bg: rgba(239, 68, 68, 0.05);
    --success-bg: rgba(34, 197, 94, 0.05);
    --toggle-bg: #e2e8f0;
    --toggle-hover: #cbd5e1;
    --toggle-icon: #475569;
}
```

#### Theme Toggle Button Styles
- Positioned absolutely in top-right corner
- 44x44px circular button with border
- Smooth icon transitions using opacity and transform
- Accessible with proper focus states
- Responsive sizing for mobile devices

#### Smooth Transitions
Added `transition` property to key elements:
- `body`, `.container`, `.sidebar`, `.chat-messages`, etc.
- 0.3s ease transitions for `background-color`, `color`, `border-color`, `box-shadow`

### 3. script.js
- **Added theme management functions:**
  - `initializeTheme()` - Loads saved theme from localStorage on page load
  - `setTheme(theme)` - Applies theme by setting data-theme attribute
  - `toggleTheme()` - Toggles between light and dark themes

- **Added event listener** for theme toggle button click

- **Theme persistence** using localStorage with key `'course-assistant-theme'`

**New code added:**
```javascript
// Theme management
const THEME_KEY = 'course-assistant-theme';

function initializeTheme() {
    const savedTheme = localStorage.getItem(THEME_KEY) || 'dark';
    setTheme(savedTheme);
}

function setTheme(theme) {
    if (theme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
    } else {
        document.documentElement.removeAttribute('data-theme');
    }
    localStorage.setItem(THEME_KEY, theme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
}
```

## Design Decisions

1. **Default Theme:** Dark theme is the default (no data-theme attribute needed)
2. **Storage Key:** `'course-assistant-theme'` used for localStorage
3. **Icon Animation:** CSS-based icon switching with smooth opacity/transform transitions
4. **Accessibility:** Button includes `aria-label` and keyboard-navigable focus states
5. **Positioning:** Toggle button is positioned absolutely in top-right corner, above the main content

## Browser Compatibility
- Uses CSS custom properties (variables) - supported in all modern browsers
- Uses localStorage API - supported in all modern browsers
- Smooth transitions degrade gracefully in older browsers
