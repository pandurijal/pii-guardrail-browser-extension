import { readFileSync } from 'fs';
import { join } from 'path';

/**
 * Drift guard for the design-token duplication between the canonical
 * stylesheet (`src/shared/styles/tokens.css`, consumed by the Svelte
 * popup/options and the overlay CSS) and the Shadow-DOM mirror
 * (`src/ui/shared/shadow-design-system.ts`).
 *
 * Shadow roots mounted from content scripts do not inherit the document
 * `:root` stylesheet, so a subset of tokens is hand-copied into a `:host`
 * block as a TS string. Until both sides are generated from one source,
 * this test keeps the copied values from silently diverging: change a
 * mirrored value in tokens.css and this fails until the shadow copy is
 * updated too.
 */

const ROOT = join(__dirname, '..', '..');
const TOKENS_CSS = readFileSync(
  join(ROOT, 'src/shared/styles/tokens.css'),
  'utf8',
);
const SHADOW_TS = readFileSync(
  join(ROOT, 'src/ui/shared/shadow-design-system.ts'),
  'utf8',
);

/**
 * Shadow `--pg-*` token → canonical `--*` token it must equal. Only
 * tokens whose values are intended to be byte-identical belong here;
 * shadow-only tokens (e.g. `--pg-color-warning`, `--pg-shadow-floating`)
 * and intentional divergences are handled separately below.
 */
const MIRRORED_TOKENS: Record<string, string> = {
  '--pg-color-header': '--color-header',
  '--pg-color-surface': '--color-surface',
  '--pg-color-card': '--color-card',
  '--pg-color-border': '--color-border-strong',
  '--pg-color-ink': '--color-ink',
  '--pg-color-muted': '--color-muted',
  '--pg-color-subtle': '--color-subtle',
  '--pg-color-accent': '--color-accent',
  '--pg-color-accent-soft': '--color-accent-soft',
  '--pg-color-success': '--color-success',
  '--pg-radius-sm': '--radius-sm',
  '--pg-radius-md': '--radius-md',
};

/**
 * Shadow tokens that share a canonical name (stripping the `pg-` prefix)
 * but intentionally hold a different value. Listed here so the
 * coverage check below doesn't flag them — and so any change is a
 * deliberate edit to this list.
 *
 * `--pg-font-sans` carries a broader fallback stack (BlinkMacSystemFont,
 * "Segoe UI") than the canonical `--font-sans` because it renders inside
 * arbitrary third-party pages rather than the extension's own surfaces.
 */
const KNOWN_DIVERGENCES = new Set<string>(['--pg-font-sans']);

/** First-wins map of `--name` → value, so a `:host`/`:root` declaration
 *  is captured ahead of any later theme override of the same token. */
function parseCustomProps(source: string): Map<string, string> {
  const out = new Map<string, string>();
  const re = /(--[\w-]+)\s*:\s*([^;]+);/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(source)) !== null) {
    const name = match[1];
    const value = match[2].replace(/\s+/g, ' ').trim();
    if (!out.has(name)) out.set(name, value);
  }
  return out;
}

const canonical = parseCustomProps(TOKENS_CSS);
const shadow = parseCustomProps(SHADOW_TS);

describe('shadow design-system token drift guard', () => {
  it('parses tokens from both sources', () => {
    expect(canonical.size).toBeGreaterThan(0);
    expect(shadow.size).toBeGreaterThan(0);
  });

  it.each(Object.entries(MIRRORED_TOKENS))(
    'shadow %s mirrors canonical %s',
    (shadowVar, canonicalVar) => {
      const shadowValue = shadow.get(shadowVar);
      const canonicalValue = canonical.get(canonicalVar);

      expect(shadowValue).toBeDefined();
      expect(canonicalValue).toBeDefined();
      expect(shadowValue).toBe(canonicalValue);
    },
  );

  it('registers every shadow token that shadows a canonical name', () => {
    const unregistered: string[] = [];
    for (const shadowVar of shadow.keys()) {
      if (!shadowVar.startsWith('--pg-')) continue;
      const canonicalName = `--${shadowVar.slice('--pg-'.length)}`;
      if (!canonical.has(canonicalName)) continue; // shadow-only token
      if (shadowVar in MIRRORED_TOKENS) continue;
      if (KNOWN_DIVERGENCES.has(shadowVar)) continue;
      unregistered.push(shadowVar);
    }

    // A token here looks like a copy of a canonical value but is neither
    // asserted equal (MIRRORED_TOKENS) nor flagged as a deliberate
    // divergence (KNOWN_DIVERGENCES). Add it to whichever applies.
    expect(unregistered).toEqual([]);
  });
});
