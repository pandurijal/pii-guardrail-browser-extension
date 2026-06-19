import { classifyBrowserMemory, decideSystemCompatibility } from '../../src/shared/system-compatibility-policy';

describe('system compatibility policy', () => {
  test.each([
    [undefined, 'unknown'],
    [0, 'unknown'],
    [2, 'critical'],
    [2.1, 'warning'],
    [4, 'warning'],
    [4.1, 'ok'],
    [8, 'ok'],
    [16, 'ok'],
  ] as const)('classifies browser-reported memory %s as %s', (memory, tier) => {
    expect(classifyBrowserMemory(memory)).toBe(tier);
  });

  test('WebGPU availability never changes the memory tier or auto-disable recommendation', () => {
    const available = decideSystemCompatibility({ browserMemoryGb: 2, webGpu: 'available' });
    const unavailable = decideSystemCompatibility({ browserMemoryGb: 2, webGpu: 'unavailable' });

    expect(available.tier).toBe('critical');
    expect(unavailable.tier).toBe('critical');
    expect(available.recommendation).toBe('auto-disable-local-ai');
    expect(unavailable.recommendation).toBe('auto-disable-local-ai');
  });

  test('CPU cores are not part of the compatibility decision', () => {
    const decision = decideSystemCompatibility({ browserMemoryGb: 16, webGpu: 'unknown' });

    expect(decision.tier).toBe('ok');
    expect(decision.notes.join(' ')).not.toMatch(/CPU core/i);
  });
});
