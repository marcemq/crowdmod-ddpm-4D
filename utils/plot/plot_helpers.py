import itertools
import re
 
NBSP = "\u00a0"  # non-breaking space: matplotlib keeps it when measuring text
 
# Long-name patterns, e.g. "..._Conic_iHeun150" (or "..._Conic_intgHeun150")
_FM_RE = re.compile(r'(Linear|Conic)_i(?:ntg)?(Euler|Heun)(\d*)')
 
_W_SHORT     = {'Linear': 'Li', 'Conic': 'Co'}
_INTEG_SHORT = {'Euler': 'Eu', 'Heun': 'He'}
_W_ORDER     = {'Linear': 0, 'Conic': 1}
_INTEG_ORDER = {'Euler': 0, 'Heun': 1}

def make_short_name(long_name: str) -> str:
    """
    Derive a short plot label from a long model directory name.
    """
    s = long_name
    s = s.replace('DDPM-UNet', 'DIF-U')
    s = s.replace('FM-UNet',   'FM-U')
    s = s.replace('ConvRNN',   'Conv')
    s = re.sub(r'sDDIMdiv(\d+)', r'DDIM_D\1', s)
    s = s.replace('gSparsity', 'gS')
    s = s.replace('gNone',     'gN')
    s = s.replace('GRUCell',   'GRU')
    s = s.replace('LSTMCell',  'LSTM')
    s = _FM_RE.sub(
        lambda m: f"{_W_SHORT[m.group(1)]}_{_INTEG_SHORT[m.group(2)]}{m.group(3)}", s
    )
    s = re.sub(r'_+', '_', s).strip('_')
    return s

def _guidance_rank(name: str) -> int:
    if 'gNone' in name:
        return 0
    if 'gSparsity' in name:
        return 1
    return 2

def model_sort_key(long_name: str):
    """
    Sort key that gives a stable, readable order (all keys have the same shape):
      0. ConvRNN
      1. DDPM sampler (no guidance first)
      2. DDIM sampler, by divider ascending (D2, D4, ..., D300)
      3. FM: weight type (Linear, Conic) -> integrator (Euler, Heun)
         -> number of integration steps ascending (25, 50, 75, ...)
      9. anything else, alphabetical
    """
    fm = _FM_RE.search(long_name)
    if fm:
        w, integ, steps = fm.groups()
        return (3, _W_ORDER.get(w, 9), _INTEG_ORDER.get(integ, 9), int(steps or 0), long_name)
 
    ddim = re.search(r'sDDIMdiv(\d+)', long_name)
    if ddim:
        return (2, 0, _guidance_rank(long_name), int(ddim.group(1)), long_name)
 
    if 'sDDPM' in long_name:
        return (1, 0, _guidance_rank(long_name), 0, long_name)
 
    if long_name.startswith('ConvRNN'):
        return (0, 0, 0, 0, long_name)
 
    return (9, 0, 0, 0, long_name)

# Backward-compatible alias (old name used by comparison_models_plot.py)
ddim_sort_key = model_sort_key
 
 
def pad_labels(long_names) -> dict:
    """
    Build {long_name: padded_short_label} where every label has the same length:
      - numbers are zero-padded per position (He25 -> He025, D2 -> D002)
      - remaining length differences are filled with non-breaking spaces (right side)
 
    Use together with a monospace font so all labels take exactly the same width.
    """
    longs  = list(long_names)
    shorts = [make_short_name(n) for n in longs]
 
    # widest number at each "n-th number in the label" position
    max_w = {}
    for s in shorts:
        for i, d in enumerate(re.findall(r'\d+', s)):
            max_w[i] = max(max_w.get(i, 0), len(d))
 
    def zero_pad(s):
        counter = itertools.count()
        return re.sub(r'\d+', lambda m: m.group().zfill(max_w[next(counter)]), s)
 
    shorts = [zero_pad(s) for s in shorts]
    width  = max(len(s) for s in shorts)
    return {n: s.ljust(width, NBSP) for n, s in zip(longs, shorts)}