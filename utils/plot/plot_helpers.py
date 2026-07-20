import re

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
    s = s.replace('Linear_intgEuler', 'LpEi')
    s = re.sub(r'_+', '_', s).strip('_')
    return s


def ddim_sort_key(long_name: str):
    """
    Sort DDIM models by divider number, non-DDIM models go last.
    """
    match = re.search(r'sDDIMdiv(\d+)', long_name)
    if match:
        return (0, int(match.group(1)))
    return (1, long_name)