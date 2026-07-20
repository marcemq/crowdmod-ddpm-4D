import argparse, json, time
from pathlib import Path
from utils.myparser import getYamlConfig
from utils.plot.plot_helpers import make_short_name, ddim_sort_key

try:
    from PIL import Image, ImageSequence
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

FRAMES_CACHE_DIRNAME = ".frames_cache"


def discover_models_exp(raw_metrics_dir: Path, subfolder: str):
    """
    Scan raw_metrics_dir for model subdirectories that contain a
    `subfolder` (default: fixed_past_samples) with mprops_seq_n.gif,
    the predicted mprops sequences ready to be compared.
    """
    models = []
    for model_dir in sorted(raw_metrics_dir.iterdir(), key=lambda p: ddim_sort_key(p.name)):
        if not model_dir.is_dir():
            continue
        seq_dir = model_dir / subfolder
        if not seq_dir.is_dir():
            continue
        if not (seq_dir / "mprops_seq_1.gif").exists():
            continue

        label = model_dir.name.replace('_mE000', '')
        models.append({
            "long_name":  label,
            "short_name": make_short_name(label),
            "dir":        model_dir.name,
            "subfolder":  subfolder,
        })
    return models


def resolve_selected_models(all_models: list, requested_dirs: list) -> list:
    """
    Filter/reorder all_models (as discovered on disk) down to the
    dirs named in requested_dirs, preserving requested_dirs' order.
    Warns (does not fail) on names that don't match any discovered dir.
    """
    by_dir = {m["dir"]: m for m in all_models}
    selected = []
    for name in requested_dirs:
        if name in by_dir:
            selected.append(by_dir[name])
        else:
            print(f"  [warn] '{name}' not found among discovered model dirs, skipping.")
    if not selected:
        available = "\n".join(f"  - {m['dir']}" for m in all_models)
        raise SystemExit(
            "None of the requested models matched a discovered model dir.\n"
            f"Available model dirs under raw-metrics-dir:\n{available}"
        )
    return selected


def count_available_seqs(raw_metrics_dir: Path, models: list, max_check: int = 200) -> int:
    """
    Determine how many mprops_seq_N.gif exist (assumes all discovered
    models were sampled with the same NSAMPLES4PLOTS/from_fixed_past run,
    so we just probe the first model).
    """
    if not models:
        return 0
    seq_dir = raw_metrics_dir / models[0]["dir"] / models[0]["subfolder"]
    n = 0
    for i in range(1, max_check + 1):
        if (seq_dir / f"mprops_seq_{i}.gif").exists():
            n = i
        else:
            break
    return n


# ------------------------------------------------------------------ #
# Frame extraction: swap the "animated GIF" for a set of real PNG
# frames on disk, so the viewer can jump to an exact frame instantly
# via a plain <img src=...>, without any client-side GIF decoding
# (which would need fetch()'ing raw bytes -- blocked by Chrome for
# file:// pages, exactly how this HTML is normally opened).
# ------------------------------------------------------------------ #
def gif_path(seq_dir: Path, seq: int, gt: bool) -> Path:
    fname = f"mprops_GT_seq_{seq}.gif" if gt else f"mprops_seq_{seq}.gif"
    return seq_dir / fname


def frame_cache_dir(seq_dir: Path, seq: int, gt: bool) -> Path:
    name = f"mprops_GT_seq_{seq}" if gt else f"mprops_seq_{seq}"
    return seq_dir / FRAMES_CACHE_DIRNAME / name


def count_frames(main_dir: Path, models: list) -> int:
    """
    Number of frames per sequence GIF (assumes PAST_LEN+FUTURE_LEN is
    consistent across all sequences for a given sampling run, so we
    just probe model #1's first sequence).
    """
    if not HAVE_PIL or not models:
        return 0
    seq_dir = main_dir / models[0]["dir"] / models[0]["subfolder"]
    probe = gif_path(seq_dir, 1, gt=False)
    if not probe.exists():
        return 0
    with Image.open(probe) as img:
        return img.n_frames


def extract_frames(gif_file: Path, out_dir: Path) -> int:
    """
    Extract every frame of gif_file into out_dir as frame_00.png,
    frame_01.png, ... Skips extraction if out_dir already looks
    populated (idempotent, same spirit as the rest of the data
    pipeline). Returns the frame count, or 0 if gif_file is missing.
    """
    if not gif_file.exists():
        return 0
    if (out_dir / "frame_00.png").exists():
        return len(list(out_dir.glob("frame_*.png")))

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with Image.open(gif_file) as img:
        for i, frame in enumerate(ImageSequence.Iterator(img)):
            frame.convert("RGBA").save(out_dir / f"frame_{i:02d}.png")
            count = i + 1
    return count


def extract_all_frames(main_dir: Path, models: list, num_seqs: int, num_frames_expected: int) -> None:
    total = len(models) * num_seqs * 2  # prediction + GT per (model, seq)
    done, skipped, mismatched = 0, 0, 0
    t0 = time.time()

    for m in models:
        seq_dir = main_dir / m["dir"] / m["subfolder"]
        for seq in range(1, num_seqs + 1):
            for gt in (False, True):
                src = gif_path(seq_dir, seq, gt)
                out_dir = frame_cache_dir(seq_dir, seq, gt)
                already_cached = (out_dir / "frame_00.png").exists()
                n = extract_frames(src, out_dir)
                done += 1
                if already_cached:
                    skipped += 1
                if n and n != num_frames_expected:
                    mismatched += 1
                    print(f"  [warn] {src} has {n} frames, expected {num_frames_expected}")
                if done % 100 == 0 or done == total:
                    print(f"  extracted frames: {done}/{total} ({skipped} already cached)")

    elapsed = time.time() - t0
    print(f"Frame extraction done in {elapsed:.1f}s "
          f"({total - skipped} newly extracted, {skipped} already cached"
          + (f", {mismatched} frame-count mismatches" if mismatched else "") + ")")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #f6f7f9;
    --panel: #ffffff;
    --border: #e0e2e6;
    --text: #1f2430;
    --muted: #7a8291;
    --accent: #3cb44b;
    --accent2: #4363d8;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
  }
  header {
    position: sticky;
    top: 0;
    z-index: 10;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 12px 20px;
  }
  header h1 {
    font-size: 18px;
    margin: 0 0 2px 0;
  }
  header .subtitle {
    color: var(--muted);
    font-size: 12.5px;
    margin-bottom: 10px;
  }
  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 18px;
  }
  .seq-nav, .frame-nav {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .frame-nav {
    padding-left: 18px;
    border-left: 1px solid var(--border);
  }
  .seq-nav button, .frame-nav button {
    padding: 5px 10px;
    border: 1px solid var(--border);
    background: var(--panel);
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
  }
  .seq-nav button:hover, .frame-nav button:hover { background: #eef1f5; }
  .frame-nav button.playing {
    color: var(--accent2);
    border-color: var(--accent2);
  }
  .frame-nav button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .seq-nav input[type=number], .frame-nav input[type=number] {
    width: 46px;
    padding: 5px 6px;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 13px;
    text-align: center;
  }
  .seq-nav input[type=range], .frame-nav input[type=range] { width: 160px; }
  .model-toggles {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
  }
  .model-toggles label {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #eef1f5;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 3px 10px 3px 6px;
    font-size: 12.5px;
    cursor: pointer;
    user-select: none;
  }
  .model-toggles label.checked {
    background: #e4f3e6;
    border-color: var(--accent);
  }
  .model-toggles input { cursor: pointer; }
  .small-btn {
    padding: 4px 9px;
    border: 1px solid var(--border);
    background: var(--panel);
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    color: var(--muted);
  }
  .small-btn:hover { color: var(--text); }
  .gt-toggle {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 12.5px;
  }
  main {
    padding: 16px 20px 60px 20px;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 14px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  .card.gt {
    border-color: var(--accent2);
  }
  .card-header {
    padding: 7px 10px;
    font-size: 12.5px;
    font-weight: 600;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 6px;
  }
  .card-header .long-name {
    font-weight: 400;
    color: var(--muted);
    font-size: 10.5px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .card img {
    width: 100%;
    display: block;
    cursor: zoom-in;
    background: #fafbfc;
  }
  .empty-note {
    color: var(--muted);
    font-size: 13px;
    padding: 30px;
    text-align: center;
  }
  /* lightbox */
  #lightbox {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(10, 12, 16, 0.85);
    z-index: 100;
    align-items: center;
    justify-content: center;
    cursor: zoom-out;
  }
  #lightbox img {
    max-width: 92vw;
    max-height: 92vh;
    border-radius: 6px;
  }
  #lightbox .caption {
    position: absolute;
    top: 16px;
    left: 50%;
    transform: translateX(-50%);
    color: #fff;
    font-size: 13px;
    background: rgba(0,0,0,0.4);
    padding: 4px 12px;
    border-radius: 12px;
  }
</style>
</head>
<body>

<header>
  <h1>__TITLE__</h1>
  <div class="subtitle">__SUBTITLE__</div>
  <div class="controls">
    <div class="seq-nav">
      <button id="prevSeqBtn" title="Previous sequence (Shift+&larr;)">&larr;</button>
      <span>Seq</span>
      <input type="number" id="seqNumber" min="1" max="__NUM_SEQS__" value="1">
      <span>/ __NUM_SEQS__</span>
      <button id="nextSeqBtn" title="Next sequence (Shift+&rarr;)">&rarr;</button>
      <input type="range" id="seqSlider" min="1" max="__NUM_SEQS__" value="1">
    </div>
    <div class="frame-nav">
      <button id="prevFrameBtn" title="Previous frame (&larr;)">&#9198;</button>
      <button id="playPauseBtn" title="Play / pause (space)">&#9654;</button>
      <button id="nextFrameBtn" title="Next frame (&rarr;)">&#9197;</button>
      <span>Frame</span>
      <input type="number" id="frameNumber" min="0" max="__NUM_FRAMES_MAX__" value="0">
      <span>/ __NUM_FRAMES__</span>
      <input type="range" id="frameSlider" min="0" max="__NUM_FRAMES_MAX__" value="0">
    </div>
    <div class="gt-toggle">
      <input type="checkbox" id="gtToggle" checked>
      <label for="gtToggle">Show Ground Truth</label>
    </div>
    <div>
      <button class="small-btn" id="selectAll">Select all</button>
      <button class="small-btn" id="selectNone">Select none</button>
    </div>
    <div class="model-toggles" id="modelToggles"></div>
  </div>
</header>

<main>
  <div class="grid" id="grid"></div>
</main>

<div id="lightbox">
  <div class="caption" id="lightboxCaption"></div>
  <img id="lightboxImg" src="">
</div>

<script>
const MODELS = __MODELS_JSON__;
const NUM_SEQS = __NUM_SEQS__;
const NUM_FRAMES = __NUM_FRAMES__;
const FRAMES_AVAILABLE = NUM_FRAMES > 0;
const FRAMES_DIRNAME = "__FRAMES_DIRNAME__";
const FRAME_INTERVAL_MS = 220;

const state = {
  seq: 1,
  frame: 0,
  selected: new Set(MODELS.map(m => m.dir)),
  showGT: true,
  playing: false,
};

const gridEl = document.getElementById('grid');
const togglesEl = document.getElementById('modelToggles');
const seqNumberEl = document.getElementById('seqNumber');
const seqSliderEl = document.getElementById('seqSlider');
const frameNumberEl = document.getElementById('frameNumber');
const frameSliderEl = document.getElementById('frameSlider');
const gtToggleEl = document.getElementById('gtToggle');
const prevFrameBtn = document.getElementById('prevFrameBtn');
const nextFrameBtn = document.getElementById('nextFrameBtn');
const playPauseBtn = document.getElementById('playPauseBtn');
const frameNavEl = document.querySelector('.frame-nav');

if (!FRAMES_AVAILABLE) {
  // No extracted frames (Pillow missing, or generated with
  // --no-frame-extract): fall back to plain animated GIFs and hide
  // the frame transport controls, since there's nothing to step to.
  frameNavEl.style.display = 'none';
}

function gifPath(model, seq, gt) {
  const fname = gt ? `mprops_GT_seq_${seq}.gif` : `mprops_seq_${seq}.gif`;
  return `${model.dir}/${model.subfolder}/${fname}`;
}

function framePath(model, seq, gt, frame) {
  const seqName = gt ? `mprops_GT_seq_${seq}` : `mprops_seq_${seq}`;
  const frameName = `frame_${String(frame).padStart(2, '0')}.png`;
  return `${model.dir}/${model.subfolder}/${FRAMES_DIRNAME}/${seqName}/${frameName}`;
}

function currentSrc(model, gt) {
  return FRAMES_AVAILABLE ? framePath(model, state.seq, gt, state.frame) : gifPath(model, state.seq, gt);
}

function buildToggles() {
  togglesEl.innerHTML = '';
  MODELS.forEach(m => {
    const label = document.createElement('label');
    label.className = state.selected.has(m.dir) ? 'checked' : '';
    label.title = m.long_name;

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = state.selected.has(m.dir);
    cb.addEventListener('change', () => {
      if (cb.checked) state.selected.add(m.dir);
      else state.selected.delete(m.dir);
      label.className = cb.checked ? 'checked' : '';
      renderGrid();
    });

    label.appendChild(cb);
    label.appendChild(document.createTextNode(m.short_name));
    togglesEl.appendChild(label);
  });
}

// Cards currently on screen. Rebuilt on renderGrid(); img src is
// refreshed in place (no DOM rebuild) whenever just the frame changes,
// so playback/stepping stays smooth.
let cardEntries = [];

function makeCard(m, gt) {
  const card = document.createElement('div');
  card.className = 'card' + (gt ? ' gt' : '');

  const head = document.createElement('div');
  head.className = 'card-header';
  const title = document.createElement('span');
  title.textContent = gt ? 'Ground Truth' : m.short_name;
  const longName = document.createElement('span');
  longName.className = 'long-name';
  longName.textContent = m.long_name;
  head.appendChild(title);
  if (!gt) head.appendChild(longName);

  const img = document.createElement('img');
  img.src = currentSrc(m, gt);
  img.alt = (gt ? 'GT ' : '') + m.long_name + ' seq ' + state.seq;
  img.addEventListener('click', () => openLightbox(img.src, (gt ? 'GT — ' : '') + m.long_name));

  card.appendChild(head);
  card.appendChild(img);

  cardEntries.push({ img, m, gt });
  return card;
}

function renderGrid() {
  gridEl.innerHTML = '';
  cardEntries = [];
  const selectedModels = MODELS.filter(m => state.selected.has(m.dir));

  if (selectedModels.length === 0) {
    const note = document.createElement('div');
    note.className = 'empty-note';
    note.textContent = 'Select at least one model above to compare.';
    gridEl.appendChild(note);
    return;
  }

  if (state.showGT) {
    gridEl.appendChild(makeCard(selectedModels[0], true));
  }
  selectedModels.forEach(m => gridEl.appendChild(makeCard(m, false)));
}

function refreshImages() {
  cardEntries.forEach(entry => { entry.img.src = currentSrc(entry.m, entry.gt); });
}

function setSeq(n) {
  n = Math.max(1, Math.min(NUM_SEQS, n));
  state.seq = n;
  state.frame = 0;
  seqNumberEl.value = n;
  seqSliderEl.value = n;
  frameNumberEl.value = 0;
  frameSliderEl.value = 0;
  renderGrid();
}

function setFrame(n) {
  if (!FRAMES_AVAILABLE) return;
  n = Math.max(0, Math.min(NUM_FRAMES - 1, n));
  state.frame = n;
  frameNumberEl.value = n;
  frameSliderEl.value = n;
  refreshImages();
}

function stepFrame(delta) {
  pausePlayback();
  setFrame(state.frame + delta);
}

let playTimer = null;
function startPlayback() {
  if (!FRAMES_AVAILABLE || state.playing) return;
  state.playing = true;
  playPauseBtn.textContent = '\u23F8'; // pause icon
  playPauseBtn.classList.add('playing');
  playPauseBtn.title = 'Pause (space)';
  playTimer = setInterval(() => {
    const next = (state.frame + 1) % NUM_FRAMES;
    setFrame(next);
  }, FRAME_INTERVAL_MS);
}

function pausePlayback() {
  if (!state.playing) return;
  state.playing = false;
  playPauseBtn.textContent = '\u25B6'; // play icon
  playPauseBtn.classList.remove('playing');
  playPauseBtn.title = 'Play (space)';
  clearInterval(playTimer);
  playTimer = null;
}

function togglePlayback() {
  if (state.playing) pausePlayback();
  else startPlayback();
}

document.getElementById('prevSeqBtn').addEventListener('click', () => setSeq(state.seq - 1));
document.getElementById('nextSeqBtn').addEventListener('click', () => setSeq(state.seq + 1));
seqNumberEl.addEventListener('change', () => setSeq(parseInt(seqNumberEl.value || '1', 10)));
seqSliderEl.addEventListener('input', () => setSeq(parseInt(seqSliderEl.value, 10)));

prevFrameBtn.addEventListener('click', () => stepFrame(-1));
nextFrameBtn.addEventListener('click', () => stepFrame(1));
playPauseBtn.addEventListener('click', togglePlayback);
frameNumberEl.addEventListener('change', () => { pausePlayback(); setFrame(parseInt(frameNumberEl.value || '0', 10)); });
frameSliderEl.addEventListener('input', () => { pausePlayback(); setFrame(parseInt(frameSliderEl.value, 10)); });

gtToggleEl.addEventListener('change', () => { state.showGT = gtToggleEl.checked; renderGrid(); });
document.getElementById('selectAll').addEventListener('click', () => {
  MODELS.forEach(m => state.selected.add(m.dir));
  buildToggles(); renderGrid();
});
document.getElementById('selectNone').addEventListener('click', () => {
  state.selected.clear();
  buildToggles(); renderGrid();
});

const lightbox = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightboxImg');
const lightboxCaption = document.getElementById('lightboxCaption');
function openLightbox(src, caption) {
  lightboxImg.src = src;
  lightboxCaption.textContent = caption;
  lightbox.style.display = 'flex';
}
lightbox.addEventListener('click', () => { lightbox.style.display = 'none'; });

document.addEventListener('keydown', (e) => {
  const typing = document.activeElement &&
    (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA');
  if (typing) return;

  if (e.key === 'Escape') { lightbox.style.display = 'none'; return; }

  if (e.shiftKey && e.key === 'ArrowRight') { setSeq(state.seq + 1); return; }
  if (e.shiftKey && e.key === 'ArrowLeft')  { setSeq(state.seq - 1); return; }
  if (e.key === 'ArrowRight') { e.preventDefault(); stepFrame(1); return; }
  if (e.key === 'ArrowLeft')  { e.preventDefault(); stepFrame(-1); return; }
  if (e.code === 'Space')     { e.preventDefault(); togglePlayback(); return; }
});

buildToggles();
renderGrid();
</script>
</body>
</html>
"""


def generate_html(models, num_seqs, num_frames, output_path: Path, title: str):
    html = HTML_TEMPLATE
    html = html.replace("__TITLE__", title)

    frame_note = (f"{num_frames} frames/sequence, globally synced &middot; "
                  f"&larr;/&rarr; step a frame, space to play/pause, Shift+&larr;/&rarr; changes sequence"
                  if num_frames > 0 else
                  "frame stepping unavailable (run without --no-frame-extract, and with Pillow installed, "
                  "to enable it) &middot; showing animated GIFs")
    html = html.replace(
        "__SUBTITLE__",
        f"{len(models)} models loaded &middot; {num_seqs} fixed-past sequences &middot; "
        f"{frame_note} &middot; click a GIF/frame to zoom"
    )
    html = html.replace("__NUM_SEQS__", str(num_seqs))
    html = html.replace("__NUM_FRAMES__", str(num_frames))
    html = html.replace("__NUM_FRAMES_MAX__", str(max(num_frames - 1, 0)))
    html = html.replace("__FRAMES_DIRNAME__", FRAMES_CACHE_DIRNAME)
    html = html.replace("__MODELS_JSON__", json.dumps(models))
    output_path.write_text(html, encoding="utf-8")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a static HTML gallery to compare mprops_seq_N.gif "
                    "across all sampled models for a given dataset's output dir, "
                    "with globally-synced frame-by-frame stepping."
    )
    parser.add_argument('--main-models-dir', type=str, default='output_hermes_bn/',
                         help='Directory containing model subfolders (same one used by comparison_models_plot.py).')
    parser.add_argument('--subfolder', type=str, default='fixed_past_samples',
                         help='Subfolder inside each model dir holding the mprops_seq_N.gif files.')
    parser.add_argument('--output-html', type=str, default=None,
                         help='Path to write the HTML file. Defaults to <main-models-dir>/gif_comparison.html '
                              'so relative paths resolve correctly.')
    parser.add_argument('--models-file', type=str, default=None,
                         help="YAML file with a top-level MODELS list of model dir names to load, "
                              "in display order (same style as config/*_datafiles.yml). "
                              "If not given, all discovered models are loaded.")
    parser.add_argument('--list-models', action='store_true',
                         help='Just print the model dir names discovered under main-models-dir and exit '
                              '(handy for building a --models-file).')
    parser.add_argument('--no-frame-extract', action='store_true',
                         help='Skip PNG frame extraction (faster to generate, but disables frame-by-frame '
                              'stepping; falls back to plain animated GIFs).')
    parser.add_argument('--title', type=str, default=None,
                         help='Page title. Defaults to the main-models-dir name.')
    args = parser.parse_args()

    main_dir = Path(args.main_models_dir)
    if not main_dir.is_dir():
        raise SystemExit(f"main-models-dir not found: {main_dir}")

    all_models = discover_models_exp(main_dir, args.subfolder)
    if not all_models:
        raise SystemExit(
            f"No model subdirectories with '{args.subfolder}/mprops_seq_1.gif' found under {main_dir}"
        )

    if args.list_models:
        print(f"Model dirs discovered under {main_dir}:")
        for m in all_models:
            print(f"  - \"{m['dir']}\"")
        raise SystemExit(0)

    if args.models_file:
        requested = getYamlConfig(args.models_file)
        requested = requested.MODELS
        models = resolve_selected_models(all_models, requested)
    else:
        print("No --models-file given, loading all discovered models.")
        models = all_models

    num_seqs = count_available_seqs(main_dir, models)
    if num_seqs == 0:
        raise SystemExit("Found model dirs but couldn't count any mprops_seq_N.gif files.")

    num_frames = 0
    if args.no_frame_extract:
        print("--no-frame-extract set: skipping PNG extraction, using animated GIFs only.")
    elif not HAVE_PIL:
        print("Pillow not available: skipping PNG extraction, using animated GIFs only. "
              "Install it (pip install pillow) to enable frame-by-frame stepping.")
    else:
        num_frames = count_frames(main_dir, models)
        if num_frames == 0:
            print("[warn] couldn't determine frame count, using animated GIFs only.")
        else:
            print(f"Extracting {num_frames} frames/sequence for {len(models)} models x {num_seqs} sequences "
                  f"(cached under '{FRAMES_CACHE_DIRNAME}/', safe to re-run)...")
            extract_all_frames(main_dir, models, num_seqs, num_frames)

    output_path = Path(args.output_html) if args.output_html else main_dir / "gif_comparison.html"
    title = args.title or f"Sampling comparison — {main_dir.name}"

    generate_html(models, num_seqs, num_frames, output_path, title)

    print(f"Loaded {len(models)} models, {num_seqs} sequences each:")
    for m in models:
        print(f"  {m['long_name']:55s} -> {m['short_name']}")
    print(f"\nWrote {output_path}")
    print("Open it directly in a browser (relative paths assume it stays next to the model folders).")

# execution example:
# python3 utils/plot/gif_comparison_viewer.py --main-models-dir=output_hermes_bn/ --models-file=config/models_list.yml