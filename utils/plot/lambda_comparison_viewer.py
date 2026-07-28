import argparse, os
from pathlib import Path
from utils.myparser import getYamlConfig


def discover_folders(parent_dir: Path, comp_plots_subdir: str):
    """
    Scan parent_dir for subdirectories that contain a `comp_plots_subdir`
    (default: comp_plots) with at least one PNG in it -- i.e. output dirs
    already processed by comparison_models_plot.py.
    """
    folders = []
    for d in sorted(parent_dir.iterdir()):
        if not d.is_dir():
            continue
        cp = d / comp_plots_subdir
        if not cp.is_dir():
            continue
        if not any(cp.glob("*.png")):
            continue
        folders.append(d.name)
    return folders


def resolve_selected_folders(all_folders: list, requested: list) -> list:
    """
    Filter/reorder all_folders down to the names in requested, preserving
    requested's order. Warns (does not fail) on names that don't match.
    """
    available = set(all_folders)
    selected = []
    for name in requested:
        if name in available:
            selected.append(name)
        else:
            print(f"  [warn] '{name}' not found among discovered folders, skipping.")
    if not selected:
        listing = "\n".join(f"  - {f}" for f in all_folders)
        raise SystemExit(
            "None of the requested folders matched a discovered one.\n"
            f"Available folders under parent-dir:\n{listing}"
        )
    return selected


def discover_plot_keys(parent_dir: Path, folders: list, comp_plots_subdir: str) -> list:
    """
    Union of every *.png stem found across all selected folders' comp_plots
    dirs (not just the first one -- different lambda runs could in principle
    have slightly different metric sets available). Sorted for a stable
    dropdown order.
    """
    keys = set()
    for name in folders:
        cp = parent_dir / name / comp_plots_subdir
        for png in cp.glob("*.png"):
            keys.add(png.stem)
    return sorted(keys)


def group_plot_keys(keys: list):
    """
    Bucket plot keys into the same families comparison_models_plot.py
    produces them in, so the dropdown reads as
    Summary metrics / Over time / BHATT / Other instead of one flat list.
    """
    groups = {"Summary metrics": [], "Over time": [], "BHATT": [], "Other": []}
    for k in keys:
        if k.startswith("summary_bhatt"):
            groups["BHATT"].append(k)
        elif k.startswith("summary_"):
            groups["Summary metrics"].append(k)
        elif "_otime" in k:
            groups["Over time"].append(k)
        else:
            groups["Other"].append(k)
    return [(label, sorted(ks)) for label, ks in groups.items() if ks]


def pick_default_plot(keys: list, requested: str = None) -> str:
    if requested:
        if requested in keys:
            return requested
        print(f"  [warn] --default-plot '{requested}' not found among discovered plots, "
              f"falling back to auto-pick.")
    for k in keys:
        if k.startswith("summary_psnr"):
            return k
    return keys[0] if keys else ""


def derive_labels(folders: list, label_mode: str, lambda_divisor: float):
    """
    Turn folder names into short display labels by stripping their longest
    common prefix (e.g. 'output_hermes_bn_004' -> '004'). With
    label_mode='lambda', additionally interpret that suffix as an integer
    scaled by lambda_divisor (default 1000), e.g. '004' -> 'λ=0.004' --
    this matches the output_hermes_bn_004/016/032/.../128 naming, but is
    just a display convenience: it's only ever a guess, so it's flagged
    here and easy to turn off with --label-mode raw.
    """
    prefix = os.path.commonprefix(folders) if len(folders) > 1 else ""
    labels = {}
    for name in folders:
        suffix = name[len(prefix):] if prefix and name.startswith(prefix) else name
        suffix = suffix.strip("_-") or name
        if label_mode == "lambda":
            try:
                val = int(suffix) / lambda_divisor
                labels[name] = f"\u03bb={val:g}"
                continue
            except ValueError:
                pass
        labels[name] = suffix
    return labels


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
  header h1 { font-size: 18px; margin: 0 0 2px 0; }
  header .subtitle { color: var(--muted); font-size: 12.5px; margin-bottom: 10px; }
  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 18px;
  }
  .plot-select {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .plot-select select {
    padding: 6px 8px;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 13px;
    background: var(--panel);
    max-width: 320px;
  }
  .folder-toggles {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
  }
  .folder-toggles label {
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
  .folder-toggles label.checked {
    background: #e4f3e6;
    border-color: var(--accent);
  }
  .folder-toggles input { cursor: pointer; }
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
  main { padding: 16px 20px 60px 20px; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(__MIN_CARD_PX__px, 1fr));
    gap: 14px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
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
  .card-header .pdf-link {
    font-size: 11px;
    color: var(--accent2);
    text-decoration: none;
    flex: none;
  }
  .card-header .pdf-link:hover { text-decoration: underline; }
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
  #lightbox img { max-width: 92vw; max-height: 92vh; border-radius: 6px; }
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
    <div class="plot-select">
      <span>Plot</span>
      <select id="plotSelect">
__PLOT_OPTIONS__
      </select>
    </div>
    <div>
      <button class="small-btn" id="selectAll">Select all</button>
      <button class="small-btn" id="selectNone">Select none</button>
    </div>
    <div class="folder-toggles" id="folderToggles"></div>
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
const FOLDERS = __FOLDERS_JSON__;
const COMP_PLOTS_SUBDIR = "__COMP_PLOTS_SUBDIR__";

const state = {
  selected: new Set(FOLDERS.map(f => f.dir)),
};

const gridEl = document.getElementById('grid');
const togglesEl = document.getElementById('folderToggles');
const plotSelectEl = document.getElementById('plotSelect');

function pngPath(folder, key) { return `${folder.dir}/${COMP_PLOTS_SUBDIR}/${key}.png`; }
function pdfPath(folder, key) { return `${folder.dir}/${COMP_PLOTS_SUBDIR}/${key}.pdf`; }

function buildToggles() {
  togglesEl.innerHTML = '';
  FOLDERS.forEach(f => {
    const label = document.createElement('label');
    label.className = state.selected.has(f.dir) ? 'checked' : '';
    label.title = f.dir;

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = state.selected.has(f.dir);
    cb.addEventListener('change', () => {
      if (cb.checked) state.selected.add(f.dir);
      else state.selected.delete(f.dir);
      label.className = cb.checked ? 'checked' : '';
      renderGrid();
    });

    label.appendChild(cb);
    label.appendChild(document.createTextNode(f.label));
    togglesEl.appendChild(label);
  });
}

function makeCard(f, key) {
  const card = document.createElement('div');
  card.className = 'card';

  const head = document.createElement('div');
  head.className = 'card-header';
  const title = document.createElement('span');
  title.textContent = f.label;
  const longName = document.createElement('span');
  longName.className = 'long-name';
  longName.textContent = f.dir;
  const pdfLink = document.createElement('a');
  pdfLink.className = 'pdf-link';
  pdfLink.href = pdfPath(f, key);
  pdfLink.textContent = 'PDF';
  pdfLink.title = 'Open the print-quality PDF version';
  head.appendChild(title);
  head.appendChild(longName);
  head.appendChild(pdfLink);

  const img = document.createElement('img');
  img.src = pngPath(f, key);
  img.alt = `${f.label} - ${key}`;
  img.addEventListener('click', () => openLightbox(img.src, `${f.label} \u2014 ${key}`));

  card.appendChild(head);
  card.appendChild(img);
  return card;
}

function renderGrid() {
  gridEl.innerHTML = '';
  const key = plotSelectEl.value;
  const selectedFolders = FOLDERS.filter(f => state.selected.has(f.dir));

  if (selectedFolders.length === 0) {
    const note = document.createElement('div');
    note.className = 'empty-note';
    note.textContent = 'Select at least one folder above to compare.';
    gridEl.appendChild(note);
    return;
  }

  selectedFolders.forEach(f => gridEl.appendChild(makeCard(f, key)));
}

plotSelectEl.addEventListener('change', renderGrid);
document.getElementById('selectAll').addEventListener('click', () => {
  FOLDERS.forEach(f => state.selected.add(f.dir));
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
  if (e.key === 'Escape') lightbox.style.display = 'none';
});

buildToggles();
renderGrid();
</script>
</body>
</html>
"""


def generate_html(folders_json_ready, plot_groups, default_plot, output_path: Path, title: str,
                   min_card_px: int, comp_plots_subdir: str, num_plot_keys: int):
    import json

    option_lines = []
    for group_label, keys in plot_groups:
        option_lines.append(f'        <optgroup label="{group_label}">')
        for key in keys:
            selected_attr = " selected" if key == default_plot else ""
            option_lines.append(f'          <option value="{key}"{selected_attr}>{key}</option>')
        option_lines.append('        </optgroup>')
    plot_options_html = "\n".join(option_lines)

    html = HTML_TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace(
        "__SUBTITLE__",
        f"{len(folders_json_ready)} folders loaded &middot; {num_plot_keys} plot types available &middot; "
        f"click a plot to zoom &middot; PDF link opens the print-quality version"
    )
    html = html.replace("__PLOT_OPTIONS__", plot_options_html)
    html = html.replace("__FOLDERS_JSON__", json.dumps(folders_json_ready))
    html = html.replace("__COMP_PLOTS_SUBDIR__", comp_plots_subdir)
    html = html.replace("__MIN_CARD_PX__", str(min_card_px))
    output_path.write_text(html, encoding="utf-8")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a static HTML page comparing comp_plots/ metric figures "
                    "(e.g. summary_psnr_bn) across several experiment output directories, "
                    "such as a LAMBDA_GUIDANCE sweep."
    )
    parser.add_argument('--parent-dir', type=str, required=True,
                         help='Directory containing the experiment subfolders (e.g. output_hermes_bn_004, '
                              'output_hermes_bn_016, ...), each already processed by comparison_models_plot.py.')
    parser.add_argument('--comp-plots-subdir', type=str, default='comp_plots',
                         help="Subfolder inside each experiment dir holding the *.png/*.pdf figures "
                              "(comparison_models_plot.py's default output dir name).")
    parser.add_argument('--output-html', type=str, default=None,
                         help='Path to write the HTML file. Defaults to <parent-dir>/lambda_comparison.html '
                              'so relative paths resolve correctly.')
    parser.add_argument('--folders-file', type=str, default=None,
                         help="YAML file with a top-level FOLDERS list of experiment dir names to load, "
                              "in display order (same style as config/*_datafiles.yml and gif_comparison_viewer's "
                              "--models-file). If not given, all discovered folders are loaded.")
    parser.add_argument('--list-folders', action='store_true',
                         help='Just print the experiment dir names discovered under parent-dir and exit '
                              '(handy for building a --folders-file).')
    parser.add_argument('--default-plot', type=str, default=None,
                         help='Plot key selected when the page first loads (e.g. summary_psnr_bn). '
                              'Defaults to the first summary_psnr_* plot found, else the first plot alphabetically.')
    parser.add_argument('--label-mode', type=str, default='lambda', choices=['lambda', 'raw'],
                         help="How to label each folder. 'lambda' (default) strips the folders' common prefix "
                              "and interprets the remaining suffix as an int scaled by --lambda-divisor, e.g. "
                              "'output_hermes_bn_004' -> 'λ=0.004' -- this is a naming assumption, disable with "
                              "--label-mode raw to just show the stripped suffix as-is.")
    parser.add_argument('--lambda-divisor', type=float, default=1000,
                         help='Divisor used to convert a folder name suffix to a lambda value in --label-mode lambda '
                              "(e.g. '004' / 1000 -> 0.004).")
    parser.add_argument('--reference-width', type=int, default=1500,
                         help='Browser width (px) the grid is calibrated to show about 4 cards per row at; '
                              'it still reflows fluidly at any actual window size.')
    parser.add_argument('--title', type=str, default=None,
                         help='Page title. Defaults to the parent-dir name.')
    args = parser.parse_args()

    parent_dir = Path(args.parent_dir)
    if not parent_dir.is_dir():
        raise SystemExit(f"parent-dir not found: {parent_dir}")

    all_folders = discover_folders(parent_dir, args.comp_plots_subdir)
    if not all_folders:
        raise SystemExit(
            f"No subdirectories with '{args.comp_plots_subdir}/*.png' found under {parent_dir}"
        )

    if args.list_folders:
        print(f"Experiment folders discovered under {parent_dir}:")
        for f in all_folders:
            print(f"  - \"{f}\"")
        raise SystemExit(0)

    if args.folders_file:
        requested_cfg = getYamlConfig(args.folders_file)
        requested = requested_cfg.FOLDERS
        folders = resolve_selected_folders(all_folders, requested)
    else:
        print("No --folders-file given, loading all discovered folders.")
        folders = all_folders

    plot_keys = discover_plot_keys(parent_dir, folders, args.comp_plots_subdir)
    if not plot_keys:
        raise SystemExit("No *.png plots found in the selected folders' comp_plots dirs.")
    default_plot = pick_default_plot(plot_keys, args.default_plot)
    plot_groups = group_plot_keys(plot_keys)

    labels = derive_labels(folders, args.label_mode, args.lambda_divisor)
    folders_json_ready = [{"dir": name, "label": labels[name]} for name in folders]

    # ~3 cards per row at the reference width, same fluid auto-fit/minmax()
    # approach as gif_comparison_viewer.py, just with a generic default
    # target since these are fixed-size matplotlib figures, not per-dataset
    # GIF sequences.
    target_cols = 3
    gap, padding = 14, 40
    usable = args.reference_width - padding
    min_card_px = max(280, int((usable - gap * (target_cols - 1)) / target_cols))

    output_path = Path(args.output_html) if args.output_html else parent_dir / "lambda_comparison.html"
    title = args.title or f"LAMBDA_GUIDANCE comparison — {parent_dir.name}"

    generate_html(folders_json_ready, plot_groups, default_plot, output_path, title,
                  min_card_px, args.comp_plots_subdir, len(plot_keys))

    print(f"Loaded {len(folders)} folders, {len(plot_keys)} plot types "
          f"(default: {default_plot}):")
    for f in folders_json_ready:
        print(f"  {f['dir']:35s} -> {f['label']}")
    print(f"\nWrote {output_path}")
    print("Open it directly in a browser (relative paths assume it stays next to the experiment folders).")

# execution example:
# python3 utils/plot/lambda_comparison_viewer.py --parent-dir=output_hermes_bn_sweep/ --default-plot=summary_psnr_bn