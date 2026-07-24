"""
ExteriorBallisticsCalc / Exterior Ballistics Calculator
========================================

Desktop exterior ballistics calculator powered by py-ballisticcalc 2.2.10.

## Tech Stack
- **UI**: Native Tkinter widgets, publication-quality interactive Matplotlib charts (blit-accelerated)
- **Engine**: py-ballisticcalc 2.2.10 + py-ballisticcalc-exts (Cython/C++ accelerated)

## Library Utilization

### Fully Leveraged
- 10 drag table options (G1~RA4 + Custom Mach-CD), all exposed in UI dropdown
- DragModelMultiBC with dynamic row add/remove
- 6 integration engines: RK4 / Euler / SciPy(odeint) / Velocity Verlet / + Cython-accelerated RK4 & Euler
- Ammo: all parameters (mv, powder_temp, temp_modifier, use_powder_sensitivity)
- Weapon: sight_height, twist, twist_direction
- Atmosphere: Atmo / Vacuum / ICAO standard atmosphere, altitude-linked
- Shot.look_angle (elevation/depression for non-level shooting)
- Calculator core flow: set_weapon_zero() → barrel_elevation_for_target() → fire()
- All 16 TrajectoryData fields
- HitResult.get_at() interpolation (used across all 5 tabs)
- 3 unit systems (metric/imperial/mixed) + dynamic switching of all 16 PreferredUnits slots
- py-ballisticcalc-exts (Cython/C++): CythonizedRK4IntegrationEngine set as default engine

### Intentionally Excluded
- **Coriolis**: requires lat/azimuth input; ~10–30 cm beyond 800 m; niche scenario
- **Multi-layer wind zones**: niche scenario
- **cant_angle / zero-atmosphere separation**: niche scenarios
- **Built-in matplotlib plot** (hit_result_as_plot): cannot inject into TkAgg; custom rendering is superior
- **Built-in pandas export** (hit_result_as_dataframe): target is ttk.Treeview, not DataFrame
- **Lofted trajectory / max range / engine tuning / dense output / SciPy method switch**: niche or non-small-arms scenarios
- **DangerSpace**: different semantics from custom PBR; not interchangeable
- **TrajectoryData.formatted()**: magic tuple indices reduce readability
- **calc_powder_sens()**: low-frequency need; requires separate UI design
- **find_apex()**: library already auto-tags APEX during integration; no extra search needed

## Features

### 5 Analysis Tabs
- **Tab 1 Trajectory**: velocity-gradient trajectory plot + velocity twin axis + 16-column table
- **Tab 2 Trajectory Analysis**: multi-ammo overlay trajectory + velocity twin axis + curve intersections + 13-column table
- **Tab 3 Energy**: multi-ammo kinetic energy curves + curve intersections + 9-column table (sectional KE with distance interpolation)
- **Tab 4 Wind Drift**: multi-ammo wind drift curves + compare (6-col) / detail (5-col) table toggle
- **Tab 5 Drag**: multi-ammo drag (Mach) curves + dF/dM derivative twin axis + G7 extrema + curve intersections + 6-column table

### Ammo Library
- JSON-based ammo storage
- Add / Duplicate / Delete / Save / Import / Remove / Clear
- List rows show drag table type + BC + bullet weight + muzzle velocity

### Real-time Parameters
- Muzzle energy / Recoil impulse / Sectional KE (J/cm²) / Sectional density (SD, lb/in²)
- BC ↔ i (form factor) bidirectional linked editing
- Sectional KE with distance interpolation in Tab 2/3 tables

### Chart Interaction
- Blit-accelerated hover across all 5 tabs; 60 ms debounce redraw
- Hover crosshair + tooltip; click to highlight and snap to key points
- Curve intersections: auto-detected, blend-color markers (excluded from legend)
- Localized Matplotlib toolbar (including chart copy to clipboard)
- Tab 1 display options: select-all linking; independent toggles for apogee/transonic/zero/bore-line/sight-line/velocity-line

### Advanced Ballistics
- **Apogee lock**: given max ordinate, binary-search reverse solve for zero distance
- **PBR**: 3 target heights (0.3 / 1.0 / 1.5 m)
- **Supersonic range**: find distance where Mach drops to 1.2
- **Vacuum mode**: no-drag simulation
- **ICAO standard atmosphere**: one-click standard data injection
- **Custom drag table**: arbitrary Mach-CD data point input
- **Look angle**: uphill/downhill shooting support
- **Twist rate + direction**: left/right-hand twist selection

### Universal Table Features
- **Right-click hide columns**: all 7 Treeviews; native checkbutton menu
- **Adaptive column width**: auto-fit based on header + first row + last row

### Unit System
- 3 unit systems — Metric / Imperial / Mixed
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import math
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.font_manager as fm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

# ---- matplotlib toolbar localization ----
class _ChineseToolbar(NavigationToolbar2Tk):
    toolitems = [
        ("Home", "Reset to original view", "home", "home"),
        ("Back", "Back to previous view", "back", "back"),
        ("Forward", "Forward to next view", "forward", "forward"),
        (None, None, None, None),
        ("Pan", "Left-drag to pan / right-drag to zoom", "move", "pan"),
        ("Zoom", "Zoom to rectangle", "zoom_to_rect", "zoom"),
        ("Subplots", "Configure subplot parameters", "subplots", "configure_subplots"),
        (None, None, None, None),
        ("Save", "Save to image file", "filesave", "save_figure"),
        ("Copy", "Copy chart to clipboard", "filesave", "copy_figure"),
    ]

    def format_coord(self, x, y):
        """Format coordinates: hide trailing zero for integers, one decimal otherwise"""
        def _fmt(v):
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            return f"{v:.1f}"
        return f"x={_fmt(x)}  y={_fmt(y)}"

    def copy_figure(self):
        """Copy current chart to clipboard"""
        import tempfile, subprocess, os
        fig = self.canvas.figure
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        try:
            tmp.close()
            fig.savefig(tmp.name)
            subprocess.run([
                'powershell', '-command',
                f'Add-Type -AssemblyName System.Windows.Forms; '
                f'[Windows.Forms.Clipboard]::SetImage([System.Drawing.Image]::FromFile("{tmp.name}"))'
            ], capture_output=True, timeout=10)
        finally:
            try: os.unlink(tmp.name)
            except OSError: pass

    def configure_subplots(self):
        parent = self.canvas.get_tk_widget().winfo_toplevel()
        dlg = tk.Toplevel(parent)
        dlg.title("Subplot Configuration")
        dlg.resizable(False, False)
        dlg.transient(parent)
        dlg.grab_set()
        dlg.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        w, h = 900, 320
        dlg.geometry(f"{w}x{h}+{px + pw//2 - w//2}+{py + ph//2 - h//2}")
        frm = ttk.Frame(dlg, padding=(12, 10))
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(2, weight=1)
        fig = self.canvas.figure
        sp = fig.subplotpars
        items = [
            ("Left:", sp.left, 0.0, 0.5), ("Bottom:", sp.bottom, 0.0, 0.5),
            ("Right:", sp.right, 0.5, 1.0), ("Top:", sp.top, 0.5, 1.0),
            ("Wspace:", sp.wspace, 0.0, 0.8), ("Hspace:", sp.hspace, 0.0, 0.8),
        ]
        entries, scales = {}, {}
        for i, (label, val, lo, hi) in enumerate(items):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky="e", padx=(0, 4), pady=2)
            var = tk.StringVar(value=f"{val:.4f}")
            ttk.Entry(frm, textvariable=var, width=10).grid(row=i, column=1, sticky="w", pady=2)
            entries[label] = var
            sv = tk.DoubleVar(value=val)
            ttk.Scale(frm, from_=lo, to=hi, variable=sv, orient="horizontal").grid(
                row=i, column=2, sticky="ew", padx=(6, 0), pady=2)
            scales[label] = sv
            def _slider_cb(sv=sv, v=var):
                def cb(*_): v.set(f"{sv.get():.4f}"); apply()
                return cb
            sv.trace_add("write", _slider_cb())
            def _entry_cb(sv=sv, v=var, lo=lo, hi=hi):
                def cb(*_):
                    try:
                        fv = float(v.get())
                        if lo <= fv <= hi: sv.set(fv)
                        apply()
                    except ValueError: pass
                return cb
            var.trace_add("write", _entry_cb())
        def apply():
            try:
                fig.subplots_adjust(
                    left=float(entries["Left:"].get()), bottom=float(entries["Bottom:"].get()),
                    right=float(entries["Right:"].get()), top=float(entries["Top:"].get()),
                    wspace=float(entries["Wspace:"].get()), hspace=float(entries["Hspace:"].get()))
                self.canvas.draw_idle()
            except ValueError: pass
        def reset():
            for k, d in zip(entries.keys(), [0.125, 0.11, 0.9, 0.88, 0.2, 0.2]): entries[k].set(f"{d:.4f}")
            apply()
        btns = ttk.Frame(frm)
        btns.grid(row=len(items), column=0, columnspan=3, pady=(12, 0))
        ttk.Button(btns, text="Reset", command=reset).pack(side="left", padx=4)
        ttk.Button(btns, text="Close", command=dlg.destroy).pack(side="left", padx=4)
        dlg.wait_window()

# ---- library imports ----
from py_ballisticcalc import (
    Calculator, Shot, Weapon, Ammo,
    DragModel, DragModelMultiBC, DragDataPoint, BCPoint, Atmo, Vacuum, Wind,
    TrajFlag, HitResult,
    PreferredUnits, Unit, Distance, Angular, Velocity, Pressure, Temperature, Weight,
    TableG1, TableG2, TableG5, TableG6, TableG7, TableG8, TableGI, TableGS, TableRA4,
    EulerIntegrationEngine, RK4IntegrationEngine,
    SciPyIntegrationEngine, VelocityVerletIntegrationEngine,
)

from py_ballisticcalc_exts import (
    CythonizedRK4IntegrationEngine, CythonizedEulerIntegrationEngine,
)

# ============================================================
# DPI & fonts
# ============================================================
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(2)
except Exception: pass

_SYS_FONTS = {f.name for f in fm.fontManager.ttflist}
_PLOT_FONT = "SimSun" if "SimSun" in _SYS_FONTS else \
             "Noto Serif SC" if "Noto Serif SC" in _SYS_FONTS else \
             "Microsoft YaHei" if "Microsoft YaHei" in _SYS_FONTS else "sans-serif"

plt.rcParams.update({
    "font.family": "serif", "font.serif": [_PLOT_FONT, "Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.alpha": 0.35, "grid.linewidth": 0.5, "figure.dpi": 120, "savefig.dpi": 300,
    "axes.unicode_minus": False,
})

# ============================================================
# constants
# ============================================================
DRAG_TABLES = {
    "G1 (Spitzer)": TableG1, "G2": TableG2, "G5": TableG5,
    "G6": TableG6, "G7 (Low-Drag)": TableG7, "G8": TableG8,
    "GI": TableGI, "GS (Sphere)": TableGS, "RA4": TableRA4,
    "Custom (Mach-CD)": None,
}
ENGINES = {
    "RK4 (Runge-Kutta 4th)": RK4IntegrationEngine, "RK4 (Cython/C++ Accelerated)": CythonizedRK4IntegrationEngine,
    "Euler": EulerIntegrationEngine, "Euler (Cython/C++ Accelerated)": CythonizedEulerIntegrationEngine,
    "SciPy (odeint)": SciPyIntegrationEngine, "Velocity Verlet": VelocityVerletIntegrationEngine,
}
TRAJ_FLAGS_UI = {"None": TrajFlag.NONE, "All": TrajFlag.ALL,
                 "Zero": TrajFlag.ZERO, "Transonic": TrajFlag.MACH, "Apogee": TrajFlag.APEX}

# ballistics magic numbers
_BISECT_MAX_ITER = 18          # max bisection iterations for apogee lock
_BISECT_FIRE_FACTOR = 1.5      # apogee-lock fire range = n × zero distance
_BISECT_STEP_DIV = 40          # bisection step divisor
_SNAP_THRESH_PX = 5            # mouse snap threshold (pixels)
_DEBOUNCE_MS = 60              # canvas redraw debounce delay (ms)

# unit presets (incl. mixed)
_UNIT_METRIC = {
    "distance": Unit.Meter, "velocity": Unit.MPS, "temperature": Unit.Celsius,
    "pressure": Unit.hPa, "sight_height": Unit.Centimeter, "drop": Unit.Centimeter,
    "adjustment": Unit.Mil, "twist": Unit.Centimeter, "weight": Unit.Gram,
    "diameter": Unit.Millimeter, "length": Unit.Millimeter, "energy": Unit.Joule,
    "ogw": Unit.Kilogram, "angular": Unit.Degree, "target_height": Unit.Centimeter,
}
_UNIT_IMPERIAL = {
    "distance": Unit.Yard, "velocity": Unit.FPS, "temperature": Unit.Fahrenheit,
    "pressure": Unit.InHg, "sight_height": Unit.Inch, "drop": Unit.Inch,
    "adjustment": Unit.MOA, "twist": Unit.Inch, "weight": Unit.Grain,
    "diameter": Unit.Inch, "length": Unit.Inch, "energy": Unit.FootPound,
    "ogw": Unit.Pound, "angular": Unit.Degree, "target_height": Unit.Inch,
}
_UNIT_MIXED = {
    "distance": Unit.Meter, "velocity": Unit.FPS, "temperature": Unit.Celsius,
    "pressure": Unit.hPa, "sight_height": Unit.Centimeter, "drop": Unit.Centimeter,
    "adjustment": Unit.Mil, "twist": Unit.Inch, "weight": Unit.Grain,
    "diameter": Unit.Inch, "length": Unit.Inch, "energy": Unit.Joule,
    "ogw": Unit.Kilogram, "angular": Unit.Degree, "target_height": Unit.Centimeter,
}
UNIT_PROFILES = {
    "Metric (m, m/s, hPa, °C)": _UNIT_METRIC,
    "Imperial (yd, fps, inHg, °F)": _UNIT_IMPERIAL,
    "Mixed (m, fps, hPa, °C)": _UNIT_MIXED,
}
DEFAULT_PROFILE = "Metric (m, m/s, hPa, °C)"
_AMMO_CONFIG_FILE = "ammo_configs.json"

# unit display labels
_UNIT_LABELS = {
    "Metric (m, m/s, hPa, °C)": {
        "weight": "g", "diameter": "mm", "length": "mm", "velocity": "m/s",
        "temperature": "°C", "pressure": "hPa", "sight_height": "cm", "twist": "cm",
        "distance": "m", "drop": "cm", "adjustment": "Mil", "wind": "m/s",
    },
    "Imperial (yd, fps, inHg, °F)": {
        "weight": "gr", "diameter": "in", "length": "in", "velocity": "fps",
        "temperature": "°F", "pressure": "inHg", "sight_height": "in", "twist": "in",
        "distance": "yd", "drop": "in", "adjustment": "MOA", "wind": "mph",
    },
    "Mixed (m, fps, hPa, °C)": {
        "weight": "gr", "diameter": "in", "length": "in", "velocity": "fps",
        "temperature": "°C", "pressure": "hPa", "sight_height": "cm", "twist": "in",
        "distance": "m", "drop": "cm", "adjustment": "Mil", "wind": "fps",
    },
}

def apply_unit_profile(profile_dict: dict):
    for k in ("distance","velocity","temperature","pressure","sight_height","drop",
              "adjustment","twist","weight","diameter","length","energy","ogw","angular","target_height"):
        setattr(PreferredUnits, k, profile_dict[k])

# ---- widget helpers ----
def _lbl(parent, text, row, col, sticky="w", span=1):
    lbl = ttk.Label(parent, text=text)
    lbl.grid(row=row, column=col, columnspan=span, sticky=sticky, padx=(3,1), pady=1)
    return lbl

def _ent(parent, default, width, row, col):
    e = ttk.Entry(parent, width=width)
    e.insert(0, default)
    e.grid(row=row, column=col, sticky="w", padx=(1,0), pady=1)
    return e

def _cbo(parent, values, default, width, row, col):
    cb = ttk.Combobox(parent, values=list(values), width=width, state="readonly")
    cb.set(default)
    cb.grid(row=row, column=col, sticky="w", padx=(1,0), pady=1)
    return cb

# ============================================================
# universal module — column header left-click sort (asc/desc)
# ============================================================
class ColumnSorter:
    """Left-click column header to cycle sort (asc → desc → restore); shared across all Treeviews"""
    def __init__(self, tree: ttk.Treeview):
        self.tree = tree
        self._sort_col = None       # current sort column ID
        self._sort_dir = None       # 'asc' | 'desc' | None
        self._original_order = []   # original row order (iid list)
        self._original_headers = {} # col → initial header text (without sort arrow)
        for col in tree['columns']:
            self._original_headers[col] = tree.heading(col, 'text') or col
        tree.bind('<Button-1>', self._on_click, add='+')

    def _on_click(self, event):
        if self.tree.identify_region(event.x, event.y) != 'heading':
            return
        col_idx = int(self.tree.identify_column(event.x).replace('#', '')) - 1
        cols = self.tree['columns']
        if col_idx < 0 or col_idx >= len(cols):
            return
        col_id = cols[col_idx]

        # State cycle: None → asc → desc → None (restore original order)
        if self._sort_col == col_id:
            if self._sort_dir is None:
                self._sort_dir = 'asc'
            elif self._sort_dir == 'asc':
                self._sort_dir = 'desc'
            else:
                self._sort_col = None
                self._sort_dir = None
        else:
            self._sort_col = col_id
            self._sort_dir = 'asc'

        self._apply()

    def _apply(self):
        # 1. Restore all header text (remove sort arrows)
        for c in self.tree['columns']:
            orig = self._original_headers.get(c, c)
            self.tree.heading(c, text=orig)

        # 2. Get all rows + sort
        items = list(self.tree.get_children())
        if self._sort_dir is not None and self._sort_col is not None:
            # Capture original order on first sort
            if not self._original_order:
                self._original_order = list(items)
            col_idx = self.tree['columns'].index(self._sort_col)
            def _key(iid):
                v = self.tree.item(iid, 'values')
                if col_idx >= len(v):
                    return (1, 0, '')
                s = str(v[col_idx])
                try:
                    return (0, float(s), s)
                except (ValueError, TypeError):
                    return (1, 0, s)
            items.sort(key=_key, reverse=(self._sort_dir == 'desc'))
        else:
            # Restore original order
            if self._original_order:
                orig_set = set(self._original_order)
                items = [i for i in self._original_order if i in orig_set]

        # 3. Move rows to sorted positions
        for iid in items:
            self.tree.move(iid, '', 'end')

        # 4. Append sort arrow indicator
        if self._sort_dir and self._sort_col:
            arrow = '▲ ' if self._sort_dir == 'asc' else '▼ '
            orig = self._original_headers.get(self._sort_col, self._sort_col)
            self.tree.heading(self._sort_col, text=arrow + orig)

    def reset(self):
        """Call after data update: clear sort state, update header snapshot"""
        self._sort_col = None
        self._sort_dir = None
        self._original_order = []
        for c in self.tree['columns']:
            self._original_headers[c] = self.tree.heading(c, 'text') or c

# ============================================================
# universal module — column visibility toggle (right-click header)
# ============================================================
class ColumnToggle:
    """Right-click column header for visibility toggle menu — shared across all Treeviews"""
    def __init__(self, tree: ttk.Treeview):
        self.tree = tree
        self._widths = {}
        self._vars = {}   # col → tk.IntVar, prevent GC
        for col in tree['columns']:
            self._widths[col] = tree.column(col, 'width')
            self._vars[col] = tk.IntVar(value=1)
        tree.bind('<Button-3>', self._popup)
        tree.bind('<Button-2>', self._popup)

    def _popup(self, event):
        if self.tree.identify_region(event.x, event.y) != 'heading':
            return
        menu = tk.Menu(self.tree, tearoff=0)
        for col in self.tree['columns']:
            vis = self.tree.column(col, 'width') > 0
            self._vars[col].set(1 if vis else 0)
            label = self.tree.heading(col, 'text') or col
            menu.add_checkbutton(
                label=label, variable=self._vars[col],
                command=lambda c=col: self._toggle(c))
        menu.tk_popup(event.x_root, event.y_root)

    def _toggle(self, col):
        if self._vars[col].get():
            self.tree.column(col, width=self._widths.get(col, 100), stretch=True)
        else:
            self.tree.column(col, width=0, stretch=False)

# ============================================================
# main app
# ============================================================
class BallisticApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ExteriorBallisticsCalc  -  py-ballisticcalc 2.2.10")
        self.root.geometry("1340x860")
        self.root.minsize(960, 700)
        self.root.configure(bg="#F0F0F0")

        style = ttk.Style()
        av = style.theme_names()
        if "vista" in av: style.theme_use("vista")
        elif "winnative" in av: style.theme_use("winnative")
        for sn in ("TLabel","TLabelframe","TButton","TEntry","TCombobox","Treeview","Treeview.Heading"):
            style.configure(sn, font=("Microsoft YaHei", 9))
        style.configure("Treeview", rowheight=26)
        style.configure("TLabelframe", background="#F0F0F0", borderwidth=1, relief="solid")
        style.configure("TLabelframe.Label", background="#F0F0F0", font=("Microsoft YaHei",9,"bold"))
        style.configure("Status.TLabel", background="#E0E0E0", font=("Microsoft YaHei",9),
                        padding=(8,3), anchor="w")
        style.configure("Title.TLabel", background="#F0F0F0", font=("Microsoft YaHei",12,"bold"))
        style.configure("Small.TLabel", background="#F0F0F0", font=("Microsoft YaHei",8),
                        foreground="#777777")
        style.configure("Placeholder.TEntry", foreground="#999999")

        self._results: HitResult | None = None
        self._all_results: list[HitResult] = []  # multi-ammo calculation results
        self._table_step: float = 100.0
        self._unit_profile: dict | None = None
        self._unit_label_widgets: list[tuple[tk.StringVar, str]] = []  # (StringVar, key)
        self._cursor_annot = None
        self._ammo_library: list[dict] = []
        self._active_indices: list[int] = []
        self._current_ammo_idx: int = -1
        self._build_ui()

    # ============================================================
    # UI structure
    # ============================================================
    def _build_ui(self):
        ttk.Separator(self.root, orient="horizontal").pack(fill="x")

        self._paned = ttk.PanedWindow(self.root, orient="horizontal")
        self._paned.pack(fill="both", expand=True, padx=6, pady=(4,2))

        # left panel
        lo = ttk.Frame(self._paned, width=420); self._paned.add(lo, weight=0)
        lc = tk.Canvas(lo, width=420, highlightthickness=0, bg="#F0F0F0")
        ls = ttk.Scrollbar(lo, orient="vertical", command=lc.yview)
        self._input_frame = ttk.Frame(lc)
        self._input_frame.bind("<Configure>", lambda e: lc.configure(scrollregion=lc.bbox("all")))
        lc.create_window((0,0), window=self._input_frame, anchor="nw")
        lc.configure(yscrollcommand=ls.set)
        lc.pack(side="left", fill="both", expand=True); ls.pack(side="right", fill="y")
        lc.bind("<MouseWheel>", lambda e: lc.yview_scroll(int(-1*(e.delta/120)), "units"))
        self._build_inputs(self._input_frame)

        # right panel
        rf = ttk.Frame(self._paned); self._paned.add(rf, weight=1)
        self._build_outputs(rf)

        # status bar
        sb = ttk.Frame(self.root, style="Status.TLabel"); sb.pack(fill="x", side="bottom")
        self._status_label = ttk.Label(sb, text="Ready", style="Status.TLabel")
        self._status_label.pack(side="left", fill="x", expand=True)

        # load saved config or initialize defaults
        self._load_ammo_configs()

        # auto-save on window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ============================================================
    # parameter input
    # ============================================================
    def _build_inputs(self, parent: ttk.Frame):
        row = 0

        # SD / i StringVar (bidirectional linked)
        self.sd_var = tk.StringVar(value="-")
        self.i_var = tk.StringVar(value="-")
        self.recoil_var = tk.StringVar(value="-")
        self.energy_var = tk.StringVar(value="-")
        self.sek_var = tk.StringVar(value="-")
        self._sd_locked = False
        self._bulk_loading = 0

        # ---- Ammo ----
        grp = ttk.LabelFrame(parent, text="Ammo", padding=(6,3))
        grp.grid(row=row, column=0, sticky="ew", padx=2, pady=(2,0)); row += 1; r = 0
        _lbl(grp, "Drag Table:", r, 0)
        self.drag_cb = _cbo(grp, DRAG_TABLES, "G7 (Low-Drag)", 14, r, 1)
        self.drag_cb.bind("<<ComboboxSelected>>", self._on_drag_table_change)
        _lbl(grp, "Ballistic Coefficient BC:", r, 2)
        self.bc_var = tk.StringVar(value="")
        self.bc_ent = ttk.Entry(grp, textvariable=self.bc_var, width=8)
        self.bc_ent.grid(row=r, column=3, sticky="w", padx=(1,0), pady=1)
        r += 1
        # Custom drag table data entry (visible only when "Custom (Mach-CD)" selected)
        self._custom_drag_frame = ttk.Frame(grp)
        self._custom_drag_frame.grid(row=r, column=0, columnspan=6, sticky="ew", padx=2, pady=2)
        ttk.Label(self._custom_drag_frame, text="Mach-CD Data Points:",
                  style="Small.TLabel").pack(anchor="w")
        self._custom_drag_text = tk.Text(self._custom_drag_frame, height=4, width=16,
                                         font=("Consolas", 9))
        self._custom_drag_text.pack(anchor="w", pady=1)
        self._custom_drag_text.insert("1.0", "Mach,CD")
        self._custom_drag_placeholder = True
        self._custom_drag_text.bind("<FocusIn>", self._on_custom_drag_focus_in)
        self._custom_drag_text.bind("<FocusOut>", self._on_custom_drag_focus_out)
        self._custom_drag_frame.grid_remove()  # hidden by default
        r += 1
        _lbl(grp, "Sectional Density SD:", r, 0)
        ttk.Label(grp, textvariable=self.sd_var, style="Small.TLabel", width=10, anchor="center",
                  background="#f0f0f0", relief="sunken").grid(row=r, column=1, sticky="ew", padx=(1,4), pady=1)
        ttk.Label(grp, text="lb/in²", style="Small.TLabel").grid(row=r, column=2, sticky="w", padx=(1,4), pady=1)
        self.i_label_var = tk.StringVar(value="Form Factor i (G7):")
        _lbl(grp, "", r, 3).configure(textvariable=self.i_label_var)
        self.i_ent = ttk.Entry(grp, textvariable=self.i_var, width=8, justify="center")
        self.i_ent.grid(row=r, column=4, sticky="w", padx=(1,0), pady=1)
        self.i_ent.bind("<FocusIn>", self._on_i_focus_in)
        self.i_ent.bind("<FocusOut>", self._on_i_focus_out)
        r += 1
        _lbl(grp, "Bullet Weight:", r, 0); self.wgt_ent = _ent(grp, "", 7, r, 1)
        self.ul_weight = tk.StringVar(value="g")
        ttk.Label(grp, textvariable=self.ul_weight, style="Small.TLabel").grid(row=r, column=2, sticky="w", padx=(1,4), pady=1)
        self._unit_label_widgets.append((self.ul_weight, "weight"))
        _lbl(grp, "Bullet Diameter:", r, 3); self.dia_ent = _ent(grp, "", 7, r, 4)
        self.ul_diameter = tk.StringVar(value="mm")
        ttk.Label(grp, textvariable=self.ul_diameter, style="Small.TLabel").grid(row=r, column=5, sticky="w", padx=(1,4), pady=1)
        self._unit_label_widgets.append((self.ul_diameter, "diameter"))
        r += 1
        _lbl(grp, "Muzzle Velocity:", r, 0); self.mv_ent = _ent(grp, "", 7, r, 1)
        self.mv_ent.bind("<FocusOut>", self._update_energy_recoil)
        self.mv_ent.bind("<Return>", self._update_energy_recoil)
        self.ul_velocity = tk.StringVar(value="m/s")
        ttk.Label(grp, textvariable=self.ul_velocity, style="Small.TLabel").grid(row=r, column=2, sticky="w", padx=(1,4), pady=1)
        self._unit_label_widgets.append((self.ul_velocity, "velocity"))
        _lbl(grp, "Bullet Length:", r, 3); self.len_ent = _ent(grp, "", 7, r, 4)
        self.ul_length = tk.StringVar(value="mm")
        ttk.Label(grp, textvariable=self.ul_length, style="Small.TLabel").grid(row=r, column=5, sticky="w", padx=(1,4), pady=1)
        self._unit_label_widgets.append((self.ul_length, "length"))
        r += 1
        _lbl(grp, "Recoil Impulse:", r, 0)
        ttk.Label(grp, textvariable=self.recoil_var, style="Small.TLabel", width=10, anchor="center",
                  background="#f0f0f0", relief="sunken").grid(row=r, column=1, sticky="ew", padx=(1,4), pady=1)
        self.ul_recoil = tk.StringVar(value="N·s")
        ttk.Label(grp, textvariable=self.ul_recoil, style="Small.TLabel").grid(row=r, column=2, sticky="w", padx=(1,4), pady=1)
        _lbl(grp, "Muzzle Energy:", r, 3)
        ttk.Label(grp, textvariable=self.energy_var, style="Small.TLabel", width=10, anchor="center",
                  background="#f0f0f0", relief="sunken").grid(row=r, column=4, sticky="ew", padx=(1,4), pady=1)
        self.ul_energy = tk.StringVar(value="J")
        ttk.Label(grp, textvariable=self.ul_energy, style="Small.TLabel").grid(row=r, column=5, sticky="w", padx=(2,4), pady=1)
        r += 1
        _lbl(grp, "Sectional KE:", r, 0)
        ttk.Label(grp, textvariable=self.sek_var, style="Small.TLabel", width=10, anchor="center",
                  background="#f0f0f0", relief="sunken").grid(row=r, column=1, sticky="ew", padx=(1,4), pady=1)
        ttk.Label(grp, text="J/cm²", style="Small.TLabel").grid(row=r, column=2, sticky="w", padx=(1,4), pady=1)
        r += 1
        _lbl(grp, "Powder Temp:", r, 0); self.ptemp_ent = _ent(grp, "", 6, r, 1)
        self.ul_temperature = tk.StringVar(value="°C")
        ttk.Label(grp, textvariable=self.ul_temperature, style="Small.TLabel").grid(row=r, column=2, sticky="w", padx=(1,4), pady=1)
        self._unit_label_widgets.append((self.ul_temperature, "temperature"))
        _lbl(grp, "Temp Modifier:", r, 3); self.tmod_ent = _ent(grp, "", 6, r, 4)
        ttk.Label(grp, text="%/15°C", style="Small.TLabel").grid(row=r, column=5, sticky="w", padx=(1,4), pady=1)
        r += 1
        self.pwdr_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(grp, text="Enable Powder Temperature Sensitivity", variable=self.pwdr_var).grid(
            row=r, column=0, columnspan=3, sticky="w", padx=2, pady=2)
        self.mbc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(grp, text="Enable Multi-BC", variable=self.mbc_var,
                        command=self._toggle_mbc).grid(row=r, column=3, columnspan=3, sticky="w", padx=2)
        r += 1
        self._mbc_frame = ttk.Frame(grp)
        self._mbc_frame.grid(row=r, column=0, columnspan=9, sticky="ew", padx=2)
        self._mbc_rows: list[tuple[ttk.Entry, ttk.Entry]] = []
        self._mbc_add_btn = ttk.Button(self._mbc_frame, text="+ BC Point",
                                       command=self._add_mbc_row, width=8)

        # SD/i linked event bindings
        self.wgt_ent.bind("<FocusOut>", self._update_sd_i)
        self.wgt_ent.bind("<Return>", self._update_sd_i)
        self.wgt_ent.bind("<FocusOut>", self._update_energy_recoil, add="+")
        self.wgt_ent.bind("<Return>", self._update_energy_recoil, add="+")
        self.dia_ent.bind("<FocusOut>", self._update_sd_i)
        self.dia_ent.bind("<Return>", self._update_sd_i)
        self.dia_ent.bind("<FocusOut>", self._update_energy_recoil, add="+")
        self.dia_ent.bind("<Return>", self._update_energy_recoil, add="+")
        self.bc_var.trace_add("write", self._update_sd_i)
        self.i_var.trace_add("write", self._on_i_change)

        # ---- weapon ----
        grp = ttk.LabelFrame(parent, text="Weapon", padding=(6,3))
        grp.grid(row=row, column=0, sticky="ew", padx=2, pady=0); row += 1; r = 0
        _lbl(grp, "Sight Height:", r, 0); self.sight_ent = _ent(grp, "0", 7, r, 1)
        self.ul_sight_height = tk.StringVar(value="cm")
        ttk.Label(grp, textvariable=self.ul_sight_height, style="Small.TLabel").grid(row=r, column=2, sticky="w", padx=(1,4), pady=1)
        self._unit_label_widgets.append((self.ul_sight_height, "sight_height"))
        _lbl(grp, "Twist Rate:", r, 3); self.twist_ent = _ent(grp, "0", 7, r, 4)
        self.ul_twist = tk.StringVar(value="cm")
        ttk.Label(grp, textvariable=self.ul_twist, style="Small.TLabel").grid(row=r, column=5, sticky="w", padx=(1,4), pady=1)
        self._unit_label_widgets.append((self.ul_twist, "twist"))
        self.twist_dir_var = tk.StringVar(value="RH Twist (+)")
        ttk.Combobox(grp, textvariable=self.twist_dir_var,
                     values=["RH Twist (+)", "LH Twist (-)"], width=8, state="readonly").grid(row=r, column=6, sticky="w", padx=2)

        # ---- Atmosphere ----
        grp = ttk.LabelFrame(parent, text="Atmosphere", padding=(6,3))
        grp.grid(row=row, column=0, sticky="ew", padx=2, pady=0); row += 1; r = 0
        _lbl(grp, "Altitude:", r, 0)
        self.alt_var = tk.StringVar(value="0")
        self.alt_ent = ttk.Entry(grp, textvariable=self.alt_var, width=7)
        self.alt_ent.grid(row=r, column=1, sticky="w", padx=(1,0), pady=1)
        self.alt_var.trace_add("write", self._on_altitude_change)
        self.alt_ent.bind("<FocusOut>", self._on_altitude_change)
        self.alt_ent.bind("<Return>", self._on_altitude_change)
        self.ul_distance2 = tk.StringVar(value="m")
        ttk.Label(grp, textvariable=self.ul_distance2, style="Small.TLabel").grid(row=r, column=2, sticky="w", padx=(1,4), pady=1)
        self._unit_label_widgets.append((self.ul_distance2, "distance"))
        _lbl(grp, "Pressure:", r, 3); self.pres_ent = _ent(grp, "1013.2", 7, r, 4)
        self.ul_pressure = tk.StringVar(value="hPa")
        ttk.Label(grp, textvariable=self.ul_pressure, style="Small.TLabel").grid(row=r, column=5, sticky="w", padx=(1,4), pady=1)
        self._unit_label_widgets.append((self.ul_pressure, "pressure"))
        self.vacuum_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(grp, text="Vacuum (No Drag)", variable=self.vacuum_var,
                        command=self._on_vacuum_toggle).grid(row=r, column=6, sticky="w", padx=(16,0))
        r += 1
        _lbl(grp, "Temperature:", r, 0); self.temp_ent = _ent(grp, "15.0", 7, r, 1)
        self.ul_temperature2 = tk.StringVar(value="°C")
        ttk.Label(grp, textvariable=self.ul_temperature2, style="Small.TLabel").grid(row=r, column=2, sticky="w", padx=(1,4), pady=1)
        self._unit_label_widgets.append((self.ul_temperature2, "temperature"))
        _lbl(grp, "Humidity:", r, 3); self.hum_ent = _ent(grp, "0", 7, r, 4)
        ttk.Label(grp, text="%", style="Small.TLabel").grid(row=r, column=5, sticky="w", padx=(1,4), pady=1)
        icao_btn = ttk.Button(grp, text="ICAO Standard", command=self._fill_icao)
        icao_btn.grid(row=r, column=6, sticky="w", padx=(16,0))

        # ---- wind ----
        grp = ttk.LabelFrame(parent, text="Wind", padding=(6,3))
        grp.grid(row=row, column=0, sticky="ew", padx=2, pady=0); row += 1; r = 0
        _lbl(grp, "Wind Speed:", r, 0); self.windv_ent = _ent(grp, "4.0", 7, r, 1)
        self.ul_wind = tk.StringVar(value="m/s")
        ttk.Label(grp, textvariable=self.ul_wind, style="Small.TLabel").grid(row=r, column=2, sticky="w", padx=(1,4), pady=1)
        self._unit_label_widgets.append((self.ul_wind, "wind"))
        _lbl(grp, "Wind Direction:", r, 3); self.windd_ent = _ent(grp, "90", 7, r, 4)
        ttk.Label(grp, text="° (90 = L→R)", style="Small.TLabel").grid(row=r, column=5, sticky="w", padx=(1,4), pady=1)

        # ---- Shot ----
        grp = ttk.LabelFrame(parent, text="Shot", padding=(6,3))
        grp.grid(row=row, column=0, sticky="ew", padx=2, pady=0); row += 1; r = 0
        self.zero_mode_var = tk.BooleanVar(value=True)
        self.apex_mode_var = tk.BooleanVar(value=False)
        def _toggle_zero():
            if self.zero_mode_var.get():
                self.apex_mode_var.set(False)
                self.apex_lock_ent.delete(0, "end")
                self.apex_lock_ent.configure(state="disabled")
                self.zero_ent.configure(state="normal")
            else:
                self.apex_mode_var.set(True)
                self.zero_ent.delete(0, "end")
                self.zero_ent.configure(state="disabled")
                self.apex_lock_ent.configure(state="normal")
        def _toggle_apex():
            if self.apex_mode_var.get():
                self.zero_mode_var.set(False)
                self.zero_ent.delete(0, "end")
                self.zero_ent.configure(state="disabled")
                self.apex_lock_ent.configure(state="normal")
            else:
                self.zero_mode_var.set(True)
                self.apex_lock_ent.delete(0, "end")
                self.apex_lock_ent.configure(state="disabled")
                self.zero_ent.configure(state="normal")
        ttk.Checkbutton(grp, text="Zero Distance", variable=self.zero_mode_var,
                        command=_toggle_zero).grid(row=r, column=0, sticky="w", padx=(1,0))
        self.zero_ent = _ent(grp, "0", 7, r, 1)
        self.ul_distance3 = tk.StringVar(value="m")
        ttk.Label(grp, textvariable=self.ul_distance3, style="Small.TLabel").grid(row=r, column=2, sticky="w", padx=(0,2), pady=1)
        self._unit_label_widgets.append((self.ul_distance3, "distance"))
        ttk.Checkbutton(grp, text="Max Ordinate", variable=self.apex_mode_var,
                        command=_toggle_apex).grid(row=r, column=3, sticky="w", padx=(4,0))
        self.apex_lock_ent = _ent(grp, "150", 7, r, 4)
        self.apex_lock_ent.bind("<Return>", self._on_apex_lock)
        self.apex_lock_ent.bind("<FocusOut>", self._on_apex_lock)
        self.ul_apex_lock = tk.StringVar(value="cm")
        ttk.Label(grp, textvariable=self.ul_apex_lock, style="Small.TLabel").grid(row=r, column=5, sticky="w", padx=(0,2), pady=1)
        self._unit_label_widgets.append((self.ul_apex_lock, "drop"))
        self.apex_lock_ent.configure(state="disabled")
        r += 1
        _lbl(grp, "Table Step:", r, 0); self.step_ent = _ent(grp, "100", 7, r, 1)
        self.ul_distance5 = tk.StringVar(value="m")
        ttk.Label(grp, textvariable=self.ul_distance5, style="Small.TLabel").grid(row=r, column=2, sticky="w", padx=(0,2), pady=1)
        self._unit_label_widgets.append((self.ul_distance5, "distance"))
        _lbl(grp, "Max Range:", r, 3); self.range_ent = _ent(grp, "1200", 7, r, 4)
        self.ul_distance4 = tk.StringVar(value="m")
        ttk.Label(grp, textvariable=self.ul_distance4, style="Small.TLabel").grid(row=r, column=5, sticky="w", padx=(0,2), pady=1)
        self._unit_label_widgets.append((self.ul_distance4, "distance"))
        r += 1
        _lbl(grp, "Look Angle:", r, 0); self.look_ent = _ent(grp, "0", 7, r, 1)
        ttk.Label(grp, text="°", style="Small.TLabel").grid(row=r, column=2, sticky="w", padx=(0,2), pady=1)
        r += 1
        _lbl(grp, "Unit System:", r, 0)
        self.unit_cb = _cbo(grp, UNIT_PROFILES, DEFAULT_PROFILE, 18, r, 1)
        self.unit_cb.bind("<<ComboboxSelected>>", self._on_unit_change)
        r += 1
        _lbl(grp, "Engine:", r, 0)
        self.engine_cb = _cbo(grp, ENGINES, "RK4 (Cython/C++ Accelerated)", 18, r, 1)
        r += 1
        ttk.Label(grp, text="Display:").grid(row=r, column=0, sticky="w", padx=2)
        # Select All
        self.show_all_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grp, text="Select All", variable=self.show_all_var,
                        command=self._on_show_all).grid(row=r, column=1, sticky="w", padx=2)
        # Apogee
        self.show_apex_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grp, text="Apogee", variable=self.show_apex_var,
                        command=self._on_show_sub).grid(row=r, column=2, sticky="w", padx=2)
        # Transonic
        self.show_mach_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grp, text="Transonic", variable=self.show_mach_var,
                        command=self._on_show_sub).grid(row=r, column=3, sticky="w", padx=2)
        r += 1
        # Bore Line
        self.show_barrel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grp, text="Bore Line", variable=self.show_barrel_var).grid(
            row=r, column=0, sticky="w", padx=2)
        # Sight Line
        self.show_sight_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grp, text="Sight Line", variable=self.show_sight_var).grid(
            row=r, column=1, sticky="w", padx=2)
        # Zero
        self.show_zero_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grp, text="Zero", variable=self.show_zero_var,
                        command=self._on_show_sub).grid(row=r, column=2, sticky="w", padx=2)
        # Velocity
        self.show_vel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grp, text="Velocity", variable=self.show_vel_var).grid(
            row=r, column=3, sticky="w", padx=2)

        # ---- buttons ----
        bf = ttk.Frame(parent)
        bf.grid(row=row, column=0, sticky="ew", padx=4, pady=(4,2)); row += 1
        self.calc_btn = ttk.Button(bf, text="Solve", command=self._calculate)
        self.calc_btn.pack(fill="x")
        self._progress = ttk.Progressbar(parent, mode="indeterminate")
        self._progress.grid(row=row, column=0, sticky="ew", padx=4, pady=2); row += 1

        # ---- active list (ammo included in calculation) ----
        grp_active = ttk.LabelFrame(parent, text="Active List", padding=(6,3))
        grp_active.grid(row=row, column=0, sticky="nsew", padx=2, pady=(6,0)); row += 1
        parent.rowconfigure(row-1, weight=0)
        active_outer = ttk.Frame(grp_active)
        active_outer.pack(fill="both", expand=True)
        active_lst_frame = ttk.Frame(active_outer)
        active_lst_frame.pack(side="left", fill="both", expand=True)
        self._active_listbox = tk.Listbox(active_lst_frame, height=4, exportselection=False,
                                          font=("Microsoft YaHei", 8), width=22)
        self._active_listbox.pack(side="left", fill="both", expand=True)
        active_sb = ttk.Scrollbar(active_lst_frame, orient="vertical",
                                  command=self._active_listbox.yview)
        self._active_listbox.configure(yscrollcommand=active_sb.set)
        active_sb.pack(side="right", fill="y")
        self._active_listbox.bind("<<ListboxSelect>>", self._on_active_select)
        active_btn = ttk.Frame(active_outer)
        active_btn.pack(side="right", fill="y", padx=(4,0))
        ttk.Button(active_btn, text="Remove", command=self._remove_from_active, width=7).pack(side="top", pady=1)
        ttk.Button(active_btn, text="Clear All", command=self._clear_active, width=7).pack(side="top", pady=1)
        ttk.Button(active_btn, text="↑ Import", command=self._import_to_active, width=7).pack(side="top", pady=1)

        # ---- ammo library (all ammo) ----
        grp_lib = ttk.LabelFrame(parent, text="Ammo Library", padding=(6,3))
        grp_lib.grid(row=row, column=0, sticky="nsew", padx=2, pady=(6,0)); row += 1
        parent.rowconfigure(row-1, weight=1)
        # name
        name_frame = ttk.Frame(grp_lib)
        name_frame.pack(fill="x", pady=(0,4))
        ttk.Label(name_frame, text="Name:").pack(side="left", padx=(0,4))
        self._ammo_name_var = tk.StringVar()
        self._ammo_name_ent = ttk.Entry(name_frame, textvariable=self._ammo_name_var, width=24)
        self._ammo_name_ent.pack(side="left", fill="x", expand=True)
        # list + buttons (left-right layout)
        lib_outer = ttk.Frame(grp_lib)
        lib_outer.pack(fill="both", expand=True)
        lib_lst_frame = ttk.Frame(lib_outer)
        lib_lst_frame.pack(side="left", fill="both", expand=True)
        self._library_listbox = tk.Listbox(lib_lst_frame, height=9, exportselection=False,
                                           font=("Microsoft YaHei", 8), width=22)
        self._library_listbox.pack(side="left", fill="both", expand=True)
        lib_sb = ttk.Scrollbar(lib_lst_frame, orient="vertical",
                               command=self._library_listbox.yview)
        self._library_listbox.configure(yscrollcommand=lib_sb.set)
        lib_sb.pack(side="right", fill="y")
        self._library_listbox.bind("<<ListboxSelect>>", self._on_library_select)
        lib_btn = ttk.Frame(lib_outer)
        lib_btn.pack(side="right", fill="y", padx=(4,0))
        ttk.Button(lib_btn, text="+ Add", command=self._add_ammo, width=7).pack(side="top", pady=1)
        ttk.Button(lib_btn, text="📋 Duplicate", command=self._duplicate_ammo, width=7).pack(side="top", pady=1)
        ttk.Button(lib_btn, text="- Delete", command=self._delete_ammo, width=7).pack(side="top", pady=1)
        ttk.Button(lib_btn, text="💾 Save", command=self._save_ammo, width=7).pack(side="top", pady=1)

    # ============================================================
    # output area
    # ============================================================
    def _build_outputs(self, parent: ttk.Frame):
        # notebook tabs
        self._notebook = ttk.Notebook(parent)
        self._notebook.pack(fill="both", expand=True)
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # ====== Tab 1: Trajectory ======
        tab1 = ttk.Frame(self._notebook)
        self._notebook.add(tab1, text="Trajectory")

        vp1 = ttk.PanedWindow(tab1, orient="vertical")
        vp1.pack(fill="both", expand=True)

        # top: chart + locate controls
        upper1 = ttk.Frame(vp1)
        vp1.add(upper1, weight=1)

        # chart
        plot_frame = ttk.Frame(upper1)
        plot_frame.pack(fill="both", expand=True)

        self._fig = Figure(figsize=(8, 4.5), dpi=120, facecolor="white")
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, plot_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        # debounce: skip per-pixel redraw when dragging sash
        self._install_draw_debounce(self._canvas, '_draw_timer1')

        self._ax.set_xlabel("Range (m)"); self._ax.set_ylabel("Height (cm)")
        self._ax.xaxis.set_major_locator(MaxNLocator(nbins=20, steps=[1, 5, 10]))
        self._ax.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
        self._ax.grid(True, alpha=0.35, linewidth=0.5)
        self._ax.axhline(y=0, color="#999999", linewidth=0.8, linestyle="--")
        self._ax.set_facecolor("white")
        self._fig.tight_layout(pad=2.0)

        # mouse hover
        self._cursor_annot = self._ax.annotate("", xy=(0,0), xytext=(12,12),
            textcoords="offset points", fontsize=8, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#AAA", alpha=0.9),
            visible=False, zorder=99, annotation_clip=False)
        self._highlight_scatter = self._ax.scatter([], [], s=80, c="none",
            edgecolors="#FF6600", linewidths=2, zorder=98, visible=False)
        self._ch_vline = self._ax.axvline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._ch_hline = self._ax.axhline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._fig.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._fig.canvas.mpl_connect("draw_event", self._on_draw_tab1)

        # bottom: toolbar + locate controls + table + summary
        lower1 = ttk.Frame(vp1)
        vp1.add(lower1, weight=0)

        toolbar = _ChineseToolbar(self._canvas, lower1)
        toolbar.update(); toolbar.pack(side="top", fill="x")
        self._tint_copy_button(toolbar)

        # locate controls
        loc_frame = ttk.Frame(lower1)
        loc_frame.pack(fill="x", padx=2, pady=(4,0))
        ttk.Label(loc_frame, text="Range:").pack(side="left", padx=(0,2))
        self.loc_dist_var = tk.StringVar()
        self.loc_dist_ent = ttk.Entry(loc_frame, textvariable=self.loc_dist_var, width=10)
        self.loc_dist_ent.pack(side="left", padx=2)
        ttk.Button(loc_frame, text="Locate", command=self._locate_by_distance, width=5).pack(side="left", padx=2)
        ttk.Label(loc_frame, text="Height:").pack(side="left", padx=(8,2))
        self.loc_height_var = tk.StringVar()
        self.loc_height_ent = ttk.Entry(loc_frame, textvariable=self.loc_height_var, width=10)
        self.loc_height_ent.pack(side="left", padx=2)
        ttk.Button(loc_frame, text="Locate", command=self._locate_by_height, width=5).pack(side="left", padx=2)
        ttk.Label(loc_frame, text="Time:").pack(side="left", padx=(8,2))
        self.loc_time_var = tk.StringVar()
        self.loc_time_ent = ttk.Entry(loc_frame, textvariable=self.loc_time_var, width=10)
        self.loc_time_ent.pack(side="left", padx=2)
        ttk.Button(loc_frame, text="Locate", command=self._locate_by_time, width=5).pack(side="left", padx=2)
        ttk.Button(loc_frame, text="Clear", command=self._clear_highlight, width=5).pack(side="left", padx=8)

        # table
        tbl_frame = ttk.LabelFrame(lower1, text="Trajectory Data", padding=(4,2))
        tbl_frame.pack(fill="both", expand=True, padx=2, pady=(4,0))
        self._table_cols = ("distance","velocity","mach","time","height",
                            "drop_angle","slant_height","slant_distance","windage",
                            "windage_angle","angle","density_ratio","drag","energy","ogw","flag")
        self._tree = ttk.Treeview(tbl_frame, columns=self._table_cols, show="headings", height=10)
        ColumnToggle(self._tree)
        self._sorter1 = ColumnSorter(self._tree)
        headers = {
            "distance":"Range","velocity":"Velocity","mach":"Mach","time":"Time / s",
            "height":"Height","drop_angle":"Drop Angle","slant_height":"Slant Drop",
            "slant_distance":"Slant Distance","windage":"Drift","windage_angle":"Drift Angle",
            "angle":"Trajectory Angle","density_ratio":"Density Ratio","drag":"Drag Coeff.",
            "energy":"Energy","ogw":"OGW","flag":"Flag",
        }
        widths = {"distance":72,"velocity":72,"mach":58,"time":70,"height":70,
                  "drop_angle":66,"slant_height":70,"slant_distance":70,"windage":62,
                  "windage_angle":62,"angle":60,"density_ratio":68,"drag":66,
                  "energy":72,"ogw":80,"flag":66}
        for col in self._table_cols:
            self._tree.heading(col, text=headers.get(col, col))
            self._tree.column(col, width=widths.get(col, 64), anchor="center")
        ts = ttk.Scrollbar(tbl_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=ts.set)
        self._tree.pack(side="left", fill="both", expand=True); ts.pack(side="right", fill="y")
        self._tree.configure(xscrollcommand="")

        # summary
        self._summary_var = tk.StringVar(value="")
        ttk.Label(lower1, textvariable=self._summary_var, font=("Microsoft YaHei",9),
                  foreground="#444", padding=(8,4)).pack(fill="x", padx=2, pady=(2,0))

        self._canvas.draw()

        # ====== Tab 2: Analysis ======
        tab2 = ttk.Frame(self._notebook)
        self._notebook.add(tab2, text="Analysis")

        vp2 = ttk.PanedWindow(tab2, orient="vertical")
        vp2.pack(fill="both", expand=True)

        cmp_plot_frame = ttk.Frame(vp2)
        vp2.add(cmp_plot_frame, weight=1)

        self._fig2 = Figure(figsize=(8, 4.5), dpi=120, facecolor="white")
        self._ax2 = self._fig2.add_subplot(111)
        self._canvas2 = FigureCanvasTkAgg(self._fig2, cmp_plot_frame)
        self._canvas2.get_tk_widget().pack(fill="both", expand=True)

        # debounce: skip per-pixel redraw when dragging sash
        self._install_draw_debounce(self._canvas2, '_draw_timer2')

        self._ax2.set_xlabel("Range (m)"); self._ax2.set_ylabel("Height (cm)")
        self._ax2.xaxis.set_major_locator(MaxNLocator(nbins=20, steps=[1, 5, 10]))
        self._ax2.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
        self._ax2.grid(True, alpha=0.35, linewidth=0.5)
        self._ax2.axhline(y=0, color="#999999", linewidth=0.8, linestyle="--")
        self._ax2.set_facecolor("white")
        self._fig2.tight_layout(pad=2.0)

        # bottom: toolbar + locate controls + table
        lower2 = ttk.Frame(vp2)
        vp2.add(lower2, weight=0)

        toolbar2 = _ChineseToolbar(self._canvas2, lower2)
        toolbar2.update(); toolbar2.pack(side="top", fill="x")
        self._tint_copy_button(toolbar2)

        # locate controls
        loc_frame2 = ttk.Frame(lower2)
        loc_frame2.pack(fill="x", padx=2, pady=(4,0))
        ttk.Label(loc_frame2, text="Range:").pack(side="left", padx=(0,2))
        self.loc_dist_var2 = tk.StringVar()
        self.loc_dist_ent2 = ttk.Entry(loc_frame2, textvariable=self.loc_dist_var2, width=10)
        self.loc_dist_ent2.pack(side="left", padx=2)
        self.loc_dist_ent2.bind("<Return>", self._locate_distance2)
        ttk.Button(loc_frame2, text="Locate", command=self._locate_distance2, width=5).pack(side="left", padx=2)
        ttk.Label(loc_frame2, text="Height:").pack(side="left", padx=(8,2))
        self.loc_height_var2 = tk.StringVar()
        self.loc_height_ent2 = ttk.Entry(loc_frame2, textvariable=self.loc_height_var2, width=10)
        self.loc_height_ent2.pack(side="left", padx=2)
        self.loc_height_ent2.bind("<Return>", self._locate_height2)
        ttk.Button(loc_frame2, text="Locate", command=self._locate_height2, width=5).pack(side="left", padx=2)
        ttk.Label(loc_frame2, text="Time:").pack(side="left", padx=(8,2))
        self.loc_time_var2 = tk.StringVar()
        self.loc_time_ent2 = ttk.Entry(loc_frame2, textvariable=self.loc_time_var2, width=10)
        self.loc_time_ent2.pack(side="left", padx=2)
        self.loc_time_ent2.bind("<Return>", self._locate_time2)
        ttk.Button(loc_frame2, text="Locate", command=self._locate_time2, width=5).pack(side="left", padx=2)
        ttk.Button(loc_frame2, text="Clear", command=self._clear_locate2, width=5).pack(side="left", padx=8)

        # comparison table
        cmp_tbl_frame = ttk.LabelFrame(lower2, text="Trajectory Analysis Data", padding=(4,2))
        cmp_tbl_frame.pack(fill="both", expand=True, padx=2, pady=(4,0))

        self._cmp_cols = ("name","mv","sek","recoil","vel_dist","energy_dist",
                          "time_dist","drop_dist","max_apex",
                          "pbr_03","pbr_1","pbr_15","supersonic")
        self._cmp_tree = ttk.Treeview(cmp_tbl_frame, columns=self._cmp_cols, show="headings", height=5)
        ColumnToggle(self._cmp_tree)
        self._sorter2 = ColumnSorter(self._cmp_tree)
        cmp_headers = {"name":"Ammo","mv":"MV / m·s⁻¹","recoil":"Recoil / N·s",
                       "sek":"Sectional KE / J·cm⁻²",
                       "vel_dist":"Velocity / m·s⁻¹","energy_dist":"Energy / J",
                       "time_dist":"Time / s","drop_dist":"Drop / cm",
                       "max_apex":"Max Ordinate / cm",
                       "pbr_03":"PBR 0.3m / m","pbr_1":"PBR 1m / m",
                       "pbr_15":"PBR 1.5m / m","supersonic":"Supersonic Range / m"}
        cmp_widths = {"name":68,"mv":60,"recoil":74,"sek":80,"vel_dist":60,
                      "energy_dist":68,"time_dist":66,"drop_dist":62,
                      "max_apex":72,"pbr_03":82,"pbr_1":76,"pbr_15":78,
                      "supersonic":68}
        for col in self._cmp_cols:
            self._cmp_tree.heading(col, text=cmp_headers.get(col, col))
            self._cmp_tree.column(col, width=cmp_widths.get(col, 64), anchor="center")
        cmp_ts = ttk.Scrollbar(cmp_tbl_frame, orient="vertical", command=self._cmp_tree.yview)
        self._cmp_tree.configure(yscrollcommand=cmp_ts.set)
        self._cmp_tree.pack(side="left", fill="both", expand=True)
        cmp_ts.pack(side="right", fill="y")

        # static locate lines/points (rebuilt at end of _update_comparison_plot)
        self._loc_vline2 = self._ax2.axvline(0, color="#000000", linewidth=0.5,
                                              linestyle="--", visible=False, zorder=85)
        self._loc_hline2 = self._ax2.axhline(0, color="#000000", linewidth=0.5,
                                              linestyle="--", visible=False, zorder=85)
        self._loc_scatter2 = self._ax2.scatter([], [], s=25, c="#FF6600",
                                                zorder=86, visible=False, marker="x",
                                                linewidths=1.2)

        # Tab2 mouse hover elements
        self._cursor_annot2 = self._ax2.annotate("", xy=(0,0), xytext=(12,12),
            textcoords="offset points", fontsize=8, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#AAA", alpha=0.9),
            visible=False, zorder=99, annotation_clip=False)
        self._highlight_scatter2 = self._ax2.scatter([], [], s=80, c="none",
            edgecolors="#FF6600", linewidths=2, zorder=98, visible=False)
        self._ch_vline2 = self._ax2.axvline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._ch_hline2 = self._ax2.axhline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._fig2.canvas.mpl_connect("motion_notify_event", self._on_hover_cmp)
        self._fig2.canvas.mpl_connect("button_press_event", self._on_click_cmp)
        self._fig2.canvas.mpl_connect("draw_event", self._on_draw_tab2)

        self._canvas2.draw()

        # ====== Tab 3: Energy ======
        tab_energy = ttk.Frame(self._notebook)
        self._notebook.add(tab_energy, text="Energy")

        vpE = ttk.PanedWindow(tab_energy, orient="vertical")
        vpE.pack(fill="both", expand=True)

        # top: energy chart
        energy_plot_frame = ttk.Frame(vpE)
        vpE.add(energy_plot_frame, weight=1)

        self._fig4 = Figure(figsize=(8, 4.5), dpi=120, facecolor="white")
        self._ax4 = self._fig4.add_subplot(111)
        self._canvas4 = FigureCanvasTkAgg(self._fig4, energy_plot_frame)
        self._canvas4.get_tk_widget().pack(fill="both", expand=True)

        self._install_draw_debounce(self._canvas4, '_draw_timer4')

        self._ax4.set_xlabel("Range (m)"); self._ax4.set_ylabel("Energy (J)")
        self._ax4.xaxis.set_major_locator(MaxNLocator(nbins=20, steps=[1, 5, 10]))
        self._ax4.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
        self._ax4.grid(True, alpha=0.35, linewidth=0.5)
        self._ax4.axhline(y=0, color="#999999", linewidth=0.8, linestyle="--")
        self._ax4.set_facecolor("white")
        self._fig4.tight_layout(pad=2.0)

        # bottom: toolbar + locate controls + table
        lowerE = ttk.Frame(vpE)
        vpE.add(lowerE, weight=0)

        toolbar4 = _ChineseToolbar(self._canvas4, lowerE)
        toolbar4.update(); toolbar4.pack(side="top", fill="x")
        self._tint_copy_button(toolbar4)

        # locate controls
        loc_frame4 = ttk.Frame(lowerE)
        loc_frame4.pack(fill="x", padx=2, pady=(4,0))
        ttk.Label(loc_frame4, text="Range:").pack(side="left", padx=(0,2))
        self.loc_dist_var4 = tk.StringVar()
        self.loc_dist_ent4 = ttk.Entry(loc_frame4, textvariable=self.loc_dist_var4, width=10)
        self.loc_dist_ent4.pack(side="left", padx=2)
        self.loc_dist_ent4.bind("<Return>", self._locate_distance4)
        ttk.Button(loc_frame4, text="Locate", command=self._locate_distance4, width=5).pack(side="left", padx=2)
        ttk.Label(loc_frame4, text="Energy:").pack(side="left", padx=(8,2))
        self.loc_energy_var4 = tk.StringVar()
        self.loc_energy_ent4 = ttk.Entry(loc_frame4, textvariable=self.loc_energy_var4, width=10)
        self.loc_energy_ent4.pack(side="left", padx=2)
        self.loc_energy_ent4.bind("<Return>", self._locate_energy4)
        ttk.Button(loc_frame4, text="Locate", command=self._locate_energy4, width=5).pack(side="left", padx=2)
        ttk.Label(loc_frame4, text="Time:").pack(side="left", padx=(8,2))
        self.loc_time_var4 = tk.StringVar()
        self.loc_time_ent4 = ttk.Entry(loc_frame4, textvariable=self.loc_time_var4, width=10)
        self.loc_time_ent4.pack(side="left", padx=2)
        self.loc_time_ent4.bind("<Return>", self._locate_time4)
        ttk.Button(loc_frame4, text="Locate", command=self._locate_time4, width=5).pack(side="left", padx=2)
        ttk.Button(loc_frame4, text="Clear", command=self._clear_locate4, width=5).pack(side="left", padx=8)

        # energy analysis table
        energy_tbl_frame = ttk.LabelFrame(lowerE, text="Energy Analysis Data", padding=(4,2))
        energy_tbl_frame.pack(fill="both", expand=True, padx=2, pady=(4,0))

        self._energy_cols = ("name","mv","sek","muzzle_energy","energy_dist",
                            "vel_dist","time_dist","drop_dist","supersonic")
        self._energy_tree = ttk.Treeview(energy_tbl_frame, columns=self._energy_cols, show="headings", height=5)
        ColumnToggle(self._energy_tree)
        self._sorter3 = ColumnSorter(self._energy_tree)
        energy_headers = {"name":"Ammo","mv":"MV","sek":"Sectional KE / J·cm⁻²",
                         "muzzle_energy":"Muzzle Energy",
                         "energy_dist":"Energy","vel_dist":"Velocity",
                         "time_dist":"Time / s","drop_dist":"Drop",
                         "supersonic":"Supersonic Range"}  # unit set dynamically by _update_energy_table
        energy_widths = {"name":68,"mv":60,"sek":80,"muzzle_energy":74,"energy_dist":68,
                        "vel_dist":60,"time_dist":66,"drop_dist":62,
                        "supersonic":68}
        for col in self._energy_cols:
            self._energy_tree.heading(col, text=energy_headers.get(col, col))
            self._energy_tree.column(col, width=energy_widths.get(col, 64), anchor="center")
        e_ts = ttk.Scrollbar(energy_tbl_frame, orient="vertical", command=self._energy_tree.yview)
        self._energy_tree.configure(yscrollcommand=e_ts.set)
        self._energy_tree.pack(side="left", fill="both", expand=True)
        e_ts.pack(side="right", fill="y")

        # static locate lines/points
        self._loc_vline4 = self._ax4.axvline(0, color="#000000", linewidth=0.5,
                                              linestyle="--", visible=False, zorder=85)
        self._loc_hline4 = self._ax4.axhline(0, color="#000000", linewidth=0.5,
                                              linestyle="--", visible=False, zorder=85)
        self._loc_scatter4 = self._ax4.scatter([], [], s=25, c="#FF6600",
                                                zorder=86, visible=False, marker="x",
                                                linewidths=1.2)

        # hover elements
        self._cursor_annot4 = self._ax4.annotate("", xy=(0,0), xytext=(12,12),
            textcoords="offset points", fontsize=8, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#AAA", alpha=0.9),
            visible=False, zorder=99, annotation_clip=False)
        self._highlight_scatter4 = self._ax4.scatter([], [], s=80, c="none",
            edgecolors="#FF6600", linewidths=2, zorder=98, visible=False)
        self._ch_vline4 = self._ax4.axvline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._ch_hline4 = self._ax4.axhline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._fig4.canvas.mpl_connect("motion_notify_event", self._on_hover_energy)
        self._fig4.canvas.mpl_connect("button_press_event", self._on_click_energy)
        self._fig4.canvas.mpl_connect("draw_event", self._on_draw_tab4)

        self._canvas4.draw()

        # ====== Tab 4: Wind Drift ======
        tab3 = ttk.Frame(self._notebook)
        self._notebook.add(tab3, text="Wind Drift")

        vp3 = ttk.PanedWindow(tab3, orient="vertical")
        vp3.pack(fill="both", expand=True)

        # top: wind drift chart
        wind_plot_frame = ttk.Frame(vp3)
        vp3.add(wind_plot_frame, weight=1)

        self._fig3 = Figure(figsize=(8, 4.5), dpi=120, facecolor="white")
        self._ax3 = self._fig3.add_subplot(111)
        self._canvas3 = FigureCanvasTkAgg(self._fig3, wind_plot_frame)
        self._canvas3.get_tk_widget().pack(fill="both", expand=True)

        # debounce
        self._install_draw_debounce(self._canvas3, '_draw_timer3')

        self._ax3.set_xlabel("Range (m)"); self._ax3.set_ylabel("Wind Drift (cm)")
        self._ax3.xaxis.set_major_locator(MaxNLocator(nbins=20, steps=[1, 5, 10]))
        self._ax3.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
        self._ax3.grid(True, alpha=0.35, linewidth=0.5)
        self._ax3.axhline(y=0, color="#999999", linewidth=0.8, linestyle="--")
        self._ax3.set_facecolor("white")
        self._fig3.tight_layout(pad=2.0)

        # bottom: toolbar + locate controls + table
        lower3 = ttk.Frame(vp3)
        vp3.add(lower3, weight=0)

        toolbar3 = _ChineseToolbar(self._canvas3, lower3)
        toolbar3.update(); toolbar3.pack(side="top", fill="x")
        self._tint_copy_button(toolbar3)

        # locate controls + toggle switch
        loc_frame3 = ttk.Frame(lower3)
        loc_frame3.pack(fill="x", padx=2, pady=(4,0))
        ttk.Label(loc_frame3, text="Range:").pack(side="left", padx=(0,2))
        self.loc_dist_var3 = tk.StringVar()
        self.loc_dist_ent3 = ttk.Entry(loc_frame3, textvariable=self.loc_dist_var3, width=10)
        self.loc_dist_ent3.pack(side="left", padx=2)
        self.loc_dist_ent3.bind("<Return>", self._locate_distance3)
        ttk.Button(loc_frame3, text="Locate", command=self._locate_distance3, width=5).pack(side="left", padx=2)
        ttk.Label(loc_frame3, text="Wind Drift:").pack(side="left", padx=(8,2))
        self.loc_windage_var3 = tk.StringVar()
        self.loc_windage_ent3 = ttk.Entry(loc_frame3, textvariable=self.loc_windage_var3, width=10)
        self.loc_windage_ent3.pack(side="left", padx=2)
        self.loc_windage_ent3.bind("<Return>", self._locate_windage3)
        ttk.Button(loc_frame3, text="Locate", command=self._locate_windage3, width=5).pack(side="left", padx=2)
        ttk.Label(loc_frame3, text="Time:").pack(side="left", padx=(8,2))
        self.loc_time_var3 = tk.StringVar()
        self.loc_time_ent3 = ttk.Entry(loc_frame3, textvariable=self.loc_time_var3, width=10)
        self.loc_time_ent3.pack(side="left", padx=2)
        self.loc_time_ent3.bind("<Return>", self._locate_time3)
        ttk.Button(loc_frame3, text="Locate", command=self._locate_time3, width=5).pack(side="left", padx=2)
        ttk.Button(loc_frame3, text="Clear", command=self._clear_locate3, width=5).pack(side="left", padx=8)
        # toggle: show all active / selected ammo only
        self._windage_scope_var = tk.BooleanVar(value=True)
        self._windage_scope_cb = ttk.Checkbutton(loc_frame3, text="Show All Active",
                                                  variable=self._windage_scope_var,
                                                  command=self._on_windage_scope_toggle)
        self._windage_scope_cb.pack(side="left", padx=8)

        # wind drift data area (dual tables: compare / detail, exclusive display)
        wind_tbl_frame = ttk.LabelFrame(lower3, text="Wind Drift Data", padding=(4,2))
        wind_tbl_frame.pack(fill="both", expand=True, padx=2, pady=(4,0))

        # -- compare table frame (when showing all ammo) --
        self._wind_cmp_frame = ttk.Frame(wind_tbl_frame)
        self._wind_cmp_cols = ("name", "mv", "wind_dist", "windage_dist",
                               "wind_angle_dist", "wind_max")
        self._wind_cmp_tree = ttk.Treeview(self._wind_cmp_frame, columns=self._wind_cmp_cols,
                                           show="headings", height=5)
        ColumnToggle(self._wind_cmp_tree)
        self._sorter_wcmp = ColumnSorter(self._wind_cmp_tree)
        cmp_wind_headers = {
            "name": "Ammo Name", "mv": "MV",
            "wind_dist": "Drift@Loc", "wind_angle_dist": "Drift Angle@Loc",
            "wind_max": "Max Drift",
            "windage_dist": "Drift Range",
        }
        cmp_wind_widths = {"name": 68, "mv": 60, "wind_dist": 66, "wind_angle_dist": 70,
                           "wind_max": 66, "windage_dist": 72}
        for col in self._wind_cmp_cols:
            self._wind_cmp_tree.heading(col, text=cmp_wind_headers.get(col, col))
            self._wind_cmp_tree.column(col, width=cmp_wind_widths.get(col, 64), anchor="center")
        wcmp_ts = ttk.Scrollbar(self._wind_cmp_frame, orient="vertical", command=self._wind_cmp_tree.yview)
        self._wind_cmp_tree.configure(yscrollcommand=wcmp_ts.set)
        self._wind_cmp_tree.pack(side="left", fill="both", expand=True)
        wcmp_ts.pack(side="right", fill="y")

        # -- detail table frame (selected ammo only) --
        self._wind_det_frame = ttk.Frame(wind_tbl_frame)
        self._wind_det_cols = ("distance", "windage", "windage_angle", "velocity", "time")
        self._wind_det_tree = ttk.Treeview(self._wind_det_frame, columns=self._wind_det_cols,
                                           show="headings", height=5)
        ColumnToggle(self._wind_det_tree)
        self._sorter_wdet = ColumnSorter(self._wind_det_tree)
        det_wind_headers = {
            "distance": "Range", "windage": "Drift", "windage_angle": "Drift Angle",
            "velocity": "Velocity", "time": "Time / s",
        }
        det_wind_widths = {"distance": 72, "windage": 66, "windage_angle": 66,
                           "velocity": 72, "time": 70}
        for col in self._wind_det_cols:
            self._wind_det_tree.heading(col, text=det_wind_headers.get(col, col))
            self._wind_det_tree.column(col, width=det_wind_widths.get(col, 64), anchor="center")
        wdet_ts = ttk.Scrollbar(self._wind_det_frame, orient="vertical", command=self._wind_det_tree.yview)
        self._wind_det_tree.configure(yscrollcommand=wdet_ts.set)
        self._wind_det_tree.pack(side="left", fill="both", expand=True)
        wdet_ts.pack(side="right", fill="y")

        # show compare table by default
        self._wind_cmp_frame.pack(fill="both", expand=True)

        # locate lines/markers
        self._loc_vline3 = self._ax3.axvline(0, color="#000000", linewidth=0.5,
                                              linestyle="--", visible=False, zorder=85)
        self._loc_windage_hline3 = self._ax3.axhline(0, color="#000000", linewidth=0.5,
                                                       linestyle="--", visible=False, zorder=85)
        self._loc_scatter3 = self._ax3.scatter([], [], s=25, c="#FF6600",
                                                zorder=86, visible=False, marker="x",
                                                linewidths=1.2)

        # hover elements
        self._cursor_annot3 = self._ax3.annotate("", xy=(0,0), xytext=(12,12),
            textcoords="offset points", fontsize=8, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#AAA", alpha=0.9),
            visible=False, zorder=99, annotation_clip=False)
        self._highlight_scatter3 = self._ax3.scatter([], [], s=80, c="none",
            edgecolors="#FF6600", linewidths=2, zorder=98, visible=False)
        self._ch_vline3 = self._ax3.axvline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._ch_hline3 = self._ax3.axhline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._fig3.canvas.mpl_connect("motion_notify_event", self._on_hover_wind)
        self._fig3.canvas.mpl_connect("button_press_event", self._on_click_wind)
        self._fig3.canvas.mpl_connect("draw_event", self._on_draw_tab3)

        self._canvas3.draw()

        # ====== Tab 5: Drag ======
        tab5 = ttk.Frame(self._notebook)
        self._notebook.add(tab5, text="Drag")

        vp5 = ttk.PanedWindow(tab5, orient="vertical")
        vp5.pack(fill="both", expand=True)

        # top: drag chart
        drag_plot_frame = ttk.Frame(vp5)
        vp5.add(drag_plot_frame, weight=1)

        self._fig5 = Figure(figsize=(8, 4.5), dpi=120, facecolor="white")
        self._ax5 = self._fig5.add_subplot(111)
        self._canvas5 = FigureCanvasTkAgg(self._fig5, drag_plot_frame)
        self._canvas5.get_tk_widget().pack(fill="both", expand=True)

        self._install_draw_debounce(self._canvas5, '_draw_timer5')

        self._ax5.set_xlabel("Mach"); self._ax5.set_ylabel("Drag (N)")
        self._ax5.xaxis.set_major_locator(MaxNLocator(nbins=20, steps=[1, 2, 5, 10]))
        self._ax5.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
        self._ax5.grid(True, alpha=0.35, linewidth=0.5)
        self._ax5.set_facecolor("white")
        self._fig5.tight_layout(pad=2.0)

        # bottom: toolbar + data table
        lower5 = ttk.Frame(vp5)
        vp5.add(lower5, weight=0)

        toolbar5 = _ChineseToolbar(self._canvas5, lower5)
        toolbar5.update(); toolbar5.pack(side="top", fill="x")
        self._tint_copy_button(toolbar5)

        # drag data table
        drag_tbl_frame = ttk.LabelFrame(lower5, text="Drag Data", padding=(4,2))
        drag_tbl_frame.pack(fill="both", expand=True, padx=2, pady=(4,0))

        self._drag_cols = ("name", "bc", "cd_mach08", "cd_mach10", "cd_mach12", "cd_avg")
        self._drag_tree = ttk.Treeview(drag_tbl_frame, columns=self._drag_cols,
                                       show="headings", height=5)
        ColumnToggle(self._drag_tree)
        self._sorter5 = ColumnSorter(self._drag_tree)
        drag_headers = {
            "name": "Ammo", "bc": "BC",
            "cd_mach08": "Drag@0.8M", "cd_mach10": "Drag@1.0M", "cd_mach12": "Drag@1.2M",
            "cd_avg": "Avg Drag (N)",
        }
        drag_widths = {"name": 64, "bc": 80, "cd_mach08": 62, "cd_mach10": 62,
                       "cd_mach12": 62, "cd_avg": 62}
        for col in self._drag_cols:
            self._drag_tree.heading(col, text=drag_headers.get(col, col))
            self._drag_tree.column(col, width=drag_widths.get(col, 56), anchor="center")
        drag_ts = ttk.Scrollbar(drag_tbl_frame, orient="vertical", command=self._drag_tree.yview)
        self._drag_tree.configure(yscrollcommand=drag_ts.set)
        self._drag_tree.pack(side="left", fill="both", expand=True)
        drag_ts.pack(side="right", fill="y")

        # hover elements
        self._cursor_annot5 = self._ax5.annotate("", xy=(0, 0), xytext=(12, 12),
            textcoords="offset points", fontsize=8, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#AAA", alpha=0.9),
            visible=False, zorder=99, annotation_clip=False)
        self._highlight_scatter5 = self._ax5.scatter([], [], s=80, c="none",
            edgecolors="#FF6600", linewidths=2, zorder=98, visible=False)
        self._fig5.canvas.mpl_connect("motion_notify_event", self._on_hover_drag)
        self._fig5.canvas.mpl_connect("button_press_event", self._on_click_drag)
        self._fig5.canvas.mpl_connect("draw_event", self._on_draw_tab5)

        self._canvas5.draw()

        # delay-adjust panedwindow sash to avoid clipping left content
        self.root.after(200, self._adjust_sash)

    def _adjust_sash(self):
        """Measure actual left-content width; position sash to avoid clipping"""
        self.root.update_idletasks()
        needed = self._input_frame.winfo_reqwidth()
        if needed > 0:
            self._paned.sashpos(0, needed + 36)

    # ============================================================
    # dynamic UI
    # ============================================================
    def _on_unit_change(self, event=None):
        name = self.unit_cb.get()
        labels = _UNIT_LABELS.get(name, _UNIT_LABELS[DEFAULT_PROFILE])
        for sv, key in self._unit_label_widgets:
            if key in labels:
                sv.set(labels[key])
        self._update_library_list()
        self._update_active_list()
        self._update_sd_i()


    def _on_vacuum_toggle(self):
        if self.vacuum_var.get():
            self.pres_ent.configure(state="disabled")
            self.alt_ent.configure(state="disabled")
            self.temp_ent.configure(state="disabled")
            self.hum_ent.configure(state="disabled")
        else:
            self.pres_ent.configure(state="normal")
            self.alt_ent.configure(state="normal")
            self.temp_ent.configure(state="normal")
            self.hum_ent.configure(state="normal")

    def _on_show_all(self):
        """Select/deselect all display options"""
        v = self.show_all_var.get()
        for var in (self.show_zero_var, self.show_apex_var, self.show_mach_var,
                     self.show_barrel_var, self.show_sight_var, self.show_vel_var):
            var.set(v)

    def _on_show_sub(self):
        """Sync select-all state on child toggle"""
        all_on = all(v.get() for v in (self.show_zero_var, self.show_apex_var,
                     self.show_mach_var, self.show_barrel_var, self.show_sight_var, self.show_vel_var))
        self.show_all_var.set(all_on)

    def _icao_update(self, alt_raw: float):
        """Core: compute standard pressure & temperature via ICAO formula and fill inputs"""
        profile = UNIT_PROFILES.get(self.unit_cb.get(), _UNIT_METRIC)
        alt_dist = PreferredUnits.distance(alt_raw)
        pres = Atmo.standard_pressure(alt_dist) >> profile["pressure"]
        temp = Atmo.standard_temperature(alt_dist) >> profile["temperature"]
        self.pres_ent.delete(0, "end"); self.pres_ent.insert(0, f"{pres:.1f}")
        self.temp_ent.delete(0, "end"); self.temp_ent.insert(0, f"{temp:.1f}")

    def _on_altitude_change(self, *args):
        """Real-time ICAO pressure & temperature when altitude changes"""
        if getattr(self, '_icao_suppress', False): return
        try:
            self._icao_update(float(self.alt_var.get()))
        except (ValueError, tk.TclError):
            pass

    def _fill_icao(self):
        """ICAO standard atmosphere button: inject sea-level standard values"""
        self.vacuum_var.set(False)
        self._on_vacuum_toggle()
        profile = UNIT_PROFILES.get(self.unit_cb.get(), _UNIT_METRIC)
        self._icao_suppress = True
        self.alt_var.set("0")
        self._icao_suppress = False
        self.pres_ent.delete(0, "end")
        self.pres_ent.insert(0, f"{Pressure.hPa(1013.25) >> profile['pressure']:.1f}")
        self.temp_ent.delete(0, "end")
        self.temp_ent.insert(0, f"{Temperature.Celsius(15) >> profile['temperature']:.1f}")
        self.hum_ent.delete(0, "end")
        self.hum_ent.insert(0, "0")

    # ---- ammo config management ----
    @staticmethod
    def _config_path():
        import os, sys
        if getattr(sys, 'frozen', False):
            return os.path.join(os.path.dirname(sys.executable), _AMMO_CONFIG_FILE)
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), _AMMO_CONFIG_FILE)

    def _load_ammo_configs(self):
        """Load ammo library & active list from JSON; fall back to defaults if missing/corrupt"""
        import os, json, sys
        path = self._config_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # backward compatibility: old format "configs" → migrate to library
                library = data.get("library", data.get("configs", []))
                active = data.get("active_indices", list(range(len(library))))
                if library:
                    # remember active ammo before sort; remap indices after sort
                    active_cfgs = [library[i] for i in active if 0 <= i < len(library)]
                    library.sort(key=lambda c: c.get("name", ""))
                    self._ammo_library = library
                    self._active_indices = [library.index(cfg) for cfg in active_cfgs]
                    if not self._active_indices:
                        self._active_indices = list(range(len(library)))
                    self._active_indices.sort()
                    # restore unit system
                    saved_unit = data.get("unit_profile", "")
                    if saved_unit and saved_unit in UNIT_PROFILES:
                        self.unit_cb.set(saved_unit)
                        self._on_unit_change()
                    self._current_ammo_idx = 0
                    self._set_fields_from_ammo(self._ammo_library[0])
                    self._update_sd_i()
                    self._update_library_list()
                    self._update_active_list()
                    self._library_listbox.selection_set(0)
                    self._status_label.configure(
                        text=f"Loaded {len(library)} ammo, {len(self._active_indices)} in active list")
                    return
            except Exception:
                pass  # corrupt save; fall back to defaults
        # no save — extract bundled ammo library from exe to exe directory
        if getattr(sys, 'frozen', False):
            bundled = os.path.join(sys._MEIPASS, _AMMO_CONFIG_FILE)
            if os.path.exists(bundled):
                import shutil
                shutil.copy2(bundled, path)
                self._load_ammo_configs()  # recurse once to load the newly copied file
                return
        # still no config; use defaults
        cfg = self._get_ammo_from_fields()
        cfg["name"] = "Default Ammo"
        if not cfg.get("diameter"): cfg["diameter"] = 7.82
        if not cfg.get("length"): cfg["length"] = 32.6
        self._ammo_library = [cfg]
        self._active_indices = [0]
        self._current_ammo_idx = 0
        self._set_fields_from_ammo(cfg)
        self._update_sd_i()
        self._update_library_list()
        self._update_active_list()

    def _save_ammo_configs(self):
        """Save ammo library & active list to JSON file"""
        import os, json
        path = self._config_path()
        data = {
            "unit_profile": self.unit_cb.get(),
            "library": self._ammo_library,
            "active_indices": self._active_indices,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _on_close(self):
        """Auto-save config on window close"""
        self._save_ammo_configs()
        self.root.destroy()

    def _on_drag_table_change(self, event=None):
        """Drag table switch → show/hide custom data entry + update i-label"""
        drag_name = self.drag_cb.get()
        if drag_name == "Custom (Mach-CD)":
            self._custom_drag_frame.grid()
            # custom drag table needs no BC scaling; BC fixed at 1.0 and disabled
            self._sd_locked = True
            self._set_entry(self.bc_ent, "1.0000")
            self.bc_ent.configure(state="disabled")
            self._sd_locked = False
            # show placeholder when no real data
            if self._custom_drag_placeholder:
                pass  # keep placeholder
            else:
                text = self._custom_drag_text.get("1.0", "end-1c").strip()
                if not text:
                    self._custom_drag_text.delete("1.0", "end")
                    self._custom_drag_text.insert("1.0", "Mach,CD")
                    self._custom_drag_text.configure(fg="#999999")
                    self._custom_drag_placeholder = True
        else:
            self._custom_drag_frame.grid_remove()
            self.bc_ent.configure(state="normal")
        self.i_label_var.set(f"Form Factor i ({drag_name.split()[0]}):")
        self._update_sd_i()

    def _on_custom_drag_focus_in(self, event=None):
        """Custom drag entry focus-in → clear placeholder & restore normal color"""
        if self._custom_drag_placeholder:
            self._custom_drag_text.delete("1.0", "end")
            self._custom_drag_text.configure(fg="#000000")
            self._custom_drag_placeholder = False

    def _on_custom_drag_focus_out(self, event=None):
        """Custom drag entry focus-out → restore placeholder if empty"""
        text = self._custom_drag_text.get("1.0", "end-1c").strip()
        if not text:
            self._custom_drag_text.delete("1.0", "end")
            self._custom_drag_text.insert("1.0", "Mach,CD")
            self._custom_drag_text.configure(fg="#999999")
            self._custom_drag_placeholder = True

    def _parse_custom_drag(self) -> list | None:
        """Parse Mach,CD data pairs from Text widget, return [DragDataPoint, ...] or None"""
        if self._custom_drag_placeholder:
            return None
        text = self._custom_drag_text.get("1.0", "end-1c").strip()
        if not text:
            return None
        points = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    mach, cd = float(parts[0]), float(parts[1])
                    points.append(DragDataPoint(mach, cd))
                except ValueError:
                    continue
        return points if len(points) >= 2 else None

    def _resolve_drag_table(self, cfg: dict):
        """Parse drag table from config: built-in name → dict lookup; custom → parse Mach-CD data"""
        drag_name = cfg.get("drag_table", "G7 (Low-Drag)")
        if drag_name == "Custom (Mach-CD)":
            custom = cfg.get("custom_drag", [])
            if custom and len(custom) >= 2:
                return [DragDataPoint(m, cd) for m, cd in custom]
            return TableG7  # fallback when custom data insufficient
        return DRAG_TABLES.get(drag_name, TableG7)

    def _update_sd_i(self, *args):
        """Recalculate SD and i when weight/dia/BC change"""
        if self._sd_locked or self._bulk_loading > 0:
            return
        self._update_energy_recoil()  # energy/impulse don't depend on diameter; compute first
        try:
            weight = float(self.wgt_ent.get())
            diameter = float(self.dia_ent.get())
            bc = float(self.bc_ent.get())
        except (ValueError, tk.TclError):
            # diameter field empty → cannot compute SD/i
            if not self.dia_ent.get().strip():
                self.sd_var.set("-")
                self.i_var.set("-")
                self.i_ent.configure(style="Placeholder.TEntry")
            return
        # update i-label to show current drag table
        drag_name = self.drag_cb.get().split()[0]  # e.g. "G7" from "G7 (Low-Drag)"
        self.i_label_var.set(f"Form Factor i ({drag_name}):")
        if diameter <= 0 or weight <= 0:
            self.sd_var.set("-")
            self.i_var.set("-")
            self.i_ent.configure(style="Placeholder.TEntry")
            return
        # convert to grain / inch (required by library sectional_density formula)
        profile = self.unit_cb.get()
        if profile == "Metric (m, m/s, hPa, °C)":
            w_gr = weight * 15.432358   # g → gr
            d_in = diameter / 25.4       # mm → in
        elif profile == "Imperial (yd, fps, inHg, °F)":
            w_gr = weight                 # already gr
            d_in = diameter               # already in
        else:  # mixed
            w_gr = weight                 # gr
            d_in = diameter               # in
        sd = w_gr / (d_in ** 2) / 7000
        i_val = sd / bc if bc != 0 else 0
        self._sd_locked = True
        self.sd_var.set(f"{sd:.4f}")
        self.i_var.set(f"{i_val:.4f}")
        self.i_ent.configure(style="TEntry")
        self._sd_locked = False

    @staticmethod
    def _compute_recoil(weight, mv, profile_name):
        """Calculate recoil impulse (N·s) per unit system; weight is raw value in current units"""
        if profile_name == "Metric (m, m/s, hPa, °C)":
            w_kg = weight / 1000
            v_mps = mv
        elif profile_name == "Imperial (yd, fps, inHg, °F)":
            w_kg = weight / 7000 * 0.453592
            v_mps = mv * 0.3048
        else:  # mixed: gr + fps
            w_kg = weight / 7000 * 0.453592
            v_mps = mv * 0.3048
        return w_kg * v_mps

    @staticmethod
    def _compute_sek(weight, mv, diameter, profile_name):
        """Calculate sectional KE (J/cm²); params are raw values in current units"""
        if profile_name == "Metric (m, m/s, hPa, °C)":
            w_kg = weight / 1000
            v_mps = mv
            d_m = diameter / 1000
        elif profile_name == "Imperial (yd, fps, inHg, °F)":
            w_kg = weight / 7000 * 0.453592
            v_mps = mv * 0.3048
            d_m = diameter * 0.0254
        else:  # mixed: gr + fps
            w_kg = weight / 7000 * 0.453592
            v_mps = mv * 0.3048
            d_m = diameter / 1000
        if d_m <= 0:
            return 0
        energy = 0.5 * w_kg * v_mps * v_mps
        area_cm2 = math.pi * (d_m / 2) ** 2 * 10000
        return energy / area_cm2

    @staticmethod
    def _sek_from_energy(energy_joules, diameter, profile_name):
        """Calculate sectional KE (J/cm²) from energy (J) and diameter (display units), for SEK at range"""
        if profile_name == "Imperial (yd, fps, inHg, °F)":
            d_m = diameter * 0.0254
        else:
            d_m = diameter / 1000
        if d_m <= 0:
            return 0
        area_cm2 = math.pi * (d_m / 2) ** 2 * 10000
        return energy_joules / area_cm2

    @staticmethod
    def _annot_offset(ax, x, y):
        """Return tooltip offset (dx, dy) based on point position on axes to avoid edge clipping.
        Right side → tooltip left; top side → tooltip below."""
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        dx = -54 if x > x_lo + (x_hi - x_lo) * 0.7 else 12
        dy = -24 if y > y_lo + (y_hi - y_lo) * 0.6 else 12
        return dx, dy

    @staticmethod
    def _vsym(p):
        """Velocity unit symbol for tooltips/charts (avoids unicode superscript rendering issues)"""
        return p["velocity"].symbol

    @staticmethod
    def _vsym_hdr(p):
        """Velocity unit symbol for table headers (supports m·s⁻¹ superscript)"""
        return p["velocity"].symbol.replace("/s", "·s⁻¹")

    def _update_energy_recoil(self, *args):
        """Recalculate muzzle energy & recoil impulse on weight/velocity change"""
        if self._bulk_loading > 0:
            return
        try:
            weight = float(self.wgt_ent.get())
            mv = float(self.mv_ent.get())
        except (ValueError, tk.TclError):
            self.recoil_var.set("-")
            self.energy_var.set("-")
            self.sek_var.set("-")
            return
        if weight <= 0 or mv <= 0:
            self.recoil_var.set("-")
            self.energy_var.set("-")
            self.sek_var.set("-")
            return
        profile_name = self.unit_cb.get()
        if profile_name == "Metric (m, m/s, hPa, °C)":
            w_kg = weight / 1000
            v_mps = mv
        elif profile_name == "Imperial (yd, fps, inHg, °F)":
            w_kg = weight / 7000 * 0.453592
            v_mps = mv * 0.3048
        else:  # mixed: gr + fps
            w_kg = weight / 7000 * 0.453592
            v_mps = mv * 0.3048
        energy = 0.5 * w_kg * v_mps * v_mps
        recoil = BallisticApp._compute_recoil(weight, mv, profile_name)
        self.recoil_var.set(f"{recoil:.2f}")
        self.energy_var.set(f"{energy:.0f}")
        # Sectional KE (J/cm²)
        try:
            dia = float(self.dia_ent.get())
        except (ValueError, tk.TclError):
            self.sek_var.set("-")
        else:
            sek = BallisticApp._compute_sek(weight, mv, dia, profile_name)
            self.sek_var.set(f"{sek:.1f}" if sek > 0 else "-")

    def _on_i_change(self, *args):
        """i edited by user → reverse-calculate BC"""
        if self._sd_locked:
            return
        try:
            i_val = float(self.i_var.get())
        except (ValueError, tk.TclError):
            return
        try:
            sd = float(self.sd_var.get())
        except (ValueError, tk.TclError):
            return
        if i_val != 0:
            bc = sd / i_val
            self._sd_locked = True
            self._set_entry(self.bc_ent, f"{bc:.4f}")
            self._sd_locked = False

    def _on_i_focus_in(self, event=None):
        """i-entry focus-in → if placeholder, clear & restore normal color"""
        if self.i_var.get() == "-":
            self.i_var.set("")
        self.i_ent.configure(style="TEntry")

    def _on_i_focus_out(self, event=None):
        """i-entry focus-out → if empty, recalculate from BC"""
        if not self.i_var.get().strip():
            self._update_sd_i()
        if self.i_var.get() == "-":
            self.i_ent.configure(style="Placeholder.TEntry")

    def _get_ammo_from_fields(self) -> dict:
        """Read all ammo parameters from input widgets into dict"""
        cfg = {
            "name": self._ammo_name_var.get().strip() or "Unnamed",
            "drag_table": self.drag_cb.get(),
            "bc": self._get_float(self.bc_ent, 0.223),
            "i": self._get_float(self.i_ent, 0),
            "weight": self._get_float(self.wgt_ent, 10.89),
            "diameter": self._get_float_or_none(self.dia_ent),
            "length": self._get_float_or_none(self.len_ent),
            "mv": self._get_float(self.mv_ent, 838),
            "powder_temp": self._get_float(self.ptemp_ent, 15),
            "temp_mod": self._get_float(self.tmod_ent, 0),
            "use_powder": self.pwdr_var.get(),
            "use_mbc": self.mbc_var.get(),
            "mbc_points": [(float(bc.get()), float(v.get())) for bc, v in self._mbc_rows
                           if bc.get() and v.get()],
        }
        # custom drag table data points (don't save placeholder)
        if cfg["drag_table"] == "Custom (Mach-CD)" and not self._custom_drag_placeholder:
            text = self._custom_drag_text.get("1.0", "end-1c").strip()
            if text:
                rows = []
                for line in text.splitlines():
                    parts = line.strip().replace(",", " ").split()
                    if len(parts) >= 2:
                        try:
                            rows.append([float(parts[0]), float(parts[1])])
                        except ValueError:
                            pass
                if rows:
                    cfg["custom_drag"] = rows
        return cfg

    def _set_entry(self, entry, value, default=""):
        """Clear and write to Entry widget"""
        entry.delete(0, "end")
        entry.insert(0, str(value) if value is not None else default)

    def _set_fields_from_ammo(self, cfg: dict):
        """Load ammo parameter dict into input widgets"""
        self._bulk_loading += 1
        try:
            self._ammo_name_var.set(cfg.get("name", ""))
            self.drag_cb.set(cfg.get("drag_table", "G7 (Low-Drag)"))
            # custom drag table data points
            self._custom_drag_text.delete("1.0", "end")
            custom = cfg.get("custom_drag")
            if custom:
                for row in custom:
                    self._custom_drag_text.insert("end", f"{row[0]},{row[1]}\n")
                self._custom_drag_text.configure(fg="#000000")
                self._custom_drag_placeholder = False
            else:
                self._custom_drag_text.insert("1.0", "Mach,CD")
                self._custom_drag_text.configure(fg="#999999")
                self._custom_drag_placeholder = True
            self._on_drag_table_change()
            self._set_entry(self.bc_ent, cfg.get("bc", 0.223))
            self._set_entry(self.wgt_ent, cfg.get("weight", 10.89))
            dia_val = cfg.get("diameter")
            self._set_entry(self.dia_ent, dia_val if dia_val else "")
            len_val = cfg.get("length")
            self._set_entry(self.len_ent, len_val if len_val else "")
            self._set_entry(self.mv_ent, cfg.get("mv", 838))
            self._set_entry(self.ptemp_ent, cfg.get("powder_temp", 15))
            self._set_entry(self.tmod_ent, cfg.get("temp_mod", 0.0))
            self.pwdr_var.set(cfg.get("use_powder", False))
            self.mbc_var.set(cfg.get("use_mbc", False))
            self._toggle_mbc()
            # multi-BC rows
            for row in self._mbc_rows[:]:
                self._del_mbc_row(0)
            for bcp in cfg.get("mbc_points", []):
                self._add_mbc_row()
                bc_e, vel_e = self._mbc_rows[-1]
                bc_e.delete(0, "end"); bc_e.insert(0, str(bcp[0]))
                vel_e.delete(0, "end"); vel_e.insert(0, str(bcp[1]))
        finally:
            self._bulk_loading -= 1

    # ---- ammo list display ----
    def _format_ammo_line(self, cfg):
        """Format single ammo as list-row text"""
        unit_name = self.unit_cb.get()
        labels = _UNIT_LABELS.get(unit_name, _UNIT_LABELS[DEFAULT_PROFILE])
        w_sym = labels.get("weight", "g")
        v_sym = labels.get("velocity", "m/s")
        name = cfg.get("name", "Unnamed")
        drag = cfg.get("drag_table", "G7").split()[0]
        if drag == "Custom":
            drag = "Custom"
        return (f"{name}: {drag} "
                f"BC={cfg.get('bc',0):.3f} {cfg.get('weight',0)}{w_sym} @{cfg.get('mv',0)}{v_sym}")

    def _update_library_list(self):
        """Refresh ammo library list"""
        self._library_listbox.delete(0, "end")
        for cfg in self._ammo_library:
            self._library_listbox.insert("end", self._format_ammo_line(cfg))
        if 0 <= self._current_ammo_idx < len(self._ammo_library):
            self._library_listbox.see(self._current_ammo_idx)

    def _update_active_list(self):
        """Refresh active list"""
        self._active_listbox.delete(0, "end")
        for i in self._active_indices:
            if 0 <= i < len(self._ammo_library):
                self._active_listbox.insert("end", self._format_ammo_line(self._ammo_library[i]))

    # ---- list selection ----
    def _select_ammo(self, lib_idx):
        """Select ammo library index → load fields, sync both list selections"""
        if lib_idx < 0 or lib_idx >= len(self._ammo_library): return
        self._current_ammo_idx = lib_idx
        self._set_fields_from_ammo(self._ammo_library[lib_idx])
        self._update_sd_i()
        # sync ammo library list selection
        self._library_listbox.selection_clear(0, "end")
        self._library_listbox.selection_set(lib_idx)
        # sync active list selection (if ammo is in active list)
        try:
            active_pos = self._active_indices.index(lib_idx)
            self._active_listbox.selection_clear(0, "end")
            self._active_listbox.selection_set(active_pos)
        except ValueError:
            self._active_listbox.selection_clear(0, "end")
        # apogee/zero mode linkage
        self._sync_mode_display(lib_idx)

    def _sync_mode_display(self, lib_idx):
        """Update apogee/zero display after ammo switch"""
        if self.apex_mode_var.get() and hasattr(self, '_per_ammo_zeros') and 0 <= lib_idx < len(self._per_ammo_zeros):
            zd = self._per_ammo_zeros[lib_idx]
            if zd:
                self.zero_ent.configure(state="normal")
                self.zero_ent.delete(0, "end")
                self.zero_ent.insert(0, f"{zd:.1f}")
                self.zero_ent.configure(state="disabled")
        if self.zero_mode_var.get() and hasattr(self, '_all_results') and 0 <= lib_idx < len(self._all_results):
            result = self._all_results[lib_idx]
            if result:
                apex = next((pt for pt in result if pt.flag & TrajFlag.APEX), None)
                if apex and hasattr(self, 'apex_lock_ent'):
                    p = self._unit_profile
                    self.apex_lock_ent.configure(state="normal")
                    self.apex_lock_ent.delete(0, "end")
                    self.apex_lock_ent.insert(0, f"{apex.height >> p['drop']:.1f}")
                    self.apex_lock_ent.configure(state="disabled")

    def _on_library_select(self, event=None):
        """Ammo library selection → load into inputs"""
        sel = self._library_listbox.curselection()
        if not sel: return
        idx = sel[0]
        if idx == self._current_ammo_idx: return
        self._select_ammo(idx)

    def _on_active_select(self, event=None):
        """Active list selection → load into inputs (mapped to library index)"""
        sel = self._active_listbox.curselection()
        if not sel: return
        pos = sel[0]
        if pos >= len(self._active_indices): return
        lib_idx = self._active_indices[pos]
        if lib_idx == self._current_ammo_idx: return
        self._select_ammo(lib_idx)

    # ---- active list operations ----
    def _import_to_active(self):
        """Import selected library ammo into active list"""
        sel = self._library_listbox.curselection()
        if not sel: return
        lib_idx = sel[0]
        if lib_idx not in self._active_indices:
            self._active_indices.append(lib_idx)
            self._update_active_list()
            self._save_ammo_configs()

    def _remove_from_active(self):
        """Remove selected ammo from active list (keep in library)"""
        sel = self._active_listbox.curselection()
        if not sel: return
        pos = sel[0]
        if pos >= len(self._active_indices): return
        if len(self._active_indices) <= 1:
            self._status_label.configure(text="Keep at least one active ammo")
            return
        del self._active_indices[pos]
        self._update_active_list()
        self._save_ammo_configs()

    def _clear_active(self):
        """Clear active list"""
        self._active_indices.clear()
        self._update_active_list()
        self._save_ammo_configs()

    # ---- ammo library operations ----
    def _save_ammo(self):
        """Write current inputs back to selected library config (auto-sync active list)"""
        idx = self._current_ammo_idx
        if idx < 0 or idx >= len(self._ammo_library): return
        cfg = self._get_ammo_from_fields()
        self._ammo_library[idx] = cfg
        self._sort_library()
        self._save_ammo_configs()
        self._status_label.configure(text=f"Saved: {cfg['name']}")

    def _sort_library(self):
        """Sort ammo library by name; sync current_ammo_idx and active_indices"""
        if not self._ammo_library: return
        cur_cfg = self._ammo_library[self._current_ammo_idx] if 0 <= self._current_ammo_idx < len(self._ammo_library) else None
        active_cfgs = [self._ammo_library[i] for i in self._active_indices if 0 <= i < len(self._ammo_library)]
        self._ammo_library.sort(key=lambda c: c.get("name", ""))
        if cur_cfg:
            self._current_ammo_idx = self._ammo_library.index(cur_cfg)
        self._active_indices = [self._ammo_library.index(cfg) for cfg in active_cfgs]
        self._active_indices.sort()
        self._update_library_list()
        self._update_active_list()
        self._library_listbox.selection_clear(0, "end")
        self._library_listbox.selection_set(self._current_ammo_idx)

    def _add_ammo(self):
        """Add new ammo to library (copy current fields)"""
        cfg = self._get_ammo_from_fields()
        cfg["name"] = f"Ammo {len(self._ammo_library) + 1}"
        self._ammo_library.append(cfg)
        self._sort_library()
        self._save_ammo_configs()

    def _duplicate_ammo(self):
        """Duplicate selected ammo in library"""
        import copy
        idx = self._current_ammo_idx
        if idx < 0 or idx >= len(self._ammo_library): return
        cfg = copy.deepcopy(self._ammo_library[idx])
        cfg["name"] = cfg.get("name", "Unnamed") + " - Copy"
        self._ammo_library.append(cfg)
        self._sort_library()
        self._set_fields_from_ammo(cfg)
        self._update_sd_i()
        self._save_ammo_configs()
        self._status_label.configure(text=f"Duplicated: {cfg['name']}")

    def _delete_ammo(self):
        """Delete selected ammo from library (keep at least one); sync clean active list"""
        if len(self._ammo_library) <= 1: return
        idx = self._current_ammo_idx
        if idx < 0: return
        del self._ammo_library[idx]
        # remove from active list
        if idx in self._active_indices:
            self._active_indices.remove(idx)
        # fix all active_indices > deleted index
        self._active_indices = [i - 1 if i > idx else i for i in self._active_indices]
        self._active_indices = [i for i in self._active_indices if 0 <= i < len(self._ammo_library)]
        if not self._active_indices:
            self._active_indices = [0]
        # adjust current selection
        self._current_ammo_idx = min(idx, len(self._ammo_library) - 1)
        self._set_fields_from_ammo(self._ammo_library[self._current_ammo_idx])
        self._update_library_list()
        self._update_active_list()
        self._library_listbox.selection_set(self._current_ammo_idx)
        self._save_ammo_configs()

    # ---- multi-BC ----
    def _toggle_mbc(self):
        if self.mbc_var.get():
            self._mbc_frame.grid()
            if not self._mbc_rows:
                self._add_mbc_row()
        else:
            self._mbc_frame.grid_remove()

    def _add_mbc_row(self):
        idx = len(self._mbc_rows)
        f = ttk.Frame(self._mbc_frame)
        f.grid(row=idx, column=0, sticky="ew", pady=1)
        ttk.Label(f, text=f"BC{idx+1}:").pack(side="left", padx=2)
        bc_e = ttk.Entry(f, width=7); bc_e.insert(0, "0.22"); bc_e.pack(side="left", padx=2)
        ttk.Label(f, text="@").pack(side="left")
        vel_e = ttk.Entry(f, width=7); vel_e.insert(0, "800"); vel_e.pack(side="left", padx=2)
        ttk.Button(f, text="X", width=2, command=lambda i=idx: self._del_mbc_row(i)).pack(side="left", padx=2)
        self._mbc_rows.append((bc_e, vel_e))
        self._mbc_add_btn.grid(row=idx+1, column=0, sticky="w", pady=2)

    def _del_mbc_row(self, idx: int):
        for w in self._mbc_frame.grid_slaves(row=idx): w.destroy()
        if idx < len(self._mbc_rows):
            self._mbc_rows.pop(idx)
        # reorder
        for i, (bc, vel) in enumerate(self._mbc_rows):
            bc.master.grid(row=i, column=0, sticky="ew", pady=1)
        self._mbc_add_btn.grid(row=len(self._mbc_rows), column=0, sticky="w", pady=2)

    # ============================================================
    # tab switching
    # ============================================================
    def _on_tab_changed(self, event):
        """Force layout refresh on tab change; fixes Tab2 table not showing"""
        self._notebook.update_idletasks()

    # ============================================================
    # apogee → reverse-solve zero distance (bisection)
    # ============================================================
    def _bisect_zero_for_apex(self, desired_raw, gun, ammo, atmo, winds,
                               look_angle, traj_range, engine_factory):
        """Bisection: given apogee height (raw_value), return required zero distance (display units)"""
        lo = 5.0
        hi = traj_range * 2.0
        calc = Calculator(engine=engine_factory)
        for _ in range(_BISECT_MAX_ITER):
            mid = (lo + hi) / 2
            shot = Shot(weapon=gun, ammo=ammo, atmo=atmo, winds=winds,
                        look_angle=look_angle)
            shot.weapon.zero_elevation = calc.barrel_elevation_for_target(shot, mid)
            fire_range = max(mid * _BISECT_FIRE_FACTOR, 100.0)
            result = calc.fire(shot, trajectory_range=fire_range,
                              trajectory_step=fire_range / _BISECT_STEP_DIV, flags=TrajFlag.APEX)
            apex = next((pt for pt in result if pt.flag & TrajFlag.APEX), None)
            if apex is None:
                lo = mid
                continue
            if apex.height.raw_value < desired_raw:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def _on_apex_lock(self, event=None):
        """Given apogee height, bisection reverse-solve for required zero distance"""
        try:
            target_apex = float(self.apex_lock_ent.get())
        except (ValueError, tk.TclError):
            return

        try:
            # ensure PreferredUnits syncs with UI unit system
            profile_name = self.unit_cb.get()
            profile = UNIT_PROFILES.get(profile_name, _UNIT_METRIC)
            apply_unit_profile(profile)
            p = profile
            desired_raw = Distance(target_apex, p["drop"]).raw_value

            # assemble shot (reuse _run_calculation params)
            sight_h = self._get_float(self.sight_ent, 0)
            twist = self._get_float(self.twist_ent, 30.0)
            if self.twist_dir_var.get().startswith("LH"): twist = -twist
            gun = Weapon(sight_height=sight_h, twist=twist)

            if self.vacuum_var.get():
                atmo = Vacuum(altitude=self._get_float(self.alt_ent, 0),
                              temperature=self._get_float(self.temp_ent, 15))
            else:
                atmo = Atmo(altitude=self._get_float(self.alt_ent, 0),
                            pressure=self._get_float(self.pres_ent, 1013.2),
                            temperature=self._get_float(self.temp_ent, 15),
                            humidity=self._get_float(self.hum_ent, 0))

            winds = [Wind(velocity=self._get_float(self.windv_ent, 4.0),
                          direction_from=self._get_float(self.windd_ent, 90))]
            traj_range = self._get_float(self.range_ent, 1200)
            look_angle = self._get_float(self.look_ent, 0)
            eng_name = self.engine_cb.get()
            engine_factory = ENGINES.get(eng_name, RK4IntegrationEngine)

            idx = self._current_ammo_idx
            cfg = self._ammo_library[idx] if 0 <= idx < len(self._ammo_library) else {}
            drag_table = self._resolve_drag_table(cfg)
            dm = DragModel(cfg.get("bc", 0.223), drag_table,
                           diameter=cfg.get("diameter") or 7.82,
                           weight=cfg.get("weight") or 10.89,
                           length=cfg.get("length") or 0)
            use_pwdr = cfg.get("use_powder", False)
            ammo = Ammo(dm=dm, mv=cfg.get("mv", 838),
                        powder_temp=cfg.get("powder_temp", 15),
                        temp_modifier=cfg.get("temp_mod", 0),
                        use_powder_sensitivity=use_pwdr)

            # bisection reverse-solve
            zero_dist = self._bisect_zero_for_apex(
                desired_raw, gun, ammo, atmo, winds, look_angle, traj_range, engine_factory)
            self._status_label.configure(
                text=f"Apogee locked → Zero distance: {zero_dist:.1f} {p['distance'].symbol}")
            self._apex_locked_value = target_apex
            # display solved zero distance in disabled entry (only this one value on page 1)
            self.zero_ent.configure(state="normal")
            self.zero_ent.delete(0, "end")
            self.zero_ent.insert(0, f"{zero_dist:.1f}")
            self.zero_ent.configure(state="disabled")
            self._in_apex_lock = True
            try:
                self._calculate()
            finally:
                self._in_apex_lock = False
        except Exception as e:
            import traceback
            messagebox.showwarning("Lock Failed", f"Cannot reverse-solve zero distance.\n\n{e}\n\n{traceback.format_exc()}")

    # ============================================================
    # calculation
    # ============================================================
    def _calculate(self):
        # Apogee mode: reverse-solve zero distance first, then compute
        if (self.apex_mode_var.get() and
            self.apex_lock_ent.get().strip() and
            not getattr(self, '_in_apex_lock', False)):
            self._on_apex_lock()
            return
        self._progress.start()
        self._status_label.configure(text="Computing...")
        self.calc_btn.configure(state="disabled")
        self.root.update_idletasks()
        def task():
            try:
                self._run_calculation()
                self.root.after(0, self._on_success)
            except Exception as exc:
                self.root.after(0, lambda e=exc: self._on_error(str(e)))
        threading.Thread(target=task, daemon=True).start()

    def _get_float(self, widget, default=0.0):
        try: return float(widget.get())
        except (ValueError, tk.TclError): return default

    def _get_float_or_none(self, widget):
        """Read Entry value; return None if empty"""
        try: return float(widget.get())
        except (ValueError, tk.TclError): return None

    def _run_calculation(self):
        self._on_unit_change()  # sync labels
        profile_name = self.unit_cb.get()
        profile = UNIT_PROFILES.get(profile_name, _UNIT_METRIC)
        apply_unit_profile(profile)
        self._unit_profile = profile

        # ---- shared params: weapon (unchanged) ----
        sight_h = self._get_float(self.sight_ent, 0)
        twist = self._get_float(self.twist_ent, 30.0)
        if self.twist_dir_var.get().startswith("LH"): twist = -twist
        gun = Weapon(sight_height=sight_h, twist=twist)

        # ---- shared params: atmosphere ----
        if self.vacuum_var.get():
            atmo = Vacuum(altitude=self._get_float(self.alt_ent, 0),
                          temperature=self._get_float(self.temp_ent, 15))
        else:
            atmo = Atmo(altitude=self._get_float(self.alt_ent, 0),
                        pressure=self._get_float(self.pres_ent, 1013.2),
                        temperature=self._get_float(self.temp_ent, 15),
                        humidity=self._get_float(self.hum_ent, 0))

        # ---- shared params: wind ----
        wind_v = self._get_float(self.windv_ent, 4.0)
        wind_d = self._get_float(self.windd_ent, 90)
        winds = [Wind(velocity=wind_v, direction_from=wind_d)]

        # ---- shared params: shot settings ----
        zero_dist = self._get_float(self.zero_ent, 0)
        traj_range = self._get_float(self.range_ent, 1200)
        table_step = self._get_float(self.step_ent, 100)
        look_angle = self._get_float(self.look_ent, 0)
        eng_name = self.engine_cb.get()
        engine_factory = ENGINES.get(eng_name, RK4IntegrationEngine)
        flags = TrajFlag.RANGE
        if self.show_zero_var.get(): flags |= TrajFlag.ZERO
        if self.show_mach_var.get(): flags |= TrajFlag.MACH
        if self.show_apex_var.get(): flags |= TrajFlag.APEX
        plot_step = max(1.0, min(5.0, table_step / 10.0))

        # ---- iterate over active list ammo ----
        n = len(self._ammo_library)
        self._all_results = [None] * n
        self._per_ammo_zeros = [0.0] * n
        self._all_drag_data = [None] * n  # (machs, cds) for Tab 5 drag
        # cache shared params for _update_comparison_table
        self._shared_gun = gun
        self._shared_atmo = atmo
        self._shared_winds = winds
        self._shared_engine_factory = engine_factory
        self._shared_traj_range = traj_range
        self._shared_look_angle = look_angle
        calc = Calculator(engine=engine_factory)

        # calc scope: active list + current selection (ensures Tab1 always has data)
        compute_set = set(self._active_indices)
        if 0 <= self._current_ammo_idx < n:
            compute_set.add(self._current_ammo_idx)
        for i in compute_set:
            cfg = self._ammo_library[i]
            # build DragModel
            drag_table = self._resolve_drag_table(cfg)
            bc = cfg.get("bc", 0.223)
            weight = cfg.get("weight") or 10.89
            diameter = cfg.get("diameter") or 7.82
            length = cfg.get("length") or 0
            mv = cfg.get("mv", 838)
            powder_t = cfg.get("powder_temp", 15)
            temp_mod = cfg.get("temp_mod", 0)
            use_pwdr = cfg.get("use_powder", False)
            mbc_points_raw = cfg.get("mbc_points", [])

            if cfg.get("use_mbc") and mbc_points_raw:
                bc_points = [BCPoint(bcv, V=Velocity.MPS(velv)) for bcv, velv in mbc_points_raw]
                dm = DragModelMultiBC(bc_points, drag_table,
                                      weight=weight, diameter=diameter, length=length)
            else:
                dm = DragModel(bc, drag_table, weight=weight, diameter=diameter, length=length)

            # cache drag data (Tab 5):
            #   multi-BC: DragModelMultiBC already modified CD to CD_ref * SD / BC_eff; use directly
            #   standard BC: CD_ref multiplied by (SD/BC) gives dimensionless actual drag coeff CD_actual
            #   SD = w_lb / d_inch², converted via PreferredUnits to ensure correctness across unit systems
            _machs = [pt.Mach for pt in dm.drag_table]
            if cfg.get("use_mbc") and mbc_points_raw:
                _cds = [pt.CD for pt in dm.drag_table]
            else:
                w_gr = PreferredUnits.weight(weight) >> Weight.Grain
                d_in = PreferredUnits.diameter(diameter) >> Distance.Inch
                sd = (w_gr / 7000) / (d_in ** 2)  # sectional density, lb/in²
                _cds = [pt.CD * sd / bc for pt in dm.drag_table]
            self._all_drag_data[i] = (_machs, _cds)

            ammo = Ammo(dm=dm, mv=mv, powder_temp=powder_t,
                        temp_modifier=temp_mod, use_powder_sensitivity=use_pwdr)

            # apogee mode: reverse-solve zero distance for this ammo
            ammo_zero = zero_dist
            if self.apex_mode_var.get() and hasattr(self, '_apex_locked_value'):
                desired_raw = Distance(self._apex_locked_value,
                                       self._unit_profile["drop"]).raw_value
                ammo_zero = self._bisect_zero_for_apex(
                    desired_raw, gun, ammo, atmo, winds,
                    look_angle, traj_range, engine_factory)

            shot = Shot(weapon=gun, ammo=ammo, atmo=atmo, winds=winds)
            if look_angle != 0:
                shot.look_angle = Angular.Degree(look_angle)

            if ammo_zero > 0:
                _ = calc.set_weapon_zero(shot, ammo_zero)

            result = calc.fire(shot, trajectory_range=traj_range,
                               trajectory_step=plot_step, flags=flags)
            self._all_results[i] = result
            self._per_ammo_zeros[i] = ammo_zero

        # default to showing currently selected ammo
        idx = self._current_ammo_idx
        if 0 <= idx < n and self._all_results[idx] is not None:
            self._results = self._all_results[idx]
        else:
            first = next((r for r in self._all_results if r is not None), None)
            self._results = first

        self._table_step = table_step

    # ============================================================
    # update
    # ============================================================
    def _on_success(self):
        self._progress.stop(); self.calc_btn.configure(state="normal")
        try:
            self._update_plot(); self._update_table(); self._update_summary()
            self._update_comparison_plot(); self._update_comparison_table()
            self._restore_locate2()
            self._update_energy_plot(); self._update_energy_table()
            self._restore_locate4()
            self._update_windage_plot(); self._update_windage_table()
            self._restore_locate3()
            self._update_drag_plot(); self._update_drag_table()
            self._clear_highlight()
            # backfill current apogee height into lock entry (zero mode only)
            if self._results and hasattr(self, 'apex_lock_ent') and self.zero_mode_var.get():
                apex = next((pt for pt in self._results if pt.flag & TrajFlag.APEX), None)
                if apex:
                    p = self._unit_profile
                    self.apex_lock_ent.delete(0, "end")
                    self.apex_lock_ent.insert(0, f"{apex.height >> p['drop']:.1f}")
            # apogee mode: show zero distance for selected ammo
            if (self.apex_mode_var.get() and hasattr(self, '_per_ammo_zeros')
                    and self._per_ammo_zeros):
                idx = self._current_ammo_idx
                if 0 <= idx < len(self._per_ammo_zeros):
                    zd = self._per_ammo_zeros[idx]
                    p = self._unit_profile
                    self.zero_ent.configure(state="normal")
                    self.zero_ent.delete(0, "end")
                    self.zero_ent.insert(0, f"{zd:.1f}")
                    self.zero_ent.configure(state="disabled")
            self._status_label.configure(text="Computation complete")
        except Exception as exc:
            import traceback
            self._status_label.configure(text="Display error")
            messagebox.showerror("Display error", f"{exc}\n\n{traceback.format_exc()}")

    def _on_error(self, msg: str):
        self._progress.stop(); self.calc_btn.configure(state="normal")
        self._status_label.configure(text="Computation error")
        messagebox.showerror("Calculation Error", f"Ballistic calculation failed. Please check input parameters.\n\n{msg}")

    def _find_mach_threshold(self, result, threshold: float):
        """First Mach crossing from above threshold to below (closer Mach point)"""
        for i in range(1, len(result)):
            if result[i-1].mach >= threshold and result[i].mach < threshold:
                return result[i-1] if abs(result[i-1].mach - threshold) < abs(result[i].mach - threshold) else result[i]
        return None

    def _find_mach12(self, result): return self._find_mach_threshold(result, 1.2)
    def _find_mach10(self, result): return self._find_mach_threshold(result, 1.0)
    def _find_mach08(self, result): return self._find_mach_threshold(result, 0.8)

    # ---- curve intersections (shared) ----
    def _add_curve_intersections(self, ax, traj_data, colors):
        """Find multi-curve intersections; plot blend-color markers (excluded from legend); return [(ix, iy), ...] for hover snap.
        Auto-skip origin (x≈0)."""
        intersections = []
        n = len(traj_data)
        if n < 2:
            return intersections
        from matplotlib.colors import to_rgb
        for i in range(n):
            for j in range(i + 1, n):
                di, yi, ri, li = traj_data[i]
                dj, yj, rj, lj = traj_data[j]
                prev_diff = yi[0] - yj[0]
                for k in range(1, min(len(yi), len(yj))):
                    curr_diff = yi[k] - yj[k]
                    # skip prev_diff==0 case: intersection already recorded; avoid duplicate flag
                    if prev_diff == 0:
                        prev_diff = curr_diff
                        continue
                    if prev_diff * curr_diff <= 0:
                        t = abs(prev_diff) / (abs(prev_diff) + abs(curr_diff))
                        ix = di[k-1] + t * (di[k] - di[k-1])
                        if ix <= 0:   # skip origin
                            prev_diff = curr_diff
                            continue
                        iy = yi[k-1] + t * (yi[k] - yi[k-1])
                        c1 = to_rgb(colors[i % len(colors)])
                        c2 = to_rgb(colors[j % len(colors)])
                        blend = tuple((a + b) / 2 for a, b in zip(c1, c2))
                        ax.scatter(ix, iy, color=blend, s=30, zorder=7,
                                  marker="o", edgecolors="white", linewidth=0.4)
                        intersections.append((ix, iy))
                    prev_diff = curr_diff
        return intersections

    def _show_cursor_intersection(self, annot, ax, ix, iy, y_label, y_symbol, source_ax=None,
                                  x_label="Range", x_symbol=None):
        """Intersection hover tooltip: show x-axis + y-axis data adaptively.
        source_ax: y-coordinate source axis (e.g. velocity axis) for locate transform; None uses ax directly."""
        p = self._unit_profile
        if x_symbol is None:
            x_symbol = p["distance"].symbol
        x_fmt = ".3f" if x_symbol == "" else ".1f"
        x_part = f"{x_label}: {ix:{x_fmt}}" + (f" {x_symbol}" if x_symbol else "")
        text = (f"{x_part}\n"
                f"{y_label}: {iy:.1f} {y_symbol}")
        annot.set_text(text)
        if source_ax is not None:
            xd, yd = source_ax.transData.transform((ix, iy))
            x_pos, y_pos = ax.transData.inverted().transform((xd, yd))
        else:
            x_pos, y_pos = ix, iy
        annot.xy = (x_pos, y_pos)
        annot.xyann = BallisticApp._annot_offset(ax, x_pos, y_pos)
        annot.set_visible(True)

    # ---- mouse snap-to shared module ----
    @staticmethod
    def _snap_points(event, ax, points, source_ax=None):
        """Find candidate point nearest to mouse.
        points: [(x, y), ...] data coordinate list
        source_ax: optional twin axis for coordinate transform (e.g. velocity axis intersection)
        Returns (index, pixel_dist) or (None, inf)"""
        best_i, best_d = None, float("inf")
        xform = source_ax.transData if source_ax is not None else ax.transData
        for i, (cx, cy) in enumerate(points):
            xd, yd = xform.transform((cx, cy))
            d = math.hypot(xd - event.x, yd - event.y)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i, best_d

    @staticmethod
    def _snap_traj(event, ax, traj_data, source_ax=None):
        """Find TrajectoryData point nearest to mouse.
        Returns (TrajectoryData, pixel_dist) or (None, inf)"""
        best_pt, best_d = None, float("inf")
        xform = source_ax.transData if source_ax is not None else ax.transData
        for dists, vals, result, _ in traj_data:
            for i, (dx, dy) in enumerate(zip(dists, vals)):
                xd, yd = xform.transform((dx, dy))
                d = math.hypot(xd - event.x, yd - event.y)
                if d < best_d:
                    best_d = d
                    best_pt = result[i]
        return best_pt, best_d

    # ---- Tab2 table helpers ----
    def _find_supersonic_distance(self, result, p):
        """Find distance where velocity first drops below Mach 1.2 (linear interpolation; avoids step artifacts)"""
        for i in range(1, len(result)):
            prev_pt, curr_pt = result[i - 1], result[i]
            if curr_pt.mach < 1.2:
                # linear interpolation: solve precisely between prev (mach>=1.2) and curr (mach<1.2)
                m_prev, m_curr = prev_pt.mach, curr_pt.mach
                if m_prev - m_curr > 0:
                    t = (1.2 - m_curr) / (m_prev - m_curr)
                else:
                    t = 0.5
                d_prev = prev_pt.distance >> p["distance"]
                d_curr = curr_pt.distance >> p["distance"]
                return d_prev + t * (d_curr - d_prev)
        return result[-1].distance >> p["distance"] if len(result) > 0 else 0

    def _compute_pbr(self, apex_h_m, gun, ammo, atmo, winds,
                      look_angle, traj_range, engine_factory, p):
        """Calculate far-zero distance (PBR) for given max ordinate (m)"""
        desired_raw = Distance(apex_h_m, Unit.Meter).raw_value
        zd = self._bisect_zero_for_apex(desired_raw, gun, ammo, atmo, winds,
                                         look_angle, traj_range, engine_factory)
        calc = Calculator(engine=engine_factory)
        shot = Shot(weapon=gun, ammo=ammo, atmo=atmo, winds=winds,
                    look_angle=look_angle)
        if zd > 0:
            shot.weapon.zero_elevation = calc.barrel_elevation_for_target(shot, zd)
        result = calc.fire(shot, trajectory_range=max(zd * 2, traj_range),
                          trajectory_step=max(zd * 2, traj_range) / _BISECT_STEP_DIV,
                          flags=TrajFlag.ZERO)
        for pt in result:
            if pt.flag & TrajFlag.ZERO_DOWN:
                return pt.distance >> p["distance"]
        return 0  # far zero not found

    # ---- chart ----
    def _update_plot(self):
        result = self._results
        if result is None or len(result) == 0: return
        p = self._unit_profile

        # Clear old twin axes
        for ax in list(self._fig.axes):
            if ax is not self._ax: ax.remove()
        self._ax.clear()

        d_sym, h_sym = p["distance"].symbol, p["drop"].symbol
        v_sym = self._vsym(p)

        dist_vals = [pt.distance >> p["distance"] for pt in result]
        height_vals = [pt.height >> p["drop"] for pt in result]
        vel_vals = [pt.velocity >> p["velocity"] for pt in result]
        max_dist = dist_vals[-1]
        min_h = min(height_vals)
        max_h = max(height_vals)

        zero_pts = [pt for pt in result if pt.flag & TrajFlag.ZERO]
        apex = next((pt for pt in result if pt.flag & TrajFlag.APEX), None)
        mach12_pt = self._find_mach12(result)
        self._mach12_pt = mach12_pt
        mach10_pt = self._find_mach10(result)
        self._mach10_pt = mach10_pt
        mach08_pt = self._find_mach08(result)
        self._mach08_pt = mach08_pt

        look_rad = result.props.look_angle_rad
        launch_rad = result[0].angle >> Angular.Radian
        sight_h_ft = result.props.sight_height_ft
        sight_h_drop = sight_h_ft * 12.0 / (1.0 if p["drop"] is Unit.Foot else
                       (1.0/2.54) if p["drop"] is Unit.Inch else 30.48)

        # ---- trajectory (CFD rainbow gradient: color linked to velocity; vmin=0 aligns with velocity axis) ----
        points = np.array([dist_vals, height_vals]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        vmax = max(vel_vals)
        lc = LineCollection(segments, cmap="jet", norm=Normalize(vmin=0, vmax=vmax * 1.05),
                           linewidth=1.2, zorder=3)
        lc.set_array(np.array(vel_vals[:-1]))
        self._ax.add_collection(lc)
        # gradient legend proxy (Line2D patch, thickness & length match trajectory)
        n_seg = 20
        jet = plt.cm.jet
        grad_segs = []
        for i in range(n_seg):
            grad_segs.append(Line2D([0],[0], color=jet(1 - i/(n_seg-1)),
                                    linewidth=1.2, solid_capstyle='butt'))

        # ---- Sight Line ----
        if self.show_sight_var.get():
            max_drop = result[-1].distance >> p["drop"]
            x_sight = [0, max_dist]
            y_sight = [0, max_drop * math.tan(look_rad)]
            self._ax.plot(x_sight, y_sight, linestyle="--", color="#7B659E", linewidth=0.9,
                          label="Sight Line")
            sight_above_bbl = y_sight[1] > (max_drop * math.tan(launch_rad) - sight_h_drop)
            angle_deg = math.degrees(math.atan((y_sight[1]-y_sight[0])/(x_sight[1]-x_sight[0]))) if x_sight[1] else 90
            self._ax.text(x_sight[1], y_sight[1], "Sight Line", fontsize=7, color="#7B659E",
                          rotation=angle_deg, rotation_mode="anchor", transform_rotates_text=True,
                          ha="right", va="bottom" if sight_above_bbl else "top")

        # ---- Bore Line ----
        if self.show_barrel_var.get():
            max_drop = result[-1].distance >> p["drop"]
            x_bbl = [0, max_dist]
            y_bbl = [-sight_h_drop, max_drop * math.tan(launch_rad) - sight_h_drop]
            _sba = y_sight[1] > (max_drop * math.tan(launch_rad) - sight_h_drop) if self.show_sight_var.get() else True
            self._ax.plot(x_bbl, y_bbl, linestyle=":", color="#B85450", linewidth=0.9,
                          label="Bore Line")
            angle_bbl = math.degrees(math.atan((y_bbl[1]-y_bbl[0])/(x_bbl[1]-x_bbl[0]))) if x_bbl[1] else 90
            self._ax.text(x_bbl[1], y_bbl[1], "Bore Line", fontsize=7, color="#B85450",
                          rotation=angle_bbl, rotation_mode="anchor", transform_rotates_text=True,
                          ha="right", va="top" if _sba else "bottom")

        # ---- zero/transonic vertical dashed lines ----
        for pt in zero_pts:
            zx = pt.distance >> p["distance"]
            zy = pt.height >> p["drop"]
            self._ax.plot([zx, zx], [min_h, zy], linestyle=":", color="#C8961E", linewidth=0.7)
            self._ax.text(zx + max_dist/100, min_h,
                          "Near Zero" if pt.flag & TrajFlag.ZERO_UP else "Far Zero",
                          fontsize=7, rotation=90, color="#C8961E")
        if self.show_mach_var.get():
            if mach12_pt:
                mx = mach12_pt.distance >> p["distance"]
                my = mach12_pt.height >> p["drop"]
                self._ax.plot([mx, mx], [min_h, my], linestyle=":", color="#D6604D", linewidth=0.7)
                self._ax.text(mx + max_dist/100, min_h, "Mach 1.2", fontsize=7,
                              rotation=90, color="#D6604D")
            if mach10_pt:
                mx = mach10_pt.distance >> p["distance"]
                my = mach10_pt.height >> p["drop"]
                self._ax.plot([mx, mx], [min_h, my], linestyle=":", color="#B2182B", linewidth=0.7)
                self._ax.text(mx + max_dist/100, min_h, "Mach 1.0", fontsize=7,
                              rotation=90, color="#B2182B")
            if mach08_pt:
                mx = mach08_pt.distance >> p["distance"]
                my = mach08_pt.height >> p["drop"]
                self._ax.plot([mx, mx], [min_h, my], linestyle=":", color="#4393C3", linewidth=0.7)
                self._ax.text(mx + max_dist/100, min_h, "Mach 0.8", fontsize=7,
                              rotation=90, color="#4393C3")

        # ---- special point markers ----
        for i, pt in enumerate(zero_pts):
            self._ax.scatter(pt.distance >> p["distance"], pt.height >> p["drop"],
                             color="#C8961E", s=36, zorder=6,
                             label="Zero" if i == 0 else "",
                             edgecolors="white", linewidth=0.4)
        if apex:
            self._ax.scatter(apex.distance >> p["distance"], apex.height >> p["drop"],
                             color="#555555", s=44, zorder=6, marker="D",
                             label="Apogee", edgecolors="white", linewidth=0.4)
        if self.show_mach_var.get():
            if mach12_pt:
                self._ax.scatter(mach12_pt.distance >> p["distance"], mach12_pt.height >> p["drop"],
                                 color="#D6604D", s=36, zorder=6, marker="s",
                                 label="Mach 1.2", edgecolors="white", linewidth=0.4)
            if mach10_pt:
                self._ax.scatter(mach10_pt.distance >> p["distance"], mach10_pt.height >> p["drop"],
                                 color="#B2182B", s=36, zorder=6, marker="s",
                                 label="Mach 1.0", edgecolors="white", linewidth=0.4)
            if mach08_pt:
                self._ax.scatter(mach08_pt.distance >> p["distance"], mach08_pt.height >> p["drop"],
                                 color="#4393C3", s=36, zorder=6, marker="s",
                                 label="Mach 0.8", edgecolors="white", linewidth=0.4)

        # ---- velocity twin axis ----
        ax_vel = None
        if self.show_vel_var.get():
            ax_vel = self._ax.twinx()
            ax_vel.plot(dist_vals, vel_vals, color="#6C8EBF", linewidth=0.9, linestyle="--",
                        label="Velocity")
            ax_vel.set_ylabel(f"Velocity ({v_sym})", color="#6C8EBF")
            ax_vel.tick_params(axis="y", colors="#6C8EBF", labelsize=8)
            ax_vel.set_ylim(0, max(vel_vals) * 1.05)
            ax_vel.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 5, 10]))
            ax_vel.spines["top"].set_visible(False)
            ax_vel.spines["left"].set_visible(False)
        self._ax_vel = ax_vel
        self._tab1_vel_data = [(dist_vals, vel_vals, result, 0)] if ax_vel is not None else None

        # ---- primary axis labels & style ----
        self._ax.axhline(y=0, color="#CCCCCC", linewidth=0.8, linestyle="-")
        self._ax.set_xlabel(f"Range ({d_sym})")
        self._ax.set_ylabel(f"Height ({h_sym})")
        self._ax.xaxis.set_major_locator(MaxNLocator(nbins=20, steps=[1, 5, 10]))
        self._ax.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
        self._ax.grid(True, alpha=0.35, linewidth=0.5)
        self._ax.set_facecolor("white")

        # legend
        handles, labels = self._ax.get_legend_handles_labels()
        if ax_vel is not None:
            h2, l2 = ax_vel.get_legend_handles_labels()
            handles += h2; labels += l2
        uniq = {}
        for h, l in zip(handles, labels):
            if l not in uniq: uniq[l] = h
        # insert gradient trajectory first, then velocity, then bore line/sight line
        all_h = [(tuple(grad_segs))]
        all_l = ["Trajectory"]
        if "Velocity" in uniq:
            all_h.append(uniq.pop("Velocity"))
            all_l.append("Velocity")
        # swap bore line & sight line positions (drawn sight-line first; manually reverse to bore-line first)
        for key in ("Bore Line", "Sight Line"):
            if key in uniq:
                all_h.append(uniq.pop(key))
                all_l.append(key)
        all_h += list(uniq.values())
        all_l += list(uniq.keys())
        if all_h:
            self._ax.legend(all_h, all_l, fontsize=8,
                            loc="lower left", framealpha=0.85, edgecolor="#CCCCCC",
                            handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})

        # ---- velocity axis rainbow bar (right side, spaced from velocity axis to prevent occlusion) ----
        cbar = self._fig.colorbar(lc, ax=ax_vel, orientation="vertical",
                                  pad=0.12, fraction=0.035, aspect=40)
        cbar.ax.tick_params(labelsize=7, colors="#666666")
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 5, 10]))

        # ---- hover elements ----
        self._cursor_annot = self._ax.annotate("", xy=(0, 0), xytext=(12, 12),
            textcoords="offset points", fontsize=8, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#AAA", alpha=0.9),
            visible=False, zorder=99, annotation_clip=False)
        self._highlight_scatter = self._ax.scatter([], [], s=100, c="none",
            edgecolors="#FF6600", linewidths=2.5, zorder=98, visible=False)
        self._ch_vline = self._ax.axvline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._ch_hline = self._ax.axhline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._ch_vline.set_animated(True)
        self._ch_hline.set_animated(True)
        self._cursor_annot.set_animated(True)

        self._fig.tight_layout(pad=2.0)
        self._canvas.draw()

        self._plot_data = (dist_vals, height_vals, result)

    # ---- chart interaction ----
    def _on_draw_tab1(self, event):
        """Capture static background for blit acceleration"""
        self._bg1 = self._fig.canvas.copy_from_bbox(self._fig.bbox)
        self._bg1_w = self._fig.bbox.width
        self._bg1_h = self._fig.bbox.height

    def _blit_tab1(self):
        """Unified blit: restore background → draw animated elements → flush"""
        if not hasattr(self, '_bg1'): return
        # skip when background size doesn't match canvas (during resize)
        if (self._fig.bbox.width, self._fig.bbox.height) != (self._bg1_w, self._bg1_h):
            return
        self._fig.canvas.restore_region(self._bg1)
        if self._ch_vline.get_visible():
            self._ax.draw_artist(self._ch_vline)
        if self._ch_hline.get_visible():
            self._ax.draw_artist(self._ch_hline)
        if self._cursor_annot.get_visible():
            self._ax.draw_artist(self._cursor_annot)
        self._fig.canvas.blit()

    def _on_hover(self, event):
        # hide crosshair & tooltip when mouse leaves chart
        if event.inaxes is None or self._results is None:
            self._ch_vline.set_visible(False)
            self._ch_hline.set_visible(False)
            self._cursor_annot.set_visible(False)
            self._blit_tab1()
            return
        if not hasattr(self, '_plot_data'): return
        dist_vals, height_vals, result = self._plot_data
        SNAP_THRESH = _SNAP_THRESH_PX

        # Update crosshair to follow mouse (pixel coords avoid twinx interference)
        mx, my = self._ax.transData.inverted().transform((event.x, event.y))
        self._ch_vline.set_xdata([mx, mx])
        self._ch_vline.set_visible(True)
        self._ch_hline.set_ydata([my, my])
        self._ch_hline.set_visible(True)

        # Find nearest point
        traj_data = [(dist_vals, height_vals, result, 0)]
        best_pt, best_d = self._snap_traj(event, self._ax, traj_data)
        # special point snap (zero / apogee / Mach)
        snap_pts = []
        snap_info = []  # [(sname, pt), ...]
        for pt in result:
            if pt.flag & TrajFlag.ZERO:
                snap_pts.append((pt.distance >> self._unit_profile["distance"], pt.height >> self._unit_profile["drop"]))
                snap_info.append(("Zero", pt))
            elif pt.flag & TrajFlag.APEX:
                snap_pts.append((pt.distance >> self._unit_profile["distance"], pt.height >> self._unit_profile["drop"]))
                snap_info.append(("Apogee", pt))
        for sname, pt in [("Mach 1.2", self._mach12_pt), ("Mach 1.0", self._mach10_pt), ("Mach 0.8", self._mach08_pt)]:
            if pt:
                snap_pts.append((pt.distance >> self._unit_profile["distance"], pt.height >> self._unit_profile["drop"]))
                snap_info.append((sname, pt))
        si, sd = self._snap_points(event, self._ax, snap_pts)
        if si is not None and sd < SNAP_THRESH * 1.5:
            sname, spt = snap_info[si]
            self._show_cursor(spt, sname); self._blit_tab1(); return
        # velocity snap (mirrors Tab2 _snap_traj + source_ax pattern)
        is_vel = False
        pt_v = None
        if self._tab1_vel_data is not None:
            pt_v, d_v = self._snap_traj(event, self._ax,
                self._tab1_vel_data, source_ax=self._ax_vel)
            if pt_v and d_v < best_d:
                best_pt, best_d, is_vel = pt_v, d_v, True
        if best_pt and best_d < SNAP_THRESH:
            if is_vel:
                self._show_cursor_vel(best_pt)
            else:
                self._show_cursor(best_pt, None)
        else:
            self._cursor_annot.set_visible(False)
        self._blit_tab1()

    def _show_cursor(self, pt, label_hint=None):
        p = self._unit_profile
        d_sym, v_sym, h_sym = p["distance"].symbol, self._vsym(p), p["drop"].symbol
        meaningful = int(pt.flag) & ~int(TrajFlag.RANGE)
        flag_str = TrajFlag.name(meaningful) if meaningful else ""
        if label_hint:
            flag_str = label_hint
        text = (f"Range: {pt.distance >> p['distance']:.1f} {d_sym}\n"
                f"Height: {pt.height >> p['drop']:.2f} {h_sym}\n"
                f"Velocity: {pt.velocity >> p['velocity']:.1f} {v_sym}\n"
                f"Time: {pt.time:.3f} s\n"
                f"{flag_str}")
        self._cursor_annot.set_text(text)
        x_data = pt.distance >> p["distance"]
        y_data = pt.height >> p["drop"]
        self._cursor_annot.xy = (x_data, y_data)
        self._cursor_annot.xyann = self._annot_offset(self._ax, x_data, y_data)
        self._cursor_annot.set_visible(True)

    def _show_cursor_vel(self, pt):
        """Velocity curve hover: show Range, Velocity, Time only (mirrors Tab2 _show_cursor_cmp_vel)"""
        p = self._unit_profile
        d_sym, v_sym = p["distance"].symbol, self._vsym(p)
        text = (f"Range: {pt.distance >> p['distance']:.1f} {d_sym}\n"
                f"Velocity: {pt.velocity >> p['velocity']:.1f} {v_sym}\n"
                f"Time: {pt.time:.3f} s")
        self._cursor_annot.set_text(text)
        dist_val = pt.distance >> p["distance"]
        vel_val = pt.velocity >> p["velocity"]
        if hasattr(self, '_ax_vel') and self._ax_vel is not None:
            xd, yd = self._ax_vel.transData.transform((dist_val, vel_val))
            x_ax, y_ax = self._ax.transData.inverted().transform((xd, yd))
            self._cursor_annot.xy = (x_ax, y_ax)
            self._cursor_annot.xyann = self._annot_offset(self._ax, x_ax, y_ax)
        self._cursor_annot.set_visible(True)

    def _on_click(self, event):
        if event.inaxes is None or self._results is None: return
        if not hasattr(self, '_plot_data'): return
        dist_vals, height_vals, result = self._plot_data
        best_i, best_d = 0, float("inf")
        SNAP_THRESH = _SNAP_THRESH_PX
        for i, (dx, dy) in enumerate(zip(dist_vals, height_vals)):
            xd, yd = self._ax.transData.transform((dx, dy))
            d = math.hypot(xd - event.x, yd - event.y)
            if d < best_d: best_d = d; best_i = i
        # special point snap
        for pt in result:
            if not (pt.flag & (TrajFlag.ZERO | TrajFlag.APEX)): continue
            sx = pt.distance >> self._unit_profile["distance"]
            sy = pt.height >> self._unit_profile["drop"]
            xd, yd = self._ax.transData.transform((sx, sy))
            if math.hypot(xd - event.x, yd - event.y) < SNAP_THRESH * 1.5:
                self._highlight_point(pt); return
        # Mach point snap
        for pt in [self._mach12_pt, self._mach10_pt, self._mach08_pt]:
            if pt is None: continue
            sx = pt.distance >> self._unit_profile["distance"]
            sy = pt.height >> self._unit_profile["drop"]
            xd, yd = self._ax.transData.transform((sx, sy))
            if math.hypot(xd - event.x, yd - event.y) < SNAP_THRESH * 1.5:
                self._highlight_point(pt); return
        if best_d < SNAP_THRESH:
            self._highlight_point(result[best_i])

    def _highlight_point(self, pt):
        p = self._unit_profile
        self._highlight_scatter.set_offsets([[pt.distance >> p["distance"],
                                              pt.height >> p["drop"]]])
        self._highlight_scatter.set_visible(True)
        self.loc_dist_var.set(f"{pt.distance >> p['distance']:.2f}")
        self.loc_height_var.set(f"{pt.height >> p['drop']:.2f}")
        self.loc_time_var.set(f"{pt.time:.4f}")
        self._canvas.draw_idle()

    def _clear_highlight(self):
        self._highlight_scatter.set_visible(False)
        self.loc_dist_var.set("")
        self.loc_height_var.set("")
        self.loc_time_var.set("")
        self._canvas.draw_idle()

    def _tint_copy_button(self, toolbar):
        """Gray out the toolbar Copy button's filesave icon to distinguish from Save"""
        btn = toolbar._buttons.get('Copy')
        if not btn or not hasattr(btn, '_ntimage'):
            return
        from PIL import Image, ImageTk
        import numpy as np
        import matplotlib.cbook as cbook
        path = cbook._get_data_path('images', 'filesave_large.png')
        with Image.open(path) as im:
            arr = np.array(im.convert('RGBA'))
            # replace black pixels with gray
            mask = (arr[..., :3] == 0).all(axis=-1)
            arr[mask, :3] = (0x99, 0x99, 0x99)
            im_gray = Image.fromarray(arr)
            size = btn._ntimage.width()
            img_gray = ImageTk.PhotoImage(im_gray.resize((size, size)), master=toolbar)
        btn.configure(image=img_gray)
        btn._ntimage = img_gray  # replace reference

    def _locate_by_distance(self):
        if self._results is None: return
        try:
            target = float(self.loc_dist_var.get())
            p = self._unit_profile
            # convert display units to library internal units (float treated as raw_value by get_at)
            target_raw = Distance(target, p["distance"]).raw_value
            pt = self._results.get_at("distance", target_raw)
            self._highlight_point(pt)
            self.loc_time_var.set(f"{pt.time:.4f}")
        except Exception as e:
            messagebox.showwarning("Locate Failed", f"Cannot find that range in the trajectory.\n{e}")

    def _locate_by_height(self):
        if self._results is None: return
        try:
            target = float(self.loc_height_var.get())
            p = self._unit_profile
            # convert display units to library internal units (float treated as raw_value by get_at)
            target_raw = Distance(target, p["drop"]).raw_value
            pt = self._results.get_at("height", target_raw)
            self._highlight_point(pt)
        except Exception as e:
            messagebox.showwarning("Locate Failed", f"Cannot find that height in the trajectory.\n{e}")

    def _locate_by_time(self):
        if self._results is None: return
        try:
            target = float(self.loc_time_var.get())
            pt = self._results.get_at("time", target)
            self._highlight_point(pt)
            p = self._unit_profile
            self.loc_dist_var.set(f"{pt.distance >> p['distance']:.2f}")
        except Exception as e:
            messagebox.showwarning("Locate Failed", f"Cannot find that time in the trajectory.\n{e}")

    # ---- Tab2 locate (Range→vline / Height→hline / Time→circle) ----
    def _locate_distance2(self, event=None):
        """Draw vertical dashed line at given range"""
        try:
            target = float(self.loc_dist_var2.get())
        except (ValueError, tk.TclError):
            return
        self._loc_vline2.set_xdata([target, target])
        self._loc_vline2.set_visible(True)
        self._canvas2.draw_idle()
        self._update_comparison_table()

    def _locate_height2(self, event=None):
        """Draw horizontal dashed line at given height"""
        try:
            target = float(self.loc_height_var2.get())
        except (ValueError, tk.TclError):
            return
        self._loc_hline2.set_ydata([target, target])
        self._loc_hline2.set_visible(True)
        self._canvas2.draw_idle()
        self._update_comparison_table()

    def _locate_time2(self, event=None):
        """Draw circles at given time on all trajectory curves"""
        if not self._all_results or not any(r is not None for r in self._all_results): return
        p = self._unit_profile
        try:
            target = float(self.loc_time_var2.get())
        except (ValueError, tk.TclError):
            return
        xs, ys = [], []
        for result in self._all_results:
            try:
                pt = result.get_at("time", target)
                xs.append(pt.distance >> p["distance"])
                ys.append(pt.height >> p["drop"])
            except Exception:
                continue
        if xs:
            self._loc_scatter2.set_offsets(list(zip(xs, ys)))
            self._loc_scatter2.set_visible(True)
        self._canvas2.draw_idle()
        self._update_comparison_table()

    def _clear_locate2(self):
        """Clear all locate lines & points"""
        self.loc_dist_var2.set("")
        self.loc_height_var2.set("")
        self.loc_time_var2.set("")
        self._loc_vline2.set_visible(False)
        self._loc_hline2.set_visible(False)
        self._loc_scatter2.set_visible(False)
        self._canvas2.draw_idle()
        self._update_comparison_table()

    def _restore_locate2(self):
        """Restore locate lines after calc (plot rebuild clears them)"""
        # range locate vline
        try:
            target = float(self.loc_dist_var2.get())
            self._loc_vline2.set_xdata([target, target])
            self._loc_vline2.set_visible(True)
        except (ValueError, tk.TclError):
            self._loc_vline2.set_visible(False)
        # height locate hline
        try:
            target = float(self.loc_height_var2.get())
            self._loc_hline2.set_ydata([target, target])
            self._loc_hline2.set_visible(True)
        except (ValueError, tk.TclError):
            self._loc_hline2.set_visible(False)
        # time locate circles
        try:
            target = float(self.loc_time_var2.get())
            if self._all_results and any(r is not None for r in self._all_results):
                p = self._unit_profile
                xs, ys = [], []
                for result in self._all_results:
                    try:
                        pt = result.get_at("time", target)
                        xs.append(pt.distance >> p["distance"])
                        ys.append(pt.height >> p["drop"])
                    except Exception:
                        continue
                if xs:
                    self._loc_scatter2.set_offsets(list(zip(xs, ys)))
                    self._loc_scatter2.set_visible(True)
                else:
                    self._loc_scatter2.set_visible(False)
            else:
                self._loc_scatter2.set_visible(False)
        except (ValueError, tk.TclError):
            self._loc_scatter2.set_visible(False)
        self._canvas2.draw_idle()

    # ---- Tab2 chart interaction ----
    def _on_draw_tab2(self, event):
        """Capture static background for blit acceleration"""
        self._bg2 = self._fig2.canvas.copy_from_bbox(self._fig2.bbox)
        self._bg2_w = self._fig2.bbox.width
        self._bg2_h = self._fig2.bbox.height

    def _blit_tab2(self):
        """Unified blit: restore background → draw animated elements → flush"""
        if not hasattr(self, '_bg2'): return
        # skip when background size doesn't match canvas (during resize)
        if (self._fig2.bbox.width, self._fig2.bbox.height) != (self._bg2_w, self._bg2_h):
            return
        self._fig2.canvas.restore_region(self._bg2)
        if self._ch_vline2.get_visible():
            self._ax2.draw_artist(self._ch_vline2)
        if self._ch_hline2.get_visible():
            self._ax2.draw_artist(self._ch_hline2)
        if self._cursor_annot2.get_visible():
            self._ax2.draw_artist(self._cursor_annot2)
        self._fig2.canvas.blit()

    def _on_hover_cmp(self, event):
        if event.inaxes is None or not self._all_results or not any(r is not None for r in self._all_results):
            self._ch_vline2.set_visible(False)
            self._ch_hline2.set_visible(False)
            self._cursor_annot2.set_visible(False)
            self._blit_tab2()
            return
        if not hasattr(self, '_cmp_traj_data'): return
        p = self._unit_profile
        SNAP_THRESH = _SNAP_THRESH_PX

        # crosshair (pixel coords avoid twinx interference)
        mx, my = self._ax2.transData.inverted().transform((event.x, event.y))
        self._ch_vline2.set_xdata([mx, mx])
        self._ch_vline2.set_visible(True)
        self._ch_hline2.set_ydata([my, my])
        self._ch_hline2.set_visible(True)

        # find nearest point across all trajectories (height curve + velocity curve)
        pt_h, d_h = self._snap_traj(event, self._ax2, self._cmp_traj_data)
        pt_v, d_v = None, float("inf")
        if hasattr(self, '_ax_vel2') and self._ax_vel2 is not None:
            pt_v, d_v = self._snap_traj(event, self._ax2,
                getattr(self, '_cmp_vel_data', []), source_ax=self._ax_vel2)
        if pt_v and d_v < d_h:
            best_pt, best_d, is_vel = pt_v, d_v, True
        else:
            best_pt, best_d, is_vel = pt_h, d_h, False

        # special point snap (zero / apogee / Mach points)
        snap_pts = []
        snap_info = []  # [(sname, pt), ...]
        for _, _, result, _ in self._cmp_traj_data:
            for pt in result:
                if pt.flag & TrajFlag.ZERO:
                    snap_pts.append((pt.distance >> p["distance"], pt.height >> p["drop"]))
                    snap_info.append(("Zero", pt))
                elif pt.flag & TrajFlag.APEX:
                    snap_pts.append((pt.distance >> p["distance"], pt.height >> p["drop"]))
                    snap_info.append(("Apogee", pt))
            for sname, fn in [("Mach 1.2", self._find_mach12), ("Mach 1.0", self._find_mach10), ("Mach 0.8", self._find_mach08)]:
                pt = fn(result)
                if pt:
                    snap_pts.append((pt.distance >> p["distance"], pt.height >> p["drop"]))
                    snap_info.append((sname, pt))
        si, sd = self._snap_points(event, self._ax2, snap_pts)
        if si is not None and sd < SNAP_THRESH * 1.5:
            sname, spt = snap_info[si]
            self._show_cursor_cmp(spt, sname); self._blit_tab2(); return

        # curve intersection snap (trajectory + velocity curves)
        hi, hd = self._snap_points(event, self._ax2,
            getattr(self, '_cmp_intersections_height', []))
        if hi is not None and hd < SNAP_THRESH * 1.5:
            ix, iy = self._cmp_intersections_height[hi]
            self._show_cursor_intersection(self._cursor_annot2, self._ax2,
                ix, iy, "Height", p['drop'].symbol)
            self._blit_tab2(); return
        if hasattr(self, '_ax_vel2') and self._ax_vel2 is not None:
            vi, vd = self._snap_points(event, self._ax2,
                getattr(self, '_cmp_intersections_vel', []), source_ax=self._ax_vel2)
            if vi is not None and vd < SNAP_THRESH * 1.5:
                ix, iy = self._cmp_intersections_vel[vi]
                self._show_cursor_intersection(self._cursor_annot2, self._ax2,
                    ix, iy, "Velocity", self._vsym(p), source_ax=self._ax_vel2)
                self._blit_tab2(); return

        if best_pt and best_d < SNAP_THRESH:
            if is_vel:
                self._show_cursor_cmp_vel(best_pt)
            else:
                self._show_cursor_cmp(best_pt)
        else:
            self._cursor_annot2.set_visible(False)
        self._blit_tab2()

    def _show_cursor_cmp_vel(self, pt):
        """Velocity curve hover: show Range, Velocity, Time only"""
        p = self._unit_profile
        d_sym, v_sym = p["distance"].symbol, self._vsym(p)
        text = (f"Range: {pt.distance >> p['distance']:.1f} {d_sym}\n"
                f"Velocity: {pt.velocity >> p['velocity']:.1f} {v_sym}\n"
                f"Time: {pt.time:.3f} s")
        self._cursor_annot2.set_text(text)
        # Velocity axis coords → display pixels → height axis coords, ensures tooltip at correct screen position
        dist_val = pt.distance >> p["distance"]
        vel_val = pt.velocity >> p["velocity"]
        if hasattr(self, '_ax_vel2') and self._ax_vel2 is not None:
            xd, yd = self._ax_vel2.transData.transform((dist_val, vel_val))
            x_ax2, y_ax2 = self._ax2.transData.inverted().transform((xd, yd))
        else:
            x_ax2, y_ax2 = dist_val, vel_val
        self._cursor_annot2.xy = (x_ax2, y_ax2)
        self._cursor_annot2.xyann = self._annot_offset(self._ax2, x_ax2, y_ax2)
        self._cursor_annot2.set_visible(True)

    def _on_click_cmp(self, event):
        if event.inaxes is None or not self._all_results or not any(r is not None for r in self._all_results): return
        if not hasattr(self, '_cmp_traj_data'): return
        p = self._unit_profile
        SNAP_THRESH = _SNAP_THRESH_PX

        best_pt, best_d = None, float("inf")
        for dists, heights, result, _ in self._cmp_traj_data:
            for i, (dx, dy) in enumerate(zip(dists, heights)):
                xd, yd = self._ax2.transData.transform((dx, dy))
                d = math.hypot(xd - event.x, yd - event.y)
                if d < best_d:
                    best_d = d; best_pt = result[i]

        # special point snap
        for _, _, result, _ in self._cmp_traj_data:
            for pt in result:
                if not (pt.flag & (TrajFlag.ZERO | TrajFlag.APEX)): continue
                sx = pt.distance >> p["distance"]
                sy = pt.height >> p["drop"]
                xd, yd = self._ax2.transData.transform((sx, sy))
                if math.hypot(xd - event.x, yd - event.y) < SNAP_THRESH * 1.5:
                    self._highlight_point_cmp(pt); return
            for pt in [self._find_mach12(result), self._find_mach10(result), self._find_mach08(result)]:
                if pt is None: continue
                sx = pt.distance >> p["distance"]
                sy = pt.height >> p["drop"]
                xd, yd = self._ax2.transData.transform((sx, sy))
                if math.hypot(xd - event.x, yd - event.y) < SNAP_THRESH * 1.5:
                    self._highlight_point_cmp(pt); return
        if best_pt and best_d < SNAP_THRESH:
            self._highlight_point_cmp(best_pt)

    def _show_cursor_cmp(self, pt, label_hint=None):
        p = self._unit_profile
        d_sym, v_sym, h_sym = p["distance"].symbol, self._vsym(p), p["drop"].symbol
        meaningful = int(pt.flag) & ~int(TrajFlag.RANGE)
        flag_str = label_hint if label_hint else (TrajFlag.name(meaningful) if meaningful else "")
        text = (f"Range: {pt.distance >> p['distance']:.1f} {d_sym}\n"
                f"Height: {pt.height >> p['drop']:.2f} {h_sym}\n"
                f"Velocity: {pt.velocity >> p['velocity']:.1f} {v_sym}\n"
                f"Time: {pt.time:.3f} s\n"
                f"{flag_str}")
        self._cursor_annot2.set_text(text)
        x_data = pt.distance >> p["distance"]
        y_data = pt.height >> p["drop"]
        self._cursor_annot2.xy = (x_data, y_data)
        self._cursor_annot2.xyann = self._annot_offset(self._ax2, x_data, y_data)
        self._cursor_annot2.set_visible(True)

    def _highlight_point_cmp(self, pt):
        p = self._unit_profile
        self._highlight_scatter2.set_offsets([[pt.distance >> p["distance"],
                                               pt.height >> p["drop"]]])
        self._highlight_scatter2.set_visible(True)
        self._canvas2.draw_idle()

    # ---- table helpers ----
    def _auto_fit_columns(self, tree, cols, first_row, last_row):
        """Auto-fit column width: max of header + first row + last row"""
        font_name = ttk.Style().lookup('Treeview', 'font')
        try:
            font = tk.font.nametofont(font_name or "TkDefaultFont")
        except Exception:
            font = tk.font.nametofont("TkFixedFont")
        for i, col in enumerate(cols):
            hdr = tree.heading(col, 'text') or ''
            w = font.measure(hdr)
            if first_row and i < len(first_row):
                w = max(w, font.measure(str(first_row[i])))
            if last_row and i < len(last_row):
                w = max(w, font.measure(str(last_row[i])))
            tree.column(col, width=min(w + 10, 120), minwidth=min(w + 2, 60))

    # ---- table (full 17-field) ----
    def _update_table(self):
        result = self._results
        for row in self._tree.get_children(): self._tree.delete(row)
        if result is None: return
        p = self._unit_profile
        table_step = max(self._table_step, 1.0)
        shown_buckets = set()

        for pt in result:
            is_special = bool(pt.flag & (TrajFlag.ZERO | TrajFlag.APEX))
            dist_raw = pt.distance >> p["distance"]
            bucket = int(dist_raw / table_step)
            key = (bucket, is_special)
            if key in shown_buckets and not is_special: continue
            shown_buckets.add(key)

            vals = (
                f"{dist_raw:.1f}",
                f"{pt.velocity >> p['velocity']:.1f}",
                f"{pt.mach:.3f}",
                f"{pt.time:.3f}",
                f"{pt.height >> p['drop']:.2f}",
                f"{pt.drop_angle >> p['adjustment']:.3f}",
                f"{pt.slant_height >> p['drop']:.2f}",
                f"{pt.slant_distance >> p['distance']:.1f}",
                f"{pt.windage >> p['drop']:.2f}",
                f"{pt.windage_angle >> p['adjustment']:.3f}",
                f"{pt.angle >> p['angular']:.2f}",
                f"{pt.density_ratio:.5f}",
                f"{pt.drag:.4f}",
                f"{pt.energy >> p['energy']:.1f}",
                f"{pt.ogw >> p['ogw']:.1f}",
                TrajFlag.name(int(pt.flag) & ~int(TrajFlag.RANGE)) if int(pt.flag) & ~int(TrajFlag.RANGE) else "",
            )
            tag = ""
            if pt.flag & TrajFlag.ZERO: tag = "zero"
            elif pt.flag & TrajFlag.APEX: tag = "apex"
            elif self._mach12_pt and pt is self._mach12_pt: tag = "mach"
            elif self._mach10_pt and pt is self._mach10_pt: tag = "mach"
            elif self._mach08_pt and pt is self._mach08_pt: tag = "mach"
            self._tree.insert("", "end", values=vals, tags=(tag,) if tag else ())

        self._tree.tag_configure("zero", background="#FDDBC7")
        self._tree.tag_configure("apex", background="#D9F0D3")
        self._tree.tag_configure("mach", background="#D1E5F0")
        # auto-fit column width
        children = self._tree.get_children()
        if children:
            first = self._tree.item(children[0], 'values')
            last = self._tree.item(children[-1], 'values')
            self._auto_fit_columns(self._tree, self._table_cols, first, last)
        self._sorter1.reset()

    # ---- summary ----
    def _update_summary(self):
        result = self._results
        if result is None or len(result) == 0: return
        p = self._unit_profile
        p0 = result[0]; plast = result[-1]
        apex = next((pt for pt in result if pt.flag & TrajFlag.APEX), None)
        zeros = [pt for pt in result if pt.flag & TrajFlag.ZERO]
        d_sym, v_sym, h_sym = p["distance"].symbol, self._vsym(p), p["drop"].symbol
        parts = [
            f"MV: {p0.velocity >> p['velocity']:.1f} {v_sym}",
            f"Terminal Velocity: {plast.velocity >> p['velocity']:.1f} {v_sym}",
            f"Time of Flight: {plast.time:.3f} s",
        ]
        if apex:
            parts.append(f"Apogee: {apex.height >> p['drop']:.1f} {h_sym}"
                         f" @ {apex.distance >> p['distance']:.0f} {d_sym}")
        for z in zeros:
            zname = "Near Zero" if z.flag & TrajFlag.ZERO_UP else "Far Zero"
            parts.append(f"{zname}: {z.distance >> p['distance']:.0f} {d_sym}")
        self._summary_var.set("  |  ".join(parts))

    # ---- Analysis ----
    _CMP_COLORS = ["#0072B2","#D55E00","#009E73","#CC79A7","#56B4E9","#E69F00","#F0E442","#000000"]

    def _update_comparison_plot(self):
        """Tab2: multi-analysis chart — fully replicates Trajectory chart style"""
        results = self._all_results
        if not results or not any(r is not None for r in results): return
        p = self._unit_profile
        if p is None: return

        for ax in list(self._fig2.axes):
            if ax is not self._ax2: ax.remove()
        self._ax2.clear()

        d_sym, h_sym = p["distance"].symbol, p["drop"].symbol
        v_sym = self._vsym(p)
        colors = self._CMP_COLORS

        # first active ammo's result (for shared lines)
        r0 = next((r for r in results if r is not None), None)

        # summary range
        all_max_dist = 0
        all_min_h = float("inf")
        all_vel_max = 0
        self._cmp_traj_data = []
        for pos, lib_idx in enumerate(self._active_indices):
            r = self._all_results[lib_idx]
            if r is None or len(r) == 0: continue
            dists = [pt.distance >> p["distance"] for pt in r]
            heights = [pt.height >> p["drop"] for pt in r]
            vels = [pt.velocity >> p["velocity"] for pt in r]
            all_max_dist = max(all_max_dist, dists[-1])
            all_min_h = min(all_min_h, min(heights))
            all_vel_max = max(all_vel_max, max(vels))
            self._cmp_traj_data.append((dists, heights, r, lib_idx))

        # shared sight line geometry (first ammo)
        look_rad = r0.props.look_angle_rad
        launch_rad = r0[0].angle >> Angular.Radian
        sight_h_ft = r0.props.sight_height_ft
        sight_h_drop = sight_h_ft * 12.0 / (1.0 if p["drop"] is Unit.Foot else
                       (1.0/2.54) if p["drop"] is Unit.Inch else 30.48)
        max_drop = r0[-1].distance >> p["drop"]

        # ---- each trajectory curve ----
        zero_seen = False; apex_seen = False
        mach12_seen = False; mach10_seen = False; mach08_seen = False
        for pos, lib_idx in enumerate(self._active_indices):
            result = self._all_results[lib_idx]
            if result is None or len(result) == 0: continue
            color = colors[pos % len(colors)]
            name = self._ammo_library[lib_idx].get("name", f"Ammo{lib_idx+1}")
            dist_vals = [pt.distance >> p["distance"] for pt in result]
            height_vals = [pt.height >> p["drop"] for pt in result]

            # trajectory curve
            self._ax2.plot(dist_vals, height_vals, color=color, linewidth=1.2,
                          label=name, zorder=3)

            # this ammo's zeros (annotate every trajectory; text only on first to avoid clutter)
            zero_pts = [pt for pt in result if pt.flag & TrajFlag.ZERO]
            for j, pt in enumerate(zero_pts):
                zx = pt.distance >> p["distance"]
                zy = pt.height >> p["drop"]
                self._ax2.plot([zx, zx], [all_min_h, zy], linestyle=":",
                              color="#C8961E", linewidth=0.7, alpha=0.5)
                if pos == 0:
                    self._ax2.text(zx + all_max_dist/100, all_min_h,
                                  "Near Zero" if pt.flag & TrajFlag.ZERO_UP else "Far Zero",
                                  fontsize=7, rotation=90, color="#C8961E")
                self._ax2.scatter(pt.distance >> p["distance"], pt.height >> p["drop"],
                                 color="#C8961E", s=36, zorder=6,
                                 label="Zero" if not zero_seen else "",
                                 edgecolors="white", linewidth=0.4)
                zero_seen = True

            # apogee (each ammo)
            apex_i = next((pt for pt in result if pt.flag & TrajFlag.APEX), None)
            if apex_i:
                self._ax2.scatter(apex_i.distance >> p["distance"], apex_i.height >> p["drop"],
                                 color="#555555", s=44, zorder=6, marker="D",
                                 label="Apogee" if not apex_seen else "",
                                 edgecolors="white", linewidth=0.4)
                apex_seen = True

            # Mach 1.2 / 1.0 / 0.8 (each ammo; 3 gray tiers; text only on first ammo)
            if self.show_mach_var.get():
                m12_i = self._find_mach12(result)
                if m12_i:
                    self._ax2.plot([m12_i.distance >> p["distance"], m12_i.distance >> p["distance"]],
                                  [all_min_h, m12_i.height >> p["drop"]],
                                  linestyle=":", color="#999999", linewidth=0.7, alpha=0.5)
                    self._ax2.scatter(m12_i.distance >> p["distance"], m12_i.height >> p["drop"],
                                     color="#999999", s=36, zorder=6, marker="s",
                                     label="Mach 1.2" if not mach12_seen else "",
                                     edgecolors="white", linewidth=0.4)
                    if pos == 0:
                        self._ax2.text((m12_i.distance >> p["distance"]) + all_max_dist/100, all_min_h,
                                      "Mach 1.2", fontsize=7, rotation=90, color="#999999")
                    mach12_seen = True

                m10_i = self._find_mach10(result)
                if m10_i:
                    self._ax2.plot([m10_i.distance >> p["distance"], m10_i.distance >> p["distance"]],
                                  [all_min_h, m10_i.height >> p["drop"]],
                                  linestyle=":", color="#666666", linewidth=0.7, alpha=0.5)
                    self._ax2.scatter(m10_i.distance >> p["distance"], m10_i.height >> p["drop"],
                                     color="#666666", s=36, zorder=6, marker="s",
                                     label="Mach 1.0" if not mach10_seen else "",
                                     edgecolors="white", linewidth=0.4)
                    if pos == 0:
                        self._ax2.text((m10_i.distance >> p["distance"]) + all_max_dist/100, all_min_h,
                                      "Mach 1.0", fontsize=7, rotation=90, color="#666666")
                    mach10_seen = True

                m08_i = self._find_mach08(result)
                if m08_i:
                    self._ax2.plot([m08_i.distance >> p["distance"], m08_i.distance >> p["distance"]],
                                  [all_min_h, m08_i.height >> p["drop"]],
                                  linestyle=":", color="#333333", linewidth=0.7, alpha=0.5)
                    self._ax2.scatter(m08_i.distance >> p["distance"], m08_i.height >> p["drop"],
                                     color="#333333", s=36, zorder=6, marker="s",
                                     label="Mach 0.8" if not mach08_seen else "",
                                     edgecolors="white", linewidth=0.4)
                    if pos == 0:
                        self._ax2.text((m08_i.distance >> p["distance"]) + all_max_dist/100, all_min_h,
                                      "Mach 0.8", fontsize=7, rotation=90, color="#333333")
                    mach08_seen = True

        # ---- curve intersections (trajectory, excluded from legend) ----
        self._cmp_intersections_height = self._add_curve_intersections(
            self._ax2, self._cmp_traj_data, colors)

        # ---- Sight Line ----
        if self.show_sight_var.get():
            x_sight = [0, all_max_dist]
            y_sight = [0, max_drop * math.tan(look_rad)]
            self._ax2.plot(x_sight, y_sight, linestyle="--", color="#7B659E", linewidth=0.9,
                          label="Sight Line")
            sight_above_bbl = y_sight[1] > (max_drop * math.tan(launch_rad) - sight_h_drop)
            angle_deg = math.degrees(math.atan((y_sight[1]-y_sight[0])/(x_sight[1]-x_sight[0]))) if x_sight[1] else 90
            self._ax2.text(x_sight[1], y_sight[1], "Sight Line", fontsize=7, color="#7B659E",
                          rotation=angle_deg, rotation_mode="anchor", transform_rotates_text=True,
                          ha="right", va="bottom" if sight_above_bbl else "top")

        # ---- velocity twin axis (shared) ----
        ax_vel2 = None
        if self.show_vel_var.get():
            ax_vel2 = self._ax2.twinx()
            self._ax_vel2 = ax_vel2
            ax_vel2.set_ylabel(f"Velocity ({v_sym})", color="#6C8EBF")
            ax_vel2.tick_params(axis="y", colors="#6C8EBF", labelsize=8)
            ax_vel2.set_ylim(0, all_vel_max * 1.05)
            ax_vel2.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
            ax_vel2.spines["top"].set_visible(False)
            ax_vel2.spines["left"].set_visible(False)

            for pos, lib_idx in enumerate(self._active_indices):
                result = self._all_results[lib_idx]
                if result is None or len(result) == 0: continue
                color = colors[pos % len(colors)]
                dist_vals = [pt.distance >> p["distance"] for pt in result]
                vel_vals = [pt.velocity >> p["velocity"] for pt in result]
                ax_vel2.plot(dist_vals, vel_vals, color=color, linewidth=0.9, linestyle="--")

            # velocity curve intersections & hover data
            self._cmp_vel_data = []
            for pos, lib_idx in enumerate(self._active_indices):
                r = self._all_results[lib_idx]
                if r is None or len(r) == 0: continue
                dists = [pt.distance >> p["distance"] for pt in r]
                vels = [pt.velocity >> p["velocity"] for pt in r]
                self._cmp_vel_data.append((dists, vels, r, lib_idx))
            self._cmp_intersections_vel = self._add_curve_intersections(
                ax_vel2, self._cmp_vel_data, colors)

        # ---- primary axis labels & style ----
        self._ax2.axhline(y=0, color="#CCCCCC", linewidth=0.8, linestyle="-")
        self._ax2.set_xlabel(f"Range ({d_sym})")
        self._ax2.set_ylabel(f"Height ({h_sym})")
        self._ax2.xaxis.set_major_locator(MaxNLocator(nbins=20, steps=[1, 5, 10]))
        self._ax2.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
        self._ax2.grid(True, alpha=0.35, linewidth=0.5)
        self._ax2.set_facecolor("white")

        # ---- legend: merge trajectory+velocity per ammo into one row (sorted by trajectory height) ----
        records = []
        for pos, lib_idx in enumerate(self._active_indices):
            result = self._all_results[lib_idx]
            if result is None or len(result) == 0: continue
            color = colors[pos % len(colors)]
            name = self._ammo_library[lib_idx].get("name", f"Ammo{lib_idx+1}")
            sort_key = sum(pt.height.raw_value for pt in result) / len(result)
            p1 = Line2D([0], [0], color=color, linewidth=1.2)
            p2 = Line2D([0], [0], color=color, linewidth=0.9, linestyle="--")
            records.append((sort_key, (p1, p2), name))
        records.sort(key=lambda r: r[0], reverse=True)
        pairs = [r[1] for r in records]
        pair_labels = [r[2] for r in records]

        # collect other legend items (zero, apogee, Mach points, sight line)
        seen_legend = set(pair_labels)
        remaining_handles = []; remaining_labels = []
        for h, l in zip(*self._ax2.get_legend_handles_labels()):
            if l and l not in seen_legend:
                seen_legend.add(l)
                remaining_handles.append(h); remaining_labels.append(l)
        # move sight line above zero
        if "Sight Line" in remaining_labels:
            idx = remaining_labels.index("Sight Line")
            remaining_handles.insert(0, remaining_handles.pop(idx))
            remaining_labels.insert(0, remaining_labels.pop(idx))

        all_handles = pairs + remaining_handles
        all_labels = pair_labels + remaining_labels
        if all_handles:
            self._ax2.legend(all_handles, all_labels, fontsize=8,
                            loc="lower left", framealpha=0.85, edgecolor="#CCCCCC",
                            handler_map={tuple: HandlerTuple(ndivide=None)})

        # ---- hover elements (recreated) ----
        self._cursor_annot2 = self._ax2.annotate("", xy=(0, 0), xytext=(12, 12),
            textcoords="offset points", fontsize=8, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#AAA", alpha=0.9),
            visible=False, zorder=99, annotation_clip=False)
        self._highlight_scatter2 = self._ax2.scatter([], [], s=100, c="none",
            edgecolors="#FF6600", linewidths=2.5, zorder=98, visible=False)
        self._ch_vline2 = self._ax2.axvline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._ch_hline2 = self._ax2.axhline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._ch_vline2.set_animated(True)
        self._ch_hline2.set_animated(True)
        self._cursor_annot2.set_animated(True)

        # rebuild locate lines/points (lost after axes clear)
        self._loc_vline2 = self._ax2.axvline(0, color="#000000", linewidth=0.5,
                                              linestyle="--", visible=False, zorder=85)
        self._loc_hline2 = self._ax2.axhline(0, color="#000000", linewidth=0.5,
                                              linestyle="--", visible=False, zorder=85)
        self._loc_scatter2 = self._ax2.scatter([], [], s=25, c="#FF6600",
                                                zorder=86, visible=False, marker="x",
                                                linewidths=1.2)

        self._fig2.tight_layout(pad=2.0)
        self._canvas2.draw()

    def _update_comparison_table(self):
        """Tab2: comparison table"""
        for row in self._cmp_tree.get_children():
            self._cmp_tree.delete(row)
        if not self._all_results or not any(r is not None for r in self._all_results): return
        p = self._unit_profile
        if p is None: return

        # dynamic column headers: with units
        self._cmp_tree.heading("mv", text=f"MV / {self._vsym_hdr(p)}")
        self._cmp_tree.heading("supersonic", text=f"Supersonic Range / {p['distance'].symbol}")

        # locate range (if user entered one)
        loc_dist = None
        try:
            loc_dist = float(self.loc_dist_var2.get())
        except (ValueError, tk.TclError):
            pass

        # dynamic headers: velocity/energy/time/drop at locate range
        if loc_dist is not None:
            d_sym = p['distance'].symbol
            self._cmp_tree.heading("vel_dist",
                text=f"{loc_dist:.0f} {d_sym}Velocity / {self._vsym_hdr(p)}")
            self._cmp_tree.heading("energy_dist",
                text=f"{loc_dist:.0f} {d_sym}Energy / {p['energy'].symbol}")
            self._cmp_tree.heading("time_dist",
                text=f"{loc_dist:.0f} {d_sym}Time / s")
            self._cmp_tree.heading("drop_dist",
                text=f"{loc_dist:.0f} {d_sym}Drop / {p['drop'].symbol}")
            self._cmp_tree.heading("sek",
                text=f"{loc_dist:.0f} {d_sym}Sectional KE / J·cm⁻²")
        else:
            self._cmp_tree.heading("vel_dist", text=f"Velocity / {self._vsym_hdr(p)}")
            self._cmp_tree.heading("energy_dist", text=f"Energy / {p['energy'].symbol}")
            self._cmp_tree.heading("time_dist", text="Time / s")
            self._cmp_tree.heading("drop_dist", text=f"Drop / {p['drop'].symbol}")
            self._cmp_tree.heading("sek", text="Sectional KE / J·cm⁻²")

        for pos, lib_idx in enumerate(self._active_indices):
            result = self._all_results[lib_idx]
            if result is None or len(result) == 0: continue
            name = self._ammo_library[lib_idx].get("name", f"Ammo{lib_idx+1}")
            p0 = result[0]
            apex = next((pt for pt in result if pt.flag & TrajFlag.APEX), None)

            # velocity / energy / time / drop (at locate range)
            if loc_dist is not None and hasattr(self, '_shared_gun'):
                try:
                    pt_d = result.get_at("distance",
                                         Distance(loc_dist, p["distance"]).raw_value)
                    vel_d = f"{pt_d.velocity >> p['velocity']:.1f}"
                    nrg_d = f"{pt_d.energy >> p['energy']:.1f}"
                    tim_d = f"{pt_d.time:.3f}"
                    drp_d = f"{pt_d.height >> p['drop']:.1f}"
                except Exception:
                    vel_d = nrg_d = tim_d = drp_d = "-"
            else:
                vel_d = nrg_d = tim_d = drp_d = "-"

            # Max Ordinate
            apex_h = f"{apex.height >> p['drop']:.1f}" if apex else "-"

            # point-blank range (PBR), using cached shared params
            if hasattr(self, '_shared_gun'):
                cfg = self._ammo_library[lib_idx] if lib_idx < len(self._ammo_library) else {}
                drag_table = self._resolve_drag_table(cfg)
                bc = cfg.get("bc", 0.223)
                weight = cfg.get("weight") or 10.89
                diameter = cfg.get("diameter") or 7.82
                length_val = cfg.get("length") or 0
                mv_val = cfg.get("mv", 838)
                powder_t = cfg.get("powder_temp", 15)
                temp_mod = cfg.get("temp_mod", 0)
                use_pwdr = cfg.get("use_powder", False)
                mbc_raw = cfg.get("mbc_points", [])
                if cfg.get("use_mbc") and mbc_raw:
                    pts = [BCPoint(bcv, V=Velocity.MPS(velv)) for bcv, velv in mbc_raw]
                    dm = DragModelMultiBC(pts, drag_table, weight=weight,
                                          diameter=diameter, length=length_val)
                else:
                    dm = DragModel(bc, drag_table, weight=weight,
                                   diameter=diameter, length=length_val)
                ammo = Ammo(dm=dm, mv=mv_val, powder_temp=powder_t,
                            temp_modifier=temp_mod, use_powder_sensitivity=use_pwdr)

                def _pbr(h):
                    try:
                        return (f"{self._compute_pbr(h, self._shared_gun, ammo,
                                  self._shared_atmo, self._shared_winds,
                                  self._shared_look_angle, self._shared_traj_range,
                                  self._shared_engine_factory, p):.1f}")
                    except Exception:
                        return "-"
                pbr03 = _pbr(0.3)
                pbr1 = _pbr(1.0)
                pbr15 = _pbr(1.5)
            else:
                pbr03 = pbr1 = pbr15 = "-"

            # Supersonic Range
            sup_dist = f"{self._find_supersonic_distance(result, p):.1f}"

            # recoil impulse & sectional KE
            cfg_r = self._ammo_library[lib_idx]
            profile_name = self.unit_cb.get()
            recoil_val = f"{BallisticApp._compute_recoil(cfg_r.get('weight', 10.89), cfg_r.get('mv', 838), profile_name):.2f}"
            # sectional KE: no range→muzzle value; range→interpolated
            if loc_dist is not None and hasattr(self, '_shared_gun'):
                try:
                    pt_d = result.get_at("distance",
                                         Distance(loc_dist, p["distance"]).raw_value)
                    sek_val = f"{BallisticApp._sek_from_energy(pt_d.energy.raw_value, cfg_r.get('diameter', 7.82), profile_name):.1f}"
                except Exception:
                    sek_val = "-"
            else:
                sek_val = f"{BallisticApp._compute_sek(cfg_r.get('weight', 10.89), cfg_r.get('mv', 838), cfg_r.get('diameter', 7.82), profile_name):.1f}"

            vals = (
                name,
                f"{p0.velocity >> p['velocity']:.1f}",
                sek_val,
                recoil_val,
                vel_d,
                nrg_d,
                tim_d,
                drp_d,
                apex_h,
                pbr03,
                pbr1,
                pbr15,
                sup_dist,
            )
            self._cmp_tree.insert("", "end", values=vals)
        # auto-fit column width
        children = self._cmp_tree.get_children()
        if children:
            first = self._cmp_tree.item(children[0], 'values')
            last = self._cmp_tree.item(children[-1], 'values')
            self._auto_fit_columns(self._cmp_tree, self._cmp_cols, first, last)
        self._sorter2.reset()

    # ============================================================
    # Tab 3: Energy — chart
    # ============================================================
    def _update_energy_plot(self):
        """Energy chart: Energy vs Range, multi-ammo overlay + 3 Mach points"""
        results = self._all_results
        if not results or not any(r is not None for r in results): return
        p = self._unit_profile
        if p is None: return

        for ax in list(self._fig4.axes):
            if ax is not self._ax4: ax.remove()
        self._ax4.clear()

        d_sym = p["distance"].symbol
        e_sym = p["energy"].symbol
        colors = self._CMP_COLORS

        all_max_dist = 0
        self._energy_traj_data = []
        for pos, lib_idx in enumerate(self._active_indices):
            r = self._all_results[lib_idx]
            if r is None or len(r) == 0: continue
            dists = [pt.distance >> p["distance"] for pt in r]
            energies = [pt.energy >> p["energy"] for pt in r]
            all_max_dist = max(all_max_dist, dists[-1])
            self._energy_traj_data.append((dists, energies, r, lib_idx))

        if not self._energy_traj_data: return

        # ---- each energy curve + 3 Mach points ----
        mach12_seen = False; mach10_seen = False; mach08_seen = False
        for pos, (dists, energies, r, lib_idx) in enumerate(self._energy_traj_data):
            color = colors[pos % len(colors)]
            name = self._ammo_library[lib_idx].get("name", f"Ammo{lib_idx+1}")

            self._ax4.plot(dists, energies, color=color, linewidth=1.2,
                          label=name, zorder=3)

            # Mach 1.2
            m12 = self._find_mach12(r)
            if m12:
                mx = m12.distance >> p["distance"]
                my = m12.energy >> p["energy"]
                self._ax4.plot([mx, mx], [0, my], linestyle=":",
                              color="#999999", linewidth=0.7, alpha=0.5)
                self._ax4.scatter(mx, my, color="#999999", s=36, zorder=6, marker="s",
                                 label="Mach 1.2" if not mach12_seen else "",
                                 edgecolors="white", linewidth=0.4)
                if pos == 0:
                    self._ax4.text(mx + all_max_dist/100, 0, "Mach 1.2",
                                  fontsize=7, rotation=90, color="#999999")
                mach12_seen = True

            # Mach 1.0
            m10 = self._find_mach10(r)
            if m10:
                mx = m10.distance >> p["distance"]
                my = m10.energy >> p["energy"]
                self._ax4.plot([mx, mx], [0, my], linestyle=":",
                              color="#666666", linewidth=0.7, alpha=0.5)
                self._ax4.scatter(mx, my, color="#666666", s=36, zorder=6, marker="s",
                                 label="Mach 1.0" if not mach10_seen else "",
                                 edgecolors="white", linewidth=0.4)
                if pos == 0:
                    self._ax4.text(mx + all_max_dist/100, 0, "Mach 1.0",
                                  fontsize=7, rotation=90, color="#666666")
                mach10_seen = True

            # Mach 0.8
            m08 = self._find_mach08(r)
            if m08:
                mx = m08.distance >> p["distance"]
                my = m08.energy >> p["energy"]
                self._ax4.plot([mx, mx], [0, my], linestyle=":",
                              color="#333333", linewidth=0.7, alpha=0.5)
                self._ax4.scatter(mx, my, color="#333333", s=36, zorder=6, marker="s",
                                 label="Mach 0.8" if not mach08_seen else "",
                                 edgecolors="white", linewidth=0.4)
                if pos == 0:
                    self._ax4.text(mx + all_max_dist/100, 0, "Mach 0.8",
                                  fontsize=7, rotation=90, color="#333333")
                mach08_seen = True

        # ---- curve intersections (excluded from legend) ----
        self._energy_intersections = self._add_curve_intersections(
            self._ax4, self._energy_traj_data, colors)

        # ---- labels & style ----
        self._ax4.axhline(y=0, color="#CCCCCC", linewidth=0.8, linestyle="-")
        self._ax4.set_xlabel(f"Range ({d_sym})")
        self._ax4.set_ylabel(f"Energy ({e_sym})")
        self._ax4.xaxis.set_major_locator(MaxNLocator(nbins=20, steps=[1, 5, 10]))
        self._ax4.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
        self._ax4.grid(True, alpha=0.35, linewidth=0.5)
        self._ax4.set_facecolor("white")

        # ---- legend (sorted by energy) ----
        records = []
        for pos, lib_idx in enumerate(self._active_indices):
            r = self._all_results[lib_idx]
            if r is None or len(r) == 0: continue
            color = colors[pos % len(colors)]
            name = self._ammo_library[lib_idx].get("name", f"Ammo{lib_idx+1}")
            sort_key = sum(pt.energy.raw_value for pt in r) / len(r)
            ln = Line2D([0], [0], color=color, linewidth=1.2)
            records.append((sort_key, (ln,), name))
        records.sort(key=lambda r: r[0], reverse=True)
        pairs = [r[1] for r in records]
        pair_labels = [r[2] for r in records]

        seen_legend = set(pair_labels)
        remaining_handles = []; remaining_labels = []
        for h, l in zip(*self._ax4.get_legend_handles_labels()):
            if l and l not in seen_legend:
                seen_legend.add(l)
                remaining_handles.append(h); remaining_labels.append(l)

        all_handles = pairs + remaining_handles
        all_labels = pair_labels + remaining_labels
        if all_handles:
            self._ax4.legend(all_handles, all_labels, fontsize=8,
                            loc="upper right", framealpha=0.85, edgecolor="#CCCCCC",
                            handler_map={tuple: HandlerTuple(ndivide=None)})

        # ---- hover elements ----
        self._cursor_annot4 = self._ax4.annotate("", xy=(0, 0), xytext=(12, 12),
            textcoords="offset points", fontsize=8, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#AAA", alpha=0.9),
            visible=False, zorder=99, annotation_clip=False)
        self._highlight_scatter4 = self._ax4.scatter([], [], s=100, c="none",
            edgecolors="#FF6600", linewidths=2.5, zorder=98, visible=False)
        self._ch_vline4 = self._ax4.axvline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._ch_hline4 = self._ax4.axhline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._ch_vline4.set_animated(True)
        self._ch_hline4.set_animated(True)
        self._cursor_annot4.set_animated(True)

        # rebuild locate lines/points
        self._loc_vline4 = self._ax4.axvline(0, color="#000000", linewidth=0.5,
                                              linestyle="--", visible=False, zorder=85)
        self._loc_hline4 = self._ax4.axhline(0, color="#000000", linewidth=0.5,
                                              linestyle="--", visible=False, zorder=85)
        self._loc_scatter4 = self._ax4.scatter([], [], s=25, c="#FF6600",
                                                zorder=86, visible=False, marker="x",
                                                linewidths=1.2)

        self._fig4.tight_layout(pad=2.0)
        self._canvas4.draw()

    # ---- energy table ----
    def _update_energy_table(self):
        """Energy table: one row per active list ammo"""
        for row in self._energy_tree.get_children():
            self._energy_tree.delete(row)
        if not self._all_results or not any(r is not None for r in self._all_results): return
        p = self._unit_profile
        if p is None: return

        # dynamic column headers: with units
        self._energy_tree.heading("mv", text=f"MV / {self._vsym_hdr(p)}")
        self._energy_tree.heading("muzzle_energy", text=f"Muzzle Energy / {p['energy'].symbol}")
        self._energy_tree.heading("supersonic", text=f"Supersonic Range / {p['distance'].symbol}")

        loc_dist = None
        try:
            loc_dist = float(self.loc_dist_var4.get())
        except (ValueError, tk.TclError):
            pass

        if loc_dist is not None:
            d_sym = p['distance'].symbol
            self._energy_tree.heading("energy_dist",
                text=f"{loc_dist:.0f} {d_sym}Energy / {p['energy'].symbol}")
            self._energy_tree.heading("vel_dist",
                text=f"{loc_dist:.0f} {d_sym}Velocity / {self._vsym_hdr(p)}")
            self._energy_tree.heading("time_dist",
                text=f"{loc_dist:.0f} {d_sym}Time / s")
            self._energy_tree.heading("drop_dist",
                text=f"{loc_dist:.0f} {d_sym}Drop / {p['drop'].symbol}")
            self._energy_tree.heading("sek",
                text=f"{loc_dist:.0f} {d_sym}Sectional KE / J·cm⁻²")
        else:
            self._energy_tree.heading("energy_dist", text=f"Energy / {p['energy'].symbol}")
            self._energy_tree.heading("vel_dist", text=f"Velocity / {self._vsym_hdr(p)}")
            self._energy_tree.heading("time_dist", text="Time / s")
            self._energy_tree.heading("drop_dist", text=f"Drop / {p['drop'].symbol}")
            self._energy_tree.heading("sek", text="Sectional KE / J·cm⁻²")

        for pos, lib_idx in enumerate(self._active_indices):
            result = self._all_results[lib_idx]
            if result is None or len(result) == 0: continue
            name = self._ammo_library[lib_idx].get("name", f"Ammo{lib_idx+1}")
            p0 = result[0]

            if loc_dist is not None:
                try:
                    pt_d = result.get_at("distance",
                                         Distance(loc_dist, p["distance"]).raw_value)
                    nrg_d = f"{pt_d.energy >> p['energy']:.1f}"
                    vel_d = f"{pt_d.velocity >> p['velocity']:.1f}"
                    tim_d = f"{pt_d.time:.3f}"
                    drp_d = f"{pt_d.height >> p['drop']:.1f}"
                except Exception:
                    nrg_d = vel_d = tim_d = drp_d = "-"
            else:
                nrg_d = vel_d = tim_d = drp_d = "-"

            sup_dist = f"{self._find_supersonic_distance(result, p):.1f}"

            cfg_r = self._ammo_library[lib_idx]
            profile_name = self.unit_cb.get()
            # sectional KE: no range→muzzle value; range→interpolated
            if loc_dist is not None:
                try:
                    pt_d = result.get_at("distance",
                                         Distance(loc_dist, p["distance"]).raw_value)
                    sek_val = f"{BallisticApp._sek_from_energy(pt_d.energy.raw_value, cfg_r.get('diameter', 7.82), profile_name):.1f}"
                except Exception:
                    sek_val = "-"
            else:
                sek_val = f"{BallisticApp._compute_sek(cfg_r.get('weight', 10.89), cfg_r.get('mv', 838), cfg_r.get('diameter', 7.82), profile_name):.1f}"

            vals = (
                name,
                f"{p0.velocity >> p['velocity']:.1f}",
                sek_val,
                f"{p0.energy >> p['energy']:.1f}",
                nrg_d,
                vel_d,
                tim_d,
                drp_d,
                sup_dist,
            )
            self._energy_tree.insert("", "end", values=vals)

        children = self._energy_tree.get_children()
        if children:
            first = self._energy_tree.item(children[0], 'values')
            last = self._energy_tree.item(children[-1], 'values')
            self._auto_fit_columns(self._energy_tree, self._energy_cols, first, last)
        self._sorter3.reset()

    # ============================================================
    # Tab 3: Energy — chart interaction
    # ============================================================
    def _on_draw_tab4(self, event):
        """Capture static background for blit acceleration"""
        self._bg4 = self._fig4.canvas.copy_from_bbox(self._fig4.bbox)
        self._bg4_w = self._fig4.bbox.width
        self._bg4_h = self._fig4.bbox.height

    def _blit_tab4(self):
        """Unified blit: restore background → draw animated elements → flush"""
        if not hasattr(self, '_bg4'): return
        if (self._fig4.bbox.width, self._fig4.bbox.height) != (self._bg4_w, self._bg4_h):
            return
        self._fig4.canvas.restore_region(self._bg4)
        if self._ch_vline4.get_visible():
            self._ax4.draw_artist(self._ch_vline4)
        if self._ch_hline4.get_visible():
            self._ax4.draw_artist(self._ch_hline4)
        if self._cursor_annot4.get_visible():
            self._ax4.draw_artist(self._cursor_annot4)
        self._fig4.canvas.blit()

    def _on_hover_energy(self, event):
        if event.inaxes is None or not hasattr(self, '_energy_traj_data') or not self._energy_traj_data:
            self._ch_vline4.set_visible(False)
            self._ch_hline4.set_visible(False)
            self._cursor_annot4.set_visible(False)
            self._blit_tab4()
            return
        p = self._unit_profile
        SNAP_THRESH = _SNAP_THRESH_PX
        mx, my = self._ax4.transData.inverted().transform((event.x, event.y))
        self._ch_vline4.set_xdata([mx, mx])
        self._ch_vline4.set_visible(True)
        self._ch_hline4.set_ydata([my, my])
        self._ch_hline4.set_visible(True)

        best_pt, best_d = self._snap_traj(event, self._ax4, self._energy_traj_data)

        # Mach point snap
        snap_pts = []
        snap_info = []  # [(sname, pt), ...]
        for _, _, result, _ in self._energy_traj_data:
            for sname, fn in [("Mach 1.2", self._find_mach12), ("Mach 1.0", self._find_mach10), ("Mach 0.8", self._find_mach08)]:
                pt = fn(result)
                if pt:
                    snap_pts.append((pt.distance >> p["distance"], pt.energy >> p["energy"]))
                    snap_info.append((sname, pt))
        si, sd = self._snap_points(event, self._ax4, snap_pts)
        if si is not None and sd < SNAP_THRESH * 1.5:
            sname, spt = snap_info[si]
            self._show_cursor_energy(spt, sname); self._blit_tab4(); return

        # curve intersection snap
        ei, ed = self._snap_points(event, self._ax4,
            getattr(self, '_energy_intersections', []))
        if ei is not None and ed < SNAP_THRESH * 1.5:
            ix, iy = self._energy_intersections[ei]
            self._show_cursor_intersection(self._cursor_annot4, self._ax4,
                ix, iy, "Energy", p['energy'].symbol)
            self._blit_tab4(); return

        if best_pt and best_d < SNAP_THRESH:
            self._show_cursor_energy(best_pt)
        else:
            self._cursor_annot4.set_visible(False)
        self._blit_tab4()

    def _show_cursor_energy(self, pt, label_hint=None):
        p = self._unit_profile
        d_sym, v_sym = p["distance"].symbol, self._vsym(p)
        text = (f"Range: {pt.distance >> p['distance']:.1f} {d_sym}\n"
                f"Energy: {pt.energy >> p['energy']:.1f} {p['energy'].symbol}\n"
                f"Velocity: {pt.velocity >> p['velocity']:.1f} {v_sym}\n"
                f"Time: {pt.time:.3f} s"
                + (f"\n{label_hint}" if label_hint else ""))
        self._cursor_annot4.set_text(text)
        x_data = pt.distance >> p["distance"]
        y_data = pt.energy >> p["energy"]
        self._cursor_annot4.xy = (x_data, y_data)
        self._cursor_annot4.xyann = self._annot_offset(self._ax4, x_data, y_data)
        self._cursor_annot4.set_visible(True)

    def _on_click_energy(self, event):
        if event.inaxes is None or not hasattr(self, '_energy_traj_data') or not self._energy_traj_data:
            return
        p = self._unit_profile
        SNAP_THRESH = _SNAP_THRESH_PX
        best_pt, best_d = None, float("inf")
        for dists, energies, result, _ in self._energy_traj_data:
            for i, (dx, dy) in enumerate(zip(dists, energies)):
                xd, yd = self._ax4.transData.transform((dx, dy))
                d = math.hypot(xd - event.x, yd - event.y)
                if d < best_d:
                    best_d = d; best_pt = result[i]
        for _, _, result, _ in self._energy_traj_data:
            for pt in [self._find_mach12(result), self._find_mach10(result), self._find_mach08(result)]:
                if pt is None: continue
                sx = pt.distance >> p["distance"]
                sy = pt.energy >> p["energy"]
                xd, yd = self._ax4.transData.transform((sx, sy))
                if math.hypot(xd - event.x, yd - event.y) < SNAP_THRESH * 1.5:
                    self._highlight_point_energy(pt); return
        if best_pt and best_d < SNAP_THRESH:
            self._highlight_point_energy(best_pt)

    def _highlight_point_energy(self, pt):
        p = self._unit_profile
        self._highlight_scatter4.set_offsets([[pt.distance >> p["distance"],
                                               pt.energy >> p["energy"]]])
        self._highlight_scatter4.set_visible(True)
        self._canvas4.draw_idle()

    # ============================================================
    # Tab 3: Energy — locate controls
    # ============================================================
    def _locate_distance4(self, event=None):
        """Draw vertical dashed line at given range"""
        try:
            target = float(self.loc_dist_var4.get())
        except (ValueError, tk.TclError):
            return
        self._loc_vline4.set_xdata([target, target])
        self._loc_vline4.set_visible(True)
        self._canvas4.draw_idle()
        self._update_energy_table()

    def _locate_energy4(self, event=None):
        """Draw horizontal dashed line at given energy"""
        try:
            target = float(self.loc_energy_var4.get())
        except (ValueError, tk.TclError):
            return
        self._loc_hline4.set_ydata([target, target])
        self._loc_hline4.set_visible(True)
        self._canvas4.draw_idle()
        self._update_energy_table()

    def _locate_time4(self, event=None):
        """Draw circles at given time on all energy curves"""
        if not hasattr(self, '_energy_traj_data') or not self._energy_traj_data: return
        p = self._unit_profile
        try:
            target = float(self.loc_time_var4.get())
        except (ValueError, tk.TclError):
            return
        xs, ys = [], []
        for _, _, result, _ in self._energy_traj_data:
            try:
                pt = result.get_at("time", target)
                xs.append(pt.distance >> p["distance"])
                ys.append(pt.energy >> p["energy"])
            except Exception:
                continue
        if xs:
            self._loc_scatter4.set_offsets(list(zip(xs, ys)))
            self._loc_scatter4.set_visible(True)
        self._canvas4.draw_idle()
        self._update_energy_table()

    def _clear_locate4(self):
        """Clear all locate lines & points"""
        self.loc_dist_var4.set("")
        self.loc_energy_var4.set("")
        self.loc_time_var4.set("")
        self._loc_vline4.set_visible(False)
        self._loc_hline4.set_visible(False)
        self._loc_scatter4.set_visible(False)
        self._canvas4.draw_idle()
        self._update_energy_table()

    def _restore_locate4(self):
        """Restore locate lines after calc (plot rebuild clears them)"""
        try:
            target = float(self.loc_dist_var4.get())
            self._loc_vline4.set_xdata([target, target])
            self._loc_vline4.set_visible(True)
        except (ValueError, tk.TclError):
            self._loc_vline4.set_visible(False)
        try:
            target = float(self.loc_energy_var4.get())
            self._loc_hline4.set_ydata([target, target])
            self._loc_hline4.set_visible(True)
        except (ValueError, tk.TclError):
            self._loc_hline4.set_visible(False)
        try:
            target = float(self.loc_time_var4.get())
            if hasattr(self, '_energy_traj_data') and self._energy_traj_data:
                p = self._unit_profile
                xs, ys = [], []
                for _, _, result, _ in self._energy_traj_data:
                    try:
                        pt = result.get_at("time", target)
                        xs.append(pt.distance >> p["distance"])
                        ys.append(pt.energy >> p["energy"])
                    except Exception:
                        continue
                if xs:
                    self._loc_scatter4.set_offsets(list(zip(xs, ys)))
                    self._loc_scatter4.set_visible(True)
                else:
                    self._loc_scatter4.set_visible(False)
            else:
                self._loc_scatter4.set_visible(False)
        except (ValueError, tk.TclError):
            self._loc_scatter4.set_visible(False)
        self._canvas4.draw_idle()

    # ============================================================
    # Tab 4: Wind Drift — chart
    # ============================================================
    def _update_windage_plot(self):
        """Tab4: Drift vs Range chart — mirrors Tab2 multi-analysis style"""
        results = self._all_results
        if not results or not any(r is not None for r in results): return
        p = self._unit_profile
        if p is None: return

        for ax in list(self._fig3.axes):
            if ax is not self._ax3: ax.remove()
        self._ax3.clear()

        d_sym, h_sym = p["distance"].symbol, p["drop"].symbol
        colors = self._CMP_COLORS
        show_all = self._windage_scope_var.get()

        # determine which ammo indices to plot
        if show_all:
            plot_indices = [i for i in self._active_indices
                           if self._all_results[i] is not None and len(self._all_results[i]) > 0]
        else:
            idx = self._current_ammo_idx
            if (0 <= idx < len(self._ammo_library)
                    and self._all_results[idx] is not None
                    and len(self._all_results[idx]) > 0):
                plot_indices = [idx]
            else:
                plot_indices = []

        # summary range
        all_max_dist = 0
        all_wind_min = float("inf")
        all_wind_max = float("-inf")
        self._wind_traj_data = []
        for lib_idx in plot_indices:
            r = self._all_results[lib_idx]
            dists = [pt.distance >> p["distance"] for pt in r]
            windages = [pt.windage >> p["drop"] for pt in r]
            all_max_dist = max(all_max_dist, dists[-1])
            all_wind_min = min(all_wind_min, min(windages))
            all_wind_max = max(all_wind_max, max(windages))
            self._wind_traj_data.append((dists, windages, r, lib_idx))

        if not self._wind_traj_data: return

        # ensure zero point is within visible range
        if all_wind_min > 0: all_wind_min = 0
        if all_wind_max < 0: all_wind_max = 0

        # ---- each drift curve ----
        for pos, (dists, windages, r, lib_idx) in enumerate(self._wind_traj_data):
            color = colors[pos % len(colors)]
            name = self._ammo_library[lib_idx].get("name", f"Ammo{lib_idx+1}")
            self._ax3.plot(dists, windages, color=color, linewidth=1.2,
                          label=name, zorder=3)

        # ---- zero line ----
        self._ax3.axhline(y=0, color="#CCCCCC", linewidth=0.8, linestyle="-")

        # ---- labels & style ----
        self._ax3.set_xlabel(f"Range ({d_sym})")
        self._ax3.set_ylabel(f"Drift ({h_sym})")
        self._ax3.xaxis.set_major_locator(MaxNLocator(nbins=20, steps=[1, 5, 10]))
        self._ax3.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
        self._ax3.grid(True, alpha=0.35, linewidth=0.5)
        self._ax3.set_facecolor("white")

        # ---- legend: one row per ammo (sorted by drift) ----
        records = []
        for pos, lib_idx in enumerate(plot_indices):
            color = colors[pos % len(colors)]
            name = self._ammo_library[lib_idx].get("name", f"Ammo{lib_idx+1}")
            _, windages, _, _ = self._wind_traj_data[pos]
            sort_key = sum(windages) / len(windages) if windages else 0
            ln = Line2D([0], [0], color=color, linewidth=1.2)
            records.append((sort_key, (ln,), name))
        records.sort(key=lambda r: r[0], reverse=True)
        pairs = [r[1] for r in records]
        pair_labels = [r[2] for r in records]

        if pairs:
            self._ax3.legend(pairs, pair_labels, fontsize=8,
                            loc="upper left", framealpha=0.85, edgecolor="#CCCCCC",
                            handler_map={tuple: HandlerTuple(ndivide=None)})

        # ---- hover elements (recreated) ----
        self._cursor_annot3 = self._ax3.annotate("", xy=(0, 0), xytext=(12, 12),
            textcoords="offset points", fontsize=8, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#AAA", alpha=0.9),
            visible=False, zorder=99, annotation_clip=False)
        self._highlight_scatter3 = self._ax3.scatter([], [], s=100, c="none",
            edgecolors="#FF6600", linewidths=2.5, zorder=98, visible=False)
        self._ch_vline3 = self._ax3.axvline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._ch_hline3 = self._ax3.axhline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._ch_vline3.set_animated(True)
        self._ch_hline3.set_animated(True)
        self._cursor_annot3.set_animated(True)

        # rebuild locate lines/points
        self._loc_vline3 = self._ax3.axvline(0, color="#000000", linewidth=0.5,
                                              linestyle="--", visible=False, zorder=85)
        self._loc_windage_hline3 = self._ax3.axhline(0, color="#000000", linewidth=0.5,
                                                       linestyle="--", visible=False, zorder=85)
        self._loc_scatter3 = self._ax3.scatter([], [], s=25, c="#FF6600",
                                                zorder=86, visible=False, marker="x",
                                                linewidths=1.2)

        self._fig3.tight_layout(pad=2.0)
        self._canvas3.draw()

    # ============================================================
    # Tab 4: Wind Drift — table
    # ============================================================
    def _update_windage_table(self):
        """Tab4: show compare or detail table per toggle"""
        results = self._all_results
        if not results or not any(r is not None for r in results): return
        p = self._unit_profile
        if p is None: return
        show_all = self._windage_scope_var.get()

        if show_all:
            self._update_windage_comparison_table(p)
        else:
            self._update_windage_detail_table(p)

    def _update_windage_comparison_table(self, p):
        """Tab4 compare table: one row per active list ammo"""
        self._wind_det_frame.pack_forget()
        self._wind_cmp_frame.pack(fill="both", expand=True)

        for row in self._wind_cmp_tree.get_children():
            self._wind_cmp_tree.delete(row)

        # dynamic headers (with units)
        self._wind_cmp_tree.heading("name", text="Ammo Name")
        self._wind_cmp_tree.heading("mv", text=f"MV / {self._vsym_hdr(p)}")
        self._wind_cmp_tree.heading("wind_max", text=f"Max Drift / {p['drop'].symbol}")

        # LocateRange / LocateDrift
        loc_dist = None
        try:
            loc_dist = float(self.loc_dist_var3.get())
        except (ValueError, tk.TclError):
            pass
        loc_windage = None
        try:
            loc_windage = float(self.loc_windage_var3.get())
        except (ValueError, tk.TclError):
            pass

        # dynamic headers: drift / drift angle / drift range
        if loc_dist is not None:
            d_sym = p['distance'].symbol
            self._wind_cmp_tree.heading("wind_dist",
                text=f"{loc_dist:.0f} {d_sym}Drift / {p['drop'].symbol}")
            self._wind_cmp_tree.heading("wind_angle_dist",
                text=f"{loc_dist:.0f} {d_sym}Drift Angle / {p['adjustment'].symbol}")
        else:
            self._wind_cmp_tree.heading("wind_dist", text=f"Drift / {p['drop'].symbol}")
            self._wind_cmp_tree.heading("wind_angle_dist", text=f"Drift Angle / {p['adjustment'].symbol}")
        if loc_windage is not None:
            self._wind_cmp_tree.heading("windage_dist",
                text=f"{loc_windage:.0f} {p['drop'].symbol}Drift Range / {p['distance'].symbol}")
        else:
            self._wind_cmp_tree.heading("windage_dist", text=f"Drift Range / {p['distance'].symbol}")

        for pos, lib_idx in enumerate(self._active_indices):
            result = self._all_results[lib_idx]
            if result is None or len(result) == 0: continue
            name = self._ammo_library[lib_idx].get("name", f"Ammo{lib_idx+1}")
            p0 = result[0]

            # max drift (point with largest absolute value)
            max_w_pt = max(result, key=lambda pt: abs(pt.windage.raw_value))
            max_wind = f"{max_w_pt.windage >> p['drop']:.2f}"

            # drift at locate range
            if loc_dist is not None:
                try:
                    pt_d = result.get_at("distance",
                                         Distance(loc_dist, p["distance"]).raw_value)
                    w_dist = f"{pt_d.windage >> p['drop']:.2f}"
                    wa_dist = f"{pt_d.windage_angle >> p['adjustment']:.3f}"
                except Exception:
                    w_dist = wa_dist = "-"
            else:
                w_dist = wa_dist = "-"

            # range at locate drift (reverse lookup)
            if loc_windage is not None:
                try:
                    pt_w = result.get_at("windage",
                                         Distance(loc_windage, p["drop"]).raw_value)
                    windage_dist = f"{pt_w.distance >> p['distance']:.1f}"
                except Exception:
                    windage_dist = "-"
            else:
                windage_dist = "-"

            vals = (
                name,
                f"{p0.velocity >> p['velocity']:.1f}",
                w_dist,
                windage_dist,
                wa_dist,
                max_wind,
            )
            self._wind_cmp_tree.insert("", "end", values=vals)

        children = self._wind_cmp_tree.get_children()
        if children:
            first = self._wind_cmp_tree.item(children[0], 'values')
            last = self._wind_cmp_tree.item(children[-1], 'values')
            self._auto_fit_columns(self._wind_cmp_tree, self._wind_cmp_cols, first, last)
        self._sorter_wcmp.reset()

    def _update_windage_detail_table(self, p):
        """Tab4 detail table: selected ammo only; one row per step showing drift"""
        self._wind_cmp_frame.pack_forget()
        self._wind_det_frame.pack(fill="both", expand=True)

        for row in self._wind_det_tree.get_children():
            self._wind_det_tree.delete(row)

        # dynamic headers (with units)
        self._wind_det_tree.heading("distance", text=f"Range / {p['distance'].symbol}")
        self._wind_det_tree.heading("windage", text=f"Drift / {p['drop'].symbol}")
        self._wind_det_tree.heading("windage_angle", text=f"Drift Angle / {p['adjustment'].symbol}")
        self._wind_det_tree.heading("velocity", text=f"Velocity / {self._vsym_hdr(p)}")
        self._wind_det_tree.heading("time", text="Time / s")

        idx = self._current_ammo_idx
        if idx < 0 or idx >= len(self._ammo_library): return
        result = self._all_results[idx]
        if result is None or len(result) == 0: return

        table_step = max(self._table_step, 1.0)
        shown_buckets = set()

        for pt in result:
            is_special = bool(pt.flag & (TrajFlag.ZERO | TrajFlag.APEX))
            dist_raw = pt.distance >> p["distance"]
            bucket = int(dist_raw / table_step)
            key = (bucket, is_special)
            if key in shown_buckets and not is_special: continue
            shown_buckets.add(key)

            vals = (
                f"{dist_raw:.1f}",
                f"{pt.windage >> p['drop']:.2f}",
                f"{pt.windage_angle >> p['adjustment']:.3f}",
                f"{pt.velocity >> p['velocity']:.1f}",
                f"{pt.time:.3f}",
            )
            tag = ""
            if pt.flag & TrajFlag.ZERO: tag = "zero"
            elif pt.flag & TrajFlag.APEX: tag = "apex"
            self._wind_det_tree.insert("", "end", values=vals, tags=(tag,) if tag else ())

        self._wind_det_tree.tag_configure("zero", background="#FDDBC7")
        self._wind_det_tree.tag_configure("apex", background="#D9F0D3")

        children = self._wind_det_tree.get_children()
        if children:
            first = self._wind_det_tree.item(children[0], 'values')
            last = self._wind_det_tree.item(children[-1], 'values')
            self._auto_fit_columns(self._wind_det_tree, self._wind_det_cols, first, last)
        self._sorter_wdet.reset()

    def _on_windage_scope_toggle(self):
        """Toggle: refresh drift chart & table"""
        if not self._all_results or not any(r is not None for r in self._all_results):
            return
        self._update_windage_plot()
        self._update_windage_table()

    # ============================================================
    # Tab 4: chart interaction
    # ============================================================
    def _on_draw_tab3(self, event):
        """Capture static background for blit acceleration"""
        self._bg3 = self._fig3.canvas.copy_from_bbox(self._fig3.bbox)
        self._bg3_w = self._fig3.bbox.width
        self._bg3_h = self._fig3.bbox.height

    def _blit_tab3(self):
        """Unified blit: restore background → draw animated elements → flush"""
        if not hasattr(self, '_bg3'): return
        if (self._fig3.bbox.width, self._fig3.bbox.height) != (self._bg3_w, self._bg3_h):
            return
        self._fig3.canvas.restore_region(self._bg3)
        if self._ch_vline3.get_visible():
            self._ax3.draw_artist(self._ch_vline3)
        if self._ch_hline3.get_visible():
            self._ax3.draw_artist(self._ch_hline3)
        if self._cursor_annot3.get_visible():
            self._ax3.draw_artist(self._cursor_annot3)
        self._fig3.canvas.blit()

    def _on_hover_wind(self, event):
        if event.inaxes is None or not hasattr(self, '_wind_traj_data') or not self._wind_traj_data:
            self._ch_vline3.set_visible(False)
            self._ch_hline3.set_visible(False)
            self._cursor_annot3.set_visible(False)
            self._blit_tab3()
            return
        p = self._unit_profile
        SNAP_THRESH = _SNAP_THRESH_PX

        # crosshair (pixel coords)
        mx, my = self._ax3.transData.inverted().transform((event.x, event.y))
        self._ch_vline3.set_xdata([mx, mx])
        self._ch_vline3.set_visible(True)
        self._ch_hline3.set_ydata([my, my])
        self._ch_hline3.set_visible(True)

        # find nearest point across all drift curves
        best_pt, best_d = self._snap_traj(event, self._ax3, self._wind_traj_data)

        if best_pt and best_d < SNAP_THRESH:
            self._show_cursor_wind(best_pt)
        else:
            self._cursor_annot3.set_visible(False)
        self._blit_tab3()

    def _show_cursor_wind(self, pt):
        p = self._unit_profile
        d_sym, h_sym = p["distance"].symbol, p["drop"].symbol
        text = (f"Range: {pt.distance >> p['distance']:.1f} {d_sym}\n"
                f"Wind Drift: {pt.windage >> p['drop']:.2f} {h_sym}\n"
                f"Drift Angle: {pt.windage_angle >> p['adjustment']:.3f} {p['adjustment'].symbol}\n"
                f"Velocity: {pt.velocity >> p['velocity']:.1f} {self._vsym(p)}\n"
                f"Time: {pt.time:.3f} s")
        self._cursor_annot3.set_text(text)
        x_data = pt.distance >> p["distance"]
        y_data = pt.windage >> p["drop"]
        self._cursor_annot3.xy = (x_data, y_data)
        self._cursor_annot3.xyann = self._annot_offset(self._ax3, x_data, y_data)
        self._cursor_annot3.set_visible(True)

    def _on_click_wind(self, event):
        if event.inaxes is None or not hasattr(self, '_wind_traj_data') or not self._wind_traj_data:
            return
        p = self._unit_profile
        SNAP_THRESH = _SNAP_THRESH_PX

        best_pt, best_d = None, float("inf")
        for dists, windages, result, _ in self._wind_traj_data:
            for i, (dx, dy) in enumerate(zip(dists, windages)):
                xd, yd = self._ax3.transData.transform((dx, dy))
                d = math.hypot(xd - event.x, yd - event.y)
                if d < best_d:
                    best_d = d; best_pt = result[i]

        if best_pt and best_d < SNAP_THRESH:
            self._highlight_point_wind(best_pt)

    def _highlight_point_wind(self, pt):
        p = self._unit_profile
        self._highlight_scatter3.set_offsets([[pt.distance >> p["distance"],
                                               pt.windage >> p["drop"]]])
        self._highlight_scatter3.set_visible(True)
        self._canvas3.draw_idle()

    # ============================================================
    # Tab 5: Drag — chart + table
    # ============================================================
    # standard atmosphere constants (for drag calculation; unified baseline across all ammo)
    _RHO_STD = 1.225       # kg/m³, ICAO sea-level standard air density
    _C_SOUND_STD = 340.3   # m/s, 15°C standard speed of sound
    _DENSE_STEP = 0.01     # drag curve resampling step (Mach)

    @staticmethod
    def _drag_force(mach, cd_eff, area_m2):
        """Drag (N) = ½ρv² × CD_eff × A,  v = Mach × speed of sound"""
        v = mach * BallisticApp._C_SOUND_STD
        return 0.5 * BallisticApp._RHO_STD * v * v * cd_eff * area_m2

    def _get_bullet_area(self, lib_idx):
        """Bullet cross-sectional area (m²), converted via PreferredUnits for all unit systems"""
        cfg = self._ammo_library[lib_idx]
        d = cfg.get("diameter") or 7.82
        d_m = PreferredUnits.diameter(d) >> Distance.Meter
        return math.pi * (d_m / 2) ** 2

    def _update_drag_plot(self):
        """Tab5: Mach vs Drag (N) line chart"""
        results = self._all_drag_data
        if not results or not any(r is not None for r in results): return
        self._ax5.clear()

        colors = self._CMP_COLORS
        plot_indices = [i for i in self._active_indices
                       if results[i] is not None and len(results[i][0]) > 0]

        # Mach reference lines
        for m_ref, ls in [(0.8, ":"), (1.0, "--"), (1.2, ":")]:
            self._ax5.axvline(x=m_ref, color="#AAAAAA", linewidth=0.8, linestyle=ls)

        # dF/dM slope zones (color temperature follows d²F/dM² from negative to positive)
        _ZONES = [
            (0.00, 0.80, "#F5EDE0", ""),  # subsonic, gradual rise
            (0.80, 1.20, "#F5D5C8", ""),  # transonic, steep rise
            (1.20, 1.45, "#F0F0F0", ""),  # valley, flat
            (1.45, 1.95, "#F5E0D0", ""),  # rapid climb
            (1.95, 2.85, "#EDEDED", ""),  # broad peak, near zero
            (2.85, 3.50, "#D8E8F5", ""),  # accelerating decline
        ]
        for x0, x1, color, _ in _ZONES:
            self._ax5.axvspan(x0, x1, facecolor=color, edgecolor="none", alpha=0.55, zorder=0)

        # each drag curve (Y = drag N)
        all_force_max = 0
        # resample to 0.01 Mach step for intersection precision & smooth hover
        dense_data = []  # (dense_machs, dense_forces, None, lib_idx) for intersection detection
        self._drag_hover_points = []
        for pos, lib_idx in enumerate(plot_indices):
            machs, cds = results[lib_idx]
            if len(machs) == 0: continue
            area = self._get_bullet_area(lib_idx)
            # resample
            d_machs, d_forces = [], []
            m = 0.0
            while m <= 3.5 + 1e-9:
                cd = self._interp_at(machs, cds, m)
                f = self._drag_force(m, cd, area)
                d_machs.append(m)
                d_forces.append(f)
                self._drag_hover_points.append((m, f))
                if f > all_force_max: all_force_max = f
                m += self._DENSE_STEP
            color = colors[pos % len(colors)]
            name = self._ammo_library[lib_idx].get("name", f"Ammo{lib_idx+1}")
            self._ax5.plot(d_machs, d_forces, color=color, linewidth=1.2, label=name, zorder=3)
            dense_data.append((d_machs, d_forces, None, lib_idx))

        # ---- slope twin axis: G7 standard bullet dF/dM (N/Mach) ----
        if hasattr(self, '_ax5_deriv') and self._ax5_deriv is not None:
            self._ax5_deriv.remove()
        ax5_deriv = self._ax5.twinx()
        self._ax5_deriv = ax5_deriv
        self._drag_deriv_points = []
        all_slope_max = 0
        # pick Mach points from any G7 ammo (all G7 identical); use standard CD table directly
        g7_machs = [pt["Mach"] for pt in TableG7]
        g7_cds   = [pt["CD"]  for pt in TableG7]
        # G7 reference bullet: diameter 1 inch, BC=1, CD_actual = CD_table
        d_ref_m = 0.0254  # 1 inch
        area_ref = math.pi * (d_ref_m / 2) ** 2
        forces_g7 = [self._drag_force(m, cd, area_ref) for m, cd in zip(g7_machs, g7_cds)]
        slopes, slope_ms = [], []
        for i in range(len(g7_machs)):
            if i == 0:
                dh = g7_machs[1] - g7_machs[0]
                s = (forces_g7[1] - forces_g7[0]) / dh
            elif i == len(g7_machs) - 1:
                dh = g7_machs[-1] - g7_machs[-2]
                s = (forces_g7[-1] - forces_g7[-2]) / dh
            else:
                dh = g7_machs[i+1] - g7_machs[i-1]
                s = (forces_g7[i+1] - forces_g7[i-1]) / dh
            if s > all_slope_max: all_slope_max = s
            M = g7_machs[i]
            if 0 < M <= 3.5:
                slope_ms.append(M)
                slopes.append(s)
                self._drag_deriv_points.append((M, s))
        ax5_deriv.plot(slope_ms, slopes, color="#6C8EBF", linewidth=1.0, linestyle="--")
        # zone extrema markers + storage for hover snap
        _EXTREMA = [
            (1.20, 1.45, "min"),   # valley
            (1.95, 2.85, "max"),   # peak
            (2.85, 3.50, "min"),   # secondary low
        ]
        self._drag_extrema = []
        for x0, x1, kind in _EXTREMA:
            pts = [(m, s) for m, s in zip(slope_ms, slopes) if x0 <= m <= x1]
            if not pts: continue
            if kind == "max":
                best = max(pts, key=lambda p: p[1]); marker = "v"
            else:
                best = min(pts, key=lambda p: p[1]); marker = "^"
            self._drag_extrema.append((best[0], best[1]))
            ax5_deriv.plot(best[0], best[1], marker=marker, color="#6C8EBF",
                           markersize=8, markeredgecolor="white", markeredgewidth=0.6,
                           linestyle="none", zorder=10)
        ax5_deriv.set_ylabel("dF/dM (G7) [N/Mach]", color="#6C8EBF")
        ax5_deriv.tick_params(axis="y", colors="#6C8EBF", labelsize=8)
        if all_slope_max > 0:
            ax5_deriv.set_ylim(0, all_slope_max * 1.05)
        ax5_deriv.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
        ax5_deriv.spines["top"].set_visible(False)
        ax5_deriv.spines["left"].set_visible(False)

        # curve intersections (excluded from legend)
        self._drag_intersections = self._add_curve_intersections(self._ax5, dense_data, colors)

        # style
        self._ax5.set_xlabel("Mach")
        self._ax5.set_ylabel("Drag (N)")
        self._ax5.xaxis.set_major_locator(MaxNLocator(nbins=20, steps=[1, 2, 5, 10]))
        self._ax5.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
        self._ax5.grid(True, alpha=0.35, linewidth=0.5)
        self._ax5.set_facecolor("white")

        # legend (descending by avg drag; dashed lines at bottom)
        handles, labels = self._ax5.get_legend_handles_labels()
        if handles:
            recs = []
            for h, l in zip(handles, labels):
                ydata = h.get_ydata()
                sort_key = sum(ydata) / len(ydata) if len(ydata) > 0 else 0
                recs.append((sort_key, h, l))
            recs.sort(key=lambda r: r[0], reverse=True)
            all_h = [r[1] for r in recs]
            all_l = [r[2] for r in recs]
            # dashed: dF/dM slope
            all_h.append(Line2D([0], [0], color="#6C8EBF", linewidth=1.0, linestyle="--"))
            all_l.append("dF/dM (G7)")
            self._ax5.legend(all_h, all_l,
                            fontsize=8, loc="upper left", framealpha=0.85, edgecolor="#CCCCCC")

        self._fig5.tight_layout(pad=2.0)

        self._ax5.set_xlim(left=0, right=3.65)
        if all_force_max > 0:
            self._ax5.set_ylim(bottom=0, top=all_force_max * 1.05)

        # hover elements (must rebuild after ax.clear())
        self._cursor_annot5 = self._ax5.annotate("", xy=(0, 0), xytext=(12, 12),
            textcoords="offset points", fontsize=8, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#AAA", alpha=0.9),
            visible=False, zorder=99, annotation_clip=False)
        self._highlight_scatter5 = self._ax5.scatter([], [], s=80, c="none",
            edgecolors="#FF6600", linewidths=2, zorder=98, visible=False)
        self._ch_vline5 = self._ax5.axvline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._ch_hline5 = self._ax5.axhline(0, color="#888888", linewidth=0.5,
                                          linestyle="--", visible=False, zorder=90)
        self._ch_vline5.set_animated(True)
        self._ch_hline5.set_animated(True)
        self._cursor_annot5.set_animated(True)
        self._canvas5.draw()

    @staticmethod
    def _interp_at(machs, values, target_mach):
        """Linear interpolation: find value at target_mach on machs curve"""
        if target_mach <= machs[0]: return values[0]
        if target_mach >= machs[-1]: return values[-1]
        for i in range(1, len(machs)):
            if machs[i] >= target_mach:
                t = (target_mach - machs[i-1]) / (machs[i] - machs[i-1])
                return values[i-1] + t * (values[i] - values[i-1])
        return values[-1]

    def _update_drag_table(self):
        """Tab5: Drag Data table — drag (N) at key Mach points"""
        for row in self._drag_tree.get_children():
            self._drag_tree.delete(row)
        if not self._all_drag_data or not any(r is not None for r in self._all_drag_data):
            return

        indices = [i for i in self._active_indices
                  if self._all_drag_data[i] is not None]

        key_machs = [0.8, 1.0, 1.2]
        for lib_idx in indices:
            machs, cds = self._all_drag_data[lib_idx]
            name = self._ammo_library[lib_idx].get("name", f"Ammo{lib_idx+1}")
            cfg = self._ammo_library[lib_idx]
            bc = cfg.get("bc", 0.223)
            drag_name = cfg.get("drag_table", "G7")
            drag_short = drag_name.split()[0]  # "G7", "G1", "GI", "RA4", ...
            if drag_short == "Custom":
                drag_short = "Custom"
            area = self._get_bullet_area(lib_idx)
            forces = [self._drag_force(m, cd, area) for m, cd in zip(machs, cds)]
            f_vals = [f"{self._interp_at(machs, forces, m):.2f}" for m in key_machs]
            f_avg = f"{sum(forces) / len(forces):.2f}"
            vals = (name, f"{drag_short} {bc:.3f}", *f_vals, f_avg)
            self._drag_tree.insert("", "end", values=vals)
        self._sorter5.reset()

    def _on_draw_tab5(self, event):
        """Capture static background for blit acceleration"""
        self._bg5 = self._fig5.canvas.copy_from_bbox(self._fig5.bbox)
        self._bg5_w = self._fig5.bbox.width
        self._bg5_h = self._fig5.bbox.height

    def _blit_tab5(self):
        """restore background → draw animated elements → blit flush"""
        if not hasattr(self, '_bg5'): return
        if (self._fig5.bbox.width, self._fig5.bbox.height) != (self._bg5_w, self._bg5_h):
            return
        self._fig5.canvas.restore_region(self._bg5)
        if self._ch_vline5.get_visible():
            self._ax5.draw_artist(self._ch_vline5)
        if self._ch_hline5.get_visible():
            self._ax5.draw_artist(self._ch_hline5)
        if self._cursor_annot5.get_visible():
            self._ax5.draw_artist(self._cursor_annot5)
        self._fig5.canvas.blit()

    def _on_hover_drag(self, event):
        if event.inaxes is None:
            self._cursor_annot5.set_visible(False)
            self._ch_vline5.set_visible(False)
            self._ch_hline5.set_visible(False)
            self._blit_tab5()
            return

        # crosshair always follows mouse position
        mx, my = self._ax5.transData.inverted().transform((event.x, event.y))
        self._ch_vline5.set_xdata([mx, mx])
        self._ch_vline5.set_visible(True)
        self._ch_hline5.set_ydata([my, my])
        self._ch_hline5.set_visible(True)

        # intersection snap (takes priority over regular data points)
        inter_pts = getattr(self, '_drag_intersections', None)
        if inter_pts:
            ii, id_ = self._snap_points(event, self._ax5, inter_pts)
            if ii is not None and id_ < _SNAP_THRESH_PX * 1.5:
                ix, iy = inter_pts[ii]
                self._show_cursor_intersection(self._cursor_annot5, self._ax5,
                    ix, iy, "Drag", "N", x_label="Mach", x_symbol="")
                self._blit_tab5(); return

        # extrema point snap (crosshair only; no tooltip)
        extrema = getattr(self, '_drag_extrema', None)
        if extrema:
            ei, ed = self._snap_points(event, self._ax5, extrema, source_ax=self._ax5_deriv)
            if ei is not None and ed < _SNAP_THRESH_PX * 1.5:
                ex, ey = extrema[ei]
                # horizontal line y: convert from derivative axis to force axis coords
                _, y_disp = self._ax5_deriv.transData.transform((0, ey))
                _, yy = self._ax5.transData.inverted().transform((0, y_disp))
                self._ch_vline5.set_xdata([ex, ex])
                self._ch_hline5.set_ydata([yy, yy])
                self._cursor_annot5.set_visible(False)
                self._blit_tab5(); return

        points = getattr(self, '_drag_hover_points', None)
        deriv_pts = getattr(self, '_drag_deriv_points', None)

        best_is_deriv = False
        best_i, best_d = None, float("inf")
        if points:
            best_i, best_d = self._snap_points(event, self._ax5, points)
        if deriv_pts:
            di, dd = self._snap_points(event, self._ax5, deriv_pts, source_ax=self._ax5_deriv)
            if di is not None and dd < best_d:
                best_i, best_d = di, dd
                best_is_deriv = True

        if best_i is None or best_d >= _SNAP_THRESH_PX:
            self._cursor_annot5.set_visible(False)
            self._blit_tab5()
            return

        if best_is_deriv:
            mx_snap, slope_snap = deriv_pts[best_i]
            self._cursor_annot5.xy = (mx_snap, slope_snap)
            self._cursor_annot5.xyann = self._annot_offset(self._ax5_deriv, mx_snap, slope_snap)
            self._cursor_annot5.set_text(f"Mach: {mx_snap:.3f}\ndF/dM: {slope_snap:.2f} N/Mach")
        else:
            mx_snap, f_snap = points[best_i]
            self._cursor_annot5.xy = (mx_snap, f_snap)
            self._cursor_annot5.xyann = self._annot_offset(self._ax5, mx_snap, f_snap)
            self._cursor_annot5.set_text(f"Mach: {mx_snap:.3f}\nDrag: {f_snap:.2f} N")
        self._cursor_annot5.set_visible(True)
        self._blit_tab5()

    def _on_click_drag(self, event):
        if event.inaxes is None: return
        points = getattr(self, '_drag_hover_points', None)
        if not points: return

        best_i, best_d = self._snap_points(event, self._ax5, points)
        if best_i is not None and best_d <= _SNAP_THRESH_PX:
            self._highlight_scatter5.set_offsets([points[best_i]])
            self._highlight_scatter5.set_visible(True)
            self._canvas5.draw_idle()

    # ============================================================
    # Tab 4: locate controls
    # ============================================================
    def _locate_distance3(self, event=None):
        """Draw vertical dashed line at given range"""
        try:
            target = float(self.loc_dist_var3.get())
        except (ValueError, tk.TclError):
            return
        self._loc_vline3.set_xdata([target, target])
        self._loc_vline3.set_visible(True)
        self._canvas3.draw_idle()
        self._update_windage_table()

    def _locate_windage3(self, event=None):
        """Draw horizontal dashed line at given drift"""
        try:
            target = float(self.loc_windage_var3.get())
        except (ValueError, tk.TclError):
            return
        self._loc_windage_hline3.set_ydata([target, target])
        self._loc_windage_hline3.set_visible(True)
        self._canvas3.draw_idle()
        self._update_windage_table()

    def _locate_time3(self, event=None):
        """Draw circles at given time on all drift curves"""
        if not hasattr(self, '_wind_traj_data') or not self._wind_traj_data: return
        p = self._unit_profile
        try:
            target = float(self.loc_time_var3.get())
        except (ValueError, tk.TclError):
            return
        xs, ys = [], []
        for _, _, result, _ in self._wind_traj_data:
            try:
                pt = result.get_at("time", target)
                xs.append(pt.distance >> p["distance"])
                ys.append(pt.windage >> p["drop"])
            except Exception:
                continue
        if xs:
            self._loc_scatter3.set_offsets(list(zip(xs, ys)))
            self._loc_scatter3.set_visible(True)
        self._canvas3.draw_idle()
        self._update_windage_table()

    def _clear_locate3(self):
        """Clear all locate lines & points"""
        self.loc_dist_var3.set("")
        self.loc_windage_var3.set("")
        self.loc_time_var3.set("")
        self._loc_vline3.set_visible(False)
        self._loc_windage_hline3.set_visible(False)
        self._loc_scatter3.set_visible(False)
        self._canvas3.draw_idle()
        self._update_windage_table()

    def _restore_locate3(self):
        """Restore locate lines after calc (plot rebuild clears them)"""
        try:
            target = float(self.loc_dist_var3.get())
            self._loc_vline3.set_xdata([target, target])
            self._loc_vline3.set_visible(True)
        except (ValueError, tk.TclError):
            self._loc_vline3.set_visible(False)
        try:
            target = float(self.loc_windage_var3.get())
            self._loc_windage_hline3.set_ydata([target, target])
            self._loc_windage_hline3.set_visible(True)
        except (ValueError, tk.TclError):
            self._loc_windage_hline3.set_visible(False)
        try:
            target = float(self.loc_time_var3.get())
            if hasattr(self, '_wind_traj_data') and self._wind_traj_data:
                p = self._unit_profile
                xs, ys = [], []
                for _, _, result, _ in self._wind_traj_data:
                    try:
                        pt = result.get_at("time", target)
                        xs.append(pt.distance >> p["distance"])
                        ys.append(pt.windage >> p["drop"])
                    except Exception:
                        continue
                if xs:
                    self._loc_scatter3.set_offsets(list(zip(xs, ys)))
                    self._loc_scatter3.set_visible(True)
                else:
                    self._loc_scatter3.set_visible(False)
            else:
                self._loc_scatter3.set_visible(False)
        except (ValueError, tk.TclError):
            self._loc_scatter3.set_visible(False)
        self._canvas3.draw_idle()


# ============================================================
# entry point
    def _install_draw_debounce(self, canvas, timer_attr: str):
        """Replace canvas.draw_idle with 60ms debounced version to avoid high-frequency redraw during drag"""
        setattr(self, timer_attr, None)
        original = canvas.draw_idle

        def debounced():
            timer = getattr(self, timer_attr, None)
            if timer is not None:
                canvas.get_tk_widget().after_cancel(timer)
            setattr(self, timer_attr, canvas.get_tk_widget().after(_DEBOUNCE_MS, original))

        canvas.draw_idle = debounced

# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = BallisticApp(root)
    root.mainloop()

# GitHub: https://github.com/Celeritas2026/ExteriorBallisticsCalc_py-ballisticcalc-gui
