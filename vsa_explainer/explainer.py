import matplotlib.pyplot as plt
from IPython.display import HTML as IPythonHTML
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw, AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D, MolToImage
from rdkit.Chem.EState import EState, EState_VSA
from rdkit.Chem.Lipinski import NumHDonors, NumHAcceptors
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
import numpy as np
import re
from collections import namedtuple
from IPython.display import SVG, display

def get_vsa_bin_bounds(descriptor_name):
    """
    Given a descriptor like "SMR_VSA3" or "SlogP_VSA1", parse its __doc__
    and return (lower_bound, upper_bound) as floats.
    """
    func = getattr(Descriptors, descriptor_name, None)
    if func is None or func.__doc__ is None:
        raise ValueError(f"No such descriptor {descriptor_name!r} or missing __doc__")
    doc = func.__doc__
    # unified pattern: matches "a <= x < b", "a < x < b", including "-inf"
    m = re.search(
        r"""\(\s*([+-]?\d*\.?\d+|[-]?inf)  # group 1: a or -inf
              \s*(?:<=|<)\s*x\s*(?:<|<=)\s*
              ([+-]?\d*\.?\d+|[-]?inf)      # group 2: b or inf
          \s*\)""",
        doc,
        flags=re.IGNORECASE | re.VERBOSE
    )
    if m:
        lb, ub = m.group(1).lower(), m.group(2).lower()
        lower = float("-inf") if lb in ("-inf",) else float(lb)
        upper = float("inf") if ub in ("inf", "+inf") else float(ub)
        return lower, upper

    # fallback: single‑sided "( x < b )"
    m2 = re.search(r"\(\s*x\s*<\s*([+-]?\d*\.?\d+)\s*\)", doc)
    if m2:
        return float("-inf"), float(m2.group(1))
    # fallback: "( a <= x )"
    m3 = re.search(r"\(\s*([+-]?\d*\.?\d+)\s*<=\s*x\s*\)", doc)
    if m3:
        return float(m3.group(1)), float("inf")

    raise ValueError(f"Could not parse bin bounds from {doc!r}")

def get_bin_bounds(idx, bins):
    """
    idx is 1‑based. bins is a sorted list of length N.
      idx == 1       → (-inf, bins[0])
      2 <= idx <= N → [bins[idx-2], bins[idx-1])
      idx == N + 1   → [bins[-1], inf)
    """
    N = len(bins)
    if idx == 1:
        return float("-inf"), bins[0]
    elif 2 <= idx <= N:
        return bins[idx-2], bins[idx-1]
    elif idx == N + 1:
        return bins[-1], float("inf")
    else:
        raise ValueError(f"Index {idx} out of range for {N}-boundary bins")

def get_peoe_charges(mol):
    """
    Compute PEOE (Partial Equalization of Orbital Electronegativity) charges.
    This uses Gasteiger charges as an approximation since RDKit doesn't have
    true PEOE charges built-in.
    """
    # Make a copy to avoid modifying the original molecule
    mol_copy = Chem.Mol(mol)
    
    # Compute Gasteiger charges (closest approximation to PEOE in RDKit)
    ComputeGasteigerCharges(mol_copy)
    
    # Extract charges
    charges = []
    for atom in mol_copy.GetAtoms():
        charge = atom.GetDoubleProp('_GasteigerCharge')
        # Handle NaN values that can occur with Gasteiger calculation
        if np.isnan(charge):
            charge = 0.0
        charges.append(charge)
    
    return charges

def get_peoe_vsa_bins():
    """
    Return the charge bins used for PEOE_VSA descriptors.
    These are the standard bins used in RDKit for PEOE_VSA calculations.
    """
    # Standard PEOE_VSA charge bins (from RDKit source)
    return [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

def _prep_mol(smiles):
    """Parse SMILES and ensure 2-D coordinates exist. Returns mol or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if not mol.GetNumConformers():
        AllChem.Compute2DCoords(mol)
    return mol


def _compute_highlights(smiles, desc):
    """
    Computes which atoms contribute to a given VSA descriptor bin and their contributions.
    Parameters: 
    smiles (str)
    desc (str, e.g. "SMR_VSA8", "EState_VSA3", "PEOE_VSA5")
    
    Returns:
    dict with keys:
        atoms        list[int]   atom indices in the descriptor bin
        contribs     list[float] VSA (or EState) contribution per atom
        total        float       sum of contribs
        lower        float       bin lower bound
        upper        float       bin upper bound
        values       list[float] the per-atom property used for binning
        desc_type    str         descriptor type ("Value" for SMR/SlogP/EState, "Charge" for PEOE)
    or None if the descriptor is unknown or the molecule is invalid.
    """
    mol = _prep_mol(smiles)
    if mol is None:
        return None
    # precompute all per-atom values/contributions
    crippen_contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    vsa_contribs     = list(rdMolDescriptors._CalcLabuteASAContribs(mol)[0])
    estate_indices   = EState.EStateIndices(mol)
    peoe_charges     = get_peoe_charges(mol)
 
    if desc.startswith("SMR_VSA") or desc.startswith("SlogP_VSA"):
        try:
            lower, upper = get_vsa_bin_bounds(desc)
        except ValueError:
            return None
        prop_idx      = 1 if desc.startswith("SMR") else 0
        values        = [c[prop_idx] for c in crippen_contribs]
        contributions = vsa_contribs
        desc_type     = "Value"
 
    elif desc.startswith("EState_VSA"):
        idx           = int(desc.split("EState_VSA")[1])
        lower, upper  = get_bin_bounds(idx, EState_VSA.estateBins)
        values        = estate_indices
        contributions = vsa_contribs
        desc_type     = "Value"
 
    elif desc.startswith("VSA_EState"):
        idx           = int(desc.split("VSA_EState")[1])
        lower, upper  = get_bin_bounds(idx, EState_VSA.vsaBins)
        values        = vsa_contribs
        contributions = estate_indices
        desc_type     = "Value"
 
    elif desc.startswith("PEOE_VSA"):
        idx           = int(desc.split("PEOE_VSA")[1])
        lower, upper  = get_bin_bounds(idx, get_peoe_vsa_bins())
        values        = peoe_charges
        contributions = vsa_contribs
        desc_type     = "Charge"
 
    else:
        return None
    
    # --- find atoms in the selected bin ---
    atoms, contribs = [], []
    for i, (val, contrib) in enumerate(zip(values, contributions)):
        if lower <= val < upper:
            atoms.append(i)
            contribs.append(contrib)
    return dict(
        atoms=atoms,        # indices of atoms in the bin
        contribs=contribs,  
        total=sum(contribs),
        lower=lower,
        upper=upper,
        values=values,      
        desc_type=desc_type,
    )

def _draw_mol_svg(smiles, atoms, highlight_colors, size=300):
    """
    Render a molecule SVG with highlighted atoms.
 
    Parameters
    highlight_colors : dict[int, tuple[float,float,float]]
        Atom index   : RGB tuple
 
    Returns SVG
    """
    mol = _prep_mol(smiles)
    if mol is None:
        return f'<svg width="{size}" height="{size}"><text x="10" y="20" fill="red">Invalid SMILES</text></svg>'
 
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    drawer.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(atoms),
        highlightAtomColors=highlight_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

def _make_highlight_colors(atoms, contribs, total, row_max=None):
    """
    Map atom contributions to RGB highlight colors.
 
    The green channel intensity is scaled by atomic contribution and the normalizer, where
    normalizer is `row_max` (per-row mode) or `total` (per-cell mode).
     
    Returns dict[int, (R, G, B)].
    """
    normalizer = row_max if row_max is not None else total
    if normalizer == 0:
        return {i: (0.7, 0.7, 0.7) for i in atoms}
    normalizer = abs(normalizer)
 
    TARGET_R, TARGET_G, TARGET_B = 0.0, 0.7, 0.0
    colors = {}
    for i, c in zip(atoms, contribs):
        frac = max(0.0, min(1.0, c / normalizer))
        # Interpolate white → (0, 0.7, 0): each channel lerps from 1.0 to its target
        r = 1.0 + frac * (TARGET_R - 1.0)
        g = 1.0 + frac * (TARGET_G - 1.0)
        b = 1.0 + frac * (TARGET_B - 1.0)
        colors[i] = (r, g, b)
    return colors

def _build_grid_html(smiles_list, descriptors, labels, normalize, mol_size=280):
    """
    Assemble a HTML grid to view several molecules (columns) and descriptors (rows)
 
    Parameters
    ----------
    normalize : "row" (normalize color intensity by all molecules in the row) or "cell" (normalize color for each molecule independently)
    """
    # Pre-compute all highlights
    # results[desc][smi] = _compute_highlights(...) or None
    results = {
        desc: {smi: _compute_highlights(smi, desc) for smi in smiles_list}
        for desc in descriptors
    }
 
    # Build column headers 
    header_cells = "<th style='padding:8px 12px;text-align:center;font-weight:500;font-size:13px;border-bottom:1px solid #ddd;min-width:80px;'>descriptor</th>\n"
    for lbl in labels:
        header_cells += (
            f"<th style='padding:8px 12px;text-align:center;font-weight:500;"
            f"font-size:13px;border-bottom:1px solid #ddd;max-width:{mol_size+20}px;"
            f"word-break:break-all;'>{lbl}</th>\n"
        )
 
    # Build rows
    rows_html = ""
    for desc in descriptors:
        # Per-row normalization: max abs(total) across all molecules in this row
        if normalize == "row":
            row_totals = [
                abs(r["total"]) for r in results[desc].values()
                if r is not None and r["atoms"]
            ]
            row_max = max(row_totals) if row_totals else None
        else:
            row_max = None  # per-cell or flat: each cell handles itself
 
        # Unit label for the Σ line depends on what the descriptor actually sums
        if desc.startswith("VSA_EState"):
            sigma_unit = ""          # summing EState indices, dimensionless
            sigma_label = "Σ EState"
        else:
            sigma_unit = " Å²"       # summing VSA surface area contributions
            sigma_label = "Σ"
 
        cells_html = (
            f"<td style='padding:8px 12px;font-weight:500;font-size:13px;"
            f"vertical-align:middle;white-space:nowrap;"
            f"border-bottom:1px solid #eee;'>{desc}</td>\n"
        )
 
        for smi in smiles_list:
            r = results[desc][smi]
            if r is None or not r["atoms"]:
                svg = _draw_mol_svg(smi, [], {}, size=mol_size)
                total_str = "no atoms in bin"
                label_color = "#999"
            else:
                if normalize == "flat":
                    hcolors = {i: (0.0, 0.7, 0.0) for i in r["atoms"]}
                else:
                    hcolors = _make_highlight_colors(
                        r["atoms"], r["contribs"], r["total"],
                        row_max=row_max,
                    )
                svg = _draw_mol_svg(smi, r["atoms"], hcolors, size=mol_size)
                total_str = f"{sigma_label} = {r['total']:.3f}{sigma_unit}"
                label_color = "#333"
 
            # Embed SVG inline; strip the XML declaration if present
            svg_inline = re.sub(r"<\?xml[^?]*\?>", "", svg).strip()
 
            cells_html += (
                f"<td style='padding:6px 10px;text-align:center;"
                f"border-bottom:1px solid #eee;vertical-align:top;'>"
                f"{svg_inline}"
                f"<div style='font-size:11px;color:{label_color};"
                f"margin-top:2px;'>{total_str}</div>"
                f"</td>\n"
            )
 
        rows_html += f"<tr>\n{cells_html}</tr>\n"
 
    # Add lgend
    norm_label = {"row": "per-row", "cell": "per-cell", "flat": "flat (uniform)"}[normalize]
    legend_html = (
        f"<div style='font-size:11px;color:#888;margin-top:6px;'>"
        f"Highlight intensity: {norm_label} normalization  |  "
        f"white = no contribution &nbsp; "
        f"<span style='color:#006600;'>&#9632;</span> = max contribution"
        f"</div>"
    )
 
    # Assemble grid
    html = f"""
    <div style='overflow-x:auto;'>
    <table style='border-collapse:collapse;font-family:sans-serif;'>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{rows_html}</tbody>
    </table>
    {legend_html}
    </div>
    """
    return html

def _visualize_single(smiles, highlight_descriptors, normalize, save_path):
    mol = _prep_mol(smiles)
    if mol is None:
        print(f"Error: Could not parse SMILES '{smiles}'")
        return
 
    for desc in highlight_descriptors:
        r = _compute_highlights(smiles, desc)
        if r is None:
            print(f"Unknown descriptor '{desc}', skipping.")
            continue
        if not r["atoms"]:
            print(f"\nNo atoms contribute to {desc} "
                  f"(range {r['lower']:.4f} to {r['upper']:.4f}).")
            continue
 
        if normalize == "cell":
            hcolors = _make_highlight_colors(r["atoms"], r["contribs"], r["total"])
        else:
            # "flat" or "row" (per-row scaling is meaningless for a single molecule,
            # so it falls back to flat — all contributing atoms at full intensity)
            hcolors = {i: (0.0, 0.7, 0.0) for i in r["atoms"]}
        
        drawer = rdMolDraw2D.MolDraw2DSVG(500, 500)
        drawer.drawOptions().addAtomIndices = True
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol,
            highlightAtoms=r["atoms"],
            highlightAtomColors=hcolors,
        )
        drawer.FinishDrawing()
        svg_text = drawer.GetDrawingText()
        display(SVG(svg_text))
 
        if save_path:
            base, _, ext = save_path.rpartition(".")
            out = f"{base}_{desc}.{ext}" if base else f"{save_path}_{desc}"
            with open(out, "w") as f:
                f.write(svg_text)
            print(f"Saved SVG to {out}")
 
        print(f"\n### {desc} Contributions — Total: {r['total']:.4f}")
        print(f"Bin range: {r['lower']:.4f} to {r['upper']:.4f}")
        print(f"{'Idx':<4s}{'Sym':<4s}{r['desc_type']:>8s}{'Contrib':>12s}{'% of total':>12s}")
        print("-" * 44)
        for i, c in zip(r["atoms"], r["contribs"]):
            sym = mol.GetAtomWithIdx(i).GetSymbol()
            val = r["values"][i]
            pct = 100 * c / r["total"] if r["total"] else 0
            print(f"{i:<4d}{sym:<4s}{val:8.3f}{c:12.3f}{pct:12.1f}%")

def _visualize_grid(smiles_list, highlight_descriptors, labels, normalize, save_path):
    if normalize not in ("flat", "cell", "row"):
        raise ValueError(f"normalize must be 'flat', 'cell', or 'row', got {normalize!r}")
    
    # default labels are truncated SMILES
    if labels is None:
        labels = [s if len(s) <= 20 else s[:18] + "…" for s in smiles_list]
    elif len(labels) != len(smiles_list):
        raise ValueError(
            f"labels has {len(labels)} entries but smiles has {len(smiles_list)}"
        )
 
    html = _build_grid_html(smiles_list, highlight_descriptors, labels, normalize)
    display(IPythonHTML(html))
 
    if save_path:
        full_html = f"<!DOCTYPE html><html><body>{html}</body></html>"
        with open(save_path, "w") as f:
            f.write(full_html)
        print(f"Saved grid HTML to {save_path}")


def visualize_vsa_contributions(
    smiles,
    highlight_descriptors=None,
    grid=False,
    labels=None,
    normalize="row",
    save_path=None,
):
    """
    Visualize per-atom VSA descriptor contributions for one or more molecules.
 
    Parameters
    ----------
    smiles : str or list[str]
        A single SMILES string, or a list of SMILES strings.
        Passing a list automatically enables grid mode.
    highlight_descriptors : list[str], optional
        Descriptor names to visualize.
        Supports SMR_VSA*, SlogP_VSA*, EState_VSA*, VSA_EState*, PEOE_VSA*.
    grid : bool, optional
    labels : list[str], optional
        Column headers for grid mode. Defaults to truncated SMILES strings.
    normalize : {"row", "cell"}, optional
        Grid mode color scaling.
        "row"  (default) - all molecules in a row share the same color scale,
                           so intensity is comparable across analogs.
        "cell" - each cell is scaled independently for maximum local contrast.
        "flat" - each contributing atom gets the full intensity
    save_path : str, optional
        If provided, save the output to this path.
        Single-molecule mode: saves SVG (one file per descriptor, suffix added).
        Grid mode: saves an HTML file.
    """
    if highlight_descriptors is None:
        highlight_descriptors = ["SMR_VSA8", "SlogP_VSA8", "PEOE_VSA8"]
 
    is_list = isinstance(smiles, (list, tuple))
    if is_list or grid:
        smiles_list = list(smiles) if is_list else [smiles]
        _visualize_grid(smiles_list, highlight_descriptors, labels, normalize, save_path)
    else:
        _visualize_single(smiles, highlight_descriptors, normalize, save_path)

# def visualize_vsa_contributions_old(smiles, highlight_descriptors=None, save_path = None):
#     """
#     Analyze and visualize VSA descriptor contributions for a molecule,
#     including SMR_VSA, SlogP_VSA, EState_VSA, VSA_EState, and PEOE_VSA families.
#     """
#     if highlight_descriptors is None:
#         highlight_descriptors = ["SMR_VSA8", "SlogP_VSA8", "PEOE_VSA8"]

#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         print(f"Error: Could not parse SMILES '{smiles}'")
#         return
#     if not mol.GetNumConformers():
#         AllChem.Compute2DCoords(mol)

#     # precompute all per-atom values/contributions
#     crippen_contribs = rdMolDescriptors._CalcCrippenContribs(mol)
#     vsa_contribs    = list(rdMolDescriptors._CalcLabuteASAContribs(mol)[0])
#     estate_indices  = EState.EStateIndices(mol)
#     peoe_charges    = get_peoe_charges(mol)

#     for desc in highlight_descriptors:
#         # --- pick the correct pairing of "values" vs "contributions" and the bin boundaries ---
#         if desc.startswith("SMR_VSA") or desc.startswith("SlogP_VSA"):
#             # SMR_VSA* or SlogP_VSA* via get_vsa_bin_bounds()
#             try:
#                 lower, upper = get_vsa_bin_bounds(desc)
#             except ValueError as e:
#                 print(e)
#                 continue
#             prop_idx   = 1 if desc.startswith("SMR") else 0
#             values     = [c[prop_idx] for c in crippen_contribs]
#             contributions = vsa_contribs

#         elif desc.startswith("EState_VSA"):
#             # EState_VSA*: sum VSA over EState bins
#             idx = int(desc.split("EState_VSA")[1]) #descriptors start from 1

#             bins = EState_VSA.estateBins
#             lower, upper = get_bin_bounds(idx, bins)
            
#             values       = estate_indices
#             contributions = vsa_contribs

#         elif desc.startswith("VSA_EState"):
#             # VSA_EState*: sum EState over VSA bins
#             idx = int(desc.split("VSA_EState")[1]) #descriptors start from 1
#             bins = EState_VSA.vsaBins
#             lower, upper = get_bin_bounds(idx, bins)
            
#             values       = vsa_contribs
#             contributions = estate_indices

#         elif desc.startswith("PEOE_VSA"):
#             # PEOE_VSA*: sum VSA over PEOE charge bins
#             idx = int(desc.split("PEOE_VSA")[1]) # descriptors start from 1
#             bins = get_peoe_vsa_bins()
#             lower, upper = get_bin_bounds(idx, bins)
            
#             values       = peoe_charges
#             contributions = vsa_contribs

#         else:
#             print(f"Unknown descriptor '{desc}', skipping.")
#             continue

#         # --- find atoms in the selected bin ---
#         atoms, contribs = [], []
#         for i, (val, contrib) in enumerate(zip(values, contributions)):
#             if lower <= val < upper:
#                 atoms.append(i)
#                 contribs.append(contrib)
#         total = sum(contribs)
#         if not atoms:
#             print(f"\nNo atoms contribute to {desc} (range {lower:.4f} to {upper:.4f}).")
#             continue

#         # --- normalize & color (green intensity here; switch channels as you like) ---
#         norm = {i: c/total for i, c in zip(atoms, contribs)}
#         highlight_colors = {i: (0.0, 0.7, 0.0) for i, v in norm.items()}

#         # --- draw SVG with atom indices ---
#         drawer = rdMolDraw2D.MolDraw2DSVG(500, 500)
#         drawer.drawOptions().addAtomIndices = True
#         rdMolDraw2D.PrepareAndDrawMolecule(
#             drawer, mol,
#             highlightAtoms=list(atoms),
#             highlightAtomColors=highlight_colors
#         )
#         drawer.FinishDrawing()
#         svg_text = drawer.GetDrawingText()
#         display(SVG(svg_text))
#         if save_path:
#             with open(save_path, "w") as f:
#                 f.write(svg_text)
#             print(f"Saved SVG to {save_path}")

#         # --- print contribution table ---
#         descriptor_type = "Charge" if desc.startswith("PEOE_VSA") else "Value"
#         print(f"\n### {desc} Contributions — Total: {total:.4f}")
#         print(f"Bin range: {lower:.4f} to {upper:.4f}")
#         print(f"{'Idx':<4s}{'Sym':<4s}{descriptor_type:>8s}{'Contrib':>12s}{'% of total':>12s}")
#         print("-"* 44)
#         for i in atoms:
#             sym   = mol.GetAtomWithIdx(i).GetSymbol()
#             val   = values[i]
#             cst   = contributions[i]
#             pct   = 100*cst/total if total else 0
#             print(f"{i:<4d}{sym:<4s}{val:8.3f}{cst:12.3f}{pct:12.1f}%") 

def highlight_top_contributing_atoms(smiles, descriptors, number_atoms=5, mode="percentage"):
    """
    Identify and highlight the top N atoms contributing most to the selected descriptors.
    
    Args:
        smiles (str): Molecule SMILES.
        descriptors (list): List of descriptor names (e.g. ['SMR_VSA8', 'PEOE_VSA5']).
        number_atoms (int): Number of top atoms to highlight.
        mode (str): 'percentage' (percentage of descriptor contributions) or 'frequency' (count of appearances).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error: Could not parse SMILES '{smiles}'")
        return
    if not mol.GetNumConformers():
        AllChem.Compute2DCoords(mol)

    # precompute all per-atom values/contributions
    crippen_contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    vsa_contribs    = list(rdMolDescriptors._CalcLabuteASAContribs(mol)[0])
    estate_indices  = EState.EStateIndices(mol)
    peoe_charges    = get_peoe_charges(mol)

    # --- helper to get bin boundaries ---
    def safe_get_bin_bounds(desc):
        if desc.startswith("SMR_VSA") or desc.startswith("SlogP_VSA"):
            return get_vsa_bin_bounds(desc)
        elif desc.startswith("EState_VSA"):
            idx = int(desc.split("EState_VSA")[1])
            return get_bin_bounds(idx, EState_VSA.estateBins)
        elif desc.startswith("VSA_EState"):
            idx = int(desc.split("VSA_EState")[1])
            return get_bin_bounds(idx, EState_VSA.vsaBins)
        elif desc.startswith("PEOE_VSA"):
            idx = int(desc.split("PEOE_VSA")[1])
            return get_bin_bounds(idx, get_peoe_vsa_bins())
        else:
            raise ValueError(f"Unknown descriptor: {desc}")

    # accumulate per-atom importance scores
    atom_scores = np.zeros(mol.GetNumAtoms())
    atom_counts = np.zeros(mol.GetNumAtoms())

    for desc in descriptors:
        try:
            lower, upper = safe_get_bin_bounds(desc)
        except ValueError as e:
            print(e)
            continue

        if desc.startswith("SMR_VSA"):
            values = [c[1] for c in crippen_contribs]
            contributions = vsa_contribs
        elif desc.startswith("SlogP_VSA"):
            values = [c[0] for c in crippen_contribs]
            contributions = vsa_contribs
        elif desc.startswith("EState_VSA"):
            values = estate_indices
            contributions = vsa_contribs
        elif desc.startswith("VSA_EState"):
            values = vsa_contribs
            contributions = estate_indices
        elif desc.startswith("PEOE_VSA"):
            values = peoe_charges
            contributions = vsa_contribs
        else:
            continue

        for i, (val, contrib) in enumerate(zip(values, contributions)):
            if lower <= val < upper:
                atom_counts[i] += 1
                atom_scores[i] += contrib

    # Choose ranking mode
    if mode == "frequency":
        scores = atom_counts
    elif mode == "percentage":
        scores = atom_scores / (atom_scores.sum() + 1e-12)
    else:
        raise ValueError("mode must be 'frequency' or 'percentage'")

    #identify top N atoms
    top_indices = np.argsort(scores)[-number_atoms:][::-1]
    top_indices = [int(i) for i in np.argsort(scores)[-number_atoms:][::-1]]
    top_scores = scores[top_indices]

    print(f"Top {number_atoms} contributing atoms ({mode} mode):")
    print(f"{'Idx':<5}{'Atom':<5}{'Score':>10}")
    print("-"*22)
    for i, s in zip(top_indices, top_scores):
        atom = mol.GetAtomWithIdx(int(i))
        print(f"{i:<5}{atom.GetSymbol():<5}{s:10.4f}")

    #highlight top atoms
    colors = {i: (0.0, 0.7, 0.0) for i in top_indices}
    drawer = rdMolDraw2D.MolDraw2DSVG(500, 500)
    drawer.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol, highlightAtoms=list(top_indices), highlightAtomColors=colors
    )
    drawer.FinishDrawing()
    svg_text = drawer.GetDrawingText()
    display(SVG(svg_text))