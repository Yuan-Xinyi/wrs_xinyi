import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


DAE_MATRIX_RE = re.compile(r"<matrix[^>]*>(.*?)</matrix>", flags=re.S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert Franka sphere origins from DAE scene coordinates to WRS link-local coordinates using per-link DAE matrices.'
    )
    parser.add_argument(
        '--input-urdf',
        type=Path,
        default=Path('wrs/robot_sim/robots/franka_research_3/franka_research_3_ccsphere.urdf'),
    )
    parser.add_argument(
        '--output-urdf',
        type=Path,
        default=Path('wrs/robot_sim/robots/franka_research_3/franka_research_3_ccsphere_wrsfix.urdf'),
    )
    return parser.parse_args()


def parse_matrix_str(matrix_str: str) -> np.ndarray:
    vals = [float(v) for v in matrix_str.split()]
    if len(vals) != 16:
        raise ValueError(f'Expected 16 values in matrix, got {len(vals)}')
    return np.array(vals, dtype=np.float64).reshape(4, 4)


def canonical_key(mat: np.ndarray) -> tuple:
    return tuple(np.round(mat.reshape(-1), 9))


def extract_candidate_matrix(dae_path: Path) -> np.ndarray:
    text = dae_path.read_text(errors='ignore')
    mats = [parse_matrix_str(' '.join(raw.split())) for raw in DAE_MATRIX_RE.findall(text)]
    if not mats:
        return np.eye(4)
    uniq = []
    seen = set()
    for mat in mats:
        key = canonical_key(mat)
        if key not in seen:
            seen.add(key)
            uniq.append(mat)
    non_identity = [mat for mat in uniq if not np.allclose(mat, np.eye(4), atol=1e-9)]
    if not non_identity:
        return np.eye(4)
    if len(non_identity) == 1:
        return non_identity[0]
    # Prefer the first non-identity matrix, but warn to stdout. This still keeps the correction link-level.
    print(f'[warn] {dae_path.name} has multiple non-identity matrices; using the first one.')
    for idx, mat in enumerate(non_identity[:5]):
        print(f'  candidate[{idx}] = {np.array2string(mat, precision=6, suppress_small=True)}')
    return non_identity[0]


def transform_point(inv_mat: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    hom = np.ones(4, dtype=np.float64)
    hom[:3] = xyz
    out = inv_mat @ hom
    return out[:3]


def fmt_xyz(xyz: np.ndarray) -> str:
    return ' '.join(f'{v:.12g}' for v in xyz.tolist())


def main() -> None:
    args = parse_args()
    tree = ET.parse(args.input_urdf)
    root = tree.getroot()

    link_to_fix = {}
    for link in root.findall('link'):
        name = link.attrib['name']
        visual = link.find('visual')
        if visual is None:
            continue
        mesh = visual.find('./geometry/mesh')
        if mesh is None:
            continue
        filename = mesh.attrib.get('filename')
        if not filename:
            continue
        dae_path = Path(filename)
        if dae_path.suffix.lower() != '.dae' or not dae_path.exists():
            continue
        mat = extract_candidate_matrix(dae_path)
        if np.allclose(mat, np.eye(4), atol=1e-9):
            continue
        link_to_fix[name] = (dae_path, mat, np.linalg.inv(mat))

    total = 0
    for link in root.findall('link'):
        name = link.attrib['name']
        if name not in link_to_fix:
            continue
        dae_path, mat, inv_mat = link_to_fix[name]
        collisions = link.findall('collision')
        changed = 0
        for collision in collisions:
            geom = collision.find('geometry/sphere')
            origin = collision.find('origin')
            if geom is None or origin is None:
                continue
            xyz = np.array([float(v) for v in origin.attrib.get('xyz', '0 0 0').split()], dtype=np.float64)
            new_xyz = transform_point(inv_mat, xyz)
            origin.attrib['xyz'] = fmt_xyz(new_xyz)
            changed += 1
        total += changed
        print(f'[fix] {name}: {changed} spheres adjusted using {dae_path.name}')
        print(f'      dae_matrix = {np.array2string(mat, precision=6, suppress_small=True)}')

    args.output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output_urdf, encoding='utf-8', xml_declaration=True)
    print(f'[saved] {args.output_urdf}')
    print(f'[summary] total adjusted spheres = {total}')


if __name__ == '__main__':
    main()
