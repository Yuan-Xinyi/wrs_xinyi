import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


DAE_MATRIX_RE = re.compile(r"<matrix[^>]*>(.*?)</matrix>", flags=re.S)
FLIP_LINKS = {"link1", "link3", "link5", "link7"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate a second-pass Franka sphere URDF fix. Use direct DAE matrices for selected odd links and inverse matrices for the others.'
    )
    parser.add_argument(
        '--input-urdf',
        type=Path,
        default=Path('wrs/robot_sim/robots/franka_research_3/franka_research_3_ccsphere.urdf'),
    )
    parser.add_argument(
        '--output-urdf',
        type=Path,
        default=Path('wrs/robot_sim/robots/franka_research_3/franka_research_3_ccsphere_wrsfix_v2.urdf'),
    )
    return parser.parse_args()


def parse_matrix_str(matrix_str: str) -> np.ndarray:
    vals = [float(v) for v in matrix_str.split()]
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
    return non_identity[0]


def transform_point(mat: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    hom = np.ones(4, dtype=np.float64)
    hom[:3] = xyz
    return (mat @ hom)[:3]


def fmt_xyz(xyz: np.ndarray) -> str:
    return ' '.join(f'{v:.12g}' for v in xyz.tolist())


def main() -> None:
    args = parse_args()
    tree = ET.parse(args.input_urdf)
    root = tree.getroot()

    link_mats = {}
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
        dae_mat = extract_candidate_matrix(dae_path)
        if np.allclose(dae_mat, np.eye(4), atol=1e-9):
            continue
        if name in FLIP_LINKS:
            applied = dae_mat
            mode = 'direct'
        else:
            applied = np.linalg.inv(dae_mat)
            mode = 'inverse'
        link_mats[name] = (dae_path, dae_mat, applied, mode)

    total = 0
    for link in root.findall('link'):
        name = link.attrib['name']
        if name not in link_mats:
            continue
        dae_path, dae_mat, applied, mode = link_mats[name]
        changed = 0
        for collision in link.findall('collision'):
            geom = collision.find('geometry/sphere')
            origin = collision.find('origin')
            if geom is None or origin is None:
                continue
            xyz = np.array([float(v) for v in origin.attrib.get('xyz', '0 0 0').split()], dtype=np.float64)
            new_xyz = transform_point(applied, xyz)
            origin.attrib['xyz'] = fmt_xyz(new_xyz)
            changed += 1
        total += changed
        print(f'[fix-v2] {name}: {changed} spheres, mode={mode}, dae={dae_path.name}')
        print(f'          dae_matrix = {np.array2string(dae_mat, precision=6, suppress_small=True)}')

    args.output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output_urdf, encoding='utf-8', xml_declaration=True)
    print(f'[saved] {args.output_urdf}')
    print(f'[summary] total adjusted spheres = {total}')


if __name__ == '__main__':
    main()
