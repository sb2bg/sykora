#!/usr/bin/env python3
"""
apply_params.py – Write tuned Texel params back into src/evaluation.zig.

Usage:
  python utils/tuning/apply_params.py tune_params.txt
  python utils/tuning/apply_params.py tune_params.txt --eval src/evaluation.zig

What it does:
  1. Reads the tuned params file (key-value format produced by texel_tune.py).
  2. For every key found, regex-replaces the corresponding default value inside
     the EvalParams struct in evaluation.zig.
  3. Also updates the backward-compat pub const lines (PAWN_VALUE etc.) so that
     search.zig and move_picker.zig stay in sync.
  4. Writes the result back in-place (original saved as .bak).

Params file format:
  pawn_value 105
  knight_table -50 -40 -30 ...   (all values space-separated on one line)
  ...
"""

import argparse
import os
import re
import shutil
import sys


EVAL_ZIG = os.path.join(
    os.path.dirname(__file__), '..', '..', 'src', 'evaluation.zig'
)

# Scalar pub const aliases that live outside EvalParams
PUBCONST_MAP = {
    'pawn_value':   'PAWN_VALUE',
    'knight_value': 'KNIGHT_VALUE',
    'bishop_value': 'BISHOP_VALUE',
    'rook_value':   'ROOK_VALUE',
    'queen_value':  'QUEEN_VALUE',
}


def load_params(path: str) -> dict:
    params = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0]
            vals = [int(x) for x in parts[1:]]
            params[key] = vals[0] if len(vals) == 1 else vals
    return params


def format_array(vals: list[int], items_per_row: int = 8) -> str:
    """Format an int list as a Zig array literal (multi-line)."""
    rows = []
    for i in range(0, len(vals), items_per_row):
        chunk = vals[i:i + items_per_row]
        rows.append('        ' + ', '.join(f'{v:4d}' for v in chunk) + ',')
    return '.{\n' + '\n'.join(rows) + '\n    }'


def replace_scalar_default(src: str, field_name: str, new_val: int) -> str:
    """
    Replace the default value of a scalar EvalParams field, e.g.:
        pawn_value: i32 = 100,
    becomes:
        pawn_value: i32 = 105,
    """
    pattern = rf'(\b{re.escape(field_name)}\s*:\s*i32\s*=\s*)-?\d+'
    replacement = rf'\g<1>{new_val}'
    new_src, count = re.subn(pattern, replacement, src)
    if count == 0:
        print(f"  [warn] scalar field '{field_name}' not found in EvalParams", file=sys.stderr)
    return new_src


def replace_array_default(src: str, field_name: str, new_vals: list[int]) -> str:
    """
    Replace the default array value of an EvalParams field, e.g.:
        pawn_table: [64]i32 = .{ ... },
    Matches from the field declaration through the closing '},'.
    """
    # Match: field_name: [N]i32 = .{...},
    # We use a non-greedy match that stops at the first '},\n'
    pattern = (
        rf'(\b{re.escape(field_name)}\s*:\s*\[\d+\]i32\s*=\s*)'
        r'\.{[^}]*?}'  # .{ ... }
    )
    # Use DOTALL to span newlines
    formatted = format_array(new_vals)
    replacement = rf'\g<1>{formatted}'
    new_src, count = re.subn(pattern, replacement, src, flags=re.DOTALL)
    if count == 0:
        print(f"  [warn] array field '{field_name}' not found in EvalParams", file=sys.stderr)
    return new_src


def replace_pubconst(src: str, const_name: str, new_val: int) -> str:
    """
    Replace the value of a pub const declaration, e.g.:
        pub const PAWN_VALUE: i32 = 100;
    becomes:
        pub const PAWN_VALUE: i32 = 105;
    """
    pattern = rf'(pub\s+const\s+{re.escape(const_name)}\s*:\s*i32\s*=\s*)-?\d+'
    replacement = rf'\g<1>{new_val}'
    new_src, count = re.subn(pattern, replacement, src)
    if count == 0:
        print(f"  [warn] pub const '{const_name}' not found", file=sys.stderr)
    return new_src


def apply_params(params: dict, eval_zig_path: str) -> None:
    with open(eval_zig_path) as f:
        src = f.read()

    original = src
    changed = 0

    for key, val in params.items():
        if isinstance(val, list):
            new_src = replace_array_default(src, key, val)
        else:
            new_src = replace_scalar_default(src, key, val)
            # Also update the pub const alias if one exists
            if key in PUBCONST_MAP:
                new_src = replace_pubconst(new_src, PUBCONST_MAP[key], val)

        if new_src != src:
            changed += 1
            src = new_src
        else:
            if isinstance(val, list):
                pass  # array warnings already printed
            # scalar warning already printed

    if src != original:
        bak = eval_zig_path + '.bak'
        shutil.copy2(eval_zig_path, bak)
        print(f"  Backup saved to {bak}")
        with open(eval_zig_path, 'w') as f:
            f.write(src)
        print(f"  Updated {eval_zig_path} ({changed} params changed)")
    else:
        print("  No changes needed.")


def dump_defaults(output_path: str, eval_zig_path: str) -> None:
    """
    Extract EvalParams defaults from evaluation.zig and write them as a
    params file.  Useful to bootstrap the tuner without a pre-existing file.
    """
    with open(eval_zig_path) as f:
        src = f.read()

    lines = []

    # Scalar fields: name: i32 = value,
    for m in re.finditer(r'\b(\w+)\s*:\s*i32\s*=\s*(-?\d+)\s*,', src):
        name, val = m.group(1), m.group(2)
        lines.append(f'{name} {val}')

    # Array fields: name: [N]i32 = .{...},
    for m in re.finditer(
        r'\b(\w+)\s*:\s*\[(\d+)\]i32\s*=\s*\.{([^}]*?)}',
        src, re.DOTALL
    ):
        name = m.group(1)
        body = m.group(3)
        # Strip inline comments (// ...) before extracting numbers
        body_no_comments = re.sub(r'//[^\n]*', '', body)
        nums = re.findall(r'-?\d+', body_no_comments)
        lines.append(f"{name} {' '.join(nums)}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Defaults written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Apply tuned params back into evaluation.zig'
    )
    parser.add_argument('params_file', nargs='?',
                        help='Tuned params file (from texel_tune.py)')
    parser.add_argument('--eval', default=None,
                        help='Path to evaluation.zig (default: auto-detect)')
    parser.add_argument('--dump-defaults', metavar='OUTPUT',
                        help='Dump current EvalParams defaults to a params file and exit')
    args = parser.parse_args()

    eval_zig = args.eval if args.eval else os.path.normpath(EVAL_ZIG)

    if not os.path.isfile(eval_zig):
        print(f"Error: {eval_zig} not found", file=sys.stderr)
        sys.exit(1)

    if args.dump_defaults:
        dump_defaults(args.dump_defaults, eval_zig)
        return

    if not args.params_file:
        parser.print_help()
        sys.exit(1)

    params = load_params(args.params_file)
    print(f"Loaded {len(params)} params from {args.params_file}")
    apply_params(params, eval_zig)
    print("\nDone. Rebuild the engine with:")
    print("  zig build -Doptimize=ReleaseFast")


if __name__ == '__main__':
    main()
