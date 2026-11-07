#!/usr/bin/env python3
"""Export a LibTorch Actor checkpoint to a dependency-free float32 blob.

The production box (gq.esrrhs.xyz) has no LibTorch and cannot reach GitHub, so
``tetris_ai_server`` is built with ``-DTETRIS_AI_WITH_TORCH=OFF`` there.  The
Actor is a plain 4-layer MLP, and ``actor.pt`` is a zip whose ``actor/data/N``
members are already raw little-endian float32 tensors.  This script unpacks
them into ``actor.bin``, which ``src/server/lite_actor.cpp`` loads directly.

Usage:
    python3 server/tools/export_actor.py                       # ckpt_bc_ppo
    python3 server/tools/export_actor.py --ckpt ckpt_bc       # another run

Blob layout (all little-endian, header = 64 bytes):
    char   magic[8]      "TETRACT1"
    int32  hidden        (256)
    int32  stateSize     (270)
    int32  actionFeatSize(18)
    int32  reserved      (0)
    int32  inDim         stateSize + actionFeatSize (288)
    int32  nTensors      (8)
    int32  counts[8]     element count of each tensor
    float32 data[]       concatenated, in the order below

Tensor order (matches ActorImpl's registered modules):
    0 l1.weight   [hidden, inDim]
    1 l1.bias     [hidden]
    2 l2.weight   [hidden, hidden]
    3 l2.bias     [hidden]
    4 l3.weight   [hidden, hidden]
    5 l3.bias     [hidden]
    6 head.weight [1, hidden]
    7 head.bias   [1]
"""

import argparse
import io
import pickletools
import struct
import sys
import zipfile
from pathlib import Path

MAGIC = b"TETRACT1"
STATE_SIZE = 270
ACTION_FEAT_SIZE = 18


def hidden_from_pkl(blob: bytes) -> int:
    """Recover the hidden width from the first weight tensor's shape."""
    # pickletools.genops yields (opcode, arg, position); BININT2 args hold dims.
    dims = [arg for op, arg, _ in pickletools.genops(blob)
            if op.name == "BININT2"]
    # First tensor is l1.weight [hidden, stateSize + actionFeatSize].
    for i in range(len(dims) - 1):
        if dims[i + 1] == STATE_SIZE + ACTION_FEAT_SIZE and dims[i] > 1:
            return dims[i]
    raise ValueError("cannot determine hidden width from data.pkl")


def storage_order(blob: bytes):
    """Cross-check that the archive's storage ids appear as 0..7 in order.

    LibTorch writes each tensor as a persistent id whose third element is the
    ``actor/data/<id>`` member name.  Bail out loudly if a future PyTorch
    version changes the serialization order, instead of silently loading
    scrambled weights.
    """
    ids = [arg for op, arg, _ in pickletools.genops(blob)
           if op.name == "BINUNICODE" and isinstance(arg, str) and arg.isdigit()]
    return ids


def export(ckpt: Path, force: bool) -> int:
    actor_pt = ckpt / "actor.pt"
    if not actor_pt.is_file():
        print(f"error: {actor_pt} not found", file=sys.stderr)
        return 1

    with zipfile.ZipFile(actor_pt) as z:
        names = set(z.namelist())
        pkl = z.read("actor/data.pkl")
        hidden = hidden_from_pkl(pkl)
        order = storage_order(pkl)
        if order != [str(i) for i in range(8)]:
            print(f"error: unexpected storage id order {order}; expected 0..7",
                  file=sys.stderr)
            return 1

        in_dim = STATE_SIZE + ACTION_FEAT_SIZE
        counts = [
            hidden * in_dim,   # l1.weight
            hidden,            # l1.bias
            hidden * hidden,   # l2.weight
            hidden,            # l2.bias
            hidden * hidden,   # l3.weight
            hidden,            # l3.bias
            hidden,            # head.weight [1, hidden]
            1,                 # head.bias
        ]
        payload = bytearray()
        for i, count in enumerate(counts):
            member = f"actor/data/{i}"
            if member not in names:
                print(f"error: {member} missing from {actor_pt}", file=sys.stderr)
                return 1
            raw = z.read(member)
            if len(raw) != count * 4:
                print(f"error: {member} is {len(raw)} bytes, expected "
                      f"{count * 4} ({count} float32)", file=sys.stderr)
                return 1
            payload.extend(raw)

    header = MAGIC + struct.pack(
        "<6i8i", hidden, STATE_SIZE, ACTION_FEAT_SIZE, 0, in_dim, 8, *counts)
    assert len(header) == 64, len(header)

    out = ckpt / "actor.bin"
    if out.exists() and not force:
        print(f"error: {out} already exists; pass --force to overwrite",
              file=sys.stderr)
        return 1
    out.write_bytes(header + payload)
    print(f"wrote {out}  ({out.stat().st_size} bytes)")
    print(f"  hidden={hidden}  state={STATE_SIZE}  action={ACTION_FEAT_SIZE}  "
          f"tensors={counts}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=Path("ckpt_bc_ppo"),
                    help="checkpoint directory containing actor.pt")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing actor.bin")
    args = ap.parse_args()
    return export(args.ckpt, args.force)


if __name__ == "__main__":
    sys.exit(main())
