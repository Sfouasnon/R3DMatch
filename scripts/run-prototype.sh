#!/bin/sh
set -eu

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <clip-or-folder> <out-dir>" >&2
  exit 1
fi

r3dmatch analyze "$1" --mode scene --out "$2"

