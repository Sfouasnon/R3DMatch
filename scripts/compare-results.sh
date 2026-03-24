#!/bin/sh
set -eu

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <analysis-dir>" >&2
  exit 1
fi

find "$1" -name '*.analysis.json' -print | sort

