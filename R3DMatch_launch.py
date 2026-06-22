"""Frozen-app entry point for py2app.

py2app needs a real top-level script to point at. This just hands off to the
package's main(). The src/ path insert is a no-op inside the bundle (that dir
won't exist there) and only helps when running this file directly from source.
"""
import os
import sys

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_src):
    sys.path.insert(0, _src)

from r3dmatch3.app import main

if __name__ == "__main__":
    main()
