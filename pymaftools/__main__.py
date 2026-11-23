"""``python -m pymaftools`` runs the command-line interface."""

import sys

from .cli import main

sys.exit(main())
