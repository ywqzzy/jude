"""`python -m jude` -> the interactive jude console."""
import sys

from jude.console import main

sys.exit(main(sys.argv[1:]))
