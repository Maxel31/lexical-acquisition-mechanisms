"""`python -m lexical_acquisition_mechanisms` で実行する入口。

学習は、torchrun からこの形で起動する。
"""

from lexical_acquisition_mechanisms.command_line import main

raise SystemExit(main())
