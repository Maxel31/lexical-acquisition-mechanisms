"""run ごとの割り当て（assignments build --experiment <実験>）。

どの語の用例を、どの語に置き換えて差し込むか（未知語の獲得では件数も）を、実験ごとに決める。
"""

import argparse

from lexical_acquisition_mechanisms import sense_addition_assignments, unknown_word_assignments
from lexical_acquisition_mechanisms.settings import Settings

BUILDERS = {
    unknown_word_assignments.EXPERIMENT: unknown_word_assignments.build_unknown_word_assignments,
    sense_addition_assignments.EXPERIMENT: (
        sense_addition_assignments.build_sense_addition_assignments
    ),
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment", choices=list(BUILDERS), required=True, help="割り当てを決める実験"
    )


def build_assignments(settings: Settings, arguments: argparse.Namespace) -> None:
    BUILDERS[arguments.experiment](settings, arguments)
