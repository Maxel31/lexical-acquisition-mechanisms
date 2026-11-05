"""run ごとの学習データ（training-data build --experiment <実験>）。"""

import argparse

from lexical_acquisition_mechanisms import unknown_word_training_data
from lexical_acquisition_mechanisms.settings import Settings

BUILDERS = {
    unknown_word_training_data.EXPERIMENT: (
        unknown_word_training_data.build_unknown_word_training_data
    ),
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment", choices=list(BUILDERS), required=True, help="学習データを作る実験"
    )


def build_training_data(settings: Settings, arguments: argparse.Namespace) -> None:
    BUILDERS[arguments.experiment](settings, arguments)
