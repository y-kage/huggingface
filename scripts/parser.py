import argparse


def parser():
    parser = argparse.ArgumentParser(
        description="Process some inputs for image and prompt handling."
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default="../DATA/car.png",
        help="Path to the image file.",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default="a cat. a remote control.",
        help="Text prompt for the image.",
    )
    parser.add_argument(
        "--points_prompt",
        type=str,
        default=[[[850, 1100], [2250, 1000]]],
        help="Points prompt for the image.",
    )
    parser.add_argument(
        "--labels_prompt",
        type=str,
        default=[[[1], [0]]],
        help="Points prompt for the image.",
    )
    parser.add_argument(
        "--boxes_prompt",
        type=str,
        default=[[[650, 900, 1000, 1250], [2050, 800, 2400, 1150]]],
        help="Boxes prompt for the image.",
    )

    return parser
