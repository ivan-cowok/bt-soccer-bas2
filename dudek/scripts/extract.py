from concurrent.futures import ThreadPoolExecutor, as_completed

from dudek.data.competition import Action
from dudek.utils.video import load_bas_videos, load_action_spotting_videos, load_competition_videos

import click

cli = click.Group()


@cli.command()
@click.option("--dataset_path", type=str, required=True)
@click.option("--resolution", type=int, default=224)
@click.option("--stride", type=int, default=2)
@click.option("--frame_target_width", type=int, default=224)
@click.option("--frame_target_height", type=int, default=224)
@click.option("--grayscale", type=bool, default=False)
@click.option("--save_all", type=bool, default=False)
@click.option("--num_workers", type=int, default=4)
def extract_bas_frames(
    dataset_path: str,
    resolution: int = 224,
    stride: int = 2,
    frame_target_width: int = 224,
    frame_target_height: int = 224,
    grayscale: bool = False,
    save_all: bool = False,
    num_workers: int = 4,
):
    assert resolution in [224, 720]

    videos = load_bas_videos(dataset_path, resolution=resolution)

    def _process(v):
        if save_all:
            v.save_all_frames(
                target_width=frame_target_width,
                target_height=frame_target_height,
                stride=stride,
                grayscale=grayscale,
            )
        else:
            v.save_frames(
                target_width=frame_target_width,
                target_height=frame_target_height,
                stride=stride,
                grayscale=grayscale,
            )

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_process, v): v for v in videos}
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                print(f"Error processing {futures[future].absolute_path}: {exc}")


@cli.command()
@click.option("--dataset_path", type=str, required=True)
@click.option("--resolution", type=int, default=224)
@click.option("--stride", type=int, default=1)
@click.option("--frame_target_width", type=int, default=224)
@click.option("--frame_target_height", type=int, default=224)
@click.option("--grayscale", type=bool, default=False)
@click.option("--num_workers", type=int, default=4)
def extract_competition_frames(
    dataset_path: str,
    resolution: int = 224,
    stride: int = 1,
    frame_target_width: int = 224,
    frame_target_height: int = 224,
    grayscale: bool = False,
    num_workers: int = 4,
):
    assert resolution in [224, 720]

    videos = load_competition_videos(dataset_path, resolution=resolution, labels_enum=Action)

    def _process(v):
        v.save_all_frames(
            target_width=frame_target_width,
            target_height=frame_target_height,
            stride=stride,
            grayscale=grayscale,
        )

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_process, v): v for v in videos}
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                print(f"Error processing {futures[future].absolute_path}: {exc}")


@cli.command()
@click.option("--dataset_path", type=str, required=True)
@click.option("--resolution", type=int, default=224)
@click.option("--stride", type=int, default=2)
@click.option("--frame_target_width", type=int, default=224)
@click.option("--frame_target_height", type=int, default=224)
@click.option("--grayscale", type=bool, default=False)
@click.option("--radius_sec", type=int, default=8)
@click.option("--num_workers", type=int, default=4)
def extract_action_spotting_frames(
    dataset_path: str,
    resolution: int = 224,
    stride: int = 2,
    frame_target_width: int = 224,
    frame_target_height: int = 224,
    grayscale: bool = False,
    radius_sec: int = 8,
    num_workers: int = 4,
):
    assert resolution in [224, 720]

    videos = load_action_spotting_videos(dataset_path, resolution=resolution)

    def _process(v):
        v.save_frames(
            target_width=frame_target_width,
            target_height=frame_target_height,
            stride=stride,
            grayscale=grayscale,
            radius_around_annotations_in_sec=radius_sec,
        )

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_process, v): v for v in videos}
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                print(f"Error processing {futures[future].absolute_path}: {exc}")


if __name__ == "__main__":
    cli()
