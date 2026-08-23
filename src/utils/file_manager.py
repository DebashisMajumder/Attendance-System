import os


def get_frames(frame_dir):

    frames = []

    for filename in sorted(os.listdir(frame_dir)):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            continue

        path = os.path.join(
            frame_dir,
            filename
        )

        frames.append(path)

    return frames