import cv2
import numpy as np
from pathlib import Path


def extract_frames(video_path, num_frames=5):
    """Extract evenly spaced frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def find_caption_end(frame):
    """Find where the caption region ends."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i in range(h // 3):
        white_ratio = np.sum(gray[i] > 230) / w
        if white_ratio < 0.1:
            for j in range(i, -1, -1):
                wr = np.sum(gray[j] > 230) / w
                if wr > 0.95:
                    return j + 1
            return i
    return 0


def unwrap_caption(caption_img):
    """Unwrap multi-line caption into a single line image.

    Finds individual text lines, crops them tightly, and concatenates
    them horizontally so the caption fits on one line.
    """
    gray = cv2.cvtColor(caption_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Find rows that contain text
    row_has_text = np.array([np.sum(gray[i] < 200) / w > 0.01 for i in range(h)])

    # Find contiguous text line regions
    lines = []
    in_text = False
    start = 0
    for i in range(h):
        if row_has_text[i] and not in_text:
            start = i
            in_text = True
        elif not row_has_text[i] and in_text:
            lines.append(caption_img[start:i, :])
            in_text = False
    if in_text:
        lines.append(caption_img[start:, :])

    if not lines:
        return caption_img

    # Crop each line tightly to actual text bounding box (both width and height)
    # Filter out tiny artifacts (< 5px tall)
    cropped_lines = []
    for line in lines:
        gray_line = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
        text_mask = gray_line < 200
        col_has_text = np.any(text_mask, axis=0)
        row_has_text_line = np.any(text_mask, axis=1)
        if np.any(col_has_text) and np.any(row_has_text_line):
            cols = np.where(col_has_text)[0]
            rows = np.where(row_has_text_line)[0]
            crop = line[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
            if crop.shape[0] >= 5:
                cropped_lines.append(crop)

    if not cropped_lines:
        return caption_img

    # Keep all lines at original size (don't scale) to preserve uniform text size.
    # Place them on a shared-height canvas, vertically centered.
    max_h = max(l.shape[0] for l in cropped_lines)
    padded = []
    for l in cropped_lines:
        if l.shape[0] < max_h:
            pad_top = (max_h - l.shape[0]) // 2
            pad_bot = max_h - l.shape[0] - pad_top
            l = cv2.copyMakeBorder(l, pad_top, pad_bot, 0, 0,
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
        padded.append(l)

    # Concatenate horizontally with a small space
    space_w = max_h // 3
    space = np.full((max_h, space_w, 3), 255, dtype=np.uint8)

    parts = []
    for i, l in enumerate(padded):
        if i > 0:
            parts.append(space)
        parts.append(l)

    return np.hstack(parts)


# Fixed caption height for all images
CAPTION_HEIGHT = 50


def create_illustration(video_path, output_path, num_frames=5, no_caption=False):
    """Create a static illustration from a video."""
    frames = extract_frames(video_path, num_frames)
    if not frames:
        print(f"  Skipping {video_path.name}: could not extract frames")
        return

    num_frames = len(frames)
    caption_end = find_caption_end(frames[0])
    content_h = frames[0].shape[0] - caption_end
    half_h = content_h // 2

    # Split all frames using consistent boundaries
    video_frames = []
    pred_frames = []
    for f in frames:
        vf = f[caption_end:caption_end + half_h, :]
        pf = f[caption_end + half_h:caption_end + half_h * 2, :]
        video_frames.append(vf)
        pred_frames.append(pf)

    frame_w = video_frames[0].shape[1]
    frame_h = video_frames[0].shape[0]

    # Build frame rows
    gap = 4
    row_w = frame_w * num_frames + gap * (num_frames - 1)
    row_video = np.full((frame_h, row_w, 3), 255, dtype=np.uint8)
    row_pred = np.full((frame_h, row_w, 3), 255, dtype=np.uint8)

    for i in range(num_frames):
        x_start = i * (frame_w + gap)
        row_video[:, x_start:x_start + frame_w] = video_frames[i]
        row_pred[:, x_start:x_start + frame_w] = pred_frames[i]

    total_w = row_w
    row_gap = np.full((4, total_w, 3), 255, dtype=np.uint8)

    if no_caption:
        result = np.vstack([row_video, row_gap, row_pred])
    else:
        # Extract and unwrap caption from first frame
        caption_img = frames[0][:caption_end, :]
        caption_single_line = unwrap_caption(caption_img)

        # Scale caption to fixed height, preserve aspect ratio, left-align
        cap_h_orig = caption_single_line.shape[0]
        cap_w_orig = caption_single_line.shape[1]
        scale = CAPTION_HEIGHT / cap_h_orig
        new_w = int(cap_w_orig * scale)
        if new_w > total_w:
            new_w = total_w
            scale = new_w / cap_w_orig

        new_h = int(cap_h_orig * scale)
        cap_resized = cv2.resize(caption_single_line, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        caption_row = np.full((CAPTION_HEIGHT, total_w, 3), 255, dtype=np.uint8)
        y_off = (CAPTION_HEIGHT - new_h) // 2
        caption_row[y_off:y_off + new_h, :new_w] = cap_resized

        result = np.vstack([caption_row, row_gap, row_video, row_gap, row_pred])

    cv2.imwrite(str(output_path), result)
    print(f"  Saved: {output_path.name}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-caption", action="store_true", help="Omit caption text row")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    input_dir = repo_root / "static" / "my_videos" / "itw"
    output_dir = repo_root / "static" / "my_images" / "itw"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(input_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} videos in {input_dir}")

    for vf in video_files:
        print(f"Processing: {vf.name}")
        out_name = vf.stem + ".png"
        out_path = output_dir / out_name
        create_illustration(vf, out_path, num_frames=5, no_caption=args.no_caption)

    print("Done!")


if __name__ == "__main__":
    main()
