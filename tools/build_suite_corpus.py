#!/usr/bin/env python3
"""Create a frozen, disjoint synthetic dev/held-out mechanics/retrieval corpus.

Speech is generated separately by create_suite_speech.ps1 (no recording).
The resulting manifest distinguishes synthetic acceptance data from benchmark
claims about real-world model quality. No model inference takes place here.
"""

import argparse
import hashlib
import json
import shutil
import wave
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speech", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image

    args.output.mkdir(parents=True, exist_ok=False)
    media = args.output / "media"
    media.mkdir()
    items = []
    queries = []

    def add(identity, split, content, caption, artifact=None):
        item = {"id": identity, "split": split, "content": content, "caption": caption}
        if artifact:
            item["artifact"] = {
                "path": artifact.relative_to(args.output).as_posix(),
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        items.append(item)
        return item

    colors = [
        ("red", [255, 0, 0]),
        ("blue", [0, 0, 255]),
        ("green", [0, 200, 0]),
        ("yellow", [255, 240, 0]),
        ("purple", [160, 0, 200]),
        ("orange", [255, 128, 0]),
        ("cyan", [0, 200, 200]),
        ("black", [0, 0, 0]),
    ]
    images = []
    videos = []
    audios = []
    for i in range(16):
        split = "dev" if i < 8 else "heldout"
        name, rgb = colors[i // 2]
        width = 144 if split == "dev" else 192
        height = 96 if split == "dev" else 144
        y, x = np.mgrid[:height, :width]
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        radius = min(height, width) // (4 if i % 2 else 3)
        shape = "circle" if i % 2 else "square"
        mask = (
            (x - width // 2) ** 2 + (y - height // 2) ** 2 <= radius**2
            if i % 2
            else ((abs(x - width // 2) <= radius) & (abs(y - height // 2) <= radius))
        )
        canvas[mask] = rgb
        file = media / f"image-{i}.png"
        Image.fromarray(canvas).save(file)
        images.append(
            add(
                f"image-{i}",
                split,
                [{"type": "image", "path": file.relative_to(args.output).as_posix()}],
                f"A {name} {shape} on a white background.",
                file,
            )
        )
    for i in range(8):
        split = "dev" if i < 4 else "heldout"
        name, rgb = colors[i]
        frame_count = 4 + i * 2
        fps = 1 if i < 4 else 2
        frames = np.full((frame_count, 96, 144, 3), 255, dtype=np.uint8)
        direction = "right" if i < 4 else "left"
        for j in range(frame_count):
            position = 8 + int(100 * j / max(1, frame_count - 1))
            if direction == "left":
                position = 120 - position
            frames[j, 36:60, position : position + 16] = rgb
        file = media / f"video-{i}.npz"
        np.savez_compressed(file, frames=frames, fps=fps)
        videos.append(
            add(
                f"video-{i}",
                split,
                [
                    {
                        "type": "video",
                        "path": file.relative_to(args.output).as_posix(),
                        "fps": fps,
                        "audio": "exclude",
                    }
                ],
                f"A {name} rectangle moving to the {direction}.",
                file,
            )
        )
    speech = json.loads((args.speech / "speech.json").read_text(encoding="utf-8-sig"))
    for i in range(16):
        split = "dev" if (i < 4 or 8 <= i < 12) else "heldout"
        file = media / f"audio-{i}.wav"
        if i < 8:
            shutil.copyfile(args.speech / f"speech-{i}.wav", file)
            caption = speech["text"][i]
        else:
            rate = (16000, 24000, 8000, 32000)[i % 4]
            seconds = 1 + (i % 3)
            t = np.arange(rate * seconds) / rate
            if i in (10, 14):
                signal = np.zeros(len(t))
                caption = "Silence."
            elif i in (11, 15):
                signal = (
                    0.25 * np.sin(2 * np.pi * 660 * t) * ((t * 4).astype(int) % 2 == 0)
                )
                caption = "A series of short electronic beeps."
            else:
                frequency = (220, 880)[i % 2]
                signal = 0.25 * np.sin(2 * np.pi * frequency * t)
                caption = (
                    "A continuous low pitched tone."
                    if frequency == 220
                    else "A continuous high pitched tone."
                )
            with wave.open(str(file), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(rate)
                out.writeframes((signal * 32767).astype("<i2").tobytes())
            caption = caption.rstrip(".") + f" Lasting {seconds} seconds."
        audios.append(
            add(
                f"audio-{i}",
                split,
                [{"type": "audio", "path": file.relative_to(args.output).as_posix()}],
                caption,
                file,
            )
        )
    topics = [
        "colors",
        "weather",
        "cooking",
        "travel",
        "geometry",
        "software",
        "music",
        "sports",
        "gardening",
        "history",
        "space",
        "medicine",
        "books",
        "animals",
        "architecture",
        "transport",
        "mathematics",
        "photography",
        "energy",
        "ocean",
        "language",
        "craft",
        "schools",
        "robots",
    ]
    for i in range(48):
        split = "dev" if i < 24 else "heldout"
        if i % 4 == 0:
            text = f"task: search result | query: informazioni su {topics[i // 2]}"
        elif i % 4 == 1:
            text = f"title: {topics[i // 2]} | text: Una guida introduttiva a {topics[i // 2]}."
        elif i % 4 == 2:
            text = f"task: code retrieval | query: implement {topics[i // 2]} lookup"
        else:
            text = f"def lookup_{topics[i // 2]}(items, key):\n    return next((x for x in items if x == key), None)"
        add(f"text-{i}", split, [{"type": "text", "text": text}], text)
    for i in range(16):
        split = "dev" if i < 8 else "heldout"
        image = images[i]
        video = videos[(i // 2) % 8]
        audio = next(
            a
            for a in audios
            if a["split"] == split and int(a["id"].split("-")[1]) % 4 == i % 4
        )
        content = [{"type": "text", "text": "Observe these colors."}] + image["content"]
        if i % 2 == 0:
            content += [{"type": "text", "text": "Movement:"}] + video["content"]
        content += [{"type": "text", "text": "Sound:"}] + audio["content"]
        add(
            f"interleaved-{i}",
            split,
            content,
            image["caption"] + " " + audio["caption"],
        )
    # Both retrieval directions have 32 queries per modality. Identical captions
    # within a split are all relevant (especially silence), not false negatives.
    for modality, members in (("image", images), ("video", videos), ("audio", audios)):
        for i in range(32):
            member = members[i % len(members)]
            relevant = [
                x["id"]
                for x in members
                if x["split"] == member["split"] and x["caption"] == member["caption"]
            ]
            template = (
                "{}",
                "Find the following scene: {}",
                "Retrieve this description: {}",
                "Look for this content: {}",
            )[i // len(members)]
            queries.append(
                {
                    "id": f"text-to-{modality}-{i}",
                    "split": member["split"],
                    "link": f"text_to_{modality}",
                    "text": template.format(member["caption"]),
                    "relevant_ids": relevant,
                }
            )
            queries.append(
                {
                    "id": f"{modality}-to-text-{i}",
                    "split": member["split"],
                    "link": f"{modality}_to_text",
                    "input_id": member["id"],
                    "relevant_text": member["caption"],
                }
            )
    prompts = [
        "A red square beside a blue circle on a white page.",
        "Una stella verde sopra un cerchio giallo.",
        "Three oranges arranged in a triangle on a wooden table.",
        "A glass bottle on a wet stone surface.",
        "A small robot reading a book in a library.",
        "Una casa bianca vicino a un lago al tramonto.",
        "A purple kite flying over a field.",
        "A black cat sitting on a blue chair.",
        'A white sign with the exact text "DS4 OPEN".',
        'Un cartello con la scritta esatta "CIAO MONDO".',
        'A poster titled "BLUE SKY" with three stars.',
        'Una copertina con il titolo "LUNA".',
        "A red circle above a blue square, a green triangle to their right.",
        "Four equal panels colored red, blue, green and yellow in that order.",
        "Two rows of three black circles on white.",
        "A yellow triangle inside a black square.",
        "Change the shape to a triangle and preserve the background.",
        "Make the central shape green and preserve its size.",
        "Add a thin black border around the central shape.",
        "Replace the white background with light grey.",
        "Combine both reference shapes side by side on one white page.",
        "Place the first reference shape above the second reference shape.",
        "A single red circle with a transparent background and clean edges.",
        "A glass cube with a transparent background and soft translucent edges.",
    ]
    recipes = []
    for i, prompt in enumerate(prompts):
        kind = (
            "generation"
            if i < 8
            else "typography"
            if i < 12
            else "layout"
            if i < 16
            else "edit_single"
            if i < 20
            else "edit_multiple"
            if i < 22
            else "alpha"
        )
        recipes.append(
            {
                "id": f"qwen-heldout-{i}",
                "split": "heldout",
                "kind": kind,
                "prompt": prompt,
                "width": (1024, 1536, 1024)[i % 3],
                "height": (1024, 1024, 1536)[i % 3],
                "reference_ids": ["image-8"]
                if kind == "edit_single"
                else ["image-8", "image-11"]
                if kind == "edit_multiple"
                else [],
                "seeds": [20261020, 20261021],
                "steps": 8,
                "cfg": 1.0,
            }
        )
    corpus = {
        "version": 1,
        "origin": "DS4 synthetic fixtures; speech synthesized, no recording",
        "speech_provenance": speech,
        "items": items,
        "retrieval": queries,
        "qwen_recipes": recipes,
        "quality_scope": "mechanics and regression on fixed synthetic data; not a real-world benchmark",
    }
    (args.output / "corpus.json").write_text(json.dumps(corpus, indent=2) + "\n")
    print(
        {
            "items": len(items),
            "retrieval_queries": len(queries),
            "qwen_heldout_recipes": len(recipes),
        }
    )


if __name__ == "__main__":
    main()
