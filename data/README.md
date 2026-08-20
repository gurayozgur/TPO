# data/

`protocols/` ships with the repository. `TPO/` does not: request the dataset as
described in the [main README](../README.md#getting-the-dataset) and unpack the
archive here, so that the frames land in `data/TPO/`.

```bash
unzip TPO.zip -d .
python ../src/check_data.py
```

## Protocols

Each protocol CSV lists the complete set of frames matching its filter. TPO is
used in its entirety as a training source, so there is no train/dev/test split.

| File | Frames | Presentations | Contents |
|---|---:|---:|---|
| `tpo_all.csv` | 57,408 | 12,480 | all vegetables, both attack types |
| `tpo_tomatoes.csv` | 19,136 | 4,160 | tomatoes only |
| `tpo_potatoes.csv` | 19,136 | 4,160 | potatoes only |
| `tpo_onions.csv` | 19,136 | 4,160 | onions only |
| `tpo_print.csv` | 32,448 | 7,488 | bona fide + print attacks |
| `tpo_replay.csv` | 32,448 | 7,488 | bona fide + replay attacks |

Columns are `image_path,label,vegetable,attack_type,cap_device,cap_scale,identity,media`.
Only the first two are read by the code; the rest describe each frame.
`image_path` is relative to the repository root and `label` is `bonafide` or `attack`.

## Presentation-level index

`tpo_index.csv` has one row per presentation (12,480 rows) with the full
acquisition metadata: `vegetable`, `label`, `attack_type` (`none`/`print`/`video`),
`src_device`, `cap_device`, `src_scale`, `cap_scale`, `identity`, `side`, `media`,
and `video_id`.

`video_id` is the frame path with the trailing `_<k>` stripped, which is exactly
the key the evaluation code groups frames by. It therefore joins one-to-one with
the video-level scores produced by `src/infer.py`, which makes it easy to break
results down by device, scale, or attack instrument.
