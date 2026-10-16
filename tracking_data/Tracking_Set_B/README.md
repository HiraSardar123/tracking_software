# Droplet tracking annotation

You will annotate droplets in short high-speed video clips of gas metal arc
welding (aluminum wire, 13,000 frames per second). Each clip shows the wire tip
at the top, the arc, and the weld pool at the bottom. Droplets detach from the
wire tip and fall to the pool.

The tool is the VGG Visual Tracker (VVT), included in this folder. It runs in
your browser from a local file: no account, no installation, no internet.

## Goal

Maintain the identity of each physical droplet throughout its visible
free-flight trajectory. One droplet = one track, from detachment to the end of
its flight.

## What is in this folder

| Item | Use |
|---|---|
| `assignment.csv` | The clips assigned to you. Annotate all of them. |
| `clips/clip_XXX.mp4` | The clips. |
| `vvt/index.html` | The annotation tool. Open it in Chrome or Firefox. |
| `vvt/projects/clip_XXX.json` | An empty project for each clip. You start from it. |
| `annotations/` | Empty. Put your finished files here. |
| `README_examples.png` | The example frame shown below. |

## Open a clip

1. Double-click `vvt/index.html`. It opens in Chrome or Firefox.
2. Click the folder icon in the top bar (**Open a VIA Project**) and choose
   `vvt/projects/clip_XXX.json`.
3. The page says **File Not Found!** That is normal. In the row
   **File Source**, click **Choose File** and select `clips/clip_XXX.mp4`.
   The clip appears with a timeline below it.

## Which droplets to annotate

- Annotate every body of liquid metal that detaches from the wire tip during
  the clip and falls toward the pool, including small secondary droplets that
  follow a larger one.
- Do not annotate a droplet that is already detached in the first frame of the
  clip.
- Do not annotate fine spatter or reflections on the pool.

![Examples](README_examples.png)

In the example frame: **1** is a droplet and **2** is a secondary droplet that
follows it, so both are annotated. **3** is fine spatter and is ignored. As a
guide, a body clearly narrower than half the wire is fine spatter.

## Annotate one droplet

1. Press `→` until the first frame where the droplet is reliably detached
   (free) from the wire tip.
2. Hold `Alt` and drag a tight box around the droplet. A new track appears:
   `Object #1`, `Object #2`, ... The name is written above the box and a row
   with the same name appears in the timeline.
3. Move forward about 10 frames (`→` ten times, or `Shift + N` twice).
4. Drag a new tight box around the same droplet, **without** `Alt`. The box is
   added to the track whose row is highlighted in the timeline.
5. Repeat steps 3 and 4 until the end of the flight. Put the last box on the
   last frame of the track (see **End**).
6. For the next droplet, start again at step 1 with `Alt`.

Every box of one droplet must carry the same name. Check the name above the
box each time. To add a box to an earlier droplet, first click its row in the
timeline.

The frames between your boxes are filled in later by a straight line, so a box
every 10 frames is enough while the droplet falls. Add boxes closer together
where it changes direction or shape quickly.

## Important rule: the automatic tracker is only a tool

After you draw or adjust a box, pressing `t` makes VVT move the box forward by
itself. It stops at the next box of the same track, or at the end of the clip.
Press any key to stop it.

- The VVT tracker is only a tool to move the bounding box faster. You are
  responsible for verifying that the track remains on the same physical
  droplet.
- The tracker must never decide the identity of a droplet. It does not know
  which droplet is which.
- If one physical object splits into two visible pieces, keep both boxes in the
   same track: draw both boxes on the same frame without `Alt`, with the same
   track row selected, then press `t`. YOLO will track the two pieces as two
   boxes under the same object name. Verify both pieces on every generated
   frame; tracking stops if either piece becomes ambiguous.
- In these clips the tracker often leaves the droplet within about 20 frames,
  and its boxes are looser than yours. Check every box it adds. Redraw or
  delete the ones that are off.
- Before pressing `t`, draw the last box of the droplet. The tracker then stops
  there instead of running to the end of the clip.

You can do the whole job without `t`.

## Fix a mistake

| Problem | Fix |
|---|---|
| A box is wrong | Click the box, press `Backspace`. Draw it again. |
| A box is slightly off | Click the box and drag it, or drag its corner. |
| A box got the wrong name | Delete it, click the correct row in the timeline, draw it again. |
| The tracker ran too far | Click the coloured segment in the track row, press `Backspace`. |
| The arrow keys move a box instead of the video | A box is selected. Click an empty part of the image. |
| The keys stop responding | Click once on the empty area below the timeline. |

## Start

Begin the track at the first frame where the droplet is reliably detached
(free) from the wire tip.

## End

Stop the track at the first of these:

- `pool`: the droplet touches the pool surface;
- `FOV_exit`: the droplet leaves the image;
- `physical_merge`: the droplet coalesces with another droplet;
- `lost_visibility`: you can no longer tell where the droplet is;
- `clip_end`: the droplet is still falling in the last frame of the clip.
  Annotate it up to the last frame.

## Multiple droplets

Every different physical droplet gets a different track, even when two fall
close together or one follows right behind another.

## Temporary visibility loss

If the droplet becomes hard to see for some frames and it is clearly the same
droplet afterward, keep the same track. Do not draw boxes on the frames where
you cannot see it.

## Physical merge

If two droplets genuinely coalesce in flight, end **both** tracks at the last
frame where they are still two separate bodies. Then start a new track
(`Alt` + drag) on the merged body and follow it to its own end. Do not carry
the identity of one droplet over to the merged body.

## Save your work

1. Click the disk icon in the top bar (**Save current VIA Project**). The
   browser downloads a file named `via_project_<date>.json`.
2. Move that file into `annotations/` and rename it
   `clip_XXX_annotations.json`, with the number of the clip.
3. Save often. Keep only your latest file for each clip.

To continue a clip later, open the tool, click **Open a VIA Project**, choose
your `annotations/clip_XXX_annotations.json`, and select the clip again in
**File Source**.

## Notes file (only when needed)

Most tracks end at the pool and need nothing else. For a track that ends for
another reason, or that you want to comment on, add a line to
`annotations/clip_XXX_notes.csv`:

```
track_id,end_reason,notes
Object #4,physical_merge,merged with Object #5
Object #9,clip_end,
```

`track_id` is the name shown above the box. `end_reason` is one of `pool`,
`FOV_exit`, `lost_visibility`, `physical_merge`, `clip_end`, `other`.

## Controls

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `→` / `←` | Next / previous frame |
| `Shift + N` / `Shift + P` | 5 frames forward / back |
| Drag | Draw a box in the highlighted track |
| `Alt` + drag | Draw a box in a new track |
| `t` | Track forward from the box just drawn or adjusted (any key stops it) |
| `Backspace` | Delete the selected box or timeline segment |
| Click a row in the timeline | Make that track the current one |

Draw boxes only while the clip is paused.

## What to return

Place all completed VVT annotation files in the `annotations/` folder: one
`clip_XXX_annotations.json` per clip, plus the notes files if you wrote any.
Send back the `annotations/` folder.

Work alone. Do not compare or share annotations with anyone until you have
delivered them. Do not label ID switches, fragmentation, or whether any
automatic tracker is right or wrong. You only mark where each droplet is.

## Metrics — for context only

You do not calculate these. The annotations will later be used for AssA, IDF1,
ID switches, fragmentation, complete-track rate and HOTA.
