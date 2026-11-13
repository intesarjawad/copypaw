# Design: the light table

The web page (`web/`) opens on an animator's light table in a dark room. The self-test dog clip shines up through a
sheet of paper in non-photo blue, the rigged drawing walks on top of it on the same red floor line, and
`footfall.json` lies on the desk beside the table as an exposure sheet, as tall as the table. One glance shows the
input, the output and the data. Below the table come the three commands that run it on your own dog, then the tool
itself ("Animate a drawing here") on the same kind of lit table: the photo hangs on a peg bar, the joints are pins
you drag, and the result plays on the glass with the dog under it.

## What it takes, and from where

- **Max Fleischer's Rotoscope** (US patent 1,242,674, 1917). Live film is projected from behind a pane of glass and
  the animator draws over it, frame by frame. Copypaw does that tracing for you, so the page shows exactly that rig:
  the clip is the lower layer, fainter and in another ink, and the drawing shares its registration and floor line.
- **The studio exposure sheet.** One ruled row per frame, a column per channel, a header of scene, drawing and frame
  count. The columns are the four paws of the dog and the four of the drawing. As an animator marks a hold, a
  planted paw is a pencil line down its column, drawn over the ruling with graphite's broken grain, with a tick
  across the frame it lands. The current frame is a grease-pencil line across the sheet, its number written just
  above the line. The sheet lies in the room, not on the glass: with the table off it falls dark with the room
  (no further than keeps its pencil at 4.5:1), and the desk lamp's beam, carrying on past the table, warms its
  near edge. Its smallest type is 12 px.
- **The table as an object.** A steel rail brushed along its length with one rolled outer edge, a pane of milky
  opal glass with a cut edge that glows green when lit, and a machined peg bar with two bolts. The sheet hangs on the
  bar's three pegs through punched slots a little below its top edge: turned steel pegs, rounded over the top, a
  glint at the upper left and the sheet's glow along the lower edge, each casting a short shadow down and right
  into its slot and across the sheet. The bar and pegs keep their proportions to the sheet, down to a thumbnail's.
  No light comes through a slot. The front rail runs deeper than the others and carries the
  plates: a rocker switch and a red push button, the lamp and the reel, as the tool's rail carries its plates. Every
  plate is one component, the same size and type in both places. A green enamelled desk lamp (weighted base, two
  tube arms, two knuckles, a cone shade, each rounded and glinting where the room's light catches it) stands on the
  desk at the table's left edge, drawn whole where the room holds it and left out of sight where it does not. Its
  base sits in a tight contact shadow and its arms and shade, held higher, throw softer shadows further off; lit,
  its bulb leaks a little light round the shade onto the desk it stands on.

Not taken: Fleischer's drawings, any studio's sheet artwork, or light through punched holes.

## The signature moment: the lamp

`web/src/lightTable/backlight.ts`, one WebGL2 fragment shader over the whole first screen.

1. **Lamp on.** Three tubes light the opal glass from below: three soft stripes at a sixth, a half and five sixths
   of the way down, with dimmer valleys between them, falling off towards the rail on every side and most at the
   tubes' ends, uneven as old tubes and a scuffed diffuser are, so the sheet reads as flat paper over tubes. The
   brightest of it blooms a little over everything on the sheet, the pencil included. The light through the paper is warmer than the bare glass, since paper scatters blue
   more than red. Bare glass around the sheet is milky and brighter than the sheet. The paper transmits a little
   unevenly: a fine formation mottle and two kinds of gently curved pulp fibre, short fine ones and a few long coarse
   ones, each thinning to its ends, clumped in flocs where the pulp settled thick and sparse where it ran thin. The glass's cut edge carries light sideways and glows faintly green. The
   glow spills onto the rail and the steel peg bar and pools on the dark desk around the table; it does not light the
   paper's own face. All of it comes from the shader: the page adds no CSS glow, so the no-WebGL still has no halo.
2. **The clip.** Only the dog is printed, as tonal film: the clip's own luminance mapped into three quiet blues
   (shadow, mid, highlight), multiplied into the light under the paper's fibres, its matte edge feathered over about
   2.5 px of screen at any scale. The dog is pale on a pale wall, so `build_light_table.py` cuts it out of each frame with GrabCut, seeded
   from the tracked skeleton: the bones and torso are surely dog, pixels near them that differ from the clip's median
   frame are probably dog, and the gap between the legs and everything below the floor line are not. The floor is
   as dark as the dog's shadow and muzzle, so only the legs may cross it, and a person stands ahead of the dog at the
   end of the clip, so nothing past the nose is dog. Below the torso and ahead of the shoulders, what matches the
   median frame and lies over two bone widths from the skeleton is wall, not dog. The largest
   piece is kept and its edge softened, and the matte atlas (`matte.jpg`) ships beside the frames. The drawing is
   registered paws over paws: per frame, the middle of its lowest ink sits over the middle of the dog's four tracked
   paws, and that frame's lowest ink row (the sheet's `drawing.lows`) sits exactly on the dog's floor line.
3. **The pencil.** Graphite catches on the paper's tooth, bleeds a little into the fibre, and its weight wanders with
   the hand's pressure. It passes almost none of the lamp's light, so lines read dark when lit from behind. A leg in
   front of the body is drawn under it, paw first, in the engine and the tab alike: the body's outline runs
   unbroken across the hip, and the upper bone covers the knee as it does on a leg behind.
4. **Switching.** The page opens with the lamp off and the tubes strike as soon as the shader is ready. On, the tube
   strikes in stutters for 420 ms and the room lights up around the glass. Off, it
   flickers once and its glow dies over 700 ms, and the desk lamp at the table's left edge takes over. Its bulb and
   rim glow, and its shade throws a wedge of light that crosses the rail before it reaches the paper and grazes it: its tooth
   stands out in relief (the depth-map relighting technique from Codrops' "Relighting images with depth maps", with
   normals from the paper's height field), the graphite stays dark with a faint sheen, and the sheet falls to near
   black away from the lamp, as the room's own light drops with the table's. The dog lingers on the paper as a
   faint blue afterimage. A mouse over the table turns the lamp to follow it; over the words it turns back. The
   eye's adaptation is modelled: reflected light counts for 45% while the lamp is on.

Frames are discrete, as rendered: the reel runs at the clip's 15 frames a second. Dragging the paper walks the dog:
the frame follows the hand at the dog's own pace across the glass, so the drawing stays under the pointer. While the
reel runs, each mark on the exposure sheet is pencilled in as its frame passes; stopped, the whole sheet shows.

### Performance and fallbacks

- The shader draws only when something changes (a new frame, the lamp settling, the desk lamp turning), so a running
  reel costs 15 draws a second. It stops when the table scrolls off screen or the tab is hidden.
- Pixel ratio is capped at 2, and is 1 on narrow or touch screens. Paper grain is in CSS pixels, so it looks the same
  at every ratio.
- The words and the exposure sheet paint first; the shader module and the three atlases (frames, matte, drawing)
  load after first paint.
- Reduced motion: the reel starts stopped on frame 19, the lamp is on from the first frame and switches at once,
  nothing loops. Every control still works, one frame at a time.
- No WebGL 2, or a texture limit under the atlas size: `web/public/light-table/still.jpg` (a photograph of the
  table alone at frame 19, with no room around it) stands in, the lamp and reel controls hide, the frame is fixed
  and the line under the table says why and that the exposure sheet still lists every frame. The tool keeps its
  CSS glass and rail, and its result sheet stays bare until there is a result.
- The canvas never holds the only copy of anything. The headline, the counter, the slider value and the full
  exposure sheet (each cell holds its `1` or `0` as text, so the sheet copies as data) are DOM.

## The promise

The page's own tool rigs and animates a drawing with the bundled walk (the motion tracked from the light table's
clip) in the tab, on a second copy of the light table
(`web/src/lightTable/toolGlass.ts`, the same shader behind the tool): the same rail, opal glass, peg bar with punched
slots, fibre and blue ink. The photo and the result are two sheets on that glass. The light table's dog lies under the
result, drawn by the shader: the bundled walk was tracked from that clip, so the same moment of it sits under each
frame. `traceOver` in `underlay.ts` scales the drawing so its front-to-back paw spread matches the dog's (the sheet's
`dog.span`, measured by `build_light_table.py`), turns it to face the way the dog walks, stands its paws over the dog's
on the red floor line and sizes the pair to fit the sheet. The result sheet is cropped to the action: the dog's
extent over every frame (the sheet's `dog.box`) and the drawing's, down to the floor line, with room above for the
peg bar. Before any photo is chosen, the dog walks alone on that sheet; while it animates, a progress bar and the
stage name sit above it and the dog keeps walking. After Animate the result takes the glass at the light table's
scale and the drawing keeps its editing size beside it, its pins still in reach. Every button is a plate on a
deeper front rail under the glass: choose a photo and Flip under the drawing, Animate, Run and the frame slot under
the result; on a phone they stack, with the slot on a row of its own across the full width. A phone's button reads "Take or choose a photo". The
uploaded drawing multiplies onto the paper with no outline round it. Pins are map pins seen from above: domed
ochre plastic heads lit from the top left, each standing on its needle, so the head's soft shadow falls clear of it
down and to the right. A pin is chosen by clicking it or by tabbing
to it; arrow keys nudge the chosen pin. With one bundled walk and no clip of your own to track, the walk is chosen
already and the choice is not shown. The drawing's heading keeps its place in every state: before a photo is
chosen a blank sheet hangs where the photo will, with the button on it. Red is for the one button to press: until
there is a drawing Animate is grey with a slip of paper taped beside it saying "Choose a photo first"; after Animate
its cap greys and Run's stays red. On a phone the rail's plates stand in one two-column grid, the slot across the
bottom. Tracking a clip of your own dog needs the local app, so the first
screen points to the three commands, which sit as a numbered list just below it. The limits of the bundled clip are
a footnote under those commands, in plain words, with the F1 figure after them.

## Numbers on the page, and where each comes from

`scripts/build_light_table.py` runs `copypaw capture`, `rig` and `animate` on a clip and a drawing (by default
`copypaw/selftest/dog_walk.mp4` and `dragon.png`; `--clip` and `--drawing` swap them, and `--name` says what the
drawing shows) and writes `web/src/lightTable/sheet.json` and the three atlases in `web/public/light-table/`. The
sample drawing is a camel, so the page's sheet is built with `--name camel`.

| Shown | Source |
|---|---|
| Scene `dog_walk`, drawing `camel` | the self-test clip's name; `--name camel` for the drawing |
| 74 frames, 4.9 s | frames rendered by `animate` at `OUTPUT_FPS` (15) |
| Planted cells, per paw and frame | `footfall.json` from `animate` (the displayed animation, after foot planting) |
| F1 0.35 | `contact_f1` in `fidelity.json` |
| Landings 1, 1, 1 and 2; at least 2 needed | `onsets()` on the capture's contacts; the rule in `is_locomotion` |
| Floor line, the drawing's place on it | `ground_line_px` and keypoints in `motion.npz`; the drawing's lowest ink row |
| The dog's silhouette | GrabCut per frame, seeded from the keypoints in `motion.npz` |

The bundled clip is 5 s, too short to show a gait: the walk check fails on it, and the page says so in the
footnote. A 10 to 15 s side-on walk is needed before the sheet can show a full stride. Rerun the script with
`--clip <that clip> --name camel`, then `npm run capture` with the dev server up to rephotograph the still, `docs/hero.jpg` and
`docs/social-preview.png`.

## Tokens

| Token | Value | Use |
|---|---|---|
| `--room`, `--room-ink`, `--room-soft` | `#141517`, `#ece7de`, `#b3aca1` | the dark room and the words on it |
| `--ink`, `--ink-soft` | `#121517`, `#3b4347` | text on the glass and the exposure sheet |
| `--paper` | `#f4f1ea` | the exposure sheet, error notes |
| `--glass`, `--glass-edge` | `#f6fafb`, `#dcdfd9` | the ring inside a checked box; the tool's milky glass without WebGL, at the rail |
| `--steel`, `--steel-edge` | `#2c3236`, `#6d777c` | the tool table's rail without WebGL |
| `--blue` | `#2a72b2` | the dog's marks on the sheet |
| `--graphite` | `#3a3f43` | the drawing's marks on the sheet |
| `--grease` | `#961a10` | the current frame, the floor line, the chosen joint pin, focus on the glass, errors |
| `--focus` | `#e0583a` | focus rings in the room |

`scripts/craft_check.py` measures each pair from the stylesheet. The words over the shader were checked from
screenshots with the words hidden, against the brightest pixel behind each block, lamp on and off and with the desk
light at the table's edge nearest them: all clear the 4.5:1 text minimum. The desk lamp's shade keeps the room
behind the headline dark, and the lamp turns to follow the mouse only over the table.

## Type

- **BBH Bogle** (display): the wordmark and the exposure sheet's header values, uppercase.
- **Mozilla Headline** (text): everything else, headline included, in sentence case.

Both are self-hosted (the page makes no request beyond its own origin) and under the SIL Open Font License 1.1.

## What must not change

- The clip, the drawing and the sheet come from one real run; nothing on the table is drawn by hand or invented.
- The drawing stands on the clip's floor line at the scale `side_by_side.mp4` gives it.
- The lamp is the only control that hides the clip, and the drawing never stops when it does.
- The light comes from under the whole sheet; the bar lies under the punched slots, so no light comes through them.
- No control promises what the page cannot do: tracking your own dog is shown as the commands that run it.
- The page's limits are stated on the page, in a footnote just below the first screen, in plain words.
- The dog under the tool's result appears only for the bundled walk tracked from that clip.
- The tool and the light table are one product: one shader, one rail, one ink.
