# Filming a pet clip

Copypaw reads the pet's pose in every frame, so the clip has to show the whole
pet clearly. A phone is enough. Most failures trace back to one of the points below.

## The short version

1. Stand square to the pet's path and film its side, with the phone at about the
   height of its back.
2. Keep the whole pet in frame from nose to tail and from paws to back, for the
   entire clip.
3. Hold the phone still, or rest it on something. Do not follow the pet with a pan.
4. Film in daylight or a bright room, with the light on the pet.
5. Capture 3 to 8 seconds of one animal walking, trotting or running.

## Camera

- **Side view.** The pose model places 17 joints in the image plane. A
  front, rear or three-quarter view hides legs behind each other, and Copypaw
  does not rebuild depth. The pet can walk left to right or right to left.
- **Level and steady.** Planted paws define the ground line, and the root's travel
  is measured against them. A camera that tilts, pans or zooms during the clip moves
  every paw at once and looks like movement of the pet.
- **Distance.** Back up until the pet fits with a margin on every side, including
  its tail. A pet that leaves the frame loses joints on those frames.
- **Orientation.** Portrait and landscape both work. Portrait clips from an iPhone
  are stored as HEVC with a rotation flag, and Copypaw applies that rotation.
  If a clip will not open, export it again as an H.264 MP4 with a constant frame rate.

## The pet and the scene

- **One animal.** The detector picks the largest animal and follows it by
  overlap from frame to frame. A second animal that walks in front can take over the track.
  People are never tracked, but keep them out of the shot, and out of any clip you plan
  to share.
- **Visible legs.** All four paws, the neck and the root of the tail must be
  seen confidently at some point in the clip. Long fur, tall grass, a fence or a
  table edge in front of the legs hides them.
- **Contrast.** A coat close to the floor's colour, such as dark fur on a dark
  floor, hides the joints from the pose model. Pick a background that differs from the coat.
- **Light.** Even light on the pet. Avoid a pet that is a silhouette against a
  window, and avoid hard shadows that cut the legs.
- **Gait.** Walking, trotting and running show repeated footfalls. Standing,
  sniffing and turning in place give the contact detector no steps to find.

## Length

Copypaw reads the first 8 seconds of a clip and ignores the rest. Aim for 3 to 8
seconds of steady walking, trotting or running.

## Check your clip

`copypaw capture clip.mp4 --out out` writes `quality.json` beside the motion.
It warns when fewer than 12 of the 17 joints are confident in more than 30% of the
frames, and it lists any of the neck, tail root or four paws that were never confident.
Both cases point back to this page: refilm with more of the pet in view, a steadier camera
and better light.

If the command stops with "found an animal in fewer than half the frames", the pet
was out of view, too small or hidden for most of the clip. Film closer, or from
farther away if the pet was cropped.

## Sharing a clip

Everything above is how a clip gets processed on your own computer. If you send a
clip to the project, read [privacy.md](privacy.md) first: it explains the consent
forms, what happens to the file, and how to withdraw it.
