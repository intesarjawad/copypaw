# Sample attribution

## dog_walk.mp4

- Title: A representative side view video clip showing limb and joint angle in a normal dog (Movie S3)
- Authors: J. Shin, B. Greer, C. Hakim, Z. Zhou, Y. Chung, Y. Duan, Z. He and D. Duan
- Source: [Wikimedia Commons file page](https://commons.wikimedia.org/wiki/File:Quantitative-Phenotyping-of-Duchenne-Muscular-Dystrophy-Dogs-by-Comprehensive-Gait-Analysis-and-pone.0059875.s003.ogv)
- Licence: [Creative Commons Attribution 2.5 Generic](https://creativecommons.org/licenses/by/2.5/)
- Changes: Trimmed to 0.000–5.000 seconds, re-encoded as H.264, and audio and source metadata removed. The source contains no audio. Playback rate remains 12 frames per second, with 60 frames total.
- Source SHA-256: `6863e5046808ff5559c5b068ec3ab942d9de63ebbdc5a5d9d14bf9e00ebdcad6`
- Sample SHA-256: `03362f03f3efc7d82a1d304e3698e674e94d42f15f75e3c359db20c17fbb744c`

The Commons file page states CC BY 2.5; its embedded metadata lists CC BY 3.0.
The MP4 container and encoder add technical identification tags.

```sh
ffmpeg -ss 0 -i source.ogv -t 5 -map 0:v:0 -an -map_metadata -1 -c:v libx264 -crf 23 -preset slow -pix_fmt yuv420p -movflags +faststart samples/pets/dog_walk.mp4
```
