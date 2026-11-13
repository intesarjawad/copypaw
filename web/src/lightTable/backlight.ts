import { program } from "../ui/draw";
import type { Atlas, Rect } from "./sheet";

const VERTEX_SHADER = `#version 300 es
void main() {
  vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}`;

// A light table in a dark room. A steel rail holds a pane of opal glass lit from below by three
// tubes; a sheet of animation paper hangs from a peg bar on the glass. The lamp's light passes
// through the clip, then the paper, which shows its fibre clouds dark against the glow; the
// pencil sits on the paper's surface. With the lamp off, a desk lamp standing low at the table's
// left edge rakes across the paper and raises its tooth.
// Everything is in CSS pixels from the top left, so the grain is the same at every pixel ratio.
const FRAGMENT_SHADER = `#version 300 es
precision highp float;

uniform sampler2D clipFrames;
uniform sampler2D clipMatte;
uniform sampler2D figureFrames;
uniform vec4 clipGrid;
uniform vec4 figureGrid;
uniform float frame;
uniform vec4 deskRect;
uniform vec4 paperRect;
uniform vec4 secondRect;
uniform vec4 clipRect;
uniform vec4 figureRect;
uniform float floorY;
uniform float lamp;
uniform vec2 deskLight;
uniform vec2 deskAim;
uniform float apron;
uniform float pixelRatio;
uniform float canvasHeight;
out vec4 colour;

const float RAIL = 16.0;
const float LAMP_HEIGHT = 70.0;
const float LAMP_REACH = 420.0;
const vec3 WARM = vec3(1.0, 0.9, 0.76);
const vec3 COOL = vec3(0.86, 0.95, 1.0);
const vec3 VIEW = vec3(0.0, 0.0, 1.0);

float hash(vec2 p) {
  p = fract(p * vec2(443.897, 441.423));
  p += dot(p, p.yx + 19.19);
  return fract((p.x + p.y) * p.x);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

float fbm(vec2 p) {
  float sum = 0.0;
  float amplitude = 0.5;
  for (int octave = 0; octave < 4; octave++) {
    sum += amplitude * noise(p);
    p = mat2(1.6, 1.2, -1.2, 1.6) * p + 7.3;
    amplitude *= 0.5;
  }
  return sum / 0.9375;
}

// One scale of pulp fibres at any angle, each gently curved. A fibre's centre stays a fifth of a
// cell inside its cell and it reaches at most 0.7 of a cell, so the cells around a point hold every
// fibre that can touch it. Where the pulp settled thick, more cells hold one.
float fibreLayer(vec2 p, float size, float thickness, float seed, float settled) {
  float strongest = 0.0;
  vec2 cell = floor(p / size);
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 c = cell + vec2(float(x), float(y)) + seed;
      float weight = hash(c);
      if (weight < settled) {
        continue;
      }
      vec2 centre = (c - seed + 0.2 + 0.6 * vec2(hash(c + 3.1), hash(c + 7.7))) * size;
      float angle = hash(c + 1.3) * 6.2832;
      vec2 along = vec2(cos(angle), sin(angle));
      float reach = size * mix(0.15, 0.7, pow(hash(c + 9.2), 1.5));
      vec2 offset = p - centre;
      float t = clamp(dot(offset, along), -reach, reach);
      float bend = (hash(c + 5.4) - 0.5) * 0.6 / reach;
      vec2 nearest = along * t + vec2(-along.y, along.x) * bend * t * t;
      float distance = length(offset - nearest);
      // A fibre thins towards its ends.
      float width = thickness * (1.0 - 0.6 * abs(t) / reach);
      float strength = mix(0.3, 1.0, (weight - settled) / (1.0 - settled));
      strongest = max(strongest, (1.0 - smoothstep(width * 0.4, width, distance)) * strength);
    }
  }
  return strongest;
}

// Short fine fibres and a few long coarse ones, clumped in flocs where the pulp settled thick and
// sparse where it ran thin.
float fibres(vec2 p) {
  float floc = smoothstep(0.35, 0.7, fbm(p / 70.0 + 23.0));
  float fine = fibreLayer(p, 9.0, 0.7, 0.0, mix(0.97, 0.5, floc));
  float coarse = fibreLayer(p, 26.0, 1.0, 41.0, mix(0.995, 0.8, floc));
  return max(fine, 0.75 * coarse);
}

// The paper's surface height: a fine grain and a coarser tooth over a faint cockle.
float toothAt(vec2 p) {
  return 0.35 * fbm(p / 1.6 + 3.0) + 0.45 * fbm(p / 5.0 + 8.0) + 0.2 * fbm(p / 14.0 + 11.0);
}

// Signed distance to a rounded box given as x, y, width, height; negative inside.
float box(vec2 p, vec4 rect, float radius) {
  vec2 halfSize = rect.zw * 0.5;
  vec2 q = abs(p - rect.xy - halfSize) - halfSize + radius;
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

// The outward direction of a distance field, in the page's y-down space.
vec2 outward(float distance) {
  vec2 slope = vec2(dFdx(distance), -dFdy(distance));
  return slope / max(length(slope), 1e-4);
}

// The peg bar's round middle peg and two flat side pegs, as one distance field, at a scale.
float pegsAt(vec2 q, float barX, float pegY, float sidePeg, float size) {
  return min(length(q - vec2(barX, pegY)) - 5.0 * size,
    min(box(q, vec4(barX - sidePeg - 13.0 * size, pegY - 4.0 * size, 26.0 * size, 8.0 * size), 4.0 * size),
        box(q, vec4(barX + sidePeg - 13.0 * size, pegY - 4.0 * size, 26.0 * size, 8.0 * size), 4.0 * size)));
}

vec2 cellUv(vec4 grid, vec2 local, float index) {
  float row = floor(index / grid.x);
  float column = index - row * grid.x;
  local = clamp(local, vec2(0.002), vec2(0.998));
  return (vec2(column, row) + local) / grid.xy;
}

vec2 p;
float spill;
// How much of the desk light's shaded cone reaches this point.
float aimed = 1.0;

// Light reflected by a surface: the dim room, the desk lamp raking in once the table's lamp is
// off, and the table's glow spilling out of the glass onto whatever is near it. The room goes
// nearly dark with the table off, so away from the desk lamp the paper falls towards black.
vec3 reflected(vec3 albedo, vec3 normal, float shine, float gloss) {
  vec3 toLamp = vec3(deskLight - p, LAMP_HEIGHT);
  float reach = dot(toLamp, toLamp);
  vec3 l = toLamp * inversesqrt(reach);
  float diffuse = max(dot(normal, l), 0.0);
  float highlight = pow(max(dot(normal, normalize(l + VIEW)), 0.0), shine) * gloss;
  float falloff = 1.0 / (1.0 + reach / (LAMP_REACH * LAMP_REACH));
  vec3 desk = WARM * aimed * (1.0 - lamp) * 9.0 * falloff * sqrt(falloff)
    * (albedo * diffuse + highlight * step(0.0, diffuse));
  vec3 room = vec3(1.0, 0.97, 0.92) * mix(0.03, 0.16, lamp) * albedo
    * (0.8 + 0.2 * dot(normal, normalize(vec3(-0.5, -0.6, 0.62))));
  return room + desk + COOL * spill * albedo;
}

float segment(vec2 x, vec2 from, vec2 to) {
  vec2 along = to - from;
  return length(x - from - along * clamp(dot(x - from, along) / dot(along, along), 0.0, 1.0));
}

// Distances to the desk lamp's base, its arms and joints, and its shade, seen from above.
vec3 lampParts(vec2 x) {
  vec2 ahead = normalize(deskAim - deskLight + vec2(1e-3, 0.0));
  vec2 q = vec2(dot(x - deskLight, ahead), dot(x - deskLight, vec2(-ahead.y, ahead.x)));
  float spread = mix(11.0, 30.0, clamp((q.x + 30.0) / 42.0, 0.0, 1.0));
  float toShade = max(abs(q.y) - spread, max(-30.0 - q.x, q.x - 12.0));
  vec2 knuckle = deskLight - ahead * 34.0;
  vec2 elbow = deskLight + vec2(-62.0, 64.0);
  vec2 foot = deskLight + vec2(-40.0, 150.0);
  float arms = min(segment(x, elbow, knuckle), segment(x, foot, elbow)) - 4.0;
  float joints = min(length(x - knuckle), length(x - elbow)) - 7.0;
  return vec3(length((x - foot) / vec2(1.0, 0.72)) - 24.0, min(arms, joints), toShade);
}

// The lamp stands on the desk: its base sits in a tight contact shadow, and its arms and shade,
// held higher, throw softer shadows further off, away from the glass.
float lampShadow() {
  float base = lampParts(p - vec2(-2.0, 3.0)).x;
  float arms = lampParts(p - vec2(-10.0, 12.0)).y;
  float shade = lampParts(p - vec2(-18.0, 20.0)).z;
  return max(1.0 - smoothstep(-3.0, 9.0, base),
    max(0.75 * (1.0 - smoothstep(-4.0, 11.0, arms)), 0.65 * (1.0 - smoothstep(-6.0, 16.0, shade))));
}

// Green enamel on a rounded surface: lit by the room from the upper left, glinting there, with the
// glass's spill on the side towards the table.
vec3 enamel(vec3 normal) {
  vec3 key = normalize(vec3(-0.5, -0.6, 0.62));
  float glint = pow(max(dot(normal, normalize(key + VIEW)), 0.0), 40.0);
  return vec3(0.07, 0.15, 0.1) * (0.45 + 0.9 * max(dot(normal, key), 0.0))
    + vec3(0.55, 0.6, 0.55) * glint * mix(0.6, 0.25, lamp)
    + COOL * spill * 0.12 * max(normal.x + 0.4, 0.0);
}

// The desk lamp seen from above: a weighted base, two arms and two joints, and an enamelled cone
// with its open end towards where it points. Lit, the bulb and the rim glow; with the table on,
// the lamp is a dark shape caught by the glass's spill.
vec4 deskLamp() {
  vec2 ahead = normalize(deskAim - deskLight + vec2(1e-3, 0.0));
  vec2 side = vec2(-ahead.y, ahead.x);
  vec2 q = vec2(dot(p - deskLight, ahead), dot(p - deskLight, side));
  float on = 1.0 - lamp;
  vec3 parts = lampParts(p);
  vec2 mouth = (q - vec2(12.0, 0.0)) / vec2(7.0, 30.0);
  float toMouth = (length(mouth) - 1.0) * 7.0;

  vec3 colour = vec3(0.0);
  float cover = 0.0;
  // The base: a heavy disc with a rolled edge.
  if (parts.x < 1.0) {
    float roll = 1.0 - smoothstep(0.0, 6.0, -parts.x);
    vec3 normal = normalize(vec3(outward(parts.x) * roll * 1.6, 1.0));
    colour = enamel(normal) + WARM * on * 0.06;
    cover = 1.0 - smoothstep(-0.6, 0.6, parts.x);
  }
  // The arms and joints: round enamelled tube and knuckles, their glint running along the top.
  if (parts.y < 1.0) {
    float depth = clamp(-parts.y / 4.0, 0.0, 1.0);
    vec3 normal = normalize(vec3(outward(parts.y) * sqrt(1.0 - depth * depth), depth + 0.15));
    colour = mix(colour, enamel(normal) * 1.15 + WARM * on * 0.12, 1.0 - smoothstep(-0.6, 0.6, parts.y));
    cover = max(cover, 1.0 - smoothstep(-0.6, 0.6, parts.y));
  }
  float shade = 1.0 - smoothstep(-0.6, 0.6, parts.z);
  if (shade > 0.0) {
    float spread = mix(11.0, 30.0, clamp((q.x + 30.0) / 42.0, 0.0, 1.0));
    float across = q.y / spread;
    // The cone is round: its normal turns away from the middle line.
    vec2 turned = side * across * 0.9;
    vec3 normal = normalize(vec3(turned, sqrt(max(1.0 - across * across * 0.81, 0.05))));
    float front = clamp((q.x + 30.0) / 42.0, 0.0, 1.0);
    vec3 shell = enamel(normal) * 1.2 + WARM * on * 0.35 * front * front;
    // The rolled rim at the open end catches the bulb's light.
    float rim = 1.0 - smoothstep(0.0, 3.0, abs(toMouth));
    shell += WARM * on * 1.6 * rim + vec3(0.2) * rim * lamp;
    float inside = 1.0 - smoothstep(-0.6, 0.6, toMouth);
    float bulb = exp(-dot(mouth - vec2(-0.3, 0.0), mouth - vec2(-0.3, 0.0)) * 2.5);
    vec3 lining = vec3(0.04) + vec3(1.0, 0.86, 0.62) * on * (0.6 + 3.5 * bulb);
    colour = mix(colour, mix(shell, lining, inside), shade);
    cover = max(cover, shade);
  }
  return vec4(colour, cover);
}

void main() {
  p = vec2(gl_FragCoord.x, canvasHeight - gl_FragCoord.y) / pixelRatio;
  // Under a table with controls on its front rail, the rail is deeper along the bottom.
  vec4 glassRect = deskRect + vec4(RAIL, RAIL, -2.0 * RAIL, -2.0 * RAIL - apron);
  float toDesk = box(p, deskRect, 10.0);
  float toGlass = box(p, glassRect, 3.0);
  // A second sheet can lie on the same glass; the peg bar and holes belong to the nearer one.
  float toFirst = box(p, paperRect, 0.5);
  float toSecond = box(p, secondRect, 0.5);
  vec4 sheetRect = toSecond < toFirst ? secondRect : paperRect;
  float toPaper = min(toFirst, toSecond);
  spill = lamp * 1.5 * exp(-max(toGlass, 0.0) / 110.0);
  // The desk lamp's shade throws a cone of light; outside it the room stays dark. Seen from above
  // the cone is a wedge from the shade, so it crosses the rail before it reaches the paper.
  vec2 beam = normalize(deskAim - deskLight + vec2(1e-3, 0.0));
  vec2 ray = (p - deskLight) / max(length(p - deskLight), 1e-3);
  aimed = mix(0.04, 1.0, smoothstep(0.8, 0.95, dot(ray, beam)));

  // Paper: a fine mottle of formation and faint fibres show in transmitted light; tooth shows in
  // raking light.
  float formation = 0.5 * fbm(p / 9.0) + 0.5 * fbm(p / 3.5 + 31.0);
  float fibre = fibres(p);
  float transmission = mix(0.9, 1.05, formation) * (1.0 - 0.09 * fibre);
  float tooth = toothAt(p);
  vec3 paperNormal = normalize(vec3(
    (tooth - toothAt(p + vec2(0.45, 0.0))) * 0.6,
    (tooth - toothAt(p + vec2(0.0, 0.45))) * 0.6,
    1.0));

  // The tubes' glow through the opal glass: three soft stripes, one over each tube at a sixth, a
  // half and five sixths of the way down, with dimmer valleys between them. The light falls off
  // towards the rail on every side, most at the tubes' ends, and is uneven as old tubes and a
  // scuffed diffuser are.
  vec2 onGlass = clamp((p - glassRect.xy) / glassRect.zw, 0.0, 1.0);
  float tubes = 0.0;
  for (int tube = 0; tube < 3; tube++) {
    float along = (onGlass.y - (2.0 * float(tube) + 1.0) / 6.0) / 0.075;
    tubes += exp(-along * along);
  }
  vec2 fromEdge = min(onGlass, 1.0 - onGlass);
  float falloff = mix(0.5, 1.0, smoothstep(0.0, 0.3, fromEdge.x)) * mix(0.72, 1.0, smoothstep(0.0, 0.16, fromEdge.y));
  float glow = (0.55 + 0.45 * min(tubes, 1.0)) * falloff * mix(0.92, 1.06, fbm(p / 160.0 + 17.0));

  // Peg bar, screwed to the glass; the sheet hangs on its three pegs through punched holes. The bar
  // and its pegs keep their proportions to the sheet, down to a thumbnail's.
  float barWidth = min(sheetRect.z * 0.46, 320.0);
  float size = clamp(barWidth / 280.0, 0.4, 1.0);
  float barX = sheetRect.x + sheetRect.z * 0.5;
  vec4 bar = vec4(barX - barWidth * 0.5, sheetRect.y - 17.0 * size, barWidth, 38.0 * size);
  float toBar = box(p, bar, 3.0 * size);
  float pegY = sheetRect.y + 14.0 * size;
  float sidePeg = barWidth * 0.36;
  float toPeg = pegsAt(p, barX, pegY, sidePeg, size);
  vec2 boltAt = vec2(barX + sign(p.x - barX) * (barWidth * 0.5 - 13.0 * size), sheetRect.y - 8.0 * size);
  float toBolt = length(p - boltAt) - 4.5 * size;
  // The sheet's punched holes are a little larger than the pegs that pass through them, and the
  // sheet hangs from the pegs, so the gap opens below each one.
  float toHole = min(toPeg, pegsAt(p - vec2(0.0, 2.5 * size), barX, pegY, sidePeg, size)) - 1.2 * size;

  vec3 light;
  if (toDesk > 0.0) {
    // The room: a dark wooden desk top.
    float grain = fbm(p / vec2(60.0, 3.0));
    light = reflected(vec3(0.08, 0.07, 0.06) * mix(0.8, 1.2, grain), VIEW, 8.0, 0.02);
    // Light from the glass pools on the desk around the table, and the bulb leaks a little round
    // the back of the shade onto the desk the lamp stands on.
    light += vec3(1.0, 0.96, 0.9) * lamp * 0.16 * exp(-toDesk / 40.0)
      + WARM * (1.0 - lamp) * 0.07 * exp(-length(p - deskLight - vec2(-30.0, 60.0)) / 90.0);
  } else if (toGlass > 0.0) {
    // The rail: steel brushed along the table's length, rolled at its outer edge, with the glass
    // set into it under a thin shadow line.
    vec2 away = outward(toDesk);
    float outer = 1.0 - smoothstep(0.0, 4.0, -toDesk);
    float inner = 1.0 - smoothstep(0.0, 2.0, toGlass);
    vec3 normal = normalize(vec3(away * outer * 1.4, 1.0));
    float brushed = noise(p * vec2(0.04, 2.0));
    light = reflected(vec3(0.42) * mix(0.85, 1.1, brushed), normal, 60.0, 0.9) * (1.0 - 0.5 * inner);
  } else {
    // Under the paper and the bare glass alike, the lamp's light comes up through the opal. The
    // glow lights the steel above the glass, not the paper and glass it shines through.
    float steelSpill = spill;
    spill = 0.0;
    float edge = 1.0 - smoothstep(0.0, 6.0, -toGlass);
    // Opal glass is milky: it spreads the tubes into a soft, even light with gentle bands.
    vec3 glass = reflected(vec3(0.62, 0.63, 0.6), VIEW, 20.0, 0.05)
      + vec3(0.95, 0.96, 0.92) * lamp * (1.05 + 0.9 * glow);
    // The glass's cut edge carries the light sideways and glows green, as plate glass does.
    glass = mix(glass, vec3(0.3, 0.75, 0.55) * (0.1 + 0.9 * lamp * glow), edge * 0.6);

    // The clip: only the dog, cut out of each frame, as one blue silhouette. The fur's own light
    // and dark show faintly through it, softened by the paper's scatter.
    vec3 film = vec3(1.0);
    vec2 local = (p - clipRect.xy) / clipRect.zw;
    if (all(greaterThan(local, vec2(0.0))) && all(lessThan(local, vec2(1.0)))) {
      vec2 uv = cellUv(clipGrid, local, frame);
      // The matte's edge is feathered over about 2.5 px of screen, whatever the clip's scale.
      float feather = log2(max(2.5 * clipGrid.z / clipRect.z, 1.0));
      float density = smoothstep(0.15, 0.85, 1.0 - textureLod(clipMatte, uv, feather).r);
      // The fur's light and dark printed in three quiet blues, as a film positive would be.
      float tone = smoothstep(0.12, 0.6, textureLod(clipFrames, uv, 0.5).r);
      vec3 blue = mix(mix(vec3(0.2, 0.32, 0.52), vec3(0.42, 0.57, 0.76), smoothstep(0.0, 0.5, tone)),
        vec3(0.7, 0.8, 0.9), smoothstep(0.5, 1.0, tone));
      // The dog walks out of the camera's view, so its edge fades rather than cut off straight.
      float inView = smoothstep(0.0, 0.07, local.x) * smoothstep(1.0, 0.93, local.x);
      film = mix(vec3(1.0), blue, density * inView);
    }

    // The pencil sinks into the paper: graphite catches on the tooth's peaks, bleeds a little
    // into the fibre, and its weight wanders with the hand's pressure.
    vec2 onFigure = (p - figureRect.xy) / figureRect.zw;
    float inside = step(0.0, min(onFigure.x, onFigure.y)) * step(max(onFigure.x, onFigure.y), 1.0);
    vec2 figureUv = cellUv(figureGrid, onFigure, frame);
    float ink = inside * (1.0 - texture(figureFrames, figureUv).r);
    float bleed = inside * (1.0 - textureLod(figureFrames, figureUv, 2.2).r);
    float soft = fwidth(ink) * 0.9;
    // Shrunk below its texels a stroke averages out with the paper, so the threshold drops with it.
    float shrink = clamp(fwidth(onFigure.x) * figureGrid.z, 1.0, 4.0);
    float stroke = smoothstep((0.3 - soft) / shrink, (0.58 + soft) / shrink, ink);
    float pressure = mix(0.72, 1.0, fbm(p / 70.0 + 5.0));
    float line = stroke * pressure * mix(0.82, 1.0, smoothstep(0.3, 0.62, tooth));
    float smudge = smoothstep(0.02, 0.3, bleed) * (1.0 - stroke) * 0.16;
    float crayon = smoothstep(0.02, 0.05, ink) * (1.0 - stroke);
    float onPaperRow = step(clipRect.x + 8.0, p.x) * step(p.x, clipRect.x + clipRect.z - 8.0);
    float redPencil = onPaperRow * (1.0 - smoothstep(0.55, 1.3, abs(p.y - floorY)))
      * mix(0.6, 1.0, smoothstep(0.25, 0.65, tooth));

    // Graphite is dark and a little glossy, so it glints where the desk light rakes it. With the
    // table off, the clip lingers on the paper as a faint blue afterimage.
    vec3 surface = vec3(0.95, 0.925, 0.87) * (1.0 - 0.06 * crayon - smudge)
      * mix(vec3(1.0), vec3(0.78, 0.2, 0.15), redPencil)
      * mix(vec3(1.0), film, 0.45 * (1.0 - lamp));
    vec3 paper = mix(reflected(surface, paperNormal, 12.0, 0.0),
      reflected(vec3(0.03, 0.033, 0.036), paperNormal, 24.0, 0.04), min(line * 1.35, 1.0));
    vec3 passes = mix(vec3(1.0), vec3(0.03), line) * (1.0 - 0.14 * crayon - 0.5 * smudge)
      * mix(vec3(1.0), vec3(0.7, 0.08, 0.06), redPencil);
    float overBar = 1.0 - 0.45 * (1.0 - smoothstep(-3.0, 2.0, toBar));
    // Paper scatters the blue end more than the red, so the glow through it is warmer.
    vec3 lit = vec3(1.0, 0.94, 0.84) * 1.9 * glow * transmission * film * passes * overBar * lamp;
    // The eye adapts to the lamp, so the reflected light counts for less while it is on.
    float adapted = mix(1.0, 0.45, lamp);
    // Veiling glare: the bright sheet blooms a little over everything on it, the pencil included.
    vec3 sheet = paper * adapted + lit + vec3(1.0, 0.95, 0.86) * 0.12 * glow * glow * lamp * transmission;
    float onSheet = (1.0 - smoothstep(-0.6, 0.6, toPaper)) * smoothstep(-0.5, 0.5, toHole);
    light = mix(glass, sheet, onSheet);

    // Each peg throws a short shadow down and to the right across the sheet.
    float pegShade = 1.0 - smoothstep(-1.5 * size, 2.5 * size, pegsAt(p - vec2(2.0, 3.0) * size, barX, pegY, sidePeg, size));
    light *= 1.0 - 0.35 * pegShade * step(0.0, toPeg) * smoothstep(-0.5, 0.5, toHole);

    // The paper casts a thin shadow on the glass with the desk light on.
    float shadow = smoothstep(0.0, 5.0, toPaper) * (1.0 - lamp);
    light *= mix(1.0, 0.9 + 0.1 * shadow, step(0.0, toPaper));

    spill = steelSpill;
    // The bar's top half, its two bolts and the three pegs through the sheet's holes.
    if (toBar < 0.0 && (p.y < sheetRect.y || toPeg < 0.0)) {
      vec3 normal = normalize(vec3(outward(toBar) * (1.0 - smoothstep(0.0, 3.0, -toBar)) * 1.3, 1.0));
      float brushed = noise(p * vec2(0.05, 2.2));
      vec3 steel = reflected(vec3(0.4) * mix(0.85, 1.12, brushed), normal, 50.0, 0.9);
      if (toBolt < 0.0) {
        vec2 dome = (p - boltAt) / 4.5;
        vec3 boltNormal = normalize(vec3(dome * 0.9, 1.0));
        float socket = 1.0 - smoothstep(1.6, 2.2, max(abs(dome.x), abs(dome.y)) * 4.5);
        steel = reflected(vec3(0.5) * (1.0 - 0.75 * socket), boltNormal, 70.0, 1.1);
      }
      if (toPeg < 0.0) {
        // A peg is turned steel, rounded over its top: lit by the room from the upper left, its
        // lower edge catching the glow of the sheet around it.
        float depth = clamp(-toPeg / (3.0 * size), 0.0, 1.0);
        vec2 away = outward(toPeg);
        vec3 pegNormal = normalize(vec3(away * sqrt(1.0 - depth * depth), depth + 0.2));
        vec3 key = normalize(vec3(-0.5, -0.6, 0.62));
        float glint = pow(max(dot(pegNormal, normalize(key + VIEW)), 0.0), 24.0);
        steel = reflected(vec3(0.5) * mix(0.9, 1.1, brushed), pegNormal, 50.0, 0.8)
          + vec3(0.9, 0.92, 0.95) * glint * lamp
          + vec3(1.0, 0.94, 0.84) * 0.45 * glow * lamp * max(away.y, 0.0) * (1.0 - depth)
          + vec3(0.2) * lamp * mix(0.25, 1.0, max(dot(pegNormal, key), 0.0));
      }
      light = mix(light, steel * mix(1.0, 0.6, lamp), 1.0 - smoothstep(-0.7, 0.7, toBar));
    } else if (toHole < 0.0) {
      // Between a peg and the edge of its hole the bar shows, its steel shadowed by the sheet
      // along the hole's edge; no light comes up through it, as the bar lies under the hole.
      // The peg casts its shadow down into the slot.
      float brushed = noise(p * vec2(0.05, 2.2));
      vec3 steel = reflected(vec3(0.3) * mix(0.85, 1.12, brushed), VIEW, 50.0, 0.4);
      float pegShadow = 1.0 - smoothstep(-1.0, 2.0, pegsAt(p - vec2(1.0, 3.0) * size, barX, pegY, sidePeg, size));
      light = steel * mix(1.0, 0.55, lamp) * mix(0.35, 0.75, smoothstep(-0.2, -2.0 * size, toHole))
        * (1.0 - 0.55 * pegShadow);
    } else if (toHole < 1.2 && toPaper < 0.0) {
      // The slot's cut edge is a hair of torn fibre, a little darker than the sheet.
      light *= mix(0.72, 1.0, smoothstep(0.0, 1.2 * size, toHole));
    }
  }

  light *= 1.0 - 0.7 * lampShadow();
  vec4 shaded = deskLamp();
  light = mix(light, shaded.rgb, shaded.a);

  colour = vec4(1.0 - exp(-1.5 * light), 1.0);
}`;

/** Off the canvas: where a sheet or drawing that is not there is put. */
export const NOWHERE: Rect = { x: -1e4, y: -1e4, width: 1, height: 1 };

export interface Scene {
  readonly frame: number;
  /** The table's outer edge, rail included. */
  readonly desk: Rect;
  readonly paper: Rect;
  /** Another sheet on the same glass, if any. */
  readonly second?: Rect | undefined;
  readonly clip: Rect;
  readonly figure: Rect;
  readonly floorY: number;
  readonly lamp: number;
  /** Where the desk lamp stands, in CSS pixels; it lights the table as the lamp dies. */
  readonly deskLight: readonly [number, number];
  /** Where the desk lamp points. */
  readonly deskAim: readonly [number, number];
  /** How much deeper the rail is along the bottom, for controls that sit on it. */
  readonly apron?: number;
}

export interface Backlight {
  resize(width: number, height: number, pixelRatio: number): void;
  draw(scene: Scene): void;
}

function upload(
  gl: WebGL2RenderingContext,
  unit: number,
  image: HTMLImageElement,
  mipmaps: boolean,
): void {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, gl.RED, gl.UNSIGNED_BYTE, image);
  if (mipmaps) {
    gl.generateMipmap(gl.TEXTURE_2D);
  }
  gl.texParameteri(
    gl.TEXTURE_2D,
    gl.TEXTURE_MIN_FILTER,
    mipmaps ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR,
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}

/** Returns null when the browser has no WebGL 2 or cannot hold the frame atlases. */
export function backlight(
  canvas: HTMLCanvasElement,
  clipFrames: HTMLImageElement,
  clipMatte: HTMLImageElement,
  clipAtlas: Atlas,
  figureFrames: HTMLImageElement,
  figureAtlas: Atlas,
): Backlight | null {
  const gl = canvas.getContext("webgl2", { antialias: false });
  if (gl === null) {
    return null;
  }
  const largest = Math.max(
    ...[clipFrames, clipMatte, figureFrames].flatMap((image) => [
      image.naturalWidth,
      image.naturalHeight,
    ]),
  );
  if (largest > (gl.getParameter(gl.MAX_TEXTURE_SIZE) as number)) {
    return null;
  }
  const shaders = program(gl, VERTEX_SHADER, FRAGMENT_SHADER);
  gl.useProgram(shaders);
  gl.bindVertexArray(gl.createVertexArray());
  upload(gl, 0, clipFrames, true);
  upload(gl, 1, figureFrames, true);
  upload(gl, 2, clipMatte, true);
  const at = (name: string): WebGLUniformLocation | null => gl.getUniformLocation(shaders, name);
  gl.uniform1i(at("clipFrames"), 0);
  gl.uniform1i(at("figureFrames"), 1);
  gl.uniform1i(at("clipMatte"), 2);
  const grid = (atlas: Atlas): [number, number, number, number] => [
    atlas.columns,
    atlas.rows,
    atlas.cellWidth,
    atlas.cellHeight,
  ];
  gl.uniform4f(at("clipGrid"), ...grid(clipAtlas));
  gl.uniform4f(at("figureGrid"), ...grid(figureAtlas));
  const rect = (r: Rect): [number, number, number, number] => [r.x, r.y, r.width, r.height];

  return {
    resize(width, height, pixelRatio) {
      canvas.width = Math.round(width * pixelRatio);
      canvas.height = Math.round(height * pixelRatio);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform1f(at("pixelRatio"), canvas.width / width);
      gl.uniform1f(at("canvasHeight"), canvas.height);
    },
    draw(scene) {
      gl.uniform1f(at("frame"), scene.frame);
      gl.uniform4f(at("deskRect"), ...rect(scene.desk));
      gl.uniform4f(at("paperRect"), ...rect(scene.paper));
      gl.uniform4f(at("secondRect"), ...rect(scene.second ?? NOWHERE));
      gl.uniform4f(at("clipRect"), ...rect(scene.clip));
      gl.uniform4f(at("figureRect"), ...rect(scene.figure));
      gl.uniform1f(at("floorY"), scene.floorY);
      gl.uniform1f(at("lamp"), scene.lamp);
      gl.uniform2f(at("deskLight"), ...scene.deskLight);
      gl.uniform2f(at("deskAim"), ...scene.deskAim);
      gl.uniform1f(at("apron"), scene.apron ?? 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    },
  };
}
