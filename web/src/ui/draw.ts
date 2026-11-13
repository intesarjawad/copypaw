import { BackendError } from "../backend";
import type { Animation } from "../compute/animate";
import { CHARACTER_DEPTH, viewProjection } from "../compute/camera";
import type { Tracing } from "../lightTable/underlay";
import { STRINGS, TABLE } from "../strings";
import { el } from "./dom";

const VERTEX_SHADER = `#version 300 es
in vec2 position;
in vec2 uv;
uniform mat4 viewProjection;
uniform float depth;
out vec2 texCoord;

void main() {
  gl_Position = viewProjection * vec4(position, depth, 1.0);
  texCoord = uv;
}`;

// As the engine's texture shader: nearly transparent texels leave the frame clear.
const FRAGMENT_SHADER = `#version 300 es
precision mediump float;
in vec2 texCoord;
uniform sampler2D drawing;
out vec4 colour;

void main() {
  vec4 texel = texture(drawing, texCoord);
  if (texel.a < 0.1) {
    discard;
  }
  colour = texel;
}`;

const MS_PER_SECOND = 1000;

export interface Player {
  readonly element: HTMLElement;
  /** Run and the frame slider, for the rail under the stage. */
  readonly controls: HTMLElement;
  stop(): void;
}

function compile(gl: WebGL2RenderingContext, type: GLenum, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (shader === null) {
    throw new Error("WebGL could not create a shader");
  }
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`Shader did not compile: ${gl.getShaderInfoLog(shader) ?? ""}`);
  }
  return shader;
}

export function program(
  gl: WebGL2RenderingContext,
  vertex: string,
  fragment: string,
): WebGLProgram {
  const linked = gl.createProgram();
  gl.attachShader(linked, compile(gl, gl.VERTEX_SHADER, vertex));
  gl.attachShader(linked, compile(gl, gl.FRAGMENT_SHADER, fragment));
  gl.linkProgram(linked);
  if (!gl.getProgramParameter(linked, gl.LINK_STATUS)) {
    throw new Error(`Shader program did not link: ${gl.getProgramInfoLog(linked) ?? ""}`);
  }
  return linked;
}

function attribute(
  gl: WebGL2RenderingContext,
  shaders: WebGLProgram,
  name: string,
  data: Float32Array,
  usage: GLenum,
): WebGLBuffer {
  const buffer = gl.createBuffer();
  const location = gl.getAttribLocation(shaders, name);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, usage);
  gl.enableVertexAttribArray(location);
  gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
  return buffer;
}

/**
 * Draws frames of an animation the way the engine does: the drawing as a
 * texture on its mesh, triangles in body-part order with no depth test,
 * back faces culled, seen through the engine's fitted camera.
 */
function renderer(
  canvas: HTMLCanvasElement,
  animation: Animation,
  drawing: ImageBitmap,
): (frame: number) => void {
  // The drawing buffer is kept so the shown frame can be read back, as a test or a save does.
  const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true });
  if (gl === null) {
    throw new BackendError({ code: "no_webgl", ...STRINGS.noWebgl });
  }
  const shaders = program(gl, VERTEX_SHADER, FRAGMENT_SHADER);
  gl.useProgram(shaders);
  gl.bindVertexArray(gl.createVertexArray());
  const vertexCount = animation.uv.length / 2;
  const positions = attribute(
    gl,
    shaders,
    "position",
    animation.positions.subarray(0, 2 * vertexCount),
    gl.DYNAMIC_DRAW,
  );
  attribute(gl, shaders, "uv", animation.uv, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, animation.indices, gl.STATIC_DRAW);

  gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, drawing);
  // Average texels when the drawing is shown smaller than it was photographed, so thin strokes keep their weight.
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  gl.uniformMatrix4fv(
    gl.getUniformLocation(shaders, "viewProjection"),
    false,
    viewProjection(animation.camera, animation.window, animation.mirrored),
  );
  gl.uniform1f(gl.getUniformLocation(shaders, "depth"), CHARACTER_DEPTH);
  gl.enable(gl.CULL_FACE);
  // Mirroring the picture turns every triangle over, so the front faces turn with it.
  gl.frontFace(animation.mirrored ? gl.CW : gl.CCW);
  gl.disable(gl.DEPTH_TEST);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0, 0, 0, 0);

  return (frame) => {
    const start = 2 * vertexCount * frame;
    gl.bindBuffer(gl.ARRAY_BUFFER, positions);
    gl.bufferSubData(
      gl.ARRAY_BUFFER,
      0,
      animation.positions.subarray(start, start + 2 * vertexCount),
    );
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawElements(gl.TRIANGLES, animation.indices.length, gl.UNSIGNED_INT, 0);
  };
}

/** The dog the drawing was traced from, to stand the drawing on as it plays. */
export interface Traced {
  readonly tracing: Tracing;
  show(frame: number, stage: HTMLElement): void;
}

/**
 * A canvas that plays the animation in a loop, with a run and stop button
 * and a frame slider. With reduced motion it waits on the first frame.
 */
export function animationPlayer(
  animation: Animation,
  drawing: ImageBitmap,
  traced?: Traced,
): Player {
  const [width, height] = animation.window;
  const canvas = el("canvas", {
    width: String(width),
    height: String(height),
    role: "img",
    "aria-label": STRINGS.animationLabel,
  });
  const label = el("span", {});
  const toggle = el(
    "button",
    { type: "button", class: "reel" },
    el("span", { class: "cap" }),
    label,
  );
  const slider = el("input", {
    type: "range",
    min: "0",
    max: String(animation.frameCount - 1),
    value: "0",
    "aria-label": STRINGS.frame,
  });
  const stage = el("div", { class: "stage" }, canvas);
  // A traced result is cropped to the walk; the canvas is sized so the crop fills the stage.
  const view = traced?.tracing.view ?? { x: 0, y: 0, width, height };
  stage.style.aspectRatio = `${String(view.width)} / ${String(view.height)}`;
  canvas.style.width = `${String((100 * width) / view.width)}%`;
  canvas.style.height = `${String((100 * height) / view.height)}%`;
  const controls = el("div", { class: "player-controls" }, toggle, slider);
  const element = el("div", { class: "player" }, stage);
  const draw = renderer(canvas, animation, drawing);
  const frameMs = animation.frameTimeS * MS_PER_SECOND;

  let frame = 0;
  let playing = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  let startedMs = performance.now();
  let request = 0;

  function show(next: number): void {
    frame = next;
    slider.value = String(next);
    draw(next);
    if (traced !== undefined) {
      const { a, d, e, f } = traced.tracing.drawingAt(next);
      // Percentages of the canvas, which fills the stage, so the placing scales with it.
      canvas.style.transform = `translate(${String((100 * (e - view.x)) / width)}%, ${String(
        (100 * (f - view.y)) / height,
      )}%) scale(${String(a)}, ${String(d)})`;
      traced.show(next, stage);
    }
  }

  function tick(nowMs: number): void {
    const next = Math.floor((nowMs - startedMs) / frameMs) % animation.frameCount;
    if (next !== frame) {
      show(next);
    }
    request = requestAnimationFrame(tick);
  }

  function setPlaying(next: boolean): void {
    playing = next;
    label.textContent = next ? TABLE.stop : TABLE.run;
    cancelAnimationFrame(request);
    if (next) {
      startedMs = performance.now() - frame * frameMs;
      request = requestAnimationFrame(tick);
    }
  }

  toggle.addEventListener("click", () => {
    setPlaying(!playing);
  });
  slider.addEventListener("input", () => {
    setPlaying(false);
    show(Number(slider.value));
  });
  show(0);
  setPlaying(playing);

  return {
    element,
    controls,
    stop: () => {
      cancelAnimationFrame(request);
    },
  };
}
