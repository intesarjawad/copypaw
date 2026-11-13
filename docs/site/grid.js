"use strict";

(function () {
  const video = document.getElementById("clip");
  const caption = document.getElementById("caption");
  const pickers = document.getElementById("pickers");
  const drawingSelect = document.getElementById("drawing");
  const motionSelect = document.getElementById("motion");
  const copyButton = document.getElementById("copy");
  const copyStatus = document.getElementById("copy-status");
  const command = document.getElementById("command");
  const built = document.getElementById("built");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const label = (name) => name.replaceAll("_", " ");

  function play() {
    if (reduceMotion.matches) {
      return;
    }
    const started = video.play();
    if (started !== undefined) {
      started.catch(() => {});
    }
  }

  function show(clip) {
    const text = `${label(clip.drawing)} drawing, ${clip.animal} ${clip.gait}, ${clip.timing}`;
    video.poster = `clips/${clip.poster}`;
    video.src = `clips/${clip.file}`;
    video.setAttribute("aria-label", text);
    caption.textContent = text;
    play();
  }

  function fill(select, names, text) {
    for (const name of names) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = text(name);
      select.append(option);
    }
  }

  function start(manifest) {
    const clips = manifest.clips;
    const gait = new Map(clips.map((clip) => [clip.motion, `${clip.animal} ${clip.gait}`]));
    fill(drawingSelect, manifest.drawings, label);
    fill(motionSelect, manifest.motions, (name) => gait.get(name));
    const pick = () =>
      clips.find(
        (clip) => clip.drawing === drawingSelect.value && clip.motion === motionSelect.value,
      );
    const change = () => {
      const clip = pick();
      if (clip !== undefined) {
        show(clip);
      }
    };
    drawingSelect.addEventListener("change", change);
    motionSelect.addEventListener("change", change);
    drawingSelect.value = clips[0].drawing;
    motionSelect.value = clips[0].motion;
    caption.textContent = `${label(clips[0].drawing)} drawing, ${clips[0].animal} ${clips[0].gait}, ${clips[0].timing}`;
    pickers.hidden = false;
    built.textContent = `Built from ${manifest.built_from}.`;
    play();
  }

  copyButton.hidden = false;
  copyButton.addEventListener("click", () => {
    const text = command.textContent.trim();
    const failed = () => {
      copyStatus.textContent = "Could not copy. Select the command and copy it by hand.";
    };
    if (!navigator.clipboard) {
      failed();
      return;
    }
    navigator.clipboard.writeText(text).then(() => {
      copyStatus.textContent = "Copied.";
    }, failed);
  });

  fetch("clips/manifest.json")
    .then((response) => (response.ok ? response.json() : Promise.reject(new Error("no manifest"))))
    .then(start)
    .catch(() => {});
})();
