/* Recording a meeting from the browser.
 *
 * Two situations, one recorder. Somebody in a room puts a laptop on the table
 * and captures the microphone; somebody in an online call captures the tab the
 * call is running in, usually mixed with their own microphone so their side is
 * in the transcript too. Both produce one audio track, which is uploaded to
 * /workspace/api/meetings/recordings and transcribed on the server.
 *
 * Why capture the *tab* rather than having a bot join the call: this product is
 * air-gapped. Every off-the-shelf meeting bot relays the audio through a third
 * party, which is exactly what the rest of this system is built to prevent. The
 * person is already in the meeting, so their machine already has the audio.
 *
 * Three things here are less obvious than they look:
 *
 * 1. Chrome only offers audio in getDisplayMedia when video is requested too.
 *    So we ask for video, never record it, and keep its track alive — stopping
 *    it would end the whole capture, audio included.
 * 2. The level meter is not decoration. The worst outcome this feature has is
 *    recording an hour of a muted microphone, and the meter is the only thing
 *    that tells somebody, while there is still time to fix it, that no sound is
 *    arriving.
 * 3. Streams are mixed through a Web Audio graph rather than by handing several
 *    tracks to MediaRecorder, because MediaRecorder records only the first
 *    audio track of a stream and would silently drop the other side of the call.
 */
(function () {
  "use strict";

  const MIME_CANDIDATES = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];

  function pickMimeType() {
    if (typeof MediaRecorder === "undefined") return null;
    return MIME_CANDIDATES.find((t) => MediaRecorder.isTypeSupported(t)) || "";
  }

  function extensionFor(mime) {
    if (!mime) return ".webm";
    if (mime.includes("ogg")) return ".ogg";
    if (mime.includes("mp4")) return ".m4a";
    return ".webm";
  }

  function formatClock(totalSeconds) {
    const s = Math.max(0, Math.floor(totalSeconds));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    const pad = (n) => String(n).padStart(2, "0");
    return h ? `${h}:${pad(m)}:${pad(sec)}` : `${pad(m)}:${pad(sec)}`;
  }

  /* One recording session: owns the streams, the MediaRecorder and the meter.
   * Kept as a class because every one of these must be released on stop — a
   * leaked microphone track leaves the browser's recording indicator lit, which
   * is alarming and looks like the app is still listening. */
  class Session {
    constructor({ useMic, useSystem, deviceId, roomMode, onLevel, onTick, onEnded, onError }) {
      this.useMic = useMic;
      this.roomMode = !!roomMode;
      this.useSystem = useSystem;
      this.deviceId = deviceId;
      this.onLevel = onLevel;
      this.onTick = onTick;
      this.onEnded = onEnded;
      this.onError = onError;
      this.streams = [];
      this.chunks = [];
      this.startedAt = null;
      this.stopped = false;
      this.systemHasAudio = false;
    }

    async start() {
      const mime = pickMimeType();
      if (mime === null) throw new Error("This browser cannot record audio. Use Chrome or Edge.");

      const context = new (window.AudioContext || window.webkitAudioContext)();
      this.context = context;
      const destination = context.createMediaStreamDestination();
      let sourceCount = 0;

      if (this.useSystem) {
        let display;
        try {
          display = await navigator.mediaDevices.getDisplayMedia({
            video: true,
            audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
          });
        } catch (err) {
          await this.release();
          throw new Error(
            err && err.name === "NotAllowedError"
              ? "Screen sharing was cancelled, so the meeting audio could not be captured."
              : "This browser could not capture tab audio: " + (err && err.message)
          );
        }
        this.streams.push(display);
        const audioTracks = display.getAudioTracks();
        if (!audioTracks.length) {
          await this.release();
          throw new Error(
            "That share has no audio. In the picker choose a browser tab (or a window " +
              "playing the call) and tick “Also share tab audio”. On macOS, Chrome " +
              "cannot capture whole-screen audio — share the meeting tab itself."
          );
        }
        this.systemHasAudio = true;
        context.createMediaStreamSource(new MediaStream(audioTracks)).connect(destination);
        sourceCount += 1;
        // If the person clicks Chrome's own "Stop sharing" button, the meeting
        // audio is gone. Finish the recording rather than carry on recording
        // silence for the rest of the call.
        audioTracks[0].addEventListener("ended", () => {
          if (!this.stopped) this.onEnded("Screen sharing ended, so the recording was stopped.");
        });
      }

      if (this.useMic) {
        let mic;
        // A laptop microphone defaults to settings tuned for one person
        // talking into it from two feet away on a call: echo cancellation and
        // noise suppression both treat a second voice across the room as noise
        // and duck it, and that is exactly the voice a meeting recording
        // exists to capture. Room mode turns both off and leans on automatic
        // gain instead, which lifts the quiet far-side speakers rather than
        // removing them.
        const shaping = this.roomMode
          ? { echoCancellation: false, noiseSuppression: false, autoGainControl: true }
          : {};
        try {
          mic = await navigator.mediaDevices.getUserMedia({
            audio: Object.assign(
              { channelCount: 1 },
              shaping,
              this.deviceId ? { deviceId: { exact: this.deviceId } } : {}
            ),
          });
        } catch (err) {
          await this.release();
          throw new Error(
            err && err.name === "NotAllowedError"
              ? "Microphone access was refused. Allow it in the address bar and try again."
              : "The microphone could not be opened: " + (err && err.message)
          );
        }
        this.streams.push(mic);
        context.createMediaStreamSource(mic).connect(destination);
        sourceCount += 1;
      }

      if (!sourceCount) {
        await this.release();
        throw new Error("Choose at least one audio source.");
      }

      // Meter taps the mixed signal, so it shows exactly what is being written.
      const analyser = context.createAnalyser();
      analyser.fftSize = 1024;
      context.createMediaStreamSource(destination.stream).connect(analyser);
      this.analyser = analyser;
      this.meterData = new Uint8Array(analyser.fftSize);

      this.recorder = new MediaRecorder(destination.stream, mime ? { mimeType: mime } : undefined);
      this.mime = this.recorder.mimeType || mime;
      this.recorder.ondataavailable = (e) => { if (e.data && e.data.size) this.chunks.push(e.data); };
      this.recorder.onerror = () => this.onError("The browser stopped recording unexpectedly.");
      // A timeslice means chunks arrive throughout rather than only at stop, so
      // a crashed tab loses seconds instead of the whole meeting.
      this.recorder.start(5000);
      this.startedAt = new Date();

      this._loop();
      return this;
    }

    _loop() {
      const tick = () => {
        if (this.stopped) return;
        if (this.analyser) {
          this.analyser.getByteTimeDomainData(this.meterData);
          let peak = 0;
          for (let i = 0; i < this.meterData.length; i += 1) {
            peak = Math.max(peak, Math.abs(this.meterData[i] - 128));
          }
          this.onLevel(Math.min(1, peak / 100));
        }
        this.onTick((Date.now() - this.startedAt.getTime()) / 1000);
        this.raf = requestAnimationFrame(tick);
      };
      this.raf = requestAnimationFrame(tick);
    }

    async stop() {
      if (this.stopped) return null;
      this.stopped = true;
      if (this.raf) cancelAnimationFrame(this.raf);
      const blob = await new Promise((resolve) => {
        if (!this.recorder || this.recorder.state === "inactive") {
          resolve(new Blob(this.chunks, { type: this.mime || "audio/webm" }));
          return;
        }
        this.recorder.onstop = () => resolve(new Blob(this.chunks, { type: this.mime || "audio/webm" }));
        this.recorder.stop();
      });
      const startedAt = this.startedAt;
      await this.release();
      return { blob, startedAt, mime: this.mime };
    }

    async release() {
      this.streams.forEach((s) => s.getTracks().forEach((t) => t.stop()));
      this.streams = [];
      if (this.context && this.context.state !== "closed") {
        try { await this.context.close(); } catch (_) {}
      }
    }
  }

  function upload({ blob, startedAt, mime, title, projectId, notes, expectedSpeakers, consentConfirmed, participants, durationSeconds, onProgress }) {
    const form = new FormData();
    form.append("audio", blob, "recording" + extensionFor(mime));
    form.append("title", title || "");
    form.append("startedAt", startedAt.toISOString());
    form.append("durationSeconds", String(Math.round(durationSeconds)));
    if (projectId) form.append("projectId", projectId);
    if (notes) form.append("notes", notes);
    if (expectedSpeakers) form.append("expectedSpeakers", String(expectedSpeakers));
    form.append("consentConfirmed", consentConfirmed ? "true" : "false");
    if (participants) form.append("participants", participants);

    // XHR rather than fetch: an hour of audio is a real upload and the person
    // deserves a progress bar, which fetch cannot report.
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/workspace/api/meetings/recordings");
      xhr.withCredentials = true;
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        let payload = null;
        try { payload = JSON.parse(xhr.responseText); } catch (_) {}
        if (xhr.status >= 200 && xhr.status < 300) resolve(payload);
        else reject(new Error((payload && payload.detail) || "The recording could not be uploaded."));
      };
      xhr.onerror = () => reject(new Error("The connection dropped while uploading the recording."));
      xhr.send(form);
    });
  }

  async function listMicrophones() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return [];
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((d) => d.kind === "audioinput");
  }

  window.BedrockRecorder = {
    Session,
    upload,
    listMicrophones,
    pickMimeType,
    formatClock,
    supported: typeof MediaRecorder !== "undefined" && !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
    canCaptureTabAudio: !!(navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia),
  };
})();
