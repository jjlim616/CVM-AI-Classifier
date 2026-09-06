<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import {
  Activity,
  ArrowRight,
  Check,
  ChevronDown,
  CircleHelp,
  ImagePlus,
  Layers3,
  LoaderCircle,
  ScanLine,
  ShieldCheck,
  Upload,
  X,
} from "lucide-vue-next";

interface Model {
  id: string;
  name: string;
  architecture: string;
  available: boolean;
  status: string;
}
interface Result {
  model_id: string;
  model_name: string;
  predicted_stage: string;
  scores: { stage: string; score: number }[];
  original: string;
  heatmap: string;
  heatmap_has_signal: boolean;
  elapsed_ms: number;
  input_size: number[];
  device: string;
}
type View = "compare" | "original" | "heatmap" | "overlay";
const examples = ["CS1", "CS3", "CS5", "CS6"];
const exampleStage = ref("");
const savedExample = ref(false);
const loadingExample = ref(false);
const query = new URLSearchParams(location.search);
const annotate = query.has("annotate");
const requestedExample = query.get("example")?.toUpperCase() ?? "";
async function loadExample(stage: string) {
  if (busy.value || loadingExample.value) return;
  loadingExample.value = true;
  error.value = "";
  try {
    const base = `/examples/${stage.toLowerCase()}`;
    const [imageResponse, resultResponse] = await Promise.all([fetch(`${base}.png`), fetch(`${base}.json`)]);
    if (!imageResponse.ok || !resultResponse.ok) throw new Error("Could not load this example.");
    const [blob, recorded] = await Promise.all([imageResponse.blob(), resultResponse.json()]);
    chooseFile(new File([blob], `${stage}-sample.png`, { type: "image/png" }));
    modelId.value = recorded.model_id;
    result.value = recorded;
    exampleStage.value = stage;
    savedExample.value = true;
  } catch (cause) { error.value = cause instanceof Error ? cause.message : "Example unavailable."; }
  finally { loadingExample.value = false; }
}
const models = ref<Model[]>([]);
const modelId = ref("");
const loadingModels = ref(true);
const connectionError = ref("");
const file = ref<File | null>(null);
const preview = ref("");
const result = ref<Result | null>(null);
const error = ref("");
const busy = ref(false);
const dragging = ref(false);
const opacity = ref(45);
const view = ref<View>("compare");
const input = ref<HTMLInputElement>();
let request: AbortController | undefined;
const selectedModel = computed(() =>
  models.value.find((m) => m.id === modelId.value),
);
const canAnalyze = computed(
  () => !!file.value && selectedModel.value?.available && !busy.value && !loadingExample.value,
);
const topScore = computed(
  () =>
    result.value?.scores.find((s) => s.stage === result.value?.predicted_stage)
      ?.score ?? 0,
);
const panels = computed(() =>
  view.value === "compare" ? ["original", "heatmap", "overlay"] : [view.value],
);
const stages = ["CS1", "CS2", "CS3", "CS4", "CS5", "CS6"];

async function loadModels() {
  loadingModels.value = true;
  connectionError.value = "";
  try {
    const response = await fetch("/api/models", {
      signal: AbortSignal.timeout(10000),
    });
    if (!response.ok) throw new Error();
    const data = await response.json();
    models.value = data.models;
    modelId.value =
      models.value.find((m) => m.available)?.id ?? models.value[0]?.id ?? "";
  } catch {
    connectionError.value =
      "The analysis service is unavailable. Start the local Python server and retry.";
  } finally {
    loadingModels.value = false;
  }
}

function chooseFile(chosen?: File) {
  if (!chosen || busy.value) return;
  error.value = "";
  if (!["image/png", "image/jpeg"].includes(chosen.type)) {
    error.value = "Choose a PNG or JPEG image.";
    return;
  }
  if (chosen.size > 10 * 1024 * 1024 || chosen.size === 0) {
    error.value = "Choose a non-empty image smaller than 10 MB.";
    return;
  }
  if (preview.value) URL.revokeObjectURL(preview.value);
  exampleStage.value = "";
  savedExample.value = false;
  file.value = chosen;
  preview.value = URL.createObjectURL(chosen);
  result.value = null;
  view.value = "compare";
}

function onFileChange(event: Event) {
  chooseFile((event.target as HTMLInputElement).files?.[0]);
  if (input.value) input.value.value = "";
}
function drop(event: DragEvent) {
  dragging.value = false;
  if ((event.dataTransfer?.files.length ?? 0) > 1) {
    error.value = "Choose one image at a time.";
    return;
  }
  chooseFile(event.dataTransfer?.files[0]);
}
function reset() {
  if (busy.value) return;
  if (preview.value) URL.revokeObjectURL(preview.value);
  preview.value = "";
  exampleStage.value = "";
  savedExample.value = false;
  file.value = null;
  result.value = null;
  error.value = "";
  if (input.value) input.value.value = "";
}
function modelChanged() {
  savedExample.value = false;
  result.value = null;
  error.value = "";
}

async function analyze() {
  if (!canAnalyze.value || !file.value) return;
  savedExample.value = false;
  busy.value = true;
  error.value = "";
  result.value = null;
  request = new AbortController();
  const timeout = window.setTimeout(() => request?.abort(), 120000);
  try {
    const body = new FormData();
    body.append("model_id", modelId.value);
    body.append("file", file.value);
    const response = await fetch("/api/predict", {
      method: "POST",
      body,
      signal: request.signal,
    });
    const data = await response
      .json()
      .catch(() => ({
        detail:
          "The analysis service could not complete this request. Please retry.",
      }));
    if (!response.ok)
      throw new Error(
        typeof data.detail === "string"
          ? data.detail
          : "Analysis failed. Please try again.",
      );
    result.value = data;
  } catch (cause) {
    error.value =
      cause instanceof Error && cause.name === "AbortError"
        ? "Analysis timed out. The server may still be finishing; try again shortly."
        : cause instanceof Error
          ? cause.message
          : "Unable to analyze this image.";
  } finally {
    window.clearTimeout(timeout);
    busy.value = false;
  }
}
onMounted(async () => {
  await loadModels();
  if ((examples as readonly string[]).includes(requestedExample)) {
    await loadExample(requestedExample);
  }
});
onUnmounted(() => {
  request?.abort();
  if (preview.value) URL.revokeObjectURL(preview.value);
});
</script>

<template>
  <div class="app-shell" :class="{ annotated: annotate }">
    <header class="topbar">
      <a href="/" class="brand" aria-label="CVM Studio home"
        ><span class="brand-mark"><ScanLine :size="23" /></span
        ><span>CVM<span class="brand-light"> Studio</span></span></a
      >
      <span class="header-divider"></span
      ><span class="project-label">Cervical vertebral maturation</span>
      <div class="header-right">
        <span class="local-dot"></span> Local workspace
        <span class="research-badge">Research demo</span>
      </div>
    </header>

    <main>
      <div class="page-heading">
        <div>
          <p class="eyebrow">IMAGE ANALYSIS</p>
          <h1>A closer look at maturation.</h1>
          <p class="intro">
            Classify an X-ray and explore the regions that influenced the model.
          </p>
        </div>
        <div class="privacy-note">
          <ShieldCheck :size="19" /><span
            >Processed locally<br /><strong
              >No image history saved</strong
            ></span
          >
        </div>
      </div>

      <div class="analysis-layout">
        <aside class="control-panel">
          <div class="section-heading">
            <span class="step">01</span>
            <h2>Set up analysis</h2>
          </div>
          <label class="field-label" for="model">Classification model</label>
          <div class="select-wrap">
            <select
              id="model"
              v-model="modelId"
              :disabled="busy || loadingModels || !models.length"
              @change="modelChanged"
            >
              <option v-if="!models.length" value="">
                {{ loadingModels ? "Loading models…" : "Models unavailable" }}
              </option>
              <option
                v-for="m in models"
                :key="m.id"
                :value="m.id"
                :disabled="!m.available"
              >
                {{ m.name }}{{ m.available ? "" : " — weights missing" }}
              </option></select
            ><ChevronDown :size="17" />
          </div>
          <p v-if="selectedModel?.available" class="model-state">
            <span class="local-dot"></span> Local weights found
            <span>6 stages</span>
          </p>
          <p v-else-if="models.length" class="helper">
            Install local model weights to enable analysis. See the project
            README.
          </p>
          <div v-if="connectionError" class="error-box" role="alert">
            {{ connectionError
            }}<button class="text-button" @click="loadModels">
              Retry connection
            </button>
          </div>

          <div class="upload-label">
            <label class="field-label" for="image-file">X-ray image</label
            ><span>PNG / JPG</span>
          </div>
          <input
            id="image-file"
            ref="input"
            class="visually-hidden"
            type="file"
            accept="image/png,image/jpeg"
            :disabled="busy"
            @change="onFileChange"
          />
          <button
            class="drop-zone"
            :class="{ dragging, 'has-file': file }"
            :disabled="busy"
            @click="input?.click()"
            @dragover.prevent="dragging = true"
            @dragleave.prevent="dragging = false"
            @drop.prevent="drop"
          >
            <template v-if="file"
              ><img :src="preview" alt="Selected X-ray thumbnail" /><span
                class="file-name"
                >{{ file.name }}</span
              ><span class="helper"
                >{{ (file.size / 1024).toFixed(0) }} KB · Click to replace</span
              ></template
            >
            <template v-else
              ><span class="upload-icon"><Upload :size="23" /></span
              ><strong>Drop an X-ray here</strong
              ><span>or <b>browse files</b></span
              ><small>One image · up to 10 MB</small></template
            >
          </button>
          <div class="sample-picker">
            <span>Or try a saved example</span>
            <div><button v-for="stage in examples" :key="stage" :disabled="busy || loadingExample" :aria-pressed="exampleStage === stage" @click="loadExample(stage)">{{ stage }}</button></div>
          </div>
          <p class="helper upload-hint">
            Use an image cropped to the cervical vertebrae, matching the
            training data.
          </p>
          <div v-if="error" class="error-box" role="alert">{{ error }}</div>
          <button
            class="analyze-button"
            :disabled="!canAnalyze"
            @click="analyze"
          >
            <LoaderCircle v-if="busy" :size="18" class="spin" /><ScanLine
              v-else
              :size="18"
            />{{ busy ? "Analyzing image…" : "Analyze image"
            }}<ArrowRight v-if="!busy" :size="18" class="button-arrow" />
          </button>
          <button
            v-if="file"
            class="reset-button"
            :disabled="busy"
            @click="reset"
          >
            <X :size="14" /> Clear image
          </button>
          <div class="input-spec">
            <Layers3 :size="17" />
            <div>
              <strong>Consistent model input</strong>
              <p>224 × 224 px · grayscale<br />ImageNet normalization</p>
            </div>
          </div>
          <details class="about">
            <summary><CircleHelp :size="16" /> About this demo</summary>
            <p>
              A final-year research project exploring six-stage CVM
              classification. Model scores are not calibrated probabilities of
              clinical correctness. This demo is not for diagnosis.
            </p>
          </details>
        </aside>

        <section class="workspace" aria-label="Image visualization and results">
          <div class="viewer-header">
            <div class="section-heading">
              <span class="step">02</span>
              <h2>Explore the image</h2>
            </div>
            <span class="viewer-status" :class="{ ready: result }"
              ><Check v-if="result" :size="14" />{{
                result
                  ? "Analysis complete"
                  : busy
                    ? "Processing"
                    : file
                      ? "Ready to analyze"
                      : "Awaiting image"
              }}</span
            >
          </div>
          <div class="viewer" :class="{ 'is-empty': !file }" :aria-busy="busy">
            <div v-if="!file" class="empty-view">
              <div class="viewfinder">
                <ScanLine :size="44" stroke-width="1" />
              </div>
              <h3>Your image, explained.</h3>
              <p>
                Upload an X-ray to compare the original image,<br
                  class="desktop-break"
                />
                Grad-CAM heatmap, and attention overlay.
              </p>
              <span class="empty-caption"
                ><ImagePlus :size="15" /> Start with an image on the left</span
              >
            </div>
            <template v-else>
              <div class="viewer-toolbar">
                <div class="view-switch" aria-label="Image view">
                  <button
                    v-for="tab in [
                      'compare',
                      'original',
                      'heatmap',
                      'overlay',
                    ] as View[]"
                    :key="tab"
                    :aria-pressed="view === tab"
                    :class="{ active: view === tab }"
                    :disabled="!result"
                    @click="view = tab"
                  >
                    {{
                      tab === "compare"
                        ? "Compare"
                        : tab === "heatmap"
                          ? "Grad-CAM"
                          : tab.charAt(0).toUpperCase() + tab.slice(1)
                    }}
                  </button>
                </div>
                <span class="viewer-meta">{{
                  result ? "224 × 224" : "Original upload"
                }}</span>
              </div>
              <div v-if="!result" class="pending-image">
                <img :src="preview" alt="Uploaded X-ray awaiting analysis" />
                <div v-if="busy" class="processing-cover">
                  <LoaderCircle :size="28" class="spin" /><strong
                    >Finding the most relevant regions…</strong
                  ><span>The first run also loads the model.</span>
                </div>
                <span v-else class="pending-caption"
                  >Image loaded. Select Analyze image to begin.</span
                >
              </div>
              <div
                v-else
                class="image-grid"
                :class="{ single: view !== 'compare' }"
              >
                <figure v-for="panel in panels" :key="panel">
                  <figcaption>
                    <span class="panel-number">{{
                      panel === "original"
                        ? "01"
                        : panel === "heatmap"
                          ? "02"
                          : "03"
                    }}</span
                    >{{
                      panel === "original"
                        ? "Model input"
                        : panel === "heatmap"
                          ? "Grad-CAM"
                          : "Overlay"
                    }}
                  </figcaption>
                  <div class="image-frame">
                    <img
                      :src="
                        panel === 'heatmap' ? result.heatmap : result.original
                      "
                      :alt="
                        panel === 'original'
                          ? 'Grayscale X-ray resized to model input'
                          : panel === 'heatmap'
                            ? 'Grad-CAM attention heatmap'
                            : 'X-ray with Grad-CAM overlay'
                      "
                    /><img
                      v-if="panel === 'overlay'"
                      class="overlay-image"
                      :src="result.heatmap"
                      alt=""
                      :style="{ opacity: opacity / 100 }"
                    />
                  </div>
                </figure>
              </div>
              <div v-if="result" class="viewer-footer">
                <div class="heatmap-scale">
                  <span>Lower attention</span><span class="gradient-bar"></span
                  ><span>Higher</span>
                </div>
                <div class="opacity-control">
                  <label for="opacity">Overlay</label
                  ><input
                    id="opacity"
                    v-model.number="opacity"
                    type="range"
                    min="0"
                    max="100"
                    :disabled="view === 'original' || view === 'heatmap'"
                  /><output for="opacity">{{ opacity }}%</output>
                </div>
              </div>
            </template>
          </div>
          <p
            v-if="result && !result.heatmap_has_signal"
            class="cam-warning"
            role="status"
          >
            No positive Grad-CAM signal was found for this prediction. The
            uniform map should not be interpreted as localized attention.
          </p>

          <p v-if="exampleStage" class="example-caption">Dataset label: <strong>{{ exampleStage }}</strong> · {{ savedExample ? 'Saved ConvNeXt Small result; no inference run on this click.' : 'Live analysis of the selected example.' }}</p>
          <section class="results-panel" aria-live="polite" aria-atomic="true">
            <div class="results-heading">
              <div class="section-heading">
                <span class="step">03</span>
                <h2>Model prediction</h2>
              </div>
              <span v-if="result" class="timing"
                >{{ savedExample ? "Saved example · " : "" }}{{ result.model_name }} ·
                {{ (result.elapsed_ms / 1000).toFixed(2) }}s · CPU</span
              >
            </div>
            <div v-if="result" class="results-content">
              <div class="prediction">
                <span class="result-label">PREDICTED STAGE</span>
                <div class="stage-result">
                  {{ result.predicted_stage
                  }}<span
                    >{{ (topScore * 100).toFixed(1) }}%<small
                      >model score</small
                    ></span
                  >
                </div>
                <p>Stage {{ result.predicted_stage.slice(2) }} of 6</p>
              </div>
              <div class="scores">
                <div class="scores-title">
                  Score distribution<span>CS1–CS6</span>
                </div>
                <div class="score-chart">
                  <div
                    v-for="item in result.scores"
                    :key="item.stage"
                    class="score-column"
                    :class="{ winner: item.stage === result.predicted_stage }"
                  >
                    <span class="score-value"
                      >{{ (item.score * 100).toFixed(1) }}%</span
                    >
                    <div class="bar-track">
                      <div
                        class="bar-fill"
                        :style="{ height: `${Math.max(item.score * 100, 1)}%` }"
                      ></div>
                    </div>
                    <span class="score-label">{{ item.stage }}</span>
                  </div>
                </div>
              </div>
            </div>
            <div v-else class="results-empty">
              <span class="result-placeholder">—</span>
              <div>
                <strong>{{
                  busy ? "Analysis in progress" : "No prediction yet"
                }}</strong>
                <p>
                  {{
                    busy
                      ? "Computing class scores and Grad-CAM."
                      : "Your predicted stage and class scores will appear here."
                  }}
                </p>
              </div>
              <div class="stage-placeholders">
                <span v-for="stage in stages" :key="stage">{{ stage }}</span>
              </div>
            </div>
          </section>
          <p class="explanation">
            <Activity :size="15" /> Grad-CAM highlights regions that influence a
            prediction. It does not establish that the model’s reasoning is
            clinically correct.
          </p>
        </section>
      </div>
      <footer class="page-footer">
        <span
          >CVM Studio <span class="footer-slash">/</span> FYP research
          project</span
        ><span>Educational use · Not for clinical diagnosis</span>
      </footer>
    </main>
  </div>
</template>
