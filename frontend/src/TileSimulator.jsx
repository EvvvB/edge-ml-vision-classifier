import { useState } from 'react'
import { detectionImageUrl } from './api.js'

// Mirrors the firmware constants in nicla/firmware/main.py.
const DEFAULT_GRID_COLUMNS = 3
const DEFAULT_GRID_ROWS = 2
const TILE_OVERLAP_PIXELS = 16

// Edge Impulse FOMO model input. The on-device Normalization stretches each
// tile ROI to this size, ignoring aspect ratio.
const MODEL_INPUT_SIZE = 96

// Same integer math as create_tile_rois() in the firmware: base cells from
// floor division, then each cell grown by the overlap and clipped to the frame.
export function createTileRois(frameWidth, frameHeight, columns, rows, overlap) {
  const rois = []
  for (let row = 0; row < rows; row++) {
    const y1 = Math.floor((row * frameHeight) / rows)
    const y2 = Math.floor(((row + 1) * frameHeight) / rows)
    for (let column = 0; column < columns; column++) {
      const x1 = Math.floor((column * frameWidth) / columns)
      const x2 = Math.floor(((column + 1) * frameWidth) / columns)
      const roiX1 = Math.max(0, x1 - overlap)
      const roiY1 = Math.max(0, y1 - overlap)
      const roiX2 = Math.min(frameWidth, x2 + overlap)
      const roiY2 = Math.min(frameHeight, y2 + overlap)
      rois.push({
        x: roiX1,
        y: roiY1,
        width: roiX2 - roiX1,
        height: roiY2 - roiY1,
      })
    }
  }
  return rois
}

// OpenMV's COLOR_RGB565_TO_GRAYSCALE luma: (r*38 + g*75 + b*15) >> 7.
function toGrayscale(imageData) {
  const pixels = imageData.data
  for (let i = 0; i < pixels.length; i += 4) {
    const luma = (pixels[i] * 38 + pixels[i + 1] * 75 + pixels[i + 2] * 15) >> 7
    pixels[i] = luma
    pixels[i + 1] = luma
    pixels[i + 2] = luma
  }
  return imageData
}

// The ROI rectangles inference actually ran on, straight from firmware
// metadata: 96x96 motion crops in motion_crops mode, the tile grid in
// full_sweep mode. Empty for uploads that predate the field.
export function inferenceRois(metadata) {
  const entries = metadata?.inference_rois
  if (!Array.isArray(entries)) return []
  return entries
    .filter(
      (roi) =>
        Array.isArray(roi) &&
        roi.length === 4 &&
        roi.every((value) => Number.isFinite(value)),
    )
    .map(([x, y, width, height]) => ({ x, y, width, height }))
    .filter((roi) => roi.width > 0 && roi.height > 0)
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('Could not load the detection image.'))
    img.src = src
  })
}

async function generateTiles(imageId, frameWidth, frameHeight, rois) {
  const img = await loadImage(detectionImageUrl(imageId))

  // The stored image should match the frame the ROIs were computed on, but
  // scale defensively in case the server ever resizes it.
  const scaleX = img.naturalWidth / frameWidth
  const scaleY = img.naturalHeight / frameHeight

  return rois.map((roi, index) => {
    const canvas = document.createElement('canvas')
    canvas.width = MODEL_INPUT_SIZE
    canvas.height = MODEL_INPUT_SIZE
    const ctx = canvas.getContext('2d')
    ctx.drawImage(
      img,
      roi.x * scaleX,
      roi.y * scaleY,
      roi.width * scaleX,
      roi.height * scaleY,
      0,
      0,
      MODEL_INPUT_SIZE,
      MODEL_INPUT_SIZE,
    )
    const gray = toGrayscale(
      ctx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
    )
    ctx.putImageData(gray, 0, 0)
    return { ...roi, index, url: canvas.toDataURL('image/png') }
  })
}

// Detection sources that carry COCO-style [x, y, w, h] boxes in frame
// coordinates. `detections` is the Nicla's on-device FOMO output (with a
// `tile` field naming the tile that produced it); the other two are the Pi's
// re-inference results.
const BOX_SOURCES = [
  { metadataKey: 'detections', color: '#5aa9f2' },
  { metadataKey: 'fomo_detections', color: '#f2a65a' },
  { metadataKey: 'yolo_detections', color: '#6fd08c' },
]

function collectBoxes(metadata) {
  const boxes = []
  for (const source of BOX_SOURCES) {
    const entries = metadata?.[source.metadataKey]
    if (!Array.isArray(entries)) continue
    for (const det of entries) {
      const box = Array.isArray(det.bbox) ? det.bbox : det.box
      if (!Array.isArray(box) || box.length !== 4) continue
      boxes.push({
        box,
        label: det.label,
        score: det.score ?? det.confidence,
        sourceTile: typeof det.tile === 'number' ? det.tile : null,
        color: source.color,
      })
    }
  }
  return boxes
}

// Translate frame-coordinate boxes into a tile's 96x96 model-input space:
// keep boxes that overlap the ROI, shift by the ROI origin, then apply the
// same non-uniform stretch the tile itself gets.
function boxesForTile(boxes, roi) {
  const scaleX = MODEL_INPUT_SIZE / roi.width
  const scaleY = MODEL_INPUT_SIZE / roi.height
  return boxes
    .filter(({ box: [x, y, w, h] }) =>
      x < roi.x + roi.width && x + w > roi.x &&
      y < roi.y + roi.height && y + h > roi.y,
    )
    .map((det) => {
      const [x, y, w, h] = det.box
      return {
        ...det,
        x: (x - roi.x) * scaleX,
        y: (y - roi.y) * scaleY,
        width: w * scaleX,
        height: h * scaleY,
      }
    })
}

function TileBoxOverlay({ boxes, tileIndex }) {
  if (boxes.length === 0) return null
  return (
    <svg
      className="tile-overlay"
      viewBox={`0 0 ${MODEL_INPUT_SIZE} ${MODEL_INPUT_SIZE}`}
      preserveAspectRatio="none"
    >
      {boxes.map((det, index) => {
        // A box in the overlap band lands on neighboring tiles too; dash it
        // there so the tile that actually reported it stands out.
        const foreign = det.sourceTile !== null && det.sourceTile !== tileIndex
        return (
          <g key={index} opacity={foreign ? 0.55 : 1}>
            <rect
              x={det.x}
              y={det.y}
              width={det.width}
              height={det.height}
              fill="none"
              stroke={det.color}
              strokeWidth="1"
              strokeDasharray={foreign ? '3 2' : undefined}
              vectorEffect="non-scaling-stroke"
            />
            <text
              x={det.x}
              y={det.y > 8 ? det.y - 1.5 : det.y + det.height + 7}
              fill={det.color}
              fontSize="6.5"
              fontFamily="ui-monospace, monospace"
              paintOrder="stroke"
              stroke="rgba(0,0,0,0.75)"
              strokeWidth="1"
            >
              {det.label ?? 'object'} {(det.score ?? 0).toFixed(2)}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

function downloadTile(tile, imageId, unit) {
  const link = document.createElement('a')
  link.href = tile.url
  link.download = `${imageId}-${unit}-${tile.index}.png`
  link.click()
}

export default function TileSimulator({ imageId, metadata }) {
  const [tiles, setTiles] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const frameWidth = Number(metadata.frame_width)
  const frameHeight = Number(metadata.frame_height)
  if (!(frameWidth > 0 && frameHeight > 0)) return null

  // In idle mode no inference ran, so there is nothing to replicate.
  if (metadata.inference_mode === 'idle') return null

  const isMotionCrops = metadata.inference_mode === 'motion_crops'
  const columns = Number(metadata.grid_columns) || DEFAULT_GRID_COLUMNS
  const rows = Number(metadata.grid_rows) || DEFAULT_GRID_ROWS

  // Prefer the ROIs the firmware says it ran on; recompute the sweep grid
  // only for uploads that predate the inference_rois field. Motion crops
  // cannot be reconstructed, so without recorded ROIs there is no button.
  const recordedRois = inferenceRois(metadata)
  const rois =
    recordedRois.length > 0
      ? recordedRois
      : isMotionCrops
        ? []
        : createTileRois(frameWidth, frameHeight, columns, rows, TILE_OVERLAP_PIXELS)
  if (rois.length === 0) return null

  const unit = isMotionCrops ? 'crop' : 'tile'
  // Three columns keeps crop tiles the same on-screen size as sweep tiles
  // (and matches the firmware's MOTION_MAX_CROPS); extra crops just wrap.
  const gridColumns = isMotionCrops ? 3 : columns
  const frameBoxes = collectBoxes(metadata)

  async function handleGenerate() {
    setBusy(true)
    setError(null)
    try {
      setTiles(await generateTiles(imageId, frameWidth, frameHeight, rois))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const generateLabel = isMotionCrops
    ? `Show motion crops (${rois.length}, ${MODEL_INPUT_SIZE}×${MODEL_INPUT_SIZE} grayscale)`
    : `Simulate Nicla tiles (${rows}×${columns}, ${MODEL_INPUT_SIZE}×${MODEL_INPUT_SIZE} grayscale)`
  const regenerateLabel = isMotionCrops
    ? 'Regenerate motion crops'
    : 'Regenerate Nicla tiles'

  return (
    <section className="tile-simulator">
      <div className="tile-simulator-header">
        <button type="button" className="ghost" onClick={handleGenerate} disabled={busy}>
          {busy ? 'Generating…' : tiles ? regenerateLabel : generateLabel}
        </button>
        {tiles && (
          <button
            type="button"
            className="ghost"
            onClick={() => tiles.forEach((tile) => downloadTile(tile, imageId, unit))}
          >
            Download all
          </button>
        )}
      </div>
      {error && <p className="dashboard-status error">{error}</p>}
      {tiles && (
        <div
          className="tile-grid"
          style={{ gridTemplateColumns: `repeat(${gridColumns}, 1fr)` }}
        >
          {tiles.map((tile) => (
            <figure key={tile.index} className="tile">
              <div className="tile-image-wrap">
                <img
                  src={tile.url}
                  alt={`${unit} ${tile.index}`}
                  title="Click to download"
                  onClick={() => downloadTile(tile, imageId, unit)}
                />
                <TileBoxOverlay
                  boxes={boxesForTile(frameBoxes, tile)}
                  tileIndex={tile.index}
                />
              </div>
              <figcaption>
                {unit} {tile.index} · {tile.width}×{tile.height} @ ({tile.x},
                {tile.y}) → {MODEL_INPUT_SIZE}×{MODEL_INPUT_SIZE}
              </figcaption>
            </figure>
          ))}
        </div>
      )}
    </section>
  )
}
