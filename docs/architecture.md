# Proposed MVP architecture

Status: planned architecture; the current MVP files are scaffolding.

```mermaid
flowchart LR
    IN([Input: image file]) --> UP[Streamlit upload]

    subgraph PIPE["Python pipeline"]
        direction LR
        D[RT-DETR detection] --> C[Crop plates]
        C --> O[PaddleOCR]
        O --> R[Prepare results and status]
    end

    UP --> D
    R --> VIEW[Streamlit results view]
    VIEW --> OUT([Output: annotated image, plate crops, text and status])

    R --> SAVE[Storage layer]
    SAVE --> DB[(SQLite metadata)]
    SAVE --> FILES[Crop files]
```

1. The image file enters through Streamlit upload. 
2. The Python pipeline groups the detection, cropping, OCR, and result-preparation components. It returns structured data to the results view, which presents the annotated image, plate crops, text, and status to the user. 
3. Storage branches from the structured results.

The detector returns boxes in original-image coordinates. The pipeline validates
and clamps coordinates before cropping. Visualization draws on a separate copy.
Load models once and reuse them across images.

Results should preserve raw and normalized OCR text, separate detection and OCR
scores, bounding boxes, status, source ID, timing, and model versions. Scores are
model outputs, not calibrated probabilities of a correct plate reading.

Storage saves observation metadata in SQLite and crop references on disk. UI
reruns must not automatically create duplicate observations. The prediction CLI
can call the same pipeline; it is omitted from this diagram to keep the primary
user journey clear. Later API or video entry points can also reuse the pipeline;
tracking and event logic remain future work.
