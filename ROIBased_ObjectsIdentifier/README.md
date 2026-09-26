## ROI Based Objects Identifier with Open CV


A simple prototype for:

1. Show an object to a camera.
2. Capture multiple views.
3. Convert each view into an embedding vector.
4. Save vectors to `object_memory.npz`.
5. Later show an object again.
6. Compare the new embedding with the stored embeddings.
7. Return MATCH or UNKNOWN.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Learn

```bash
python learn_mode.py --camera 0 --samples 24 --interval 0.35
```

Enter an object name, then:

- Press SPACE to choose an ROI.
- Slowly rotate the object.
- Each SPACE after ROI selection captures one sample.
- At least several different views are recommended.

The learned data is saved to:

```text
object_memory.npz
```

## Inference

```bash
python inference_mode.py --camera 0 --threshold 0.80
```

Press SPACE and select the object.

The program compares the query embedding with all stored views.

## Notes

This prototype uses pretrained MobileNetV3-Small feature embeddings.
It is intended as a starting point, not a production-grade object
re-identification model.

For a stronger system later:

- replace the embedding model with a model trained for instance retrieval;
- add automatic object detection/tracking;
- calibrate the threshold using positive and negative samples;
- quantize/export the embedding model to ONNX/TFLite;
- move inference to an ARM64/NPU/MCU target.
