# OmniParser

OCR utility with PaddleOCR and EasyOCR support.


For chinese user:
```bash
export HF_ENDPOINT=https://hf-mirror.com

```
Ensure you have the V2 weights downloaded in weights folder (ensure caption weights folder is called icon_caption_florence). If not download them with:

```bash
   for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do poetry run huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done
```