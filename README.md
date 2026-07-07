# Emotion-to-Image Generation using Generative AI

A generative AI pipeline that reads free-form text, detects the underlying emotion, and synthesizes a realistic image that visually reflects that emotional context - combining NLP-based emotion classification with diffusion-based image generation.

## Overview

This project bridges natural language understanding and visual generative AI in a single end-to-end pipeline:

1. **Emotion Detection** - A RoBERTa transformer, fine-tuned on the GoEmotions dataset, classifies input text into one of 7 core emotions.
2. **Prompt Generation** - A dynamic prompt generator converts the detected emotion and scene keywords from the text into a rich, descriptive image prompt.
3. **Image Synthesis** - The prompt is passed to a Stable Diffusion model (Realistic Vision V5.1) to generate a photorealistic image.
4. **Verification & Ranking** - Generated images are captioned using BLIP and scored against the original prompt using CLIP, ensuring semantic alignment between text, emotion, and image.

## Architecture

```
 Input Text
     │
     ▼
┌─────────────────────┐
│  RoBERTa Classifier  │  → fine-tuned on GoEmotions (28 → 7 emotions)
│  (Emotion Detection) │
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  Dynamic Prompt       │  → maps emotion + scene keywords → text prompt
│  Generator            │
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  Stable Diffusion     │  → Realistic Vision V5.1 + DPM++ scheduler
│  (Image Generation)   │
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  BLIP Captioning +    │  → validates semantic alignment
│  CLIP Ranking         │
└─────────┬────────────┘
          ▼
   Final Ranked Image(s)
```

## Repository Structure

```
├── emotion-image-pipeline.ipynb      # Stage 1: Train & evaluate the emotion classifier
├── emotion-image-generation.ipynb    # Stage 2: Generate & rank emotion-driven images
└── README.md
```

## Notebook 1 - Emotion Classification (`emotion-image-pipeline.ipynb`)

Fine-tunes a RoBERTa model for 7-class emotion classification.

Step
Description

Dataset
[GoEmotions](https://huggingface.co/datasets/go_emotions) (58k Reddit comments, 28 fine-grained emotions)

Label mapping
28 original labels grouped into 7 core emotions

Model
`roberta-base` (`RobertaForSequenceClassification`)

Training
AdamW optimizer, linear warmup scheduler, class-weighted cross-entropy loss

Evaluation
Classification report + F1 score, training/validation loss plots

Output
Fine-tuned emotion classifier checkpoint used in Notebook 2

## Notebook 2 — Image Generation (`emotion-image-generation.ipynb`)

Uses the fine-tuned emotion model to drive image synthesis.

Step
Description

Emotion detection
Loads fine-tuned RoBERTa model from Notebook 1

Prompt generation
Keyword-based scene mapping (e.g. "lecture" → "student after lecture") combined with detected emotion

Image generation
`SG161222/Realistic_Vision_V5.1_noVAE` via `diffusers`, using DPM++ Karras scheduler for higher quality in fewer steps

Captioning
BLIP (`Salesforce/blip-image-captioning-base`) generates captions for each output image

Ranking
CLIP scores each image against the prompt to select the best match

Output
Ranked images + saved JSON result with prompt, caption, and scores

## Tech Stack

- **NLP:** Hugging Face Transformers, RoBERTa
- **Image Generation:** Diffusers, Stable Diffusion (Realistic Vision V5.1), DPM-Solver++
- **Verification:** BLIP (image captioning), OpenCLIP (image-text ranking)
- **Training:** PyTorch, scikit-learn
- **Dataset:** GoEmotions (Google)

## Setup

```
pip install transformers==4.40.0 datasets scikit-learn accelerate
pip install transformers==4.44.2 diffusers==0.30.3 huggingface_hub==0.24.7 \
            accelerate==0.34.2 peft==0.13.2 safetensors==0.4.5 \
            open-clip-torch==2.24.0 ftfy==6.2.3 Pillow matplotlib
```

> A CUDA-enabled GPU is strongly recommended for both training and image generation.

## Usage

1. Run `emotion-image-pipeline.ipynb` end-to-end to train and save the emotion classifier.
2. Run `emotion-image-generation.ipynb`, pointing `EMOTION_MODEL_PATH` to the checkpoint saved in step 1.
3. Provide input text — the pipeline will detect the emotion, generate a prompt, synthesize candidate images, caption and rank them, and output the best match.

**Example:**

```
result = run_pipeline("I just finished my final exam and I can't stop smiling!")
# → Detected emotion: joy
# → Generated prompt: "student smiling after exam, joyful expression, realistic photo"
# → Output: ranked images + captions + CLIP scores
```

## Future Improvements

- Expand emotion taxonomy beyond 7 classes for finer-grained control
- Fine-tune the diffusion model (e.g. via LoRA) directly on emotion-labeled image datasets
- Add a web-based demo (Gradio/Streamlit) for interactive use
- Batch evaluation metrics across a labeled emotion-image test set
