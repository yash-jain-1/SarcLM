
# New version for detail_23k.json, text-only, BLEU/ROUGE metrics

# LLaVA text-only evaluation with dummy image, BLEU/ROUGE metrics
import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from PIL import Image
import numpy as np

DATA_PATH = '../datasets/detail_23k.json'
BASE_MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_dummy_image():
    # Create a 224x224 black image (LLaVA default size)
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)

def main():
    print("Loading dataset...")
    data = load_dataset(DATA_PATH)
    print(f"Loaded {len(data)} samples.")

    print(f"Loading LLaVA model: {BASE_MODEL_NAME}")
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
    model = LlavaForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)

    bleu_scores = []
    rouge_scores = []
    rouge = Rouge()
    smooth = SmoothingFunction().method1

    dummy_image = get_dummy_image()

    for item in tqdm(data):
        # Find human and gpt turns
        human_turn = next((c['value'] for c in item['conversations'] if c['from'] == 'human'), None)
        gpt_turn = next((c['value'] for c in item['conversations'] if c['from'] == 'gpt'), None)
        if not human_turn or not gpt_turn:
            continue
        prompt = human_turn.replace('<image>', '').strip()
        reference = gpt_turn.strip()
        # Prepare input for LLaVA
        inputs = processor(prompt, images=dummy_image, return_tensors="pt")
        output_ids = model.generate(**inputs, max_new_tokens=128)
        output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        # Remove prompt from output if present
        if output.lower().startswith(prompt.lower()):
            output = output[len(prompt):].strip()
        # BLEU
        bleu = sentence_bleu([reference.split()], output.split(), smoothing_function=smooth)
        bleu_scores.append(bleu)
        # ROUGE
        rouge_score = rouge.get_scores(output, reference)[0]
        rouge_scores.append(rouge_score)

    # Aggregate metrics
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_rouge = {k: sum(d[k] for d in rouge_scores) / len(rouge_scores) for k in rouge_scores[0]} if rouge_scores else {}

    print(f"\nAverage BLEU: {avg_bleu:.4f}")
    print("Average ROUGE:")
    for k, v in avg_rouge.items():
        print(f"  {k}: {v:.4f}")

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
