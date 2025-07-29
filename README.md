---

## 🧠 LLM Fine-tuning Integration (Axolotl + phi-2)

This project includes a sub-experiment for fine-tuning the phi-2 model using CoT-style trajectory prediction.

📂 Path: `experiments-linux/`

- ✅ Training format: Alpaca-style JSONL with (instruction, input, output)
- ✅ Model: `phi-2`, LoRA fine-tuning via Axolotl
- ✅ Checkpoints and config: See `outputs/phi2-CoT-finetune/`, `configs/`

To reproduce fine-tuning:

```bash
cd experiments-linux/
python -m axolotl.cli.train config_phi2_cot.yml
```
