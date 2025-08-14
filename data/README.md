# Custom Dataset

This directory should contain the custom domain-specific dataset used for fine-tuning the model.

## Data Format

The model expects the data to be in a **JSON Lines (`.jsonl`)** format. Each line in the file should be a valid JSON object representing a single data entry.

For instruction-based fine-tuning, each JSON object should contain a `text` field formatted as a prompt. The structure should be clear and consistent, allowing the model to learn the desired response pattern.

### Example (`custom_dataset.jsonl`):

```json
{"text": "<s>[INST] What is the capital of France? [/INST] Paris</s>"}
{"text": "<s>[INST] Who wrote 'To Kill a Mockingbird'? [/INST] Harper Lee</s>"}
```

Replace the example `custom_dataset.jsonl` file with your actual domain-specific data following this format.