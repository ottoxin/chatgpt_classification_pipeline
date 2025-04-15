# ChatGPT Classification Pipeline

This repository provides a complete framework for using OpenAI's GPT models to classify text into themes or custom categories. It includes both **theoretical foundations** and a **fully working engineering pipeline** using the new [OpenAI Responses API](https://platform.openai.com/docs/guides/structured-outputs).

## Repo Contents

| File                                 | Description                                |
|--------------------------------------|--------------------------------------------|
| `chatgpt_general_classification.ipynb` | Theme classification with confidence        |
| `chatgpt_account_type_classification.ipynb` | Specialized task: classify Twitter bios     |
| `README.md`                          | This documentation                         |

---

## Part 1: Theory

### 1. Data Collection
- Collect relevant textual data for the classification task (e.g., tweets, forum posts, news headlines).

---

### 2. Prompt Engineering

#### 2.1. Prompt Building

**1. Task Instruction**  
Clearly define the task.  
- Example: *“Classify the sentiment of the following text as Positive, Negative, or Neutral.”*

**2. Contextual Framing**  
Briefly describe the purpose or context.  
- Example: *“You are analyzing political tweets about a recent policy change.”*

**3. Label Definitions**  
Clarify what each label means.  
- Example:  
  - **Support** = Expresses agreement  
  - **Oppose** = Expresses disagreement  
  - **Neutral** = No clear stance  

**4. In-context Learning (Few-shot Examples)**  
Provide a few labeled examples.  
> Tweet: “This policy is amazing!” → Label: Support  
>  
> Tweet: “I can’t believe they passed this!” → Label: Oppose  
>  
> Tweet: “I read about the policy but haven’t formed an opinion.” → Label: Neutral  

**5. Output Format Specification**  
- Example:  
  *“Respond with one of the following labels: Support, Oppose, Neutral.”*

#### 2.2. Prompt Optimization
Use [example selection techniques](https://arxiv.org/html/2409.01466v2) to boost performance.

#### 2.3. Prompt Tuning
Use prompt embeddings or reformulations when zero-shot fails.

#### 2.4. Prompt Stability Testing
Refer to [this paper](https://arxiv.org/pdf/2407.02039) to test consistency across variants.

| Prompt     | Run 1   | Run 2   | Run 3   |
|------------|---------|---------|---------|
| Baseline   | Neutral | Neutral | Neutral |
| Prompt A   | Neutral | Neutral | Negative |
| Prompt B   | Negative | Neutral | Negative |
| Prompt C   | Neutral | Neutral | Neutral |

---

### 3. Model Selection

- Recommended: `gpt-4o`, `gpt-4o-mini`
- Temperature: 0 - 0.3 for deterministic outputs

---

### 4. In-context learning

- **Few-shot or Zero-shot** via prompt design
- Optional **fine-tuning** for specific domains

---

### 5. Evaluation

- Metrics: Accuracy, F1, Stability Variance
- Manually inspect edge cases
- Use `jsonl` or CSV formats to evaluate at scale

---

## Part 2: Engineering

### General Theme Classification (`chatgpt_general_classification.ipynb`)

This notebook uses the `responses.create()` method with **structured JSON output** to classify a text into themes with **confidence scores**.

#### Schema Example

```json
{
  "themes": [
    {
      "label": "Environment",
      "confidence": 0.92
    },
    {
      "label": "Politics",
      "confidence": 0.87
    }
  ],
  "reasoning": "Mentions carbon emissions (Environment) and legislation (Politics)."
}
```

#### Python Code

```python
import json
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-2024-08-06",
    input=[
        {"role": "system", "content": "You classify text by theme with confidence scores."},
        {"role": "user", "content": (
            "Classify the following text into themes: Politics, Economy, Environment, Health, Technology, Culture, Other.\n\n"
            "Text: The government passed a new bill to regulate carbon emissions and promote clean energy investments.\n"
            "Include confidence scores and explain the reasoning."
        )}
    ],
    text={
        "format": {
            "type": "json_schema",
            "name": "theme_classifier_with_confidence",
            "description": "Theme classification with confidence and reasoning",
            "schema": {
                "type": "object",
                "properties": {
                    "themes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "enum": ["Politics", "Economy", "Environment", "Health", "Technology", "Culture", "Other"]
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0
                                }
                            },
                            "required": ["label", "confidence"]
                        }
                    },
                    "reasoning": {
                        "type": "string"
                    }
                },
                "required": ["themes", "reasoning"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
)

output = json.loads(response.output_text)
print(output)
```

---

### Special Case: Account Type Classification (`chatgpt_account_type_classification.ipynb`)

This notebook adapts the pipeline to infer **Twitter/X account types** (e.g., Individual, Bot, Organization) using profile bios.

---

## References

- [Prompt Optimization](https://arxiv.org/html/2409.01466v2)  
- [Prompt Stability Evaluation](https://arxiv.org/pdf/2407.02039)
