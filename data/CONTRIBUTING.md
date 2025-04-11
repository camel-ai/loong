# 🐉 CONTRIBUTING.md — Seed Dataset Contributions

Welcome to **Project Loong 🐉** — our mission is to generate **verifiable synthetic data** to improve the reasoning capabilities of LLM agents. To achieve our goal of generating high quality synthetic data, we need **clean, trustworthy seed datasets** from real humans. That's where you come in.

This doc explains how to contribute seed datasets. These seed datasets are synced with a HuggingFace dataset, which can be found [here](https://github.com/camel-ai/loong).

---

## 📌 What Is a Seed Dataset?

A **seed dataset** is a set of manually created or human-curated examples in a specific domain. Each datapoint must contain:
- A question
- A final answer (the ground truth)
- A rationale (preferably as code that leads to the final answer)
- A `metadata` dict 

These datasets are used to generate large-scale synthetic data via prompting, fine-tuning, and RL. Quality here matters more than quantity.


Note that the final answer must be in a verifiable format or it must be extractable into a verifiable format. To learn more about extractors, read [here](TODO). The format depends on the verifier that you want to later use. For example, if you want to use the `PythonVerifier`, a Python expression would constitute a verifiable format. An example on how to use the `PythonVerifier` can be found [here](TODO).

---

## 📐 Required Schema

Each datapoint **must** follow this schema:

```json
{
  "question": "What is the mean of [1, 2, 3, 4, 5] multiplied by the median of the same list?",
  "final_answer": "15.0",
  "rationale": "import numpy as np\nimport pandas as pd\n\ndata = [1, 2, 3, 4, 5]\nmean = np.mean(data)\nmedian = pd.Series(data).median()\nresult = mean * median\nprint(result)",
  "metadata": {
    "license": "CC BY 4.0",
    "source": "Basic Arithmetic, 1992",
    "domain": "Mathematics",
    "python_version": "3.10.0",
    "required_dependencies": ["numpy==2.2.4", "pandas==2.2.4"],
    "name": "Loong_Mathematics",
    "contributor": "Zeyu Zhang",
    "date_created": "2025-04-07",
    "difficulty": 1,
    "tags": "arithmetic"
  }
}
```

### Field Descriptions:

- `question`: The prompt posed to the LLM. Must be entirely self-contained and not make any assumptions without stating them unambigiuously.
- `final_answer`: Ground-truth answer in a verifiable format. Must match output of rationale.
- `rationale`: Code that generates the answer — this is what gets verified.
- `metadata` (REQUIRED):
  - `license`: Licensing info (e.g., `MIT`, `CC BY 4.0`)
  - `source`: Origin or reference of the data (for example URL)
  - `domain`: Must match one of our supported domains (e.g. `"Physics"`, `"Finance"`)
  - `python_version`: The version of the python. (e.g. `3.11.5`)
  - `required_dependencies`: List of dependencies with version required to run the rationale code
  - `name`: Name of the dataset (e.g., `"Loong_Mathematics"`)
  - `contributor`: Name of the dataset contributor
  - `date_created`: Date when the dataset was created (YYYY-MM-DD format)
- `metadata` (OPTIONAL): Arbitrary metadata: difficulty (numerical scale), tags, units, subdomains, domain-specific metadata, etc.

---

## 🧾 File Structure

Organize your dataset like this:

```
/data/{domain}/{your_dataset_name}/
├── data.json         # your seed dataset, list of dicts
└── README.md         # (optional) explanation of dataset
```

### `data.json` Example (truncated):

```json
[
  {
    "question": "...",
    "final_answer": "...",
    "rationale": "...",
    "metadata": {
      "license": "...",
      "source": "...",
      "domain": "...",
      "python_version": "...",
      "required_dependencies": ["...", "..."],
      "name": "...",
      "contributor": "...",
      "date_created": "..."
    }
  },
  ...
]
```

---

## ✅ Submission Checklist

Before submitting a dataset:

- [ ] Each entry follows the required schema exactly.
- [ ] All answers are **verifiable** via the rationale.
- [ ] Code in `rationale` runs and outputs the `final_answer`.
- [ ] `metadata.license` is permissive (MIT, Apache-2.0, CC-BY, etc.)
- [ ] `source` is real and cited.
- [ ] Domain is clear and consistent.
- [ ] No copyrighted, proprietary, or questionable data.

---

## 🚫 What Not to Submit

- ❌ Proprietary content (e.g. textbooks, private corpora).
- ❌ Data scraped from platforms like StackOverflow, Reddit, or Wikipedia.
- ❌ Natural language rationales with unverifiable answers.
- ❌ Vague or subjective questions (e.g. “What is beauty?”).

---

## 📤 How to Submit

1. Fork the [Loong dataset repo](https://github.com/camel-ai/loong).
2. Add your dataset to:  
   `/data/{domain}/{your_dataset_name}/`
3. Include:
   - `data.json`
   - Optional: `README.md`
4. Open a pull request with a short description.
5. We’ll review for:
   - Format compliance
   - Licensing
   - Verifiability
   - Domain coverage

If accepted, you’ll be credited in `CONTRIBUTORS.md` and your data will contribute to improving the reasoning capabilities of LLM agents, and by extension to achieving AGI.
