# Qwen Distillation Tutorial

An interactive, dark-mode tutorial explaining three ways to distill a Qwen data-analysis agent: successful teacher transcripts, teacher token probabilities, and on-policy teacher supervision. It includes the reported experiment results, mathematical intuition, source figures, a step-through training diagram, and a temperature explorer.

**[Read the live tutorial](https://az9713.github.io/qwen-distillation-tutorial/)**

## Source

Based on [“I Trained Qwen 4B with Distillation to Catch Qwen 27B” by Prompt Engineering](https://www.youtube.com/watch?v=hOitZIkMT2Y) and the accompanying [SmolDataEnvs experiment repository](https://github.com/PromtEngineer/smoldataenvs-grpo-spark). Timestamped references and additional background sources appear in the tutorial. Reported measurements belong to the original experiment; this project is an educational explanation, not an independent training reproduction. Video screenshots are attributed excerpts; interactive diagrams and explanatory examples are teaching additions.

## View and edit

Open `index.html` in a browser, or serve this directory:

```sh
python -m http.server 8000
```

Visit `http://localhost:8000`. The site is static, with local images, interactive JavaScript, and bundled KaTeX math rendering. No API keys, model downloads, or GPU are needed.

Edit `tutorial.md`, `assets/site.css`, or the HTML visualizations in `assets/`, then rebuild:

```sh
python -m pip install -r requirements.txt
python build.py
```

GitHub Pages publishes the committed `index.html` from the root of `main`. Commit the rebuilt HTML with any source changes.
