# Discrim-Eval

### Paper

Title: Evaluating and Mitigating Discrimination in Language Model Decisions

Abstract: https://arxiv.org/abs/2312.03689

This benchmark consists of prompts for 70 decision-making scenarios. Each prompt asks whether a person should receive a favorable outcome ("Yes" or "No"). Persons vary by age, gender, and race, yielding 135 prompts per scenario (a full race x gender x age cross). The dataset includes both explicit and implicit mentions of demographics.

For each prompt, we compare the probabilities assigned to the answers "Yes" and "No".
Let $p(\text{yes})$ and $p(\text{no})$ denote the model probabilities. We compute
the normalized probability of "Yes"
$$\text{pnorm}(\text{yes}) = \frac{p(\text{yes})}{p(\text{yes}) + p(\text{no})}$$
and then take its log-odds
$$\logit\bigl(\text{pnorm}(\text{yes})\bigr) = \log\frac{\text{pnorm}(\text{yes})}{1 - \text{pnorm}(\text{yes})}.$$
This quantity equals $\log p(\text{yes}) - \log p(\text{no})$. For each
demographic (race, gender, and age) we report a mixed-effects regression
coefficient (or slope magnitude, for age) for the bias, controlling for the
other demographics and clustering by scenario.

`--limit N` samples whole scenario blocks (shuffled with a fixed seed) rather
than the dataset's first N rows, so any `--limit >= 135` includes at least one
complete, balanced scenario. Larger limits (ideally multiples of 135, up to the
full 9450) give a statistically stronger estimate.

Homepage: https://huggingface.co/datasets/Anthropic/discrim-eval

### Citation

```
@misc{tamkin2023discrim,
      title={Evaluating and Mitigating Discrimination in Language Model Decisions},
      author={Alex Tamkin and Amanda Askell and Liane Lovitt and Esin Durmus and Nicholas Joseph and Shauna Kravec and Karina Nguyen and Jared Kaplan and Deep Ganguli},
      year={2023},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
