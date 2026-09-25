# AGRVF: Automated Ground-truth and Reasoning Verification Framework

AGRVF verifies consumer product claims (skincare, supplements, household items) against retrieved evidence rather than trusting marketing language. Given a free-text claim, it identifies the referenced product, retrieves evidence from a local knowledge base and, when necessary, the live web, scores every evidence item's credibility with AECS (Adaptive Evidence Credibility Scoring), and produces one of three verdicts, SUPPORTED, REFUTED, or INSUFFICIENT_UNVERIFIABLE, together with a confidence score and a generated explanation.

The system is built around two design decisions that most fact-verification pipelines do not make: evidence credibility is scored by a category-adaptive, multi-factor formula rather than treated as uniform, and the verdict rule can abstain when evidence is insufficient rather than being forced to always classify.

## Architecture

The pipeline is organized into four sequential, independently checkpointed layers.

- **Layer A, data construction.** Mines candidate product claims from Amazon Reviews'23 using a TF-IDF plus logistic regression classifier, and processes the reference datasets used downstream.
- **Layer B, claim and product understanding.** Product identification with BGE-large embeddings and FAISS, claim understanding and category classification with Qwen3-1.7B, deterministic product and claim extraction for arbitrary free text.
- **Layer C, local evidence retrieval.** Retrieves evidence from the Amazon product catalog, Open Beauty Facts ingredient records, and FEVER and SciFact scientific passages. No web access at this stage.
- **Layer D, web evidence, AECS, and reasoning.** Resolves and scrapes the official product page when the product is not in the local catalog, scores every evidence item with AECS, classifies stance per item, determines the verdict from AECS-weighted evidence mass, and generates the final explanation.

## Repository structure

```
notebooks/                       the two Kaggle notebooks, run in order
data/processed/
  claims/                        final claim dataset and sentence dataset (Layer A)
  amazon/                        product catalog (Layer A)
  fever/                         FEVER subset used as scientific evidence (Layer A)
  scifact/                       SciFact subset used as scientific evidence (Layer A)
  openbeautyfacts/                ingredient records (Layer A)
  layer_b/                       structured claim and product understanding output
  layer_d/                       final verdicts, confidence, reasoning, provenance
results/
  layer_c/                       local evidence retrieval output
  figures/                       chart images used in the paper
  data_validation/               data quality and validation reports (Layer A)
models/                          trained claim-candidate classifier and TF-IDF vectorizer
paper/                           the paper on this project
```

## Running this project

1. Open `notebooks/agrvf-project-layera.ipynb` in Kaggle, attach the raw source datasets, run top to bottom. Produces everything under `data/processed/claims`, `amazon`, `fever`, `scifact`, `openbeautyfacts`, plus the trained classifier in `models/`.
2. Open `notebooks/agrvf.ipynb`, attach the Layer A output as a Kaggle dataset, run top to bottom. Produces `data/processed/layer_b`, `results/layer_c`, `data/processed/layer_d`, and `results/figures`.

Both notebooks checkpoint their heavy stages and detect existing output before recomputing, so an interrupted run resumes instead of starting over.

## Dataset scale

Layer A and Layer B ran across the complete corpus, 106,073 candidate claims mined from Amazon Reviews'23 All Beauty. Layer D, which requires live evidence scoring and LLM-based stance classification, was evaluated on a random sample of 5,000 claims drawn with a fixed seed.

## Results summary

Verdict distribution on the 5,000-claim sample: SUPPORTED 1455, REFUTED 758, INSUFFICIENT_UNVERIFIABLE 2787.
Confidence distribution: High 2474, Medium 1077, Low 1449.
Claim-candidate classifier: 85 percent accuracy, F1 0.79, against a 36.2 percent accuracy heuristic baseline.

## Known limitations

- Official web page resolution is unreliable in this environment. The search backend returns an anti-bot challenge response rather than results when queried from a datacenter IP, confirmed by inspecting the raw HTTP response. This does not affect the results above, since 100 percent of the sampled claims resolved against the local catalog and never required the web fallback path.
- A qualitative review found that a measurable subset of REFUTED verdicts were driven by SciFact scientific evidence with only coincidental topical overlap to the claim rather than genuine relevance. This is disclosed and quantified in the paper's limitations section. A post hoc correction was attempted and reverted after validation showed it overcorrected; a reliable fix requires recalibrating the claim-relevance component of AECS for this evidence type specifically, left as future work.
- The ingredient function reference table is a curated list of approximately 50 common cosmetic ingredients. Ingredients outside this list fall back to retrieved scientific evidence where available, which inherits the relevance limitation above.

## Data sources

This repository contains only derived outputs produced by the authors. Raw third-party datasets are not redistributed here and should be obtained from their original sources.

**Amazon Reviews'23**, used for both review text and item metadata.
https://amazon-reviews-2023.github.io/

```
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

**FEVER**, used as one of two scientific/factual evidence sources in Layer C.
https://fever.ai/dataset/fever.html

```
@inproceedings{Thorne18Fever,
  author = {Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Mittal, Arpit},
  title = {{FEVER}: a Large-scale Dataset for Fact Extraction and {VERification}},
  booktitle = {NAACL-HLT},
  year = {2018}
}
```

**SciFact**, used as the second scientific evidence source in Layer C.
https://huggingface.co/datasets/allenai/scifact

```
@inproceedings{wadden-etal-2020-fact,
  title = "Fact or Fiction: Verifying Scientific Claims",
  author = "Wadden, David and Lin, Shanchuan and Lo, Kyle and Wang, Lucy Lu and
            van Zuylen, Madeleine and Cohan, Arman and Hajishirzi, Hannaneh",
  booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
  month = nov,
  year = "2020",
  address = "Online",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2020.emnlp-main.609",
  doi = "10.18653/v1/2020.emnlp-main.609",
  pages = "7534--7550",
}
```

**Open Beauty Facts**, used as the ingredient evidence source in Layer C.
https://world.openbeautyfacts.org


## Citation

If you use this work, please cite the accompanying paper in `paper/`. A citable code and data DOI will be linked here once archived on Zenodo.

## License

Code in this repository is released under the MIT License. Raw third-party datasets referenced above retain their own original licenses and are not covered by this license.
