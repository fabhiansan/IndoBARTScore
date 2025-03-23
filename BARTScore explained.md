# BARTScore Explained

## What is BARTScore?

BARTScore is an evaluation metric for natural language generation (NLG) tasks that frames text evaluation as a text generation problem. Introduced in the paper [BARTScore: Evaluating Generated Text as Text Generation](https://arxiv.org/abs/2106.11520) by Yuan, Neubig, and Liu (NeurIPS 2021), it offers a new approach to automated text evaluation.

Unlike traditional metrics like BLEU, ROUGE, or METEOR which focus on n-gram matching, BARTScore leverages the power of pre-trained language models (specifically BART) to assess text quality based on generation probability.

## How BARTScore Works

### Conceptual Framework

BARTScore represents a paradigm shift in automatic evaluation by formulating **evaluation as a generation task**. While previous approaches have framed evaluation as:

- **Matching task** (e.g., ROUGE, BERTScore): Measuring semantic equivalence using token-level matching in different representation spaces
- **Regression task** (e.g., BLEURT): Learning to predict human judgments via supervised regression
- **Ranking task** (e.g., COMET): Learning a scoring function that ranks better hypotheses higher than worse ones

BARTScore instead treats evaluation as a **text generation probability problem**. It asks: "How likely would the pre-trained model generate text B given text A?"

### Technical Implementation

At its core, BARTScore uses the BART model to measure the log-likelihood of generating a target text given a source text. 

The process works as follows:

1. A pre-trained BART model encodes the source text
2. The model then calculates the log-likelihood of generating the target text token by token
3. These log-likelihoods are averaged to produce a final score

Mathematically, this is represented as:

```
BARTScore(src → tgt) = log P(tgt | src)
```

The higher this score (closer to 0 as log probabilities are negative), the better the quality of the generated text according to the model.

## Variants of BARTScore

BARTScore comes in several variants depending on the pre-training and fine-tuning:

1. **BART-base**: Uses the original BART model pre-trained on large text corpora
2. **BART-CNN**: Fine-tuned on the CNN/DailyMail summarization dataset
3. **BART-ParaBank**: Fine-tuned on the ParaBank2 paraphrase dataset (recommended version)

Additionally, BARTScore can be calculated in different directions:

- **src → tgt**: Measures faithfulness/coverage (does the generated text cover the source content?)
- **tgt → src**: Measures precision (is the generated content in the source?)
- **Bidirectional**: Combines both directions for a balanced score

## Using BARTScore

BARTScore is straightforward to implement using the provided code. Here's a basic example:

```python
from bart_score import BARTScorer

# Initialize with CNN/DailyMail fine-tuned model
bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')

# For the ParaBank version (recommended)
# bart_scorer.load(path='bart.pth')

# Score a source-target pair
scores = bart_scorer.score(['This is the source text.'], ['This is the target text.'])
```

For multiple references, use the multi-reference scoring function:

```python
srcs = ["I'm super happy today.", "This is a good idea."]
tgts = [["I feel good today.", "I feel happy today."], ["Not bad.", "Sounds like a good idea."]]
scores = bart_scorer.multi_ref_score(srcs, tgts, agg="max")
```

## Advantages of BARTScore

BARTScore offers several advantages over traditional metrics:

1. **Contextual Understanding**: Leverages BART's deep contextual understanding rather than surface-level matching
2. **Directionality**: Can measure both precision and recall-like qualities through bidirectional scoring
3. **Flexibility**: Works across multiple NLG tasks (summarization, translation, data-to-text, etc.)
4. **Customizability**: Can be fine-tuned on domain-specific data for better performance
5. **Multi-reference Support**: Can handle multiple references for each source text

## Applications

BARTScore has been successfully applied to evaluate:

- Summarization (shown to correlate well with human judgments)
- Machine translation
- Data-to-text generation
- Other text generation tasks

## Interpreting Scores

Since BARTScore uses log-likelihood, the scores are always negative (as probabilities are between 0 and 1). 

- Higher scores (closer to 0) indicate better quality
- Lower scores (more negative) indicate worse quality

For example, if Text A receives a score of -1.5 and Text B receives a score of -5.0, the model considers Text A to be of better quality.

## Limitations

BARTScore does have some limitations:

1. It depends on the quality and biases of the pre-trained BART model
2. It may not capture all aspects of text quality that humans value
3. The scores are not normalized and can be difficult to interpret in absolute terms

Despite these limitations, BARTScore represents a significant advancement in automatic evaluation methods for NLG tasks and has been shown to correlate well with human judgments across multiple tasks.
