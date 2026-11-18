# Reproduction

## 1. Environment

    python -m venv .venv
    .venv\Scripts\activate          # Windows
    source .venv/bin/activate        # Linux and macOS
    pip install -r requirements.txt
    pip install -e .

The paper trained on a single RTX 4060: about 24 hours of feature extraction
and 2 hours of training for five folds.

## 2. Data

Follow docs/data.md. The two numbers to measure are data.audio_len and
data.visual_len; for the paper corpus they are 1193 and 7168.

## 3. Check the geometry before training

    python -c "from cdd_avdl.config import Config; from cdd_avdl.models.cdd_avdl import CDDAVDL; import json; m = CDDAVDL(Config.from_yaml('configs/default.yaml')); print(json.dumps(m.describe(), indent=2, default=str))"

Expected for the paper geometry:

    audio_conv_len        1192
    visual_block_lengths  [7161, 7157, 7155]
    fusion_stage_channels [73, 147, 296]
    fusion_out_dim        512
    feature_dim           12
    num_parameters        24106680

The paper reports 23.11 M parameters; 24.11 M is the count of the released
architecture with the AST backbone included.

## 4. Train

One fold:

    python scripts/train.py --config configs/default.yaml --fold 0

Five-fold cross validation:

    python scripts/train.py --config configs/default.yaml --cross-validate

Training uses SGD with momentum 0.9 and learning rate 1e-4, batch size 8, and
NLLLoss on the log-probabilities the model returns. The best checkpoint by
validation loss is written to runs/<name>/best.pt and the per-epoch history to
runs/<name>/history.json. Cross validation additionally writes
runs/<name>/cross_validation.json with the mean, minimum and maximum of every
metric.

## 5. Evaluate

    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint runs/default/best.pt --fold 0

The output contains loss, accuracy, TPR, FPR, precision, recall, F1 and the
confusion matrix counts.

## 6. Reported numbers

The paper reports, for CDD-AVDL on the West China Hospital corpus:

    TPR   98.33 %
    FPR    1.9354 %
    parameters  23.11 M

## 7. Tests

    python -m pytest tests -q

test_shapes.py checks that the paper geometry falls out of the configuration
and that other sequence lengths work. test_equivalence.py transplants the
weights of the released implementation into the refactored model and asserts
that both produce identical log-probabilities.
