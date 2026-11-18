# assets

Figures and tables reproduced from the paper, used by the top-level README to
document what the code implements.

| file | source | what it shows |
| --- | --- | --- |
| fig2_framework.png | Fig. 2 | the overall CDD-AVDL pipeline, capture to probability |
| fig3_model.png | Fig. 3 | the four stages: source features, dimension reduction, time-sequential features, fusion |
| table2_fusion.png | TABLE II | the three convolution-layer fusion stages, patch sizes and outputs |
| table4_performance.png | TABLE IV | performance of every model and source-feature combination |
| table5_complexity.png | TABLE V | FLOPs and parameter size per configuration |
| table7_comparison.png | TABLE VII | CDD-AVDL against classical classifiers and audio-visual baselines |

## Provenance

All six images are page crops of the published article, rendered at 300 dpi
from the author PDF and cropped to the figure or table boundary:

    L. Liu, S. Chen, K. Chen, J. Xu, X. Chen,
    Catching the Blackdog Easily: A Convenient Depression Diagnosis Method
    based on Audio-Visual Deep Learning,
    IEEE Transactions on Affective Computing, 2025.
    doi: 10.1109/TAFFC.2025.3571697

They are included so that a reader can compare the implementation against the
published description without opening the PDF. Copyright remains with IEEE;
the images are not covered by the MIT license that applies to the code in this
repository. Remove them if you redistribute this repository in a context where
that is not acceptable, and cite the paper instead.