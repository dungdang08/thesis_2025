# Attribution — Bayes DistNet

This folder is a modified copy of **Bayes DistNet**, <https://github.com/tuero/BayesianDistNet>,
the code accompanying:

> J. Tuero, M. Buro. *Bayes DistNet – A Robust Neural Network for Algorithm Runtime Distribution
> Predictions*. AAAI 2021, 35(14):12418–12426.

The original repository states no license; all rights to the original code remain with its authors.
The upstream README is kept as [README.md](README.md).

`src/layers/BBBLinear.py`, `src/layers/BBBLinear2.py` and `src/layers/misc.py` originate from
[PyTorch-BayesianCNN](https://github.com/kumar-shridhar/PyTorch-BayesianCNN) (MIT License,
Copyright (c) Kumar Shridhar), as noted in their file headers.

Please cite both papers (BibTeX in [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).

## Modifications in the bachelor thesis (Hadar Shavit M.Sc (supervisor), Hoang Dung Dang (thesis student))

The full diff against upstream is in [fix.patch](fix.patch). Summary:

- `src/models/bayes_distnet.py`: added bias terms
- `src/layers/BBBLinear2.py`: new variant of `BBBLinear` used by the model
- `src/create_dataframes.py` and others: fixes for newer library versions, create export folders automatically
