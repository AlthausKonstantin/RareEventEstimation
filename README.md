# rareeventestimation

Estimate rare events with consensus based sampling and other methods.

## Installation

```bash
$ pip install rareeventestimation
```

## Usage

Have a look at the juypter notebooks in `docs/figures_thesis`.
These notebooks reproduce the plots an tables in my thesis.

The data for these notebooks is hosted on [archive.org](https://archive.org/details/konstantinalthaus-rareeventestimation-data) and will be loaded automatically.
The data is encoded as `.json` files and you can have look at those online before loading them.
Alternatively you can compile all the data yourself by running the scripts in
`docs/figures_thesis/data`.
Each notebooks contains an inactive cell that can load and aggregate
the locally produced data.

## License

`rareeventestimation` was created by Konstantin Althaus. It is licensed under the terms of the MIT license.

## Credits

This package contains code I have not written myself.

### Code from the Engineering Risk Analysis Group, Technical University of Munich
All files in `rareeventestimation/src/era` and `rareeventestimation/src/sis` are
written by the [ERA Group](https://www.cee.ed.tum.de/era/era-group/) and
licensed under the MIT license. I have added minor changes.

### Code from Dr. Fabian Wagner
All files in `rareeventestimation/src/enkf` and `src/rareeventestimation/problem/diffusion.py` are written by [Dr. Fabian Wagner](https://www-m2.ma.tum.de/bin/view/Allgemeines/FabianWagner). I have added minor changes.

### Stackoverflow
I have used several snippets from the almighty community on [stackoverflow](https://stackoverflow.com).

`rareeventestimation` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).


