# Secure multiparty computations in floating-point arithmetic

## Install

Clone the repo

```
git clone git@github.com:fairinternal/mpcfp.git
```

Dependencies:

- [PyTorch](https://pytorch.org/get-started/locally/).
- Matplotlib (`conda install matplotlib`)
- Seaborn (`conda install seaborn`)
- [Latex](https://www.latex-project.org/get/) for producing plots.

## Run


To run a plain-text baseline:

```
python binary_classification.py --plaintext
```

To run in MPC:

```
./launch_private.sh <arg1> <arg2> ...
```

For multi-class classification use `multi_classification.py` instead.

To see the list of available options pass `--help` to the corresponding script:

```
python binary_classification.py --help
```

## Reproduce Results

After installing the dependencies, all results and figures in "Secure
multiparty computations in floating-point arithmetic" can be generated with the
`run_all.sh` script. From the repo root run:

```
./run_all.sh
```

Note, this is a long-running job that may take a day or more to complete. The
figures will be generated as PDFs and saved in `./results`.


### List of figures

The following is a key mapping the figure number as found in the manuscript to
the corresponding file name in `results`:

|Figure Number | File name
| --- | --- 
| 1 | reciprocal.pdf
| 2 | invsqrt.pdf
| 3 | inv8root.pdf
| 4 | absval.pdf
| 5 | synth_widths_weightdiffs.pdf
| 6 | synth_terms_weightdiffs.pdf
| 7 | mnist_widths_train_loss.pdf
| 8 | mnist_widths_test_loss.pdf
| 9 | mnist_multiclass_accuracy.pdf
| 10 | covtype_widths_train_loss.pdf
| 11 | covtype_widths_test_loss.pdf
| 12 | covtype_multiclass_accuracy.pdf


## License

mpcfp is MIT licensed, as found in the LICENSE file.
