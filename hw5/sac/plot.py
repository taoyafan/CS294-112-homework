import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

"""
Using the plotter:

Call it from the command line, and supply it with logdirs to experiments.
Suppose you ran an experiment with name 'test', and you ran 'test' for 10 
random seeds. The runner code stored it in the directory structure

    data
    L test_EnvName_DateTime
      L  0
        L log.txt
        L params.json
      L  1
        L log.txt
        L params.json
       .
       .
       .
      L  9
        L log.txt
        L params.json

To plot learning curves from the experiment, averaged over all random
seeds, call

    python plot.py data/test_EnvName_DateTime --value AverageReturn

and voila. To see a different statistics, change what you put in for
the keyword --value. You can also enter /multiple/ values, and it will 
make all of them in order.


Suppose you ran two experiments: 'test1' and 'test2'. In 'test2' you tried
a different set of hyperparameters from 'test1', and now you would like 
to compare them -- see their learning curves side-by-side. Just call

    python plot.py data/test1 data/test2

and it will plot them both! They will be given titles in the legend according
to their exp_name parameters. If you want to use custom legend titles, use
the --legend flag and then provide a title for each logdir.

"""

def plot_data(data, value="AverageReturn", name="AverageReturn"):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot(data=data, time="Iteration", value=value, unit="Unit", condition="Condition")
    plt.ylabel(name)
    plt.legend(loc='best')#.draggable()
    plt.show()


def get_datasets(fpath, condition=None):
    unit = 0
    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'log.txt' in files:
            param_path = open(os.path.join(root, 'params.json'))
            params = json.load(param_path)
            exp_name = params['exp_name']
            
            log_path = os.path.join(root, 'log.txt')
            experiment_data = pd.read_table(log_path)
            if 'real' in log_path:
                experiment_data['MaxEpReturn'] = experiment_data['MaxEpReturn']*5/5000 - 4
            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
                )        
            experiment_data.insert(
                len(experiment_data.columns),
                'Condition',
                condition or exp_name
                )

            datasets.append(experiment_data[0:500])
            unit += 1

    return datasets


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--value', default='LastEpReturn', nargs='*')
    parser.add_argument('--name', nargs='*')
    args = parser.parse_args()

    use_legend = False
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True

    data = []
    if use_legend:
        for logdir, legend_title in zip(args.logdir, args.legend):
            data += get_datasets(logdir, legend_title)
    else:
        for logdir in args.logdir:
            data += get_datasets(logdir)

    if isinstance(args.value, list):
        values = args.value
    else:
        values = [args.value]

    if args.name is None:
        args.name = args.value

    if args.name[0] == 'alpha':
        for i, da in enumerate(data):
            data[i].alpha = da.alpha.apply(lambda x: float(x[1:-1]) if type(x) is str and '[' in x else float(x))
    for value in values:
        plot_data(data, value=value, name=args.name[0])


if __name__ == "__main__":
    main()
