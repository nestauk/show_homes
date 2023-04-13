# Heat pump show homes

<img src="./docs/data_tool_example.png">

## Table of Content<a id='top'></a>

- [Introduction](#intro)
- [Data work](#datawork)
  - [Speed testing](#speed)
  - [Modelling network of show homes](#model)
  - [Future work](#future)
- [Setup](#setup)
- [How to use](#setup)
- [Contributor guidelines](#contributor)

---

## Introduction<a id='intro'></a>

[[back to top]](#top)

While more and more people become aware of heat pumps as a low-carbon techonology, only few people in the UK have actually seen one in action. Having the opportunity to visit a functioning heat pump in a real home - not a show room - could help many potential adopters to get a better understanding for heat pumps and what it is like to have one at home.

The Sustainable Future team at Nesta tested the idea of show homes for heat pumps with two show homes in London and Glasgow with promising results. Both visitors and hosts were extremely happy with the experience.

Find out more about the project and sign up for the next round of show home events on <a href="visitaheatpump.com/" title="VisitAHeatPump">visitaheatpump.com</a>.

---

## Data work<a id='datawork'></a>

[[back to top]](#top)

### Speed testing<a id='speed'></a>

This data work supports the 'real-life' trials by analysing the capacity and reach of potential networks of show homes in different areas. Our initial data work showed that _on average_ any home is no further than a 15min drive away from the nearest heat pump, which inspired confidence in our project. You can play around with the <a href="https://nestauk.github.io/show_homes/Distances_similar" title="Distance to nearest show home">interactive map</a> that shows the distance to the closest property with a heat pump for a randomly selected sample of visitor and host homes. And find more insights from our initial speed-testing project [here](https://www.nesta.org.uk/a-network-of-show-homes-for-heat-pumps/).

### Modelling a network of show hoems<a id='model'></a>

In further research, we built a data tool that matches visitor homes to show homes (still using model data) while considering maximum driving distance and capacity of the show homes. The mock example below shows how certain visitor homes (gray) cannot be matched with a suitable host home (blue), either because the host home is too far away or because it's already booked out - in this simple example, each show home can only host three visitors.

<img src="./docs/mock_network_example.png"  width="400"  height="300">

The data tool lets you explore different scenarios and compute the impact and reach of a network. For example, you can set the number of visitors per slot, adjust the maximum driving distance or play around with different numbers of hosts and visitors.

We found different patterns for different areas and property types when modelling a potential network, suggesting that show home networks may not work equally well in all areas and may require the addition of strategically placed show homes in remote areas or show rooms (e.g. in DIY stores) in regions with high demand.

<!-- <p float="left">
<img src="./docs/network_example_1.png"  width="250"  height="180">
<img src="./docs/network_example_2.png"  width="250"  height="180">
</p>

<p float="left">
<img src="./docs/network_example_3.png"  width="250" height='180'>
<img src="./docs/network_example_4.png"  width="220"   height="180">
</p> -->

<img src="./docs/network_examples.png"  width="800"  height="580">

### Future work<a id='future'></a>

Future research may include an agent-based model that investigates and models how a network would develop over time in different areas.

---

## Setup<a id='setup'></a>

[[back to top]](#top)

Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:

- Install: `git-crypt`, `direnv`, and `conda`
- Have a Nesta AWS account configured with `awscli`

```
git clone https://github.com/nestauk/show_homes.git
cd show_homes
git checkout dev
make install
```

`make install` may take a while and and could potentially throw an error for AWS+Metaflow like below, but that's fine.

```
May throw error at megaflop Configuring Metaflow + AWSAWS + Metaflow setup failed, check .cookiecutter/state/setup-metaflow for more info
make: *** [.cookiecutter/state/setup-metaflow] Error 1
```

Next, activate the environment and download the necessary files (~3.6 GB):

```
conda activate show_homes
aws s3 sync s3://asf-show-homes/inputs ./inputs
aws s3 sync s3://asf-show-homes/outputs/data ./outputs/data
```

Alternatively, you can manually download the necessary files from the S3 bucket named [asf-show-homes](https://s3.console.aws.amazon.com/s3/buckets/asf-show-homes?region=eu-west-2&tab=objects). If not interested in analysing the speed testing results in detail, downloading the following two files will be sufficient:

- `inputs/kepler_configs/network_gradio_config.txt`
- `inputs/data/epc_for_show_homes.csv`

---

## How to use<a id='use'></a>

[[back to top]](#top)

For inspecting and re-creating the results from the **speed testing**, e.g. estimating the average driving distance to the nearest heat pump, use the Jupyter notebook [`speed_testing_results.ipynb`](https://github.com/nestauk/show_homes/blob/dev/show_homes/analysis/speed_testing_results.ipynb).

To open the tool for **modelling a show homes network**, you have two options:
a) Run the Jupyter notebook [`show_home_network.ipynb.ipynb`](https://github.com/nestauk/show_homes/blob/dev/show_homes/analysis/show_home_network.ipynb.ipynb) and either use the gradio window that opens within Jupyter notebook or click on the local or public link to open the tool in a web browser.

b) Run `python show_homes/pipeline/show_homes_network.py ` in a terminal and click on either the local or public URL provided by the output to open the tool in a web browser.

Once the tool is open, select the parameters on the lefthand side to create the scenario of choice and click _submit_. Information about your selection and network measures will appear in the top box on the righthand side. The network map will be shown below. Zoom into the map and switch to 3D map (second option in sidebar) to inspect the network more closely. Orange dots represent show homes and purples one stand for visitor homes. The first option in the sidebar _Show layer panels_ reveals the map's layers, which you can activate or deactivate by clicking on them.

The map will also be saved as an HTML file under `show_homes/analysis/maps`. The filename reflects the selected parameters. The HTML file is more suitable for thorough exploration or sharing than the map view in the tool.

---

## Contributor guidelines<a id='contributor'></a>

[[back to top]](#top)

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
