# Heat pump show homes

<img src="./docs/data_tool_example.png">
<br><br>

While heat pumps are becoming more and more popular, only few people in the UK have actually seen one in action. Having the opportunity to visit a running heat pump in a real home - not a show room - could help many potential adopters to get a better understanding for heat pumps and what it is like to have one at home.

The Sustainable Future team at Nesta tested the idea of show homes for heat pumps with two show homes in London and Glasgow with promising results. Both visitors and hosts were extremely happy with the experience.

Find out more about the project and sign up for the next round of show home events on <a href="visitaheatpump.com/" title="VisitAHeatPump">visitaheatpump.com</a>.

This data work supports the 'real-life' trials by analysing the capacity and reach of potential networks of show homes in different areas. Our initial data work showed that on average a person is no further than a 15min drive away from the nearest heat pump, which inspired confidence in our project. You can play around with the <a href="https://nestauk.github.io/show_homes/Distances_similar" title="Distance to nearest show home">interactive map</a> that shows the distance to the closest property with a heat pump for a randomly selected sample of visitor and host homes.

In further research, we built a data tool matches visitor home to show homes (still using model data) while considering maximum diving distance and capacity of the show homes. The mock example below shows how certain visitor homes (gray) cannot be matched with a suitable host home (blue), either because the host home is too far away or because it's already occupied by other visitors.

<img src="./docs/mock_network_example.png"  width="270"  height="200">

The data tool lets you explore different scenarios and compute the impact and reach of a network. For example, you can set the number of visitors per slot, adjust the maximum driving distance or play around with different numbers of hosts and visitors.

We found different patterns for different areas and prooperty types when modelling a potential network, suggesting that show home networks might work better in some areas or may require the addition of strategically placed show homes in remote areas or show rooms (e.g. in DIY stores) in regions with high demand.

<p float="left">
<img src="./docs/network_example_1.png"  width="250"  height="180">
<img src="./docs/network_example_2.png"  width="250"  height="180">
</p>

<p float="left">
<img src="./docs/network_example_3.png"  width="250" height='180'>
<img src="./docs/network_example_4.png"  width="220"   height="180">
</p>

Future research may include an agent-based model that investigates and models how a network would develop over time in different areas.

---

## Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `git-crypt`, `direnv`, and `conda`
  - Have a Nesta AWS account configured with `awscli`
- Run `make install` to configure the development environment:

  - Setup the conda environment
  - Configure pre-commit
  - Configure metaflow to use AWS

- Download the inputs and outputs

```
aws s3 sync s3://asf-show-homes/inputs ./inputs
aws s3 sync s3://asf-show-homes/outputs ./outputs
```

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
