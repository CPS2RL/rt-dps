# rt-dps
The code and readme file will be added later.

## ap_ant_pos_pin.py
To generate the percentage of task sets vulnerable under DPS-only, TS-only, both, and neither category, we use this code. In this code we have the option to change the period of the attacker task. This code is used for anterior, posterior, and pincer attack simulations. We generate 200 task sets for utilization 0.1-0.8. We get the result in XLSX file format. Later we analyze the XLSX file to get our desired categorized percentage of attack.


## ap_butterfly.py
To generate the percentage of task sets vulnerable under DPS-only, TS-only, both, and neither category, we use this code. This code is used for butterfly attack simulation. We generate 200 task sets for utilization 0.1-0.8. We get the result in XLSX file format. Later we analyze the XLSX file to get our desired categorized percentage of attack.

## normalized.py
We utilize this script to generate necessary data to calculate normalized attack percentage.


## delta_eta_sweep.py
We utilize this script to check how different scheduled-based attacks react to changing the inter-arrival time sensitivity parameters.


# example_plot
In the example_plot folder, we have a script to generate a schedule plot for the example used in our paper.
