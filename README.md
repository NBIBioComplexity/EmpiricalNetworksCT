# EmpiricalNetworksCT
Simulation code for disease spread and contact tracing in empirical contact networks from the Copenhagen Networks Study.

The main simulation programme is CT_sim.py.
Several command line options are available. Their names and current values are shown when the programme is run in the command line.
At the moment, the input network file is specified directly in the source code, i.e. in CT_sim.py. 
The input format for the temporal network is what we call "timechunks". Such a timechunk file can be generated using the csv_to_timechunks.py script.
