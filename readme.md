# Comparative Analysis between Infomap and Label Propagation for Community Detection

ComparativeAnalysis.py generates statistics regarding the execution of the algorithms and the quality of their
partition and outputs it into report.txt and stdout.
Please note that in the main function edgeListModels describes the paths of the graphs to be analysed as lists of
edges and infoMapArgumentsList and labelPropagationArgumentsList, as the name suggests, refer to the parameters
to pass to the Infomap and Label Propagation algorithms. 
Please also note that n, tau1, tau2, mu and average_degree also in main refer to the parameters for the
artificial graphs generated through the LFR_benchmark_graph function.
These variables make up the degree of customization of the project. The values used by default are the 
ones described in the report. 

## Structure

The graph class represents a graph with the partition calculated over it by one of the algorithms. It also stores
methods for graph reading and creation and a classify function to ease the output of the graph's metrics. 
The analyser class represents the evaluating entity which runs the test suite on the graph class. It is able
of running on both LFR and data graphs. Its InfoMap and LabelPropagation methods are our implementations of the 
algorithms with the use of existing libraries, but with some modifications, like, for instance, when we used only 
the leaf-nodes of the hierarchical tree to set the partition in the graph class. The analyser class also has the
ratePartition method which calls adaptedMancoridisAndExpansionMetric, partitionQuality, modularity and
triangleParticipationRatio which write the results of their computations to the report and pass it further to the
analysis variable in order to be used in the renderAnalysis method of the analyser where meaningful plots are drawn.

## Installation

In order to install the necessary dependencies from the main directory run:

```bash
npm install requirements.txt
```

Following this, to run the project from the main directory simply run the main python file as follows:

```bash
python3 ComparativeAnalysis.py mode
```
Where 'mode' is 0 for the use of the graphs stored in the data foler and 1 for the LFR synthetic graphs 

## Aditional Remarks

If hardware resources are lacking creating the LFR graphs might be very time consuming.
(See line 323 in ComparativeAnalysis.py)
Please note that the LFR graphs for testing were only created in our PCs after several trials, doing so
with the specified parameters is a very resourse intensive task possibly stalling in some parameter combination
