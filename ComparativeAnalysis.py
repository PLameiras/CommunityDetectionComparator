from math import sqrt
import uuid
import sys

import networkx
import collections
import infomap
import numpy
import statistics
import matplotlib.pyplot as plt

from networkx.algorithms.community.label_propagation import asyn_lpa_communities
from timeout_decorator import timeout
from networkx.algorithms.community.quality import partition_quality
from networkx.algorithms.community import modularity

from timeit import default_timer as timer
from itertools import repeat, product

class Graph:
    def __init__(self, name):
        self.graph = networkx.Graph()
        self.name = name
        self.partition = []

    def getGraph(self):
        return self.graph
    
    def getName(self):
        return self.name
    
    def getPartition(self):
        return self.partition
    
    def setPartition(self, partition):
        self.partition = partition

    # Creates a partition as sets of edges from the community attribute in the graph set by the algorithms
    def updatePartition(self):
        self.partition = [[] for i in repeat(None,times=len(
        collections.Counter(list(networkx.get_node_attributes(self.graph, 'community').values())).keys()))]
        for k, v in networkx.get_node_attributes(self.getGraph(), 'community').items(): 
            self.partition[v].append(k)

    def createGraphFromEdgeList(self, filename):

        file = open(filename, 'r')    
        type = file.readline().split()

        for line in file.readlines():
            if type[0] == "d":
                self.graph = self.graph.to_directed()

            vertices = line.split()
            edge = (int(vertices[0]), int(vertices[1]))

            if type[1] == "w":
                self.graph.add_edge(*edge, weight = float(vertices[2]))
            else:
                self.graph.add_edge(*edge)

    # Remove timeout if needed 
    @timeout(600)
    def createGraphLFR(self, argumentsLFR):
            
        self.graph = networkx.LFR_benchmark_graph(argumentsLFR[0], argumentsLFR[1], argumentsLFR[2],
                                                    argumentsLFR[3], argumentsLFR[4], max_degree=30, max_iters=200000)
    
    def classify(self, report, argumentsLFR=None):
        analysis = "-----Analyses of the network \"" + self.getName()[6:-4] + "\"-----\n" +\
                        "Nodes: {}, Edges: {}, Self Loops: {}".format(self.graph.number_of_nodes(), self.graph.number_of_edges(), networkx.number_of_selfloops(self.graph)) + "\n" +\
                        "Graph Type: " + ("Directed and " if self.graph.is_directed() == True else "Undirected and ") +\
                        ("Weighted" if networkx.is_weighted(self.graph) == True else "Non-Weighted") + "\n"      

        #Calculate Graph's degree distribution
        degrees = [val for (node, val) in self.graph.degree()]
        degreeSequence = sorted(degrees, reverse=True)        

        analysis += "Graph's Degrees mean: " + str(statistics.mean(degreeSequence)) + "\n"
        analysis += "Graph's Degrees standard deviation: " + str(statistics.stdev(degreeSequence)) + "\n"
        analysis += "Graph's Degrees quartiles: " + str(numpy.quantile(degreeSequence, [0,0.25,0.5,0.75,1])) + "\n"

        if argumentsLFR != None:
            analysis += "LFR n, tau1, tau2, mu, average_degree: " + ', '.join(str(item) for item in argumentsLFR) + "\n"

        # We did non test this metric as our computers were not able to run it
        """
        degreeCentralities = networkx.betweenness_centrality(self.graph)
        degreeCentralityMean = statistics.mean(degreeCentralities)
        degreeCentralityStandardDeviation = statistics.stdev(degreeCentralities)

        centralityOutliers = 0
        threshold = 2

        for degreeCentrality in degreeCentralities:
            z_score = (degreeCentrality - degreeCentralityMean) / degreeCentralityStandardDeviation
            if abs(z_score) > threshold and degreeCentrality > degreeCentralityMean: # Positive outliers
                centralityOutliers += 1

        analysis += "Number of degree betweenness centrality positive edge outliers: " + str(centralityOutliers) + "\n"
        """

        if self.graph.is_directed() == False and networkx.is_weighted(self.graph) == False:
            analysis += "Degree assortativity: " + str(networkx.degree_assortativity_coefficient(self.graph)) + "\n" 

        report.write(analysis)


class Analyser:
    def runTestSuite(self, infoMapArgumentsList, labelPropagationArgumentsList, analyser, graph, report):
        # InfoMap
        analysis = {}
        for infoMapArguments in infoMapArgumentsList:
            start = timer()
            analyser.InfoMap(graph, report, infoMapArguments)
            end = timer()

            report.write("----------InfoMap Stats----------\n" +
                        "InfoMap Parameters: " +
                         str(infoMapArguments[0]) +
                         "\nProcessing time: " +
                         str(end - start) +
                         "s" +
                         "\n")

            # Analyze partition quality
            analysis["infomap"] = analyser.ratePartition(graph, report)

        # Label Propagation 
        for labelPropagationArguments in labelPropagationArgumentsList:
            start = timer()
            analyser.LabelPropagation(graph, labelPropagationArguments)
            end = timer()

            report.write("------Label Propagation Stats------\n" +
                         "Processing time " +
                         str(end - start) +
                         "s" +
                         "\n")

            # Analyze partition quality
            analysis["label_prop"] = analyser.ratePartition(graph, report)

        report.write("\n")
        return analysis
        
    def InfoMap(self, graph, report, infoMapArguments):
        # Default multi-level network
        infomapWrapper = infomap.Infomap(flow_model="undirected", use_node_weights_as_flow=True)

        for edge in graph.getGraph().edges():
            infomapWrapper.network.addLink(*edge)

        infomapWrapper.run()

        report.write("%d modules with codelength %f found\n" % (infomapWrapper.numTopModules(),
                                                                infomapWrapper.codelength))
        
        # See Report Section 2.3
        # Set the community of each node in the graph
        communities = {}
        for node in infomapWrapper.iterLeafNodes():
            communities[node.physicalId] = node.moduleIndex()

        networkx.set_node_attributes(graph.getGraph(), name='community', values=communities)
        
        # Make partition accessible by other methods
        graph.updatePartition()

    def LabelPropagation(self, graph, labelPropagationArguments):
        graph.setPartition(
            [
                list(community) for community in asyn_lpa_communities(
                    graph.getGraph(), labelPropagationArguments[0], labelPropagationArguments[1]
                )
            ]
        )
    
    def ratePartition(self, graph, report):
        sumIntraDensity, sumInterDensity, diffSumIntraInterDensities, \
            meanExpansion, stdevExpansion = self.adaptedMancoridisAndExpansionMetric(graph, report)
        print("     -> Mancoridis & Expansion Analyzed")
        coverage, performance = self.partitionQuality(graph, report)
        print("     -> Coverage & Performance Analyzed")
        modularity = self.modularity(graph, report)
        print("     -> Modularity Analyzed")
        # Please note that due to limitations in our own hardware resources we were not able to test this metric
        # triangle_mean = 0
        # triangle_stdev = 0
        # triangle_mean, triangle_stdev = self.triangle_participation_ratio(graph, report)
        # print("     -> Triangle Participation Analyzed")

        return {
            "A. Mancoridis": diffSumIntraInterDensities,
            "Expansion Mean": meanExpansion,
            "Expansion Deviation": stdevExpansion,
            "Modularity": modularity,
            "Coverage": coverage,
            "Performance": performance,
            # "triangle_participation_mean": triangle_mean,
            # "triangle_participation_deviation": triangle_stdev
        }

    def adaptedMancoridisAndExpansionMetric(self, graph, report):
        sumIntraClusterDensity = 0
        sumInterClusterDensity = 0

        expansion = []
        numCommunities = len(graph.getPartition())
        for community in graph.getPartition():
            communityNodeNumber = len(community)

            # Single node communities not taken into account
            if len(community) == 1:
                continue                

            # Also known as Internal Density
            internalEdges = graph.getGraph().subgraph(community).number_of_edges()
            maxPossibleInternalEdges = communityNodeNumber * (communityNodeNumber - 1) / (1 if graph.getGraph().is_directed() == True else 2)
            sumIntraClusterDensity += internalEdges / maxPossibleInternalEdges

            # Also known as Cut Ration
            interClusterEdges = len(list(networkx.edge_boundary(graph.getGraph(), community, [x for x in graph.getGraph().nodes() if x not in community])))
            maxPossibleInterClusterEdges = communityNodeNumber * (graph.getGraph().size() - communityNodeNumber) * (2 if graph.getGraph().is_directed() == True else 1)
            sumInterClusterDensity += interClusterEdges / maxPossibleInterClusterEdges

            expansion += [interClusterEdges / communityNodeNumber]
 
        report.write(
            "Adapted Mancoridis metric: Average Intra Cluster Density {} - Average Inter Cluster Density {} = {}\n".
            format(sumIntraClusterDensity / numCommunities,
                   sumInterClusterDensity / numCommunities,
                   sumIntraClusterDensity / numCommunities - sumInterClusterDensity / numCommunities)
             )
        
        stdevExpansion = 0
        if len(expansion) > 1:
            stdevExpansion = statistics.stdev(expansion)
        meanExpansion = statistics.mean(expansion)

        report.write("Community Expansion Average: {}\n".format(stdevExpansion))
        report.write("Community Expansion Deviation: {}\n".format(meanExpansion))

        return sumIntraClusterDensity / numCommunities, sumInterClusterDensity / numCommunities, sumIntraClusterDensity / numCommunities - sumInterClusterDensity / numCommunities, \
            meanExpansion, stdevExpansion

    def partitionQuality(self, graph, report):
        quality = partition_quality(graph.getGraph(), graph.getPartition())
        report.write("Coverage: {}\n".format(quality[0]))
        report.write("Performance: {}\n".format(quality[1]))
        return quality[0], quality[1]

    def modularity(self, graph, report):
        mod = modularity(graph.getGraph(), graph.getPartition(), resolution=1)
        report.write("Modularity: {}\n".format(mod))
        return mod

    # Please note that due to limited hardware resources we were not able to test this metric
    def triangleParticipationRatio(self, graph, report):
        triangleParticipationRatio = []

        for community in graph.getPartition():

            if len(community) < 4:
                triangleParticipationRatio += [0]
                continue     

            matrix = networkx.to_numpy_array(graph.getGraph().to_undirected().subgraph(community))
            if networkx.is_weighted(graph.getGraph()):
                matrix[matrix != 0] = 1

            triangleParticipationRatio += [numpy.trace(numpy.linalg.matrix_power(matrix, 3)) / 6 / len(community)]

        mean = statistics.mean(triangleParticipationRatio)
        stdev = 0
        if len(triangleParticipationRatio) > 1:
            stdev = statistics.stdev(triangleParticipationRatio)
        report.write("Triangle Participation Ratio Average: {}\n".format(mean))
        report.write("Triangle Participation Standard Deviation: {}\n".format(stdev))
        return mean, stdev
    
    def renderAnalysis(self, analysis):
        for results in analysis.values():
            metrics = results['infomap'].keys()
            infomapMetrics = [results['infomap'][metric] for metric in metrics]
            labelPropagationMetrics = [results['label_prop'][metric] for metric in metrics]

            x_axis = numpy.arange(len(metrics))
            plt.figure().set_figwidth(20)

            plt.bar(x_axis - 0.2, infomapMetrics, 0.4, label='Infomap', color='blue')
            plt.bar(x_axis + 0.2, labelPropagationMetrics, 0.4, label='Label Propagation', color='red')

            plt.xticks(x_axis, metrics, rotation=0)
            plt.xlabel("Quality Metric")
            plt.ylabel("Value")
            plt.title("Comparative Analysis")
            plt.legend()
            plt.show()


def main():
    report = open("report.txt", 'a')
    analyser = Analyser()
    analysis = dict()
    
    edgeListModels = ["data//facebook_combined.txt", "data//bio-SC-TS.txt"]
    infoMapArgumentsList = [[True]]
    labelPropagationArgumentsList = [[None, None]]

    n = [170]
    tau1 = [2, 3]
    tau2 = [1.05, 2]
    mu = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    average_degree = [7]
    argumentsLFR = list(list(combination) for combination in product(*[n, tau1, tau2, mu, average_degree]))
    
    if len(sys.argv) < 2:
        print("Please specify mode [0/1] as a command-line argument.")
        exit()
    if int(sys.argv[1]) == 0:
        for edgeListModel in edgeListModels:
            graph = Graph(edgeListModel)
            graph.createGraphFromEdgeList(edgeListModel)
            graph.classify(report)
            analysis[uuid.uuid4()] = analyser.runTestSuite(infoMapArgumentsList, labelPropagationArgumentsList, analyser, graph, report)
    elif int(sys.argv[1]) == 1:
        for argumentLFR in argumentsLFR:
            graph = Graph("LFR")
            graph.createGraphLFR(argumentLFR)
            graph.classify(report, argumentLFR)
            analysis[uuid.uuid4()] = analyser.runTestSuite(infoMapArgumentsList, labelPropagationArgumentsList, analyser, graph, report)
    else:
        exit()

    analyser.renderAnalysis(analysis)
    
if __name__ == '__main__':
    main()
    