import numpy as np
import networkx as nx
from networkx.algorithms.flow import boykov_kolmogorov


def CreateArbitraryLabelling_yn(graph_init, ref):
    graph = graph_init.copy()
    nodes = graph.nodes
    labels = {}
    card = len(nodes)//2
    for n in nodes:
        if n == ref:
            labels[n] = 'yes'
        else: 
            if n<=card:
                labels[n]='yes'
            else  : 
                labels[n] = 'no'
    nx.set_node_attributes(graph,values = labels,name='label')
    return graph


def ComputeEnergy_yn(graph_init, graph_labelled, ref, SP_max):
    pred_ref, dist_ref = nx.dijkstra_predecessor_and_distance(graph_init, ref)
    norm = SP_max
#     norm = np.max([list(dist_ref.items())[i][1] for i in graph_init.nodes])

    E_data = 0
    for n in graph_labelled.nodes:
        label = graph_labelled.nodes[n]['label']
        if label == 'yes':
            E_data += np.abs(dist_ref[n]/norm)
        else:
            E_data += np.abs((1-dist_ref[n]/norm))

    E_smooth = 0
    for e in graph_init.edges:
        label_p = graph_labelled.nodes[e[0]]['label']
        label_q = graph_labelled.nodes[e[1]]['label']
        if label_p == label_q:
            E_smooth += 0
        else:
            dcc = graph_init.edges[e[0],e[1]]['weight']
            E_smooth += norm/dcc

    return E_data, E_smooth, E_data + E_smooth


def alpha_beta_swap_bk_yn(graph_init,graph_labelled,ref,SP_max):
    #find nodes labelled alpha or beta
    working_nodes = graph_init.nodes
    pred_ref, dist_ref = nx.dijkstra_predecessor_and_distance(graph_init, ref)
    norm = SP_max
#     norm =np.max([list(dist_ref.items())[i][1] for i in graph_init.nodes])

    #create new graph to work on
    work_graph = nx.DiGraph()
    work_graph.add_node('yes')
    work_graph.add_node('no')
    for n in working_nodes:
        work_graph.add_node(n)
        work_graph.add_edge('yes',n,weight=np.abs(dist_ref[n]/norm))
        work_graph.add_edge(n,'no',weight=np.abs((1-dist_ref[n]/norm)))
#         print(n,dist_ref[n]/norm,np.abs((1-dist_ref[n]/norm)))  
    for e in graph_init.edges:
        label_p = graph_labelled.nodes[e[0]]['label']
        label_q = graph_labelled.nodes[e[1]]['label']
        if label_p == label_q:
            weight = 0
        else:
            dcc = graph_init.edges[e[0], e[1]]['weight']
            weight = norm/dcc
        if e[0] < e[1]:
            work_graph.add_edge(e[0], e[1], weight = weight)
        else:
            work_graph.add_edge(e[1], e[0], weight = weight)  
    #calculating the max flow /min cut
    R = boykov_kolmogorov(work_graph, 'yes', 'no', capacity='weight')
    source_tree, target_tree = R.graph['trees']
    partition = (set(source_tree), set(work_graph) - set(source_tree))
    #assign new labels
    new_graph = graph_labelled.copy()
    labels = {}
    for i in working_nodes:
        I = int(i)
        if I in partition[0] and 'yes' in partition[0]:
            labels[I] = 'no'
        elif I in partition[0] and 'no' in partition[0]:
            labels[I] = 'yes'
        elif I in partition[1] and 'yes' in partition[1]:
            labels[I] = 'no'
        elif I in partition[1] and 'no' in partition[1]:
            labels[I] = 'yes'
        else:   
            print("error")
    nx.set_node_attributes(new_graph, values = labels, name='label')
#     print(nx.get_node_attributes(new_graph,'label'))
    return new_graph, R


def swap_minimization_yn(graph_init, cycles, ref, SP_max):
    E_datas=[]
    E_smooths = []
    E_tot = []
    E_c = []
    current_graph_labelled = CreateArbitraryLabelling_yn(graph_init, ref)
    Ed, Es, current_energy = ComputeEnergy_yn(graph_init, current_graph_labelled, ref, SP_max)
    E_datas.append(Ed)
    E_smooths.append(Es)
    E_tot.append(Ed+Es)
    E_c.append(current_energy)
    #cycles
    for k in range(cycles):
        new_graph_labelled, R = alpha_beta_swap_bk_yn(graph_init, current_graph_labelled, ref, SP_max)
        Ed, Es, new_energy = ComputeEnergy_yn(graph_init, current_graph_labelled, ref, SP_max)
#         print(current_energy,new_energy)
#         print(nx.get_node_attributes(new_graph_labelled,'label'))
        if new_energy <= current_energy:
            current_energy = new_energy
            current_graph_labelled = new_graph_labelled
        E_datas.append(Ed)
        E_smooths.append(Es)
        E_tot.append(Ed+Es)
        E_c.append(current_energy)
    plot_energy(E_datas, E_smooths, E_tot, E_c)
    return current_graph_labelled, R
