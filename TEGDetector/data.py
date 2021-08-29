import os
import csv
import numpy as np
from itertools import islice
import random
import scipy.sparse as sp

path = 'Data/Phishing first-order nodes'
normal_path = 'Data/Normal first-order nodes'

def append_node(node, nodes, phishing_node):

    if node in phishing_node:
        y = 1
    else:
        y = 0
        if [node,y] not in nodes:
            nodes.append([node, y])

def get_link( max_num, path):
    nodes = []
    links = []
    phishing_node = []
    num = 0
    for root, dirs, files in os.walk(path):
        for filespath in files:
            csv_path = os.path.join(root, filespath)
            csv_file = open(csv_path)
            csv_reader_lines = csv.reader(csv_file)
            if 'first' in path:
                if csv_path.split('/')[2].split('.')[0] not in phishing_node:
                    phishing_node.append(csv_path.split('/')[2].split('.')[0])
                    num +=1
            else:
                if csv_path.split('/')[2] not in phishing_node:
                    phishing_node.append(csv_path.split('/')[2])
                    num += 1
            if num > max_num:
                break
            # print(num)
            for row in islice(csv_reader_lines, 1, None):
                print(num, csv_path.split('/')[2])
                from_node = row[4]
                to_node = row[5]
                trans = [from_node, to_node, row[3], row[6]]
                append_node(from_node, nodes, phishing_node)
                append_node(to_node, nodes, phishing_node)
                links.append(trans)
    return nodes, links
# nodes, links = get_link(2, path)
# get_link(nodes, links, normal_path)

# path2 = 'Data/Phishing second-order nodes'
# # for root, dirs, files in os.walk(path2):
# #     for filespath in files:
# #         csv_path = os.path.join(root, filespath)
# #         get_link(nodes, links, csv_path)
# get_link(nodes, links, path2)
# np.savez('all_1', nodes = nodes, links = links)


# data = np.load('all_1.npz')
# nodes = data['nodes'].tolist()
# links = data['links'].tolist()
def normal_time(links):
    times = np.zeros(len(links))
    for i in range(len(links)):
        times[i] = int(links[i][3])
    max_time = np.max(times)
    min_time = np.min(times)
    for i in range(len(links)):
        links[i][3] = (int(links[i][3]) - min_time)/(max_time - min_time)
    links = sorted(links, key=lambda x: x[3])
    return links


def multi2single(links):
    single_links = []
    curr_node_pair = (links[0][0], links[0][1])
    curr_link_infos = []
    for link in links:
        if (link[0], link[1]) == curr_node_pair:
            curr_link_infos.append(link[2:])
        else:
            sum = 0
            for info in curr_link_infos:
                sum += float(info[0])
            curr_link_infos.sort(key=lambda x: float(x[-1]))
            timestamp = curr_link_infos[-1][-1]
            (node1, node2) = curr_node_pair
            single_links.append([node1, node2, sum, timestamp])

            curr_node_pair = (link[0], link[1])
            curr_link_infos = []
            curr_link_infos.append(link[2:])

    sum = 0
    for info in curr_link_infos:
        sum += float(info[0])
    curr_link_infos.sort(key=lambda x: float(x[-1]))
    timestamp = curr_link_infos[-1][-1]
    (node1, node2) = curr_node_pair
    single_links.append([node1, node2, sum, timestamp])

    return single_links

# def get_subgraph1(nodes, links, path):
#
#     csv_file = open(path)
#     csv_reader_lines = csv.reader(csv_file)
#     for row in islice(csv_reader_lines, 1, None):
#         from_node = row[4]
#         to_node = row[5]
#         trans = [from_node, to_node, row[6], float(row[3])]
#         if from_node not in nodes:
#             nodes.append(from_node)
#         if to_node not in nodes:
#             nodes.append(to_node)
#         links.append(trans)

def get_subgraph1(node, nodes, links, dict, path, p):
    # dict = {}
    csv_file = open(path)
    csv_reader_lines = csv.reader(csv_file)
    for row in islice(csv_reader_lines, 1, None):
        from_node = row[4]
        to_node = row[5]
        # print(row)
        if from_node == node:
            dict[from_node] = p
        else:
            if from_node not in dict:
                dict[from_node] = row[9]
        if to_node == node:
            dict[to_node] = p
        else:
            if to_node not in dict:
                dict[to_node] = row[9]

        trans = [from_node, to_node, row[6], float(row[3])]
        if from_node not in nodes:
            nodes.append(from_node)
        if to_node not in nodes:
            nodes.append(to_node)
        links.append(trans)

def get_subgraph2(node, nodes, links,dict, path, p):
    for root, dirs, files in os.walk(path):
        for filespath in files:
            csv_path = os.path.join(root, filespath)
            csv_file = open(csv_path)
            csv_reader_lines = csv.reader(csv_file)
            for row in islice(csv_reader_lines, 1, None):
                from_node = row[4]
                to_node = row[5]
                if from_node == node:
                    dict[from_node] = p
                else:
                    if from_node not in dict:
                        dict[from_node] = row[9]

                if to_node == node:
                    dict[to_node] = p
                else:
                    if to_node not in dict:
                        dict[to_node] = row[9]


                trans = [from_node, to_node, row[6], float(row[3])]
                if from_node not in nodes:
                    nodes.append(from_node)
                if to_node not in nodes:
                    nodes.append(to_node)
                links.append(trans)

def get_subgraph22(node, nodes, links, dict, path1, path2, p):
    for n in nodes:
        if n == node:
            continue
        p = int(dict[n])
        if p == 0:
            path = 'data/Normal first-order nodes/' + n + '.csv'
        else:
            path = 'data/Phishing first-order nodes/' + n + '.csv'
        get_subgraph1(n, nodes, links, dict, path, p)
    # for root, dirs, files in os.walk(path):
    #     for filespath in files:
    #         csv_path = os.path.join(root, filespath)
    #         csv_file = open(csv_path)
    #         csv_reader_lines = csv.reader(csv_file)
    #         for row in islice(csv_reader_lines, 1, None):
    #             from_node = row[4]
    #             to_node = row[5]
    #             if from_node == node:
    #                 dict[from_node] = p
    #             else:
    #                 if from_node not in dict:
    #                     dict[from_node] = row[9]
    #
    #             if to_node == node:
    #                 dict[to_node] = p
    #             else:
    #                 if to_node not in dict:
    #                     dict[to_node] = row[9]
    #
    #
    #             trans = [from_node, to_node, row[6], float(row[3])]
    #             if from_node not in nodes:
    #                 nodes.append(from_node)
    #             if to_node not in nodes:
    #                 nodes.append(to_node)
    #             links.append(trans)


def con_graph(node, label, max_num):
    nodes = []
    links = []
    Nodes = []
    Labels = []
    dict = {}
    if label == 0:
        path1 = 'data/Normal first-order nodes/'+node+'.csv'
        path2 = 'data/Normal second-order nodes/'+ node
        p = 0
    else:
        path1 = 'data/Phishing first-order nodes/'+node+'.csv'
        path2 = 'data/Phishing second-order nodes/'+ node
        p = 1

    get_subgraph1(node, nodes, links, dict, path1, p)
    get_subgraph2(node, nodes, links, dict, path2, p)

    normal_time(links)
    Link = sorted(links, key=lambda x: x[3], reverse=True)
    # links = multi2single(links)
    Link = sorted(multi2single(links), key = lambda x: x[3], reverse=True)[:max_num]
    for l in Link:
        if l[0] not in Nodes:
            Nodes.append(l[0])
            Labels.append(int(dict[l[0]]))
        if l[1] not in Nodes:
            Nodes.append(l[1])
            Labels.append(int(dict[l[1]]))
    adj = np.zeros((len(Nodes), len(Nodes)),dtype=np.float32())
    fea = np.zeros((len(Nodes), 2))
    for l in Link:
        adj[Nodes.index(l[0]),Nodes.index(l[1])] = l[2]
    for n in range(len(Labels)):
        fea[n][Labels[n]]=1

    return adj, fea, len(Link)

def get_node(path, path1, num, txt_path):
    if os.path.exists(txt_path):
        print('Node_set has already exists!')
        node = []
        f = open(txt_path,'r')
        for line in f:
            l = line.split(' ')
            node.append([l[0],int(l[1])])
    else:
        print('Get Node Set......')
        n0 = []
        n1 = []
        with open(txt_path, 'w') as f:
            for root, dirs, files in os.walk(path):
                dirs = os.listdir(path)
                # num = 10
                sample = random.sample(dirs, num)
                for i in sample:
                    node = i.split('.')[0]
                    n0.append([node,0])
                    # f.write(node + ' 0' + '\n')
            for root, dirs1, files in os.walk(path1):
                dirs1 = os.listdir(path1)
                sample1 = random.sample(dirs1, num)
                for i in sample1:
                    node = i.split('.')[0]
                    n1.append([node, 1])
            node = n0 + n1
            random.shuffle(node)
            for i in range(len(node)):
                f.write(node[i][0] + ' '+str(node[i][1]) + '\n')
    return node
        # f.write(node + ' 1' + '\n')

def get_node_test(path, path1, num, node_list, txt_path):
    if os.path.exists(txt_path):
        print('Node_set has already exists!')
        node = []
        f = open(txt_path,'r')
        for line in f:
            l = line.split(' ')
            node.append([l[0],int(l[1])])
    else:
        print('Get Test Node Set......')
        n0 = []
        n1 = []
        with open(txt_path, 'w') as f:
            for root, dirs, files in os.walk(path):
                dirs = os.listdir(path)
                # num = 10
                sample = random.sample(dirs, num)
                for i in sample:
                    node = i.split('.')[0]
                    trans = [node,0]
                    if trans not in node_list:
                        n0.append([node,0])
                    # f.write(node + ' 0' + '\n')
            for root, dirs1, files in os.walk(path1):
                dirs1 = os.listdir(path1)
                sample1 = random.sample(dirs1, num)
                for i in sample1:
                    node = i.split('.')[0]
                    trans1 = [node,1]
                    if trans1 not in node_list:
                        n1.append([node,1])
            node = n0 + n1
            random.shuffle(node)
            for i in range(len(node)):
                f.write(node[i][0] + ' '+str(node[i][1]) + '\n')
    return node


def split_data(Graph, begin, num):
    dataset = {}
    dataset['adj'] = Graph['Adj'][begin:begin + num]
    dataset['fea'] = Graph['Fea'][begin:begin + num]
    dataset['label'] = Graph['Label'][begin:begin + num]
    dataset['batch_num_nodes'] = Graph['Batch_num_nodes'][begin:begin + num]
    dataset['assign_input'] = Graph['Fea'][begin:begin + num]
    return dataset

def split(Graph):
    N = len(Graph['Adj'])
    train_num = int(N / 10 * 7)
    val_num = int(N / 10 * 1)
    test_num = int(N / 10 * 2)
    train_dataset = split_data(Graph, 0, train_num)
    val_dataset = split_data(Graph, train_num, val_num)
    test_dataset = split_data(Graph, train_num + val_num, test_num)
    return train_dataset, val_dataset, test_dataset,  train_num, val_num, test_num

def con_Gset(node_path, save_path, batch_num, max_link, max_n=None):
    sum_node = 0
    sum_link = 0
    max = 100
    if os.path.exists(save_path):
        print('G_set has already exists!')
        Graph = {}
        Data = np.load(save_path)
        Graph['Adj'] = Data['adj']
        Graph['Fea']  = Data['fea']
        Graph['Label']  = Data['label']
        Graph['Batch_num_nodes']  = Data['batch_num_nodes']
        Graph['max_num'] = Data['adj'].shape[-1]
    else:
        print('Constrct Graph Set......')

        Graph ={}
        Nodes = []
        Labels = []
        f = open(node_path, 'r')
        for i in f:
            node = i.split(' ')[0]
            label = float((i.split(' ')[1]).split('\n')[0])
            Nodes.append(node)
            Labels.append(label)
        N = len(Nodes)//batch_num
        # Nodes = Nodes[0:N*batch_num]
        # Labels = Labels[0:N * batch_num]
        Nodes = Nodes[0:max]
        Labels = Labels[0:max]
        num = 0
        A = []
        L = []
        F = []
        N = []
        Adj = []
        Label = []
        Fea = []
        Batch_num_nodes = []
        if max_n is None:
            max_num = 0
        else:
            max_num = max_n
        # n = '0x92e14a71b86d4d0c5fa37174b19f1fed5c591863'
        # adj, fea = con_graph(n, Labels[Nodes.index(n)], max_link)
        # 构建图
        for node in Nodes:
            if num == 0:
                A.append([])
                L.append([])
                F.append([])
                N.append([])
            if num < batch_num - 1:
                num += 1
            else:
                num = 0
            adj, fea, len_l = con_graph(node, Labels[Nodes.index(node)], max_link)

            node_num = adj.shape[0]
            sum_node += node_num
            sum_link += len_l
            if max_n is None:
                if node_num > max_num:
                    max_num = node_num
            if adj.shape[0] > max_num:
                adj = adj[0:max_num,0:max_num]
                fea = fea[0:max_num]
                node_num = max_num
            A[-1].append(adj)
            L[-1].append(Labels[Nodes.index(node)])
            F[-1].append(fea)
            N[-1].append(node_num)
        # 规定尺寸
        for i in range(len(A)):
            Adj.append(np.zeros((batch_num, max_num, max_num), dtype=np.float32()))
            Fea.append(np.zeros((batch_num, max_num, 2), dtype=np.float32()))
            for j in range(batch_num):
                Adj[i][j][:A[i][j].shape[0], :A[i][j].shape[0]] = A[i][j]
                Fea[i][j][:A[i][j].shape[0], :] = F[i][j][0:max_num]
            Label.append(np.array(L[i]))  # ,dtype=np.float32()
            Batch_num_nodes.append(np.array(N[i]))
            # Fea.append(np.array(F[i]))
        np.savez(save_path, adj=Adj, fea=Fea, label=Label, batch_num_nodes=Batch_num_nodes)
        Graph['Adj'] = Adj
        Graph['Fea'] = Fea
        Graph['Label'] = Label
        Graph['Batch_num_nodes'] = Batch_num_nodes
        Graph['max_num'] = max_num


    return Graph, sum_node/max, sum_link/max


def dynamic_G(node, label, max_num, his = False):
    # node = '0xd532ebeb6ab531d230b6ad18c93e265ff0101dea'
    nodes = []
    links = []
    Nodes = []
    Node = []
    Labels = []
    Label = []
    node_num = []
    adj = []
    fea = []
    dict = {}
    # f = open(all_node_path, 'r+')
    # for line in f.readlines():
    #     l = line.split(' ')
    #     if node == l[0]:
    #         label = int(l[1].split('\n')[0])
    no_zero = 0
    if label == 0:
        path1 = 'data/Normal first-order nodes/' + node + '.csv'
        path2 = 'data/Normal second-order nodes/' + node
        p = 0
    else:
        path1 = 'data/Phishing first-order nodes/' + node + '.csv'
        path2 = 'data/Phishing second-order nodes/' + node
        p = 1

    get_subgraph1(node, nodes, links, dict, path1, p)
    get_subgraph2(node, nodes, links, dict, path2, p)

    normal_time(links)
    Link = sorted(links, key=lambda x: x[3], reverse=True)
    dy_links = []
    dy_links_single = []
    # 切片
    for i in range(10):
        dy_links.append([])
        for l in Link:
            if not his:
                if l[3] <= (i + 1) * 0.1 and l[3] >= (i) * 0.1:
                    dy_links[i].append(l)
            else:
                if l[3] <= (i + 1) * 0.1:
                    dy_links[i].append(l)
    # links = multi2single(links)
    # 同一时刻求和
    for index in range(len(dy_links)):
        if len(dy_links[index]) > 0:
            dy_links_single.append(sorted(multi2single(dy_links[index]), key=lambda x: x[3], reverse=True)[:max_num])
        else:
            dy_links_single.append([])
    All_links = []
    for i in range(len(dy_links_single)):
        All_links += dy_links_single[i]

    for l in All_links:
        if l[0] not in Node:
            Node.append(l[0])
            Label.append(int(dict[l[0]]))
        if l[1] not in Node:
            Node.append(l[1])
            Label.append(int(dict[l[1]]))
    max_node = len(Node)
    # max_node = 0
    for index in range(len(dy_links_single)):
        Nodes.append([])
        Labels.append([])
        adj.append([])
        fea.append([])
        if len(dy_links_single[index]) > 0:
            no_zero +=1
            for l in dy_links_single[index]:
                if l[0] not in Nodes[index]:
                    Nodes[index].append(l[0])
                    Labels[index].append(int(dict[l[0]]))
                if l[1] not in Nodes[index]:
                    Nodes[index].append(l[1])
                    Labels[index].append(int(dict[l[1]]))
            adj[index] = np.zeros((len(Node), len(Node)), dtype=np.float32())
            fea[index] = np.zeros((len(Node), 2))
            for l in dy_links_single[index]:
                adj[index][Node.index(l[0]), Node.index(l[1])] = l[2]
            for n in range(len(Label)):
                fea[index][n][Label[n]] = 1
            # if len(Nodes[index]) > max_node:
            #
            # node_num.append(len(Node))
        else:
            # adj[index] = np.eye(max_node)
            adj[index] = np.zeros((max_node,max_node))
            fea[index] = np.zeros((max_node, 2))
            for i in range(max_node):
                fea[index][i][0] = 1
            # node_num.append(max_node)
    return adj, fea, max_node, len(links),no_zero


def con_dynamic_Gset(node_path, save_path, batch_num, max_link, max_n=None, his = False):
    if os.path.exists(save_path):
        print('G_set has already exists!')
        Graph = {}
        Data = np.load(save_path, allow_pickle=True)
        Graph['Adj'] = Data['adj']
        Graph['Fea']  = Data['fea']
        Graph['Label']  = Data['label']
        Graph['Batch_num_nodes']  = Data['batch_num_nodes']
        Graph['max_num'] = Data['adj'].shape[-1]
        Graph['link_num'] = Data['link_num']
        Graph['degree'] = Data['D']
        Graph['no_zero'] = Data['no_zero']
    else:
        print('Constrct Graph Set......')
        Graph ={}
        Nodes = []
        Labels = []
        Link_num = []
        No_zero = []
        f = open(node_path, 'r')
        for i in f:
            node = i.split(' ')[0]
            label = float((i.split(' ')[1]).split('\n')[0])
            Nodes.append(node)
            Labels.append(label)
        N = len(Nodes)//batch_num
        Nodes = Nodes[0:N * batch_num]
        Labels = Labels[0:N * batch_num]
        # Nodes = Nodes[:100]
        # Labels = Labels[:100]
        num = 0
        A = []
        L = []
        F = []
        N = []
        D =[]
        Adj = []
        Label = []
        Fea = []
        Batch_num_nodes = []
        if max_n is None:
            max_num = 0
        else:
            max_num = max_n
        # n = '0x92e14a71b86d4d0c5fa37174b19f1fed5c591863'
        # adj, fea = con_graph(n, Labels[Nodes.index(n)], max_link)
        # 构建图
        for node in Nodes:
            if num == 0:
                A.append([])
                L.append([])
                F.append([])
                N.append([])
                Adj.append([])
                Fea.append([])
            if num < batch_num - 1:
                num += 1
            else:
                num = 0
            # print(node)
            # adj, fea = con_graph(node, Labels[Nodes.index(node)], max_link)
            adj, fea, node_num, link_num, no_zero = dynamic_G(node, Labels[Nodes.index(node)], max_link, his)
            a = np.sign(adj)
            d = np.sum(a,axis=2)
            Degree = np.sum(np.sum(d, axis=0))/node_num
            Link_num.append(link_num)
            D.append(Degree)
            No_zero.append(no_zero)
            # node_num = adj.shape[0]
            batch_node_num = []
            for i in range(len(adj)):
                if max_n is None:
                    if node_num > max_num:
                        max_num = node_num
                if node_num >= max_num:
                    # print(node)

                    adj[i] = adj[i][0:max_num,0:max_num]
                    fea[i] = fea[i][0:max_num]
                    node_num = max_num
                    batch_node_num.append(max_num)
                else:
                    batch_node_num.append(node_num)
            A[-1].append(adj)


            L[-1].append(Labels[Nodes.index(node)])
            F[-1].append(fea)
            N[-1].append(batch_node_num)

        for bath in range(len(Adj)):
            for dy_time in range(len(adj)):
                Adj[bath].append([])
                Fea[bath].append([])
        # 规定尺寸

        for i in range(len(Adj)):
            for j in range(batch_num):
                for n in range(len(adj)):
                    Adj[i][j].append(np.zeros((max_num, max_num), dtype=np.float32()))
                    Fea[i][j].append(np.zeros(( max_num, 2), dtype=np.float32()))
                    # print(A[i][j][n].shape[0])
                    Adj[i][j][n][:A[i][j][n].shape[0], :A[i][j][n].shape[0]] = A[i][j][n]
                    Fea[i][j][n][:A[i][j][n].shape[0], :] = F[i][j][n][0:max_num]
                    Adj[i][j][n] = sp.csr_matrix(Adj[i][j][n])
                    Fea[i][j][n] = sp.csr_matrix(Fea[i][j][n])
            Adj[i] = np.array(Adj[i])
            Fea[i] = np.array(Fea[i])
            Label.append(np.array(L[i]))  # ,dtype=np.float32()
            Batch_num_nodes.append(np.array(N[i]))
            Batch_num_nodes[i] = np.array(Batch_num_nodes[i])
            # Fea.append(np.array(F[i]))
        np.savez(save_path, adj=Adj, fea=Fea, label=Label, batch_num_nodes=Batch_num_nodes, link_num=Link_num, D=D, no_zero=No_zero)
        Graph['Adj'] = Adj
        Graph['Fea'] = Fea
        Graph['Label'] = Label
        Graph['Batch_num_nodes'] = Batch_num_nodes
        Graph['max_num'] = max_num
        Graph['link_num'] = Link_num
        Graph['degree'] = D
        Graph['no_zero'] = No_zero

    return Graph


def dynamic_links(node, label,max_num, his = False):
    # node = '0xd532ebeb6ab531d230b6ad18c93e265ff0101dea'
    nodes = []
    links = []
    dict = {}

    if label == 0:
        path1 = 'data/Normal first-order nodes/' + node + '.csv'
        path2 = 'data/Normal second-order nodes/' + node
        p = 0
    else:
        path1 = 'data/Phishing first-order nodes/' + node + '.csv'
        path2 = 'data/Phishing second-order nodes/' + node
        p = 1

    get_subgraph1(node, nodes, links, dict, path1, p)
    get_subgraph2(node, nodes, links, dict, path2, p)

    normal_time(links)
    # Link = sorted(links, key=lambda x: x[3], reverse=True)
    # links = multi2single(links)
    Link = sorted(multi2single(links), key = lambda x: x[2], reverse=True)[:max_num]

    # normal_time(links)
    Link = sorted(links, key=lambda x: x[3], reverse=True)
    dy_links = []
    # dy_links_single = []
    # 切片
    for i in range(10):
        dy_links.append([])
        for l in Link:
            if not his:
                if l[3] <= (i + 1) * 0.1 and l[3] >= (i) * 0.1:
                    dy_links[i].append(l)
            else:
                if l[3] <= (i + 1) * 0.1:
                    dy_links[i].append(l)


    return dy_links