from block import *
from TEGDetector import *
from data import *
def arg_parse_1():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset',
                           help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
                                  help='Name of the benchmark dataset')
    io_parser.add_argument('--pkl', dest='pkl_fname',
                           help='Name of the pkl data file')

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
                                 help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
                                 help='number of pooling layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
                        const=True, default=False,
                        help='Whether link prediction side objective is used')

    parser.add_argument('--datadir', dest='datadir',
                        help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
                        help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
                        help='CUDA.')
    parser.add_argument('--max-links', dest='max_links', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--num-node', dest='num_nodes', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
                        help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
                        help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
                        help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
                        help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
                        help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
                        help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
                        help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--no-log-graph', dest='log_graph', action='store_const',
                        const=False, default=True,
                        help='Whether disable log graph')

    parser.add_argument('--method', dest='method',
                        help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix',
                        help='suffix added to the output filename')

    parser.set_defaults(datadir='data',
                        logdir='log',
                        dataset='bmname',  # syn1v2
                        bmname='att_Ethereum_dynamic_1k_64',
                        num_nodes = 1000,
                        max_links=2000,
                        cuda='4',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=10,
                        num_epochs=200,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=64,
                        output_dim=64,
                        num_classes=2,
                        num_gc_layers=3,
                        dropout=0.0,
                        method='diffpool',
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1
                        )
    return parser.parse_args()

args = arg_parse_1()
path = os.path.join(args.logdir, gen_prefix(args))
# if os.path.isdir(path):
#     print('Remove existing log dir: ', path)
#     shutil.rmtree(path)
writer = SummaryWriter(path)
# writer = None

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
os.system('nvidia-smi')
print('CUDA', args.cuda)

# if prog_args.bmname is not None:
#     benchmark_task_val(prog_args, writer=writer)


writer.close()


path = './data/Normal first-order nodes'
path1 = './data/Phishing first-order nodes'
# all_node_path = "./graph_nodes.txt"
all_node_path = "./data/graph_nodes_2000.txt"
test_node_path = "./data/graph_test_nodes_2000.txt"
data_path = "./data/TEGs(2k).npz"
test_data_path = "./data/ori_normal_dy.npz"

node_list = get_node(path, path1, args.num_nodes, all_node_path)
test_nodes = get_node_test(path, path1, args.num_nodes, node_list, test_node_path)
# node_list =  get_node(path,path1, args.num_nodes, all_node_path)
# node = '0x92e14a71b86d4d0c5fa37174b19f1fed5c591863'
# adj1, fea1 = dynamic_G(node, 0, 2000, all_node_path)

Dynamic_Gset = con_dynamic_Gset(all_node_path,data_path, args.batch_size, args.max_links, max_n = args.max_links) #,max_n = args.max_links
train_dataset, val_dataset, test_dataset, train_num, val_num, test_num = split(Dynamic_Gset)
# max_num_nodes = Dynamic_Gset['Adj'][0][0][0].shape[0]
Dynamic_Graph_test = con_dynamic_Gset(test_node_path, test_data_path, args.batch_size, args.max_links)  #, max_n=max_num_nodes
test_batch = len(Dynamic_Graph_test['Adj'])
max_num_nodes = Dynamic_Graph_test['Adj'][0][0][0].shape[0]
test_G = split_data(Dynamic_Graph_test, 0, test_batch)
# test_G_num = len(test_G['adj'])
# train_dataset, val_dataset, test_dataset, train_num, val_num, test_num = split(Dynamic_Gset)


input_dim = 2
assign_input_dim = 2

print(max_num_nodes)
model = TEGDetector(
    max_num_nodes,
    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
    assign_input_dim=assign_input_dim).cuda()

#
_, val_accs = train_phishing_detector_dy(train_dataset, model, train_num, val_num, test_num, args, val_dataset=None, test_dataset=test_dataset,
                    writer=writer)



# mth = os.path.join(args.logdir, gen_prefix(args), str(60) + '.pth')
# checkpoint = torch.load(mth)
# model.load_state_dict(checkpoint['net'])
# acc = 0
# for i in range(10):
#     acc += evaluate_dynamic(test_G, test_batch, model, args, name='Test')
# print(acc / 10)

