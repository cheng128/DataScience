import sys
from collections import defaultdict

min_sup = float(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]

class fp_node():
    def __init__(self, node_name, num_count, node_parent):
        self.name = node_name
        self.count = num_count
        self.next = None
        self.tail = None
        self.parent = node_parent
        self.children = {}
            
            
def get_data(file):
    data = []
    item_count_dict = defaultdict(lambda: 0)
    with open(file, 'r') as f:
        for line in f.readlines():
            items = line.replace('\n','').split(',')
            data.append(items)
            for item in items:
                item_count_dict[item] += 1
    return data, item_count_dict

def update_header(head, item, node):
    if head[item][1] == None:
        head[item][1] = node
        head[item][1].tail = node
    else:
        head[item][1].tail.next = node
    head[item][1].tail = node
    
def insert_tree(root, items, count, head):
    if items[0] in root.children:
        root.children[items[0]].count += count
    else:
        root.children[items[0]] = fp_node(items[0], count, root)
        update_header(head, items[0], root.children[items[0]])
            
    if len(items)>1:
        insert_tree(root.children[items[0]], items[1:], count, head)
        

def construct_tree(DB, item_num, count=[1]):
    freq_item = {item[0]:item[1] for item in sorted(item_num.items(), key=lambda x:x[1], reverse=True)
                 if item[1]>=min_sup}
    if len(freq_item)==0:
        return None, None
    
    header = {item: [freq_item[item], None] for item in freq_item} 
    fp_root = fp_node('root', 0, None)
    
    if len(count) != len(DB):
        count = count*len(DB)
        
    for trans, cnt in zip(DB, count):
        item_set = [i for i in freq_item if i in trans]
        if item_set:
            insert_tree(fp_root, item_set, cnt, header)

    return fp_root, header

def ascending(node):
    temp = []
    while node.parent != None:
        temp.append(node.name)
        node = node.parent
    return temp[1:]


def find_path(head, node_name, path_list, count_list, count_dict):
    current = head[node_name][1]
    while current != None:
        path = ascending(current)
        if path:
            path_list.append(path)
            count_list.append(current.count)
            for item in path:
                count_dict[item] += current.count
        current = current.next
        
def mine_fp_tree(head_table, prefix_set, item_set):
    for s in list(head_table.keys())[::-1]:
        prefix_path, frequency = [], []
        item_count = defaultdict(lambda: 0)
        new_freq = prefix_set.copy()
        new_freq.add(int(s))
        item_set.append((sorted(new_freq), head_table[s][0]))        
        find_path(head_table, s, prefix_path, frequency, item_count)
        cond_tree, sub_header = construct_tree(prefix_path, item_count, frequency)
        if sub_header != None:
            mine_fp_tree(sub_header, new_freq, item_set)
             

main_data, item_count = get_data(input_file)
min_sup *= len(main_data)
fp_tree, header_table = construct_tree(main_data, item_count)
freq_ptn = []
mine_fp_tree(header_table, set(), freq_ptn)

freq_ptn = sorted(freq_ptn, key=lambda x:x[0])
freq_ptn = sorted(freq_ptn, key=lambda x:len(x[0]))

with open(output_file, 'w') as g:
    for ptn, sup in freq_ptn:
        ptn_str = [str(i) for i in ptn] 
        g.write(','.join(ptn_str)+':'+'%.4f'%(sup/len(main_data))+'\n')