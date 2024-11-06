def best_first_search(edge_list, source, target):
    open_list = []
    closed_list = []
    visited = [ False for _ in range(num_edges)]
    visited[source] = True
    closed_list.append(source)
    
    for item in edge_list[source]:
        open_list.append(item)
    open_list = sorted(open_list, key=lambda x: x[1])

    while len(open_list) != 0:
        print(source, open_list, '\t\t', closed_list)
        # print(source, visited)
        
        if source == target:
            break
        
        item = open_list.pop(0)
        # print(item)
        # break
        if visited[item[0]] == False:
            source = item[0]
            # print(item[0])
            visited[item[0]] = True
            for it in edge_list[item[0]]:
                open_list.append(it)
            # open_list.append(item)
            open_list = sorted(open_list, key=lambda x: x[1])
            closed_list.append(item[0])



def add_edge(edge_list, x, y, h_val):
    edge_list[x].append((y, h_val))
    edge_list[y].append((x, h_val))


if __name__ == "__main__":

    num_nodes = 14

    num_edges = 14
    edge_list = [[] for _ in range(num_edges)]
    source = 0
    target = 9



    
    add_edge(edge_list, 0, 1, 3)
    add_edge(edge_list, 0, 2, 6)
    add_edge(edge_list, 0, 3, 5)
    add_edge(edge_list, 1, 4, 9)
    add_edge(edge_list, 1, 5, 8)
    add_edge(edge_list, 2, 6, 12)
    add_edge(edge_list, 2, 7, 14)
    add_edge(edge_list, 3, 8, 7)
    add_edge(edge_list, 8, 9, 5)
    add_edge(edge_list, 8, 10, 6)
    add_edge(edge_list, 9, 11, 1)
    add_edge(edge_list, 9, 12, 10)
    add_edge(edge_list, 9, 13, 2)
    
    print(edge_list)
    print()
    print()
    
    best_first_search(edge_list, source, target)