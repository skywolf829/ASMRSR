import torch
from typing import List, Tuple

class OctreeNode_nodata:
    def __init__(self, node_list, 
    shape : List[int],
    start_position : List[int],
    depth : int, index : int):
        self.node_list = node_list
        self.shape = shape
        self.start_position : List[int] = start_position
        self.depth : int = depth
        self.index : int = index

        self.last_loss : float = 0

    def __str__(self) -> str:
        return "{ data_shape: " + str(self.shape) + ", " + \
        "depth: " + str(self.depth) + ", " + \
        "index: " + str(self.index) + "}" 

    def split(self):
        nodes = []
        k = 0
        for x_quad_start in range(0, self.shape[2], int(self.shape[2]/2)):
            if(x_quad_start == 0):
                x_size = int(self.shape[2]/2)
            else:
                x_size = self.shape[2] - int(self.shape[2]/2)
            for y_quad_start in range(0, self.shape[3], int(self.shape[3]/2)):
                if(y_quad_start == 0):
                    y_size = int(self.shape[3]/2)
                else:
                    y_size = self.shape[3] - int(self.shape[3]/2)
                if(len(self.shape) == 5):
                    for z_quad_start in range(0, self.shape[4], int(self.shape[4]/2)):
                        if(z_quad_start == 0):
                            z_size = int(self.shape[4]/2)
                        else:
                            z_size = self.shape[4] - int(self.shape[4]/2)
                        n_quad = OctreeNode_nodata(
                            self.node_list,
                            [
                                self.shape[0], self.shape[1], 
                                x_size, y_size, z_size
                            ],
                            [
                                self.start_position[0]+x_quad_start, 
                                self.start_position[1]+y_quad_start,
                                self.start_position[2]+z_quad_start
                            ],
                            self.depth+1,
                            self.index*8 + k
                        )
                        nodes.append(n_quad)
                        k += 1     
                else:
                    n_quad = OctreeNode_nodata(
                        self.node_list,
                        [
                            self.shape[0], self.shape[1], 
                            x_size, y_size
                        ],
                        [
                            self.start_position[0]+x_quad_start, 
                            self.start_position[1]+y_quad_start,
                        ],
                        self.depth+1,
                        self.index*4 + k
                    )
                    nodes.append(n_quad)
                    k += 1       
        return nodes

    def data(self):
        if(len(self.shape) == 4):
            return self.node_list.data[:,:,
                self.start_position[0]:self.start_position[0]+self.shape[2],
                self.start_position[1]:self.start_position[1]+self.shape[3]]
        else:
            return self.node_list.data[:,:,
                self.start_position[0]:self.start_position[0]+self.shape[2],
                self.start_position[1]:self.start_position[1]+self.shape[3],
                self.start_position[2]:self.start_position[2]+self.shape[4]]

    def size(self) -> float:
        return (self.data.element_size() * self.data.numel()) / 1024.0

class OctreeNodeList:
    def __init__(self, data):
        self.node_list = [OctreeNode_nodata(self, data.shape, [0, 0], 0, 0)]
        self.data = data
        self.max_depth = 0

    def append(self, n : OctreeNode_nodata):
        self.node_list.append(n)

    def insert(self, i : int, n: OctreeNode_nodata):
        self.node_list.insert(i, n)

    def pop(self, i : int) -> OctreeNode_nodata:
        return self.node_list.pop(i)

    def remove(self, item : OctreeNode_nodata) -> bool:
        found : bool = False
        i : int = 0
        while(i < len(self.node_list) and not found):
            if(self.node_list[i] is item):
                self.node_list.pop(i)
                found = True
            i += 1
        return found

    def split(self, item : OctreeNode_nodata):
        found : bool = False
        i : int = 0
        index : int = 0
        while(i < len(self.node_list) and not found):
            if(self.node_list[i] is item):
                found = True
                index = i
            i += 1
        split_nodes = self.node_list[index].split()

        for i in range(len(split_nodes)):
            self.append(split_nodes[i])
    
    def split_index(self, ind : int):
        split_nodes = self.node_list[ind].split()

        for i in range(len(split_nodes)):
            self.append(split_nodes[i])

    def next_depth_level(self):
        node_indices_to_split = []
        for i in range(len(self.node_list)):
            if self.node_list[i].depth == self.max_depth:
                node_indices_to_split.append(i)

        for i in range(len(node_indices_to_split)):
            self.split_index(node_indices_to_split[i])
        
        self.max_depth += 1

    def split_from_error(self, loss_value = 0.01):
        node_indices_to_split = []
        for i in range(len(self.node_list)):
            if self.node_list[i].last_loss > loss_value:
                node_indices_to_split.append(i)

        for i in range(len(node_indices_to_split)):
            self.split_index(node_indices_to_split[i])

    def get_blocks_at_depth(self, depth_level):
        blocks = []
        for i in range(len(self.node_list)):
            if self.node_list[i].depth == depth_level:
                blocks.append(self.node_list[i])
        return blocks

    def depth_to_blocks_and_block_positions(self, depth_level):
        blocks = self.get_blocks_at_depth(depth_level)
        block_positions = []
        for i in range(len(blocks)):
            p = []
            for j in range(len(blocks[i].start_position)):
                p_i = blocks[i].start_position[j] / self.data.shape[2+j]
                p_i = p_i - 0.5
                p_i = p_i * 2
                p_i = p_i + (1/2**depth_level)
                p.append(p_i)
            block_positions.append(p)
        
        return blocks, block_positions

    def __len__(self) -> int:
        return len(self.node_list)

    def __getitem__(self, key : int) -> OctreeNode_nodata:
        return self.node_list[key]

    def __str__(self):
        s : str = "["
        for i in range(len(self.node_list)):
            s += str(self.node_list[i])
            if(i < len(self.node_list)-1):
                s += ", "
        s += "]"
        return s

    def total_size(self):
        nbytes = 0.0
        for i in range(len(self.node_list)):
            nbytes += self.node_list[i].size()
        return nbytes 
