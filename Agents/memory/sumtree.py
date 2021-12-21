import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity               # Number of leaf nodes 
        self.tree = np.zeros(2 * capacity - 1) # Total number of nodes in the tree    
        self.leaf_pointer = 0
        self.data_pointer = 0
    
    def reset(self):
        self.tree = np.zeros(2 * self.capacity - 1)  
        self.leaf_pointer = 0
        self.data_pointer = 0
    
    def add(self, priority):
        """
        Insert priority score into the sumtree
        """
        # Next available leaf node
        leaf_idx = self.data_pointer + self.capacity - 1
        
        # Update the leaf
        self.update(leaf_idx, priority)
        
        # Add 1 to data_pointer
        self.data_pointer += 1
        
        # If we reach capacity loop to begining of the leaf nodes.
        if self.data_pointer >= self.capacity:  
            self.data_pointer = 0
            
    def update(self, tree_index, priority):
        """
        Update priority score and the propagate change through tree
        """
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # then propagate the change back up through tree - takes time O(ln(capacity))
        while tree_index != 0:               
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    def get_leaf(self, priority_value):
        """
        Get leaf_index and the priority value
        """
            
        parent = 0
        
        # search for the given priority value
        while True: 
            left_child = 2 * parent + 1
            right_child = left_child + 1
            
            # If we reach a leaf node, end the search
            if len(self.tree) <= left_child:
                leaf_index = parent
                break
            
            # search for a higher priority node
            else: 
                if priority_value <= self.tree[left_child]:
                    parent = left_child
                else:
                    priority_value -= self.tree[left_child]
                    parent = right_child
            
        data_index = leaf_index - self.capacity + 1

        return leaf_index, data_index, self.tree[leaf_index]
    
    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node
 
