class BinarySearchTreeNode:
    def __init__(self, data): #initial node of the tree
        self.data = data
        self.left = None
        self.right = None
        
    def add_child(self, data):
         if data == self.data:
             return 
         
         if data < self.data:
         # add data in left subtree
            if self.left: #Checking on left side
                self.left.add_child(data)
            else:
                self.left = BinarySearchTreeNode(data)
            
         else:
            #add in right subtree
            if self.right:
                self.right.add_child(data)
            else:
                self.right = BinarySearchTreeNode(data)
                
                
    def in_order_traversal(self):
        elements = []
        
        if self.left:
            elements += self.left.in_order_traversal()
        
        #visit base node
        elements.append(self.data)
        
        #visit right subtree
        
        if self.right:
            elements += self.right.in_order_traversal()
        
        
        return elements
    
def build_tree(elements):
        root = BinarySearchTreeNode(elements[0])
        
        for i in range(1,len(elements)):
            root.add_child(elements[i])
            
        return root
    
if __name__ == '__main__':
    numbers = [17,4,1,3,5,7,8]
    numbers_tree = build_tree(numbers)
    print(numbers_tree.in_order_traversal())