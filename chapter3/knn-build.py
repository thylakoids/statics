import pprint
#  build a kd tree
class node:
    def __init__(self, point):
        self.left = None
        self.right = None
        self.point = point
        pass
    def __repr__(self):
        return pprint.pformat((self.left,self.point,self.right))

def median(lst):
    # [1 2 3 4]--->[3]
    m = len(lst) / 2
    return lst[m], m
 
def build_kdtree(data, d=0):
    try:
        k = len(data[0]) # assumes all points have the same dimension
    except IndexError as e:
        return None
    # Select axis based on depth so that axis cycles through all valid values
    axis = d%k
    # Sort point list and choose median as pivot element
    data = sorted(data, key=lambda x: x[axis])
    p, m = median(data)
    tree = node(p)
    tree.left = build_kdtree(data[:m], d+1)
    tree.right = build_kdtree(data[m+1:], d+1)
    return tree
if __name__=='__main__':
    T = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    kd_tree = build_kdtree(T)
    print kd_tree